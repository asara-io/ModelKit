open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let regression_dataset () =
  let rows = 24 in
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         (Array.init rows (fun row ->
              let value = Float.of_int row in
              [| value; sin value |])))
    ~y:
      (regression
         (Array.init rows (fun row ->
              let value = Float.of_int row in
              (2.5 *. value) +. (0.2 *. sin value))))
    ()
  |> get_data

let classification_dataset classes =
  let rows = classes * 12 in
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         (Array.init rows (fun row ->
              let label = row mod classes in
              [|
                (Float.of_int label *. 5.0)
                +. (Float.of_int (row / classes) *. 0.01);
                Float.of_int (label * label);
              |])))
    ~y:(Target.classification (Array.init rows (fun row -> row mod classes)))
    ()
  |> get_data

let ridge_pipeline () =
  let ( let* ) = Result.bind in
  let* scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* pipeline = Pipeline.add_transformer Pipeline.empty scaler in
  let* specification = Ridge_regression.create ~alpha:0.1 () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator pipeline estimator

let classifier_pipeline () =
  let ( let* ) = Result.bind in
  let* specification = Ridge_classifier.create ~alpha:1.0 () in
  let* estimator =
    Pipeline.estimator ~name:"ridge-classifier"
      (module Ridge_classifier)
      specification
  in
  Pipeline.set_estimator Pipeline.empty estimator

let k_fold folds =
  K_fold.create ~folds () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let stratified folds =
  Stratified_k_fold.create ~folds ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let test_regression_report_and_determinism () =
  let specification = Permutation_test.create ~permutations:12 () |> get in
  Alcotest.(check int)
    "permutation count" 12
    (Permutation_test.permutation_count specification);
  Alcotest.(check (option int))
    "no fit ceiling" None
    (Permutation_test.max_fits specification);
  let evaluate () =
    Permutation_test.Regression.evaluate ~return_indices:true ~specification
      ~splitter:(k_fold 4) ~scorer:Regression_scorer.neg_mean_squared_error
      ~seed:(Seed.of_int 501)
      (ridge_pipeline () |> get)
      (regression_dataset ())
    |> get
  in
  let report = evaluate () in
  let scores = Permutation_test.permutation_scores report in
  Alcotest.(check int) "permutation scores" 12 (Array.length scores);
  Alcotest.(check bool)
    "observed signal exceeds null scores" true
    (Array.for_all
       (fun score -> Permutation_test.observed_score report > score)
       scores);
  Alcotest.check (Alcotest.float 0.0) "corrected p-value" (1.0 /. 13.0)
    (Permutation_test.p_value report);
  let folds =
    Permutation_test.observed_evaluation report |> Cross_validation.folds
  in
  Alcotest.(check int) "observed folds" 4 (Array.length folds);
  Array.iter
    (fun fold ->
      Alcotest.(check bool)
        "models omitted" true
        (Option.is_none fold.Cross_validation.model);
      Alcotest.(check bool)
        "indices retained" true
        (Option.is_some fold.Cross_validation.train_indices
        && Option.is_some fold.Cross_validation.test_indices))
    folds;
  scores.(0) <- infinity;
  Alcotest.(check bool)
    "score array copied" true
    (Float.is_finite (Permutation_test.permutation_scores report).(0));
  let repeated = evaluate () in
  Alcotest.(check bool)
    "fixed-seed scores" true
    (Permutation_test.observed_score report
     = Permutation_test.observed_score repeated
    && Permutation_test.permutation_scores report
       = Permutation_test.permutation_scores repeated
    && Permutation_test.p_value report = Permutation_test.p_value repeated)

module Group_guard = struct
  type t = unit
  type params = unit
  type fitted = Feature_schema.t
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let fit_calls = ref 0
  let clone = Fun.id
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y () =
    incr fit_calls;
    let values = Target.regression_values y in
    let rec validate row =
      if row = Matrix.rows x then Ok feature_schema
      else
        let feature_group = int_of_float (Matrix.get x row 0) in
        let target_group = int_of_float (Vector.get values row /. 100.0) in
        if feature_group = target_group then validate (row + 1)
        else
          Error
            (Error.make
               ~remediation:"permute target values only within dataset groups"
               (Error.Validation
                  {
                    name = "group permutation";
                    reason = "a target crossed a group boundary";
                  }))
    in
    validate 0

  let predict _ ~feature_schema:_ ~x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"return finite predictions" error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted
end

let group_guard_pipeline () =
  Pipeline.estimator ~name:"group-guard" (module Group_guard) ()
  |> get
  |> Pipeline.set_estimator Pipeline.empty
  |> get

let grouped_dataset ?(with_groups = true) () =
  let rows = 12 in
  let groups =
    Groups.create ~expected_length:rows (Array.init rows (fun row -> row / 4))
    |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite
    ?groups:(if with_groups then Some groups else None)
    ~x:(matrix (Array.init rows (fun row -> [| Float.of_int (row / 4) |])))
    ~y:
      (regression
         (Array.init rows (fun row ->
              Float.of_int ((row / 4 * 100) + (row mod 4)))))
    ()
  |> get_data

let test_within_group_permutations () =
  Group_guard.fit_calls := 0;
  let permutations = 8 in
  let specification = Permutation_test.create ~permutations () |> get in
  let report =
    Permutation_test.Regression.evaluate ~specification ~splitter:(k_fold 3)
      ~scorer:Regression_scorer.neg_mean_squared_error ~seed:(Seed.of_int 91)
      (group_guard_pipeline ()) (grouped_dataset ())
    |> get
  in
  Alcotest.(check int)
    "every observed and permuted fold fitted"
    ((permutations + 1) * 3)
    !Group_guard.fit_calls;
  Alcotest.(check int)
    "all permutations scored" permutations
    (Array.length (Permutation_test.permutation_scores report))

let test_permutation_failure_is_typed () =
  let specification = Permutation_test.create ~permutations:3 () |> get in
  match
    Permutation_test.Regression.evaluate ~specification ~splitter:(k_fold 3)
      ~scorer:Regression_scorer.neg_mean_squared_error ~seed:(Seed.of_int 91)
      (group_guard_pipeline ())
      (grouped_dataset ~with_groups:false ())
  with
  | Ok _ -> Alcotest.fail "global target shuffle unexpectedly preserved groups"
  | Error error ->
      Alcotest.(check bool)
        "permutation context" true
        (List.exists
           (function
             | Error.Stage "permutation 0" -> true
             | Error.Stage _ | Error.Fold _ | Error.Candidate _
             | Error.Feature _ ->
                 false)
           (Error.context error))

let test_classification_families () =
  let specification = Permutation_test.create ~permutations:3 () |> get in
  let binary =
    Permutation_test.Binary_classification.evaluate ~specification
      ~splitter:(stratified 3) ~scorer:Binary_classification_scorer.accuracy
      ~seed:(Seed.of_int 18)
      (classifier_pipeline () |> get)
      (classification_dataset 2)
    |> get
  in
  let multiclass =
    Permutation_test.Multiclass_classification.evaluate ~specification
      ~splitter:(stratified 3) ~scorer:Multiclass_classification_scorer.accuracy
      ~seed:(Seed.of_int 18)
      (classifier_pipeline () |> get)
      (classification_dataset 3)
    |> get
  in
  List.iter
    (fun report ->
      Alcotest.(check int)
        "classification permutations" 3
        (Array.length (Permutation_test.permutation_scores report));
      Alcotest.(check bool)
        "valid p-value" true
        (Permutation_test.p_value report >= 0.25
        && Permutation_test.p_value report <= 1.0))
    [ binary; multiclass ]

let validation_error = function
  | Error error -> (
      match Error.kind error with
      | Error.Validation _ -> ()
      | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
      | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
      | Error.Artifact _ | Error.Callback_failure _ | Error.Cancelled ->
          Alcotest.fail (Error.to_string error))
  | Ok _ -> Alcotest.fail "expected validation error"

let test_specification_and_fit_bound () =
  validation_error (Permutation_test.create ~permutations:0 ());
  validation_error (Permutation_test.create ~max_fits:0 ());
  Group_guard.fit_calls := 0;
  let specification =
    Permutation_test.create ~permutations:3 ~max_fits:11 () |> get
  in
  validation_error
    (Permutation_test.Regression.evaluate ~specification ~splitter:(k_fold 3)
       ~scorer:Regression_scorer.neg_mean_squared_error ~seed:(Seed.of_int 8)
       (group_guard_pipeline ()) (grouped_dataset ()));
  Alcotest.(check int) "fit limit is preflighted" 0 !Group_guard.fit_calls

let () =
  Alcotest.run "permutation significance tests"
    [
      ( "evaluation",
        [
          Alcotest.test_case "regression report and determinism" `Quick
            test_regression_report_and_determinism;
          Alcotest.test_case "within-group shuffling" `Quick
            test_within_group_permutations;
          Alcotest.test_case "typed permutation failure" `Quick
            test_permutation_failure_is_typed;
          Alcotest.test_case "classification families" `Quick
            test_classification_families;
          Alcotest.test_case "specification and fit bound" `Quick
            test_specification_and_fit_bound;
        ] );
    ]
