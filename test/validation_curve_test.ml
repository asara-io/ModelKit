open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let regression_dataset () =
  let x =
    Array.init 15 (fun row ->
        let value = Float.of_int (row - 4) in
        [| value; value *. value |])
  in
  let y =
    [|
      15.2;
      9.1;
      5.4;
      2.8;
      1.2;
      0.7;
      1.5;
      3.4;
      6.8;
      11.1;
      16.9;
      23.7;
      31.8;
      41.0;
      51.6;
    |]
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix x)
    ~y:(regression y) ()
  |> get_data

let classification_dataset classes rows =
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         (Array.init rows (fun row ->
              let label = row mod classes in
              [|
                (Float.of_int label *. 5.0)
                +. (Float.of_int (row / classes) *. 0.02);
                Float.of_int (label * label);
              |])))
    ~y:(Target.classification (Array.init rows (fun row -> row mod classes)))
    ()
  |> get_data

type ridge_configuration = { alpha : float; fit_intercept : bool }

let ridge_pipeline configuration =
  let ( let* ) = Result.bind in
  let* scale =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* pipeline = Pipeline.add_transformer Pipeline.empty scale in
  let* specification =
    Ridge_regression.create ~alpha:configuration.alpha
      ~fit_intercept:configuration.fit_intercept ()
  in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator pipeline estimator

let ridge_curve ?max_fits values =
  Validation_curve.create ?max_fits ~name:"alpha"
    ~base:{ alpha = 1.0; fit_intercept = true }
    ~values
    ~encode:(fun value -> Grid_search.Float value)
    ~set:(fun configuration alpha -> Ok { configuration with alpha })
    ~build:ridge_pipeline ()
  |> get

module Counting_k_fold = struct
  type t = K_fold.t
  type params = K_fold.params
  type target = unit
  type rng = Rng.t

  let calls = ref 0
  let clone = K_fold.clone
  let params = K_fold.params

  let split specification ~rng ?groups ~x ~y () =
    incr calls;
    K_fold.split specification ~rng ?groups ~x ~y ()
end

let counting_splitter folds =
  Counting_k_fold.calls := 0;
  K_fold.create ~folds ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module Counting_k_fold)

let stratified_splitter folds =
  Stratified_k_fold.create ~folds ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let test_regression_points_and_shared_folds () =
  let values = [| 0.0; 0.5; 5.0 |] in
  let specification = ridge_curve values in
  values.(0) <- 99.0;
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "values copied" [| 0.0; 0.5; 5.0 |]
    (Validation_curve.parameter_values specification);
  Alcotest.(check string)
    "parameter name" "alpha"
    (Validation_curve.parameter_name specification);
  let report =
    Validation_curve.Regression.evaluate ~return_indices:true ~specification
      ~splitter:(counting_splitter 3)
      ~scorers:
        [| Regression_scorer.neg_mean_squared_error; Regression_scorer.r2 () |]
      ~seed:(Seed.of_int 41) (regression_dataset ())
    |> get
  in
  Alcotest.(check int) "splitter called once" 1 !Counting_k_fold.calls;
  let points = Validation_curve.points report in
  Alcotest.(check int) "point count" 3 (Array.length points);
  Array.iteri
    (fun index point ->
      Alcotest.(check int)
        "stable point index" index point.Validation_curve.point_index;
      Alcotest.check (Alcotest.float 0.0) "typed value"
        [| 0.0; 0.5; 5.0 |].(index)
        point.Validation_curve.parameter_value;
      Alcotest.(check bool)
        "encoded parameter" true
        (point.Validation_curve.parameter
        = {
            Grid_search.parameter_name = "alpha";
            parameter_value =
              Grid_search.Float point.Validation_curve.parameter_value;
          });
      Alcotest.(check bool)
        "nonnegative timings" true
        (point.Validation_curve.mean_fit_time >= 0.0
        && point.Validation_curve.mean_score_time >= 0.0);
      Alcotest.(check int)
        "score summaries" 2
        (Array.length point.Validation_curve.scores);
      Alcotest.(check bool)
        "training aggregates" true
        (Array.for_all
           (fun summary ->
             match summary.Grid_search.train with
             | Some result -> Result.is_ok result
             | None -> false)
           point.Validation_curve.scores);
      Alcotest.(check bool)
        "test aggregates" true
        (Array.for_all
           (fun summary -> Result.is_ok summary.Grid_search.test)
           point.Validation_curve.scores);
      let folds =
        Cross_validation.folds (Option.get point.Validation_curve.evaluation)
      in
      Alcotest.(check int) "fold count" 3 (Array.length folds);
      Array.iter
        (fun fold ->
          Alcotest.(check bool)
            "models omitted" true
            (Option.is_none fold.Cross_validation.model);
          Alcotest.(check bool)
            "train score present" true
            (Array.for_all
               (fun score -> Option.is_some score.Cross_validation.train_score)
               fold.Cross_validation.scores))
        folds;
      if index > 0 then
        let previous =
          Cross_validation.folds
            (Option.get points.(index - 1).Validation_curve.evaluation)
        in
        Array.iteri
          (fun fold_index fold ->
            Alcotest.(check (array int))
              "shared train rows"
              (Option.get previous.(fold_index).Cross_validation.train_indices)
              (Option.get fold.Cross_validation.train_indices);
            Alcotest.(check (array int))
              "shared test rows"
              (Option.get previous.(fold_index).Cross_validation.test_indices)
              (Option.get fold.Cross_validation.test_indices))
          folds)
    points;
  points.(0).Validation_curve.scores.(0) <-
    points.(0).Validation_curve.scores.(1);
  let copied = Validation_curve.points report in
  Alcotest.(check string)
    "score arrays copied" "neg_mean_squared_error"
    copied.(0).Validation_curve.scores.(0).Grid_search.scorer_name

let classifier_pipeline alpha =
  let ( let* ) = Result.bind in
  let* specification = Ridge_classifier.create ~alpha () in
  let* estimator =
    Pipeline.estimator ~name:"ridge-classifier"
      (module Ridge_classifier)
      specification
  in
  Pipeline.set_estimator Pipeline.empty estimator

let classifier_curve () =
  Validation_curve.create ~name:"alpha" ~base:1.0 ~values:[| 0.1; 2.0 |]
    ~encode:(fun value -> Grid_search.Float value)
    ~set:(fun _ value -> Ok value)
    ~build:classifier_pipeline ()
  |> get

let test_classification_families () =
  let specification = classifier_curve () in
  let binary =
    Validation_curve.Binary_classification.evaluate ~specification
      ~splitter:(stratified_splitter 3)
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 9)
      (classification_dataset 2 12)
    |> get |> Validation_curve.points
  in
  let multiclass =
    Validation_curve.Multiclass_classification.evaluate ~specification
      ~splitter:(stratified_splitter 3)
      ~scorers:[| Multiclass_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 9)
      (classification_dataset 3 18)
    |> get |> Validation_curve.points
  in
  let all_succeeded points =
    Array.for_all
      (fun point ->
        match point.Validation_curve.evaluation with
        | Some evaluation ->
            Cross_validation.successful_fold_count evaluation = 3
        | None -> false)
      points
  in
  Alcotest.(check bool) "binary points succeeded" true (all_succeeded binary);
  Alcotest.(check bool)
    "multiclass points succeeded" true (all_succeeded multiclass)

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
  validation_error
    (Validation_curve.create ~name:" " ~base:0 ~values:[| 1 |]
       ~encode:(fun value -> Grid_search.Int value)
       ~set:(fun _ value -> Ok value)
       ~build:(fun _ -> ridge_pipeline { alpha = 1.0; fit_intercept = true })
       ());
  validation_error
    (Validation_curve.create ~name:"value" ~base:0 ~values:[||]
       ~encode:(fun value -> Grid_search.Int value)
       ~set:(fun _ value -> Ok value)
       ~build:(fun _ -> ridge_pipeline { alpha = 1.0; fit_intercept = true })
       ());
  validation_error
    (Validation_curve.create ~max_fits:0 ~name:"value" ~base:0 ~values:[| 1 |]
       ~encode:(fun value -> Grid_search.Int value)
       ~set:(fun _ value -> Ok value)
       ~build:(fun _ -> ridge_pipeline { alpha = 1.0; fit_intercept = true })
       ());
  let set_calls = ref 0 and build_calls = ref 0 in
  let specification =
    Validation_curve.create ~max_fits:8 ~name:"value" ~base:0
      ~values:[| 1; 2; 3 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ value ->
        incr set_calls;
        Ok value)
      ~build:(fun value ->
        incr build_calls;
        ridge_pipeline { alpha = Float.of_int value; fit_intercept = true })
      ()
    |> get
  in
  validation_error
    (Validation_curve.Regression.evaluate ~specification
       ~splitter:(counting_splitter 3)
       ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
       ~seed:(Seed.of_int 4) (regression_dataset ()));
  Alcotest.(check int) "setters not called" 0 !set_calls;
  Alcotest.(check int) "builders not called" 0 !build_calls

let failure message =
  Error
    (Error.make ~remediation:"exercise validation-curve failure handling"
       (Error.Validation { name = "test configuration"; reason = message }))

let test_record_and_abort_configuration_failures () =
  let specification =
    Validation_curve.create ~name:"value" ~base:0 ~values:[| 1; 2; 3 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ value ->
        if value = 2 then failure "setter failure" else Ok value)
      ~build:(fun value ->
        if value = 3 then failure "builder failure"
        else ridge_pipeline { alpha = Float.of_int value; fit_intercept = true })
      ()
    |> get
  in
  let recorded =
    Validation_curve.Regression.evaluate ~failure_policy:Cross_validation.Record
      ~specification ~splitter:(counting_splitter 3)
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int 5) (regression_dataset ())
    |> get |> Validation_curve.points
  in
  Alcotest.(check bool)
    "first evaluates" true
    (Option.is_some recorded.(0).Validation_curve.evaluation);
  Alcotest.(check bool)
    "setter failure retained" true
    (Option.is_some recorded.(1).Validation_curve.build_error);
  Alcotest.(check bool)
    "builder failure retained" true
    (Option.is_some recorded.(2).Validation_curve.build_error);
  match
    Validation_curve.Regression.evaluate ~failure_policy:Cross_validation.Abort
      ~specification ~splitter:(counting_splitter 3)
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int 5) (regression_dataset ())
  with
  | Ok _ -> Alcotest.fail "abort policy retained the setter failure"
  | Error error ->
      Alcotest.(check bool)
        "candidate context" true
        (List.mem (Error.Candidate 1) (Error.context error))

let () =
  Alcotest.run "validation curves"
    [
      ( "evaluation",
        [
          Alcotest.test_case "typed points and shared folds" `Quick
            test_regression_points_and_shared_folds;
          Alcotest.test_case "classification families" `Quick
            test_classification_families;
        ] );
      ( "validation",
        [
          Alcotest.test_case "specification and fit bound" `Quick
            test_specification_and_fit_bound;
          Alcotest.test_case "record and abort configuration failures" `Quick
            test_record_and_abort_configuration_failures;
        ] );
    ]
