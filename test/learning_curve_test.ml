open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let regression_dataset rows =
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         (Array.init rows (fun row ->
              let value = Float.of_int row in
              [| value; value *. value |])))
    ~y:
      (regression
         (Array.init rows (fun row ->
              let value = Float.of_int row in
              (0.5 *. value *. value) -. value +. 3.0)))
    ()
  |> get_data

let classification_dataset classes rows =
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         (Array.init rows (fun row ->
              let label = row mod classes in
              [|
                (Float.of_int label *. 4.0)
                +. (Float.of_int (row / classes) *. 0.01);
                Float.of_int (label * label);
              |])))
    ~y:(Target.classification (Array.init rows (fun row -> row mod classes)))
    ()
  |> get_data

let linear_pipeline () =
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let logistic_pipeline () =
  let specification = Logistic_regression.create ~c:10.0 () |> get in
  let estimator =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~predict_proba:Logistic_regression.predict_proba
      ~classes:Logistic_regression.classes specification
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let multinomial_pipeline () =
  let specification =
    Multinomial_logistic_regression.create ~c:10.0 ~max_iterations:200 () |> get
  in
  let estimator =
    Pipeline.estimator ~name:"multinomial"
      (module Multinomial_logistic_regression)
      ~predict_proba:Multinomial_logistic_regression.predict_proba
      ~classes:Multinomial_logistic_regression.classes specification
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

module Size_failing_regressor = struct
  type t = unit
  type params = unit
  type fitted = Feature_schema.t
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let clone = Fun.id
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    if Matrix.rows x = 4 then
      Error
        (Error.make ~remediation:"exercise learning-curve failure recording"
           (Error.Validation
              { name = "test estimator"; reason = "selected size failure" }))
    else Ok feature_schema

  let predict _ ~feature_schema:_ ~x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"produce finite predictions" error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted
end

let size_failing_pipeline () =
  let estimator =
    Pipeline.estimator ~name:"size-failing" (module Size_failing_regressor) ()
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let k_fold folds =
  K_fold.create ~folds () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let stratified_k_fold folds =
  Stratified_k_fold.create ~folds ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let schedule ?shuffle ?max_fits sizes =
  Learning_curve.schedule ?shuffle ?max_fits sizes |> get

let check_nested_prefix smaller larger =
  let rec matches index =
    index = Array.length smaller
    || (larger.(index) = smaller.(index) && matches (index + 1))
  in
  Alcotest.(check bool)
    "nested prefix" true
    (Array.length smaller <= Array.length larger && matches 0)

let test_regression_schedule_and_indices () =
  let original =
    [|
      Learning_curve.Fraction 0.25;
      Learning_curve.Fraction 0.5;
      Learning_curve.Fraction 1.0;
    |]
  in
  let schedule = schedule original in
  original.(0) <- Learning_curve.Count 7;
  Alcotest.(check bool)
    "schedule copied" true
    (Learning_curve.requested_sizes schedule
    = [|
        Learning_curve.Fraction 0.25;
        Learning_curve.Fraction 0.5;
        Learning_curve.Fraction 1.0;
      |]);
  let report =
    Learning_curve.Regression.evaluate ~return_indices:true ~schedule
      ~splitter:(k_fold 3)
      ~scorers:
        [| Regression_scorer.neg_mean_absolute_error; Regression_scorer.r2 () |]
      ~seed:(Seed.of_int 17) (linear_pipeline ()) (regression_dataset 12)
    |> get
  in
  let points = Learning_curve.points report in
  Alcotest.(check (array int))
    "resolved sizes" [| 2; 4; 8 |]
    (Array.map (fun point -> point.Learning_curve.training_samples) points);
  Array.iteri
    (fun point_index point ->
      let folds = Cross_validation.folds point.Learning_curve.evaluation in
      Alcotest.(check int) "fold count" 3 (Array.length folds);
      Array.iter
        (fun fold ->
          Alcotest.(check int)
            "training rows" point.Learning_curve.training_samples
            (Array.length (Option.get fold.Cross_validation.train_indices));
          Alcotest.(check int)
            "validation rows" 4
            (Array.length (Option.get fold.Cross_validation.test_indices));
          Alcotest.(check int)
            "two scorers" 2
            (Array.length fold.Cross_validation.scores);
          Alcotest.(check bool)
            "training scores returned" true
            (Array.for_all
               (fun score -> Option.is_some score.Cross_validation.train_score)
               fold.Cross_validation.scores))
        folds;
      if point_index > 0 then
        let previous =
          Cross_validation.folds
            points.(point_index - 1).Learning_curve.evaluation
        in
        Array.iteri
          (fun fold_index fold ->
            check_nested_prefix
              (Option.get previous.(fold_index).Cross_validation.train_indices)
              (Option.get fold.Cross_validation.train_indices);
            Alcotest.(check (array int))
              "validation fold is unchanged"
              (Option.get previous.(fold_index).Cross_validation.test_indices)
              (Option.get fold.Cross_validation.test_indices))
          folds)
    points

let training_index_signature report =
  Learning_curve.points report
  |> Array.map (fun point ->
      Cross_validation.folds point.Learning_curve.evaluation
      |> Array.map (fun fold -> Option.get fold.Cross_validation.train_indices))

let test_seeded_shuffle () =
  let run seed =
    Learning_curve.Regression.evaluate ~return_indices:true
      ~schedule:
        (schedule ~shuffle:true
           [| Learning_curve.Count 3; Learning_curve.Count 8 |])
      ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int seed) (linear_pipeline ()) (regression_dataset 12)
    |> get |> training_index_signature
  in
  let first = run 91 in
  let repeated = run 91 in
  let changed = run 92 in
  Alcotest.(check bool) "same seed" true (first = repeated);
  Alcotest.(check bool) "different seed" true (first <> changed);
  Array.iteri
    (fun fold_index _ ->
      check_nested_prefix first.(0).(fold_index) first.(1).(fold_index))
    first.(0)

let test_classification_families () =
  let binary =
    Learning_curve.Binary_classification.evaluate
      ~schedule:(schedule [| Learning_curve.Count 4; Learning_curve.Count 8 |])
      ~splitter:(stratified_k_fold 3)
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 4) (logistic_pipeline ())
      (classification_dataset 2 12)
    |> get |> Learning_curve.points
  in
  let multiclass =
    Learning_curve.Multiclass_classification.evaluate
      ~schedule:(schedule [| Learning_curve.Count 6; Learning_curve.Count 12 |])
      ~splitter:(stratified_k_fold 3)
      ~scorers:[| Multiclass_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 4) (multinomial_pipeline ())
      (classification_dataset 3 18)
    |> get |> Learning_curve.points
  in
  Alcotest.(check int) "binary points" 2 (Array.length binary);
  Alcotest.(check int) "multiclass points" 2 (Array.length multiclass);
  Array.iter
    (fun point ->
      Alcotest.(check int)
        "binary folds succeeded" 3
        (Cross_validation.successful_fold_count point.Learning_curve.evaluation))
    binary;
  Array.iter
    (fun point ->
      Alcotest.(check int)
        "multiclass folds succeeded" 3
        (Cross_validation.successful_fold_count point.Learning_curve.evaluation))
    multiclass

let expect_schedule_error sizes =
  match Learning_curve.schedule sizes with
  | Error error ->
      Alcotest.(check bool)
        "validation error" true
        (match Error.kind error with
        | Error.Validation _ -> true
        | Error.Data _ | Error.Shape_mismatch _
        | Error.Feature_schema_mismatch _ | Error.Numerical _
        | Error.Convergence _ | Error.Compatibility _ | Error.Artifact _
        | Error.Callback_failure _ | Error.Cancelled ->
            false)
  | Ok _ -> Alcotest.fail "invalid schedule was accepted"

let test_validation_before_fitting () =
  expect_schedule_error [||];
  expect_schedule_error [| Learning_curve.Count 0 |];
  expect_schedule_error [| Learning_curve.Fraction nan |];
  expect_schedule_error [| Learning_curve.Fraction 1.1 |];
  let dataset = regression_dataset 12 in
  let reject schedule =
    match
      Learning_curve.Regression.evaluate ~schedule ~splitter:(k_fold 3)
        ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
        ~seed:(Seed.of_int 1) (linear_pipeline ()) dataset
    with
    | Error error ->
        Alcotest.(check bool)
          "evaluation validation error" true
          (match Error.kind error with
          | Error.Validation _ -> true
          | Error.Data _ | Error.Shape_mismatch _
          | Error.Feature_schema_mismatch _ | Error.Numerical _
          | Error.Convergence _ | Error.Compatibility _ | Error.Artifact _
          | Error.Callback_failure _ | Error.Cancelled ->
              false)
    | Ok _ -> Alcotest.fail "invalid evaluated schedule was accepted"
  in
  reject (schedule [| Learning_curve.Count 4; Learning_curve.Fraction 0.5 |]);
  reject (schedule [| Learning_curve.Count 9 |]);
  reject
    (schedule ~max_fits:5 [| Learning_curve.Count 2; Learning_curve.Count 4 |])

let test_record_and_abort_failures () =
  let dataset = regression_dataset 12 in
  let schedule =
    schedule
      [|
        Learning_curve.Count 2; Learning_curve.Count 4; Learning_curve.Count 8;
      |]
  in
  let recorded =
    Learning_curve.Regression.evaluate ~failure_policy:Cross_validation.Record
      ~schedule ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int 77) (size_failing_pipeline ()) dataset
    |> get |> Learning_curve.points
  in
  Alcotest.(check int)
    "first point succeeds" 3
    (Cross_validation.successful_fold_count
       recorded.(0).Learning_curve.evaluation);
  Alcotest.(check int)
    "failed point recorded" 0
    (Cross_validation.successful_fold_count
       recorded.(1).Learning_curve.evaluation);
  Alcotest.(check int)
    "later point continues" 3
    (Cross_validation.successful_fold_count
       recorded.(2).Learning_curve.evaluation);
  match
    Learning_curve.Regression.evaluate ~failure_policy:Cross_validation.Abort
      ~schedule ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int 77) (size_failing_pipeline ()) dataset
  with
  | Ok _ -> Alcotest.fail "abort policy recorded a fitting failure"
  | Error error ->
      Alcotest.(check bool)
        "size context" true
        (List.mem (Error.Stage "learning curve with 4 training rows")
           (Error.context error))

let () =
  Alcotest.run "learning curves"
    [
      ( "evaluation",
        [
          Alcotest.test_case "regression schedule and nested folds" `Quick
            test_regression_schedule_and_indices;
          Alcotest.test_case "seeded shuffle" `Quick test_seeded_shuffle;
          Alcotest.test_case "classification families" `Quick
            test_classification_families;
        ] );
      ( "validation",
        [
          Alcotest.test_case "preflight failures" `Quick
            test_validation_before_fitting;
          Alcotest.test_case "record and abort fold failures" `Quick
            test_record_and_abort_failures;
        ] );
    ]
