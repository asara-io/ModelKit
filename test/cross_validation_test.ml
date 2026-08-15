open Modelkit
open Cross_validation

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let classification values = Target.classification values

let regression_dataset values targets =
  Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix values)
    ~y:(regression targets) ()
  |> get_data

let classification_dataset values targets =
  Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix values)
    ~y:(classification targets) ()
  |> get_data

let linear_pipeline () =
  let terminal =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

let logistic_pipeline () =
  let specification = Logistic_regression.create ~c:10.0 () |> get in
  let terminal =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~decision_function:Logistic_regression.decision_function
      ~predict_proba:Logistic_regression.predict_proba
      ~classes:Logistic_regression.classes specification
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

let logistic_pipeline_without_classes () =
  let specification = Logistic_regression.create ~c:10.0 () |> get in
  let terminal =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~predict_proba:Logistic_regression.predict_proba specification
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

let k_fold folds =
  K_fold.create ~folds () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let stratified_k_fold folds =
  Stratified_k_fold.create ~folds ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let result_value message = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (message ^ ": " ^ Error.to_string error)
  | None -> Alcotest.fail (message ^ ": missing result")

let check_nonnegative message value =
  Alcotest.(check bool) message true (value >= 0.0)

let test_regression_report () =
  let dataset =
    regression_dataset
      (Array.init 12 (fun row -> [| Float.of_int row |]))
      (Array.init 12 (fun row -> (2.0 *. Float.of_int row) +. 1.0))
  in
  let report =
    Cross_validation.Regression.cross_validate ~return_train_score:true
      ~return_models:true ~return_indices:true ~splitter:(k_fold 3)
      ~scorers:
        [| Regression_scorer.neg_mean_absolute_error; Regression_scorer.r2 () |]
      ~seed:(Seed.of_int 42) (linear_pipeline ()) dataset
    |> get
  in
  let folds = Cross_validation.folds report in
  Alcotest.(check int) "fold count" 3 (Array.length folds);
  Alcotest.(check int)
    "successful folds" 3
    (Cross_validation.successful_fold_count report);
  Array.iteri
    (fun index (fold : _ Cross_validation.fold) ->
      Alcotest.(check int) "stable fold index" index fold.fold_index;
      check_nonnegative "fit time" fold.fit_time;
      check_nonnegative "score time" fold.score_time;
      Alcotest.(check int) "scorer count" 2 (Array.length fold.scores);
      Alcotest.(check string)
        "scorer order" "neg_mean_absolute_error" fold.scores.(0).name;
      Alcotest.(check string) "scorer order" "r2" fold.scores.(1).name;
      Alcotest.check (Alcotest.float 1e-10) "test MAE" 0.0
        (result_value "test MAE" fold.scores.(0).test_score);
      Alcotest.check (Alcotest.float 1e-10) "train MAE" 0.0
        (result_value "train MAE" fold.scores.(0).train_score);
      Alcotest.check (Alcotest.float 1e-10) "test R-squared" 1.0
        (result_value "test R-squared" fold.scores.(1).test_score);
      Alcotest.(check bool) "model retained" true (Option.is_some fold.model);
      Alcotest.(check int)
        "train index count" 8
        (Array.length (Option.get fold.train_indices));
      Alcotest.(check int)
        "test index count" 4
        (Array.length (Option.get fold.test_indices));
      Alcotest.(check int) "no failures" 0 (Array.length fold.failures))
    folds

let test_binary_response_dispatch () =
  let values =
    [|
      [| -6.0 |];
      [| -5.0 |];
      [| -4.0 |];
      [| -3.0 |];
      [| -2.0 |];
      [| -1.0 |];
      [| 1.0 |];
      [| 2.0 |];
      [| 3.0 |];
      [| 4.0 |];
      [| 5.0 |];
      [| 6.0 |];
    |]
  in
  let dataset =
    classification_dataset values [| 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1 |]
  in
  let report =
    Cross_validation.Binary_classification.cross_validate
      ~return_train_score:true ~splitter:(stratified_k_fold 3)
      ~scorers:
        [|
          Binary_classification_scorer.accuracy;
          Binary_classification_scorer.roc_auc ();
        |]
      ~seed:(Seed.of_int 17) (logistic_pipeline ()) dataset
    |> get
  in
  let folds = Cross_validation.folds report in
  Alcotest.(check int) "binary fold count" 3 (Array.length folds);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      Alcotest.(check string)
        "label scorer first" "accuracy" fold.scores.(0).name;
      Alcotest.(check string)
        "probability scorer second" "roc_auc" fold.scores.(1).name;
      let accuracy = result_value "accuracy" fold.scores.(0).test_score in
      Alcotest.(check bool)
        "fold accuracy is valid" true
        (Float.is_finite accuracy && accuracy >= 0.0 && accuracy <= 1.0);
      Alcotest.check (Alcotest.float 1e-12) "perfect fold ROC AUC" 1.0
        (result_value "ROC AUC" fold.scores.(1).test_score);
      Alcotest.(check bool)
        "models omitted by default" true
        (Option.is_none fold.model);
      Alcotest.(check bool)
        "indices omitted by default" true
        (Option.is_none fold.train_indices && Option.is_none fold.test_indices))
    folds

let test_record_and_abort_failures () =
  let dataset =
    regression_dataset
      [| [| 0.0 |]; [| 1.0 |]; [| 2.0 |]; [| 3.0 |] |]
      [| 1.0; 3.0; 5.0; 7.0 |]
  in
  let scorer = Regression_scorer.r2 () in
  let recorded =
    Cross_validation.Regression.cross_validate
      ~failure_policy:Cross_validation.Record ~splitter:(k_fold 4)
      ~scorers:[| scorer |] ~seed:(Seed.of_int 5) (linear_pipeline ()) dataset
    |> get
  in
  let folds = Cross_validation.folds recorded in
  Alcotest.(check int) "all folds retained" 4 (Array.length folds);
  Alcotest.(check int)
    "no successful folds" 0
    (Cross_validation.successful_fold_count recorded);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      (match fold.scores.(0).test_score with
      | Some (Error _) -> ()
      | None | Some (Ok _) ->
          Alcotest.fail "undefined R-squared was not recorded");
      Alcotest.(check bool)
        "structured scorer failure" true
        (Array.exists
           (fun failure ->
             match failure.Cross_validation.phase with
             | Cross_validation.Materialization | Cross_validation.Fitting ->
                 false
             | Cross_validation.Prediction Cross_validation.Train
             | Cross_validation.Prediction Cross_validation.Test ->
                 false
             | Cross_validation.Scoring
                 { partition = Cross_validation.Train; scorer = _ } ->
                 false
             | Cross_validation.Scoring
                 { partition = Cross_validation.Test; scorer } ->
                 String.equal scorer "r2")
           fold.failures))
    folds;
  match
    Cross_validation.Regression.cross_validate ~splitter:(k_fold 4)
      ~scorers:[| scorer |] ~seed:(Seed.of_int 5) (linear_pipeline ()) dataset
  with
  | Ok _ -> Alcotest.fail "abort policy accepted a failed scorer"
  | Error error ->
      Alcotest.(check bool)
        "lowest fold context" true
        (Error.context error = [ Error.Fold 0; Error.Stage "r2" ])

let test_stable_ordering () =
  let dataset =
    regression_dataset
      (Array.init 9 (fun row -> [| Float.of_int row |]))
      (Array.init 9 (fun row -> Float.of_int row))
  in
  let run () =
    Cross_validation.Regression.cross_validate ~return_indices:true
      ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 99) (linear_pipeline ()) dataset
    |> get |> Cross_validation.folds
  in
  let first = run () in
  let second = run () in
  Array.iteri
    (fun index (fold : _ Cross_validation.fold) ->
      Alcotest.(check (array int))
        "repeatable train indices"
        (Option.get fold.Cross_validation.train_indices)
        (Option.get second.(index).train_indices);
      Alcotest.(check (array int))
        "repeatable test indices"
        (Option.get fold.test_indices)
        (Option.get second.(index).test_indices);
      Alcotest.check (Alcotest.float 0.0) "repeatable score"
        (result_value "first score" fold.scores.(0).test_score)
        (result_value "second score" second.(index).scores.(0).test_score))
    first

let test_scorer_validation () =
  let dataset = regression_dataset [| [| 0.0 |]; [| 1.0 |] |] [| 0.0; 1.0 |] in
  let expect_error scorers =
    match
      Cross_validation.Regression.cross_validate ~splitter:(k_fold 2) ~scorers
        ~seed:(Seed.of_int 1) (linear_pipeline ()) dataset
    with
    | Error _ -> ()
    | Ok _ -> Alcotest.fail "invalid scorer set was accepted"
  in
  expect_error [||];
  expect_error
    [|
      Regression_scorer.neg_mean_absolute_error;
      Regression_scorer.neg_mean_absolute_error;
    |]

let test_probability_class_contract () =
  let dataset =
    classification_dataset
      [| [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |] |]
      [| 0; 0; 1; 1 |]
  in
  match
    Cross_validation.Binary_classification.cross_validate
      ~splitter:(stratified_k_fold 2)
      ~scorers:[| Binary_classification_scorer.roc_auc () |]
      ~seed:(Seed.of_int 1)
      (logistic_pipeline_without_classes ())
      dataset
  with
  | Ok _ -> Alcotest.fail "probability scoring accepted an unknown class order"
  | Error error ->
      Alcotest.(check bool)
        "class-order failure is compatible" true
        (match Error.kind error with
        | Error.Compatibility _ -> true
        | Error.Data _ | Error.Shape_mismatch _
        | Error.Feature_schema_mismatch _ | Error.Validation _
        | Error.Numerical _ | Error.Convergence _ | Error.Artifact _
        | Error.Cancelled ->
            false);
      Alcotest.(check bool)
        "class-order failure has fold and stage context" true
        (Error.context error = [ Error.Fold 0; Error.Stage "logistic" ])

let () =
  Alcotest.run "cross validation"
    [
      ( "reports",
        [
          Alcotest.test_case "regression" `Quick test_regression_report;
          Alcotest.test_case "binary response dispatch" `Quick
            test_binary_response_dispatch;
          Alcotest.test_case "stable ordering" `Quick test_stable_ordering;
        ] );
      ( "failures",
        [
          Alcotest.test_case "record and abort" `Quick
            test_record_and_abort_failures;
          Alcotest.test_case "scorer validation" `Quick test_scorer_validation;
          Alcotest.test_case "probability class contract" `Quick
            test_probability_class_contract;
        ] );
    ]
