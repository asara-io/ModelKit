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

module Fold_classifier = struct
  type t = unit
  type params = unit
  type fitted = { schema : Feature_schema.t; classes : int array }
  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  let clone = Fun.id
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y () =
    let labels = Target.classification_values y in
    Array.sort Int.compare labels;
    let classes =
      Array.to_list labels |> List.sort_uniq Int.compare |> Array.of_list
    in
    Ok { schema = feature_schema; classes }

  let predict fitted ~feature_schema:_ ~x =
    Ok (Target.classification (Array.make (Matrix.rows x) fitted.classes.(0)))

  let predict_proba fitted ~feature_schema:_ ~x =
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length fitted.classes)
      (fun _ column -> if column = 0 then 1.0 else 0.0)
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide valid test features" error)

  let classes fitted = Array.copy fitted.classes
  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

let fold_classifier_pipeline () =
  let terminal =
    Pipeline.estimator ~name:"fold-classifier"
      (module Fold_classifier)
      ~predict_proba:Fold_classifier.predict_proba
      ~classes:Fold_classifier.classes ()
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

module Sometimes_failing_regressor = struct
  type t = unit
  type params = unit
  type fitted = Feature_schema.t
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let clone = Fun.id
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    if Matrix.get x 0 0 > 0.0 then
      Error
        (Error.make ~remediation:"exercise recorded fold failures"
           (Error.Validation
              { name = "test estimator"; reason = "selected fold failure" }))
    else Ok feature_schema

  let predict _ ~feature_schema:_ ~x =
    Target.regression
      (Vector.of_array
         (Array.init (Matrix.rows x) (fun row -> Matrix.get x row 0)))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide finite test features" error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted
end

let sometimes_failing_pipeline () =
  let terminal =
    Pipeline.estimator ~name:"sometimes-failing"
      (module Sometimes_failing_regressor)
      ()
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
        | Error.Callback_failure _ | Error.Cancelled ->
            false);
      Alcotest.(check bool)
        "class-order failure has fold and stage context" true
        (Error.context error = [ Error.Fold 0; Error.Stage "logistic" ])

let test_regression_out_of_fold_order () =
  let dataset =
    regression_dataset
      (Array.init 15 (fun row -> [| Float.of_int row |]))
      (Array.init 15 (fun row -> (2.0 *. Float.of_int row) +. 1.0))
  in
  let splitter =
    K_fold.create ~folds:5 ~shuffle:true ()
    |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let report =
    Cross_validation.Regression.cross_val_predict ~splitter
      ~seed:(Seed.of_int 31) (linear_pipeline ()) dataset
    |> get
  in
  let folds = Cross_validation.prediction_folds report in
  Alcotest.(check int) "prediction fold count" 5 (Array.length folds);
  Alcotest.(check int)
    "successful prediction folds" 5
    (Cross_validation.successful_prediction_fold_count report);
  Array.iteri
    (fun index (fold : _ Cross_validation.prediction_fold) ->
      Alcotest.(check int) "logical fold index" index fold.prediction_fold_index;
      check_nonnegative "fit time" fold.prediction_fit_time;
      check_nonnegative "predict time" fold.predict_time;
      match fold.prediction_result with
      | Ok prediction ->
          Alcotest.(check int)
            "fold prediction length"
            (Array.length fold.prediction_test_indices)
            (Target.length prediction)
      | Error failure -> Alcotest.fail (Error.to_string failure.error))
    folds;
  let predictions =
    Cross_validation.out_of_fold_predictions report
    |> Result.get_ok |> Target.regression_values
  in
  Array.iteri
    (fun row expected ->
      Alcotest.check (Alcotest.float 1e-9) "restored source row order" expected
        (Vector.get predictions row))
    (Target.regression_values (Dataset.target dataset) |> Vector.to_array)

let test_classification_out_of_fold_responses () =
  let binary =
    classification_dataset
      (Array.init 12 (fun row -> [| Float.of_int (row - 6) |]))
      [| 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1 |]
  in
  let binary_report =
    Cross_validation.Binary_classification.cross_val_predict
      ~response:Cross_validation.Labels ~splitter:(stratified_k_fold 3)
      ~seed:(Seed.of_int 19) (logistic_pipeline ()) binary
    |> get
  in
  let binary_labels =
    Cross_validation.out_of_fold_predictions binary_report
    |> Result.get_ok |> Multiclass_prediction.labels |> Option.get
  in
  Alcotest.(check int) "binary label rows" 12 (Target.length binary_labels);
  let multiclass =
    classification_dataset
      (Array.init 9 (fun row -> [| Float.of_int row |]))
      [| 10; 10; 10; 20; 20; 20; 30; 30; 30 |]
  in
  let report =
    Cross_validation.Multiclass_classification.cross_val_predict
      ~response:Cross_validation.Probabilities ~splitter:(k_fold 3)
      ~seed:(Seed.of_int 23)
      (fold_classifier_pipeline ())
      multiclass
    |> get
  in
  let prediction =
    Cross_validation.out_of_fold_predictions report |> Result.get_ok
  in
  Alcotest.(check (array int))
    "global class order" [| 10; 20; 30 |]
    (Multiclass_prediction.classes prediction |> Option.get);
  let probabilities =
    Multiclass_prediction.probabilities prediction |> Option.get
  in
  Array.iteri
    (fun row expected ->
      Array.iteri
        (fun column value ->
          Alcotest.check (Alcotest.float 0.0) "aligned probability" value
            (Matrix.get probabilities row column))
        expected)
    [|
      [| 0.; 1.; 0. |];
      [| 0.; 1.; 0. |];
      [| 0.; 1.; 0. |];
      [| 1.; 0.; 0. |];
      [| 1.; 0.; 0. |];
      [| 1.; 0.; 0. |];
      [| 1.; 0.; 0. |];
      [| 1.; 0.; 0. |];
      [| 1.; 0.; 0. |];
    |]

let test_out_of_fold_coverage_and_failures () =
  let dataset =
    regression_dataset
      (Array.init 9 (fun row -> [| Float.of_int row |]))
      (Array.init 9 Float.of_int)
  in
  let reject splitter =
    match
      Cross_validation.Regression.cross_val_predict ~splitter
        ~seed:(Seed.of_int 7) (linear_pipeline ()) dataset
    with
    | Ok _ -> Alcotest.fail "non-partitioning splitter was accepted"
    | Error error ->
        Alcotest.(check bool)
          "coverage validation" true
          (match Error.kind error with
          | Error.Validation _ -> true
          | Error.Data _ | Error.Shape_mismatch _
          | Error.Feature_schema_mismatch _ | Error.Numerical _
          | Error.Convergence _ | Error.Compatibility _ | Error.Artifact _
          | Error.Callback_failure _ | Error.Cancelled ->
              false)
  in
  reject
    (Holdout.create () |> get
    |> Cross_validation.target_independent_splitter (module Holdout));
  reject
    (Repeated_k_fold.create ~folds:3 ~repeats:2 ()
    |> get
    |> Cross_validation.target_independent_splitter (module Repeated_k_fold));
  let report =
    Cross_validation.Regression.cross_val_predict
      ~failure_policy:Cross_validation.Record ~splitter:(k_fold 3)
      ~seed:(Seed.of_int 7)
      (sometimes_failing_pipeline ())
      dataset
    |> get
  in
  Alcotest.(check int)
    "two folds succeed" 2
    (Cross_validation.successful_prediction_fold_count report);
  let failures =
    match Cross_validation.out_of_fold_predictions report with
    | Ok _ -> Alcotest.fail "assembled predictions ignored a failed fold"
    | Error failures -> failures
  in
  Alcotest.(check int) "one recorded failure" 1 (Array.length failures);
  Alcotest.(check bool)
    "fitting phase" true
    (match failures.(0).phase with
    | Fitting -> true
    | Materialization | Prediction _ | Scoring _ -> false);
  Alcotest.(check bool)
    "fold context" true
    (Error.context failures.(0).error
    = [ Error.Fold 0; Error.Stage "sometimes-failing" ]);
  match
    Cross_validation.Regression.cross_val_predict ~splitter:(k_fold 3)
      ~seed:(Seed.of_int 7)
      (sometimes_failing_pipeline ())
      dataset
  with
  | Ok _ -> Alcotest.fail "abort policy recorded a failed fold"
  | Error error ->
      Alcotest.(check bool)
        "lowest failed fold" true
        (Error.context error = [ Error.Fold 0; Error.Stage "sometimes-failing" ])

let () =
  Alcotest.run "cross validation"
    [
      ( "reports",
        [
          Alcotest.test_case "regression" `Quick test_regression_report;
          Alcotest.test_case "binary response dispatch" `Quick
            test_binary_response_dispatch;
          Alcotest.test_case "stable ordering" `Quick test_stable_ordering;
          Alcotest.test_case "out-of-fold regression order" `Quick
            test_regression_out_of_fold_order;
          Alcotest.test_case "out-of-fold classification responses" `Quick
            test_classification_out_of_fold_responses;
        ] );
      ( "failures",
        [
          Alcotest.test_case "record and abort" `Quick
            test_record_and_abort_failures;
          Alcotest.test_case "scorer validation" `Quick test_scorer_validation;
          Alcotest.test_case "probability class contract" `Quick
            test_probability_class_contract;
          Alcotest.test_case "out-of-fold coverage and failures" `Quick
            test_out_of_fold_coverage_and_failures;
        ] );
    ]
