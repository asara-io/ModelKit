open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let regression values =
  values |> Vector.of_array |> Target.regression |> get_data

let classification values = Target.classification values

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let check_float message expected observed =
  Alcotest.check (Alcotest.float 1e-12) message expected observed

let is_validation = function
  | Error.Validation _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_shape = function
  | Error.Shape_mismatch _ -> true
  | Error.Data _ | Error.Feature_schema_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_regression_metrics () =
  let truth = regression [| 1.0; 2.0; 4.0 |] in
  let prediction = regression [| 1.0; 3.0; 2.0 |] in
  let sample_weight = weights [| 1.0; 2.0; 1.0 |] in
  Regression_metrics.mean_absolute_error ~sample_weight ~truth ~prediction ()
  |> get
  |> check_float "weighted MAE" 1.0;
  Regression_metrics.mean_squared_error ~sample_weight ~truth ~prediction ()
  |> get
  |> check_float "weighted MSE" 1.5;
  Regression_metrics.root_mean_squared_error ~sample_weight ~truth ~prediction
    ()
  |> get
  |> check_float "weighted RMSE" (Float.sqrt 1.5);
  Regression_metrics.r2 ~sample_weight ~truth ~prediction ()
  |> get
  |> check_float "weighted R-squared" (1.0 -. (6.0 /. 4.75));
  let curve = Regression_metrics.residual_curve ~truth ~prediction () |> get in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "residual predictions" [| 1.0; 3.0; 2.0 |]
    (Vector.to_array curve.Regression_metrics.predictions);
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "residual values" [| 0.0; -1.0; 2.0 |]
    (Vector.to_array curve.Regression_metrics.residuals)

let test_regression_undefined () =
  let truth = regression [| 2.0; 2.0 |] in
  let perfect = regression [| 2.0; 2.0 |] in
  let imperfect = regression [| 2.0; 3.0 |] in
  Regression_metrics.r2 ~truth ~prediction:perfect ()
  |> expect_error is_validation;
  let nan =
    Regression_metrics.r2 ~undefined:Undefined_metric_policy.Return_nan ~truth
      ~prediction:perfect ()
    |> get
  in
  Alcotest.(check bool) "NaN policy" true (Float.is_nan nan);
  Regression_metrics.r2 ~undefined:Undefined_metric_policy.Use_fallback ~truth
    ~prediction:perfect ()
  |> get
  |> check_float "perfect constant fallback" 1.0;
  Regression_metrics.r2 ~undefined:Undefined_metric_policy.Use_fallback ~truth
    ~prediction:imperfect ()
  |> get
  |> check_float "imperfect constant fallback" 0.0

let test_label_metrics () =
  let truth = classification [| 0; 0; 1; 1 |] in
  let prediction = classification [| 0; 1; 1; 1 |] in
  Binary_classification_metrics.accuracy ~truth ~prediction ()
  |> get
  |> check_float "accuracy" 0.75;
  Binary_classification_metrics.balanced_accuracy ~truth ~prediction ()
  |> get
  |> check_float "balanced accuracy" 0.75;
  Binary_classification_metrics.precision ~truth ~prediction ()
  |> get
  |> check_float "precision" (2.0 /. 3.0);
  Binary_classification_metrics.recall ~truth ~prediction ()
  |> get |> check_float "recall" 1.0;
  Binary_classification_metrics.f1 ~truth ~prediction ()
  |> get |> check_float "F1" 0.8;
  let custom_truth = classification [| -1; -1; 5; 5 |] in
  let custom_prediction = classification [| -1; 5; 5; 5 |] in
  Binary_classification_metrics.f1 ~positive_label:5 ~truth:custom_truth
    ~prediction:custom_prediction ()
  |> get
  |> check_float "custom positive label" 0.8

let test_probability_metrics_and_curves () =
  let truth = classification [| 0; 0; 1; 1 |] in
  let probabilities = Vector.of_array [| 0.1; 0.8; 0.7; 0.9 |] in
  let expected_loss =
    (-.Float.log 0.9 -. Float.log 0.2 -. Float.log 0.7 -. Float.log 0.9) /. 4.0
  in
  Binary_classification_metrics.log_loss ~truth
    ~positive_probabilities:probabilities ()
  |> get
  |> check_float "log loss" expected_loss;
  Binary_classification_metrics.roc_auc ~truth
    ~positive_probabilities:probabilities ()
  |> get |> check_float "ROC AUC" 0.75;
  let roc =
    Binary_classification_metrics.roc_curve ~truth
      ~positive_probabilities:probabilities ()
    |> get
  in
  let thresholds =
    Vector.to_array roc.Binary_classification_metrics.thresholds
  in
  Alcotest.(check bool)
    "initial ROC threshold is infinity" true
    (Float.is_infinite thresholds.(0));
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "ROC false-positive rates"
    [| 0.0; 0.0; 0.5; 0.5; 1.0 |]
    (Vector.to_array roc.Binary_classification_metrics.false_positive_rates);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "ROC true-positive rates"
    [| 0.0; 0.5; 0.5; 1.0; 1.0 |]
    (Vector.to_array roc.Binary_classification_metrics.true_positive_rates);
  let precision_recall =
    Binary_classification_metrics.precision_recall_curve ~truth
      ~positive_probabilities:probabilities ()
    |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "precision-recall thresholds" [| 0.1; 0.7; 0.8; 0.9 |]
    (Vector.to_array
       precision_recall.Binary_classification_metrics.decision_thresholds);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "precision-recall precision"
    [| 0.5; 2.0 /. 3.0; 0.5; 1.0; 1.0 |]
    (Vector.to_array precision_recall.Binary_classification_metrics.precisions);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "precision-recall recall"
    [| 1.0; 1.0; 0.5; 0.5; 0.0 |]
    (Vector.to_array precision_recall.Binary_classification_metrics.recalls)

let test_undefined_classification_metrics () =
  let truth = classification [| 0; 1 |] in
  let negative = classification [| 0; 0 |] in
  Binary_classification_metrics.precision ~truth ~prediction:negative ()
  |> expect_error is_validation;
  Binary_classification_metrics.precision
    ~undefined:Undefined_metric_policy.Use_fallback ~truth ~prediction:negative
    ()
  |> get
  |> check_float "precision fallback" 0.0;
  let single_class = classification [| 1; 1 |] in
  let probabilities = Vector.of_array [| 0.2; 0.8 |] in
  Binary_classification_metrics.roc_auc ~truth:single_class
    ~positive_probabilities:probabilities ()
  |> expect_error is_validation;
  Binary_classification_metrics.roc_auc
    ~undefined:Undefined_metric_policy.Use_fallback ~truth:single_class
    ~positive_probabilities:probabilities ()
  |> get
  |> check_float "ROC AUC fallback" 0.5

let test_scorers () =
  let regression_truth = regression [| 1.0; 3.0 |] in
  let regression_prediction = regression [| 2.0; 3.0 |] in
  Regression_scorer.score Regression_scorer.neg_mean_absolute_error
    ~truth:regression_truth ~prediction:regression_prediction ()
  |> get
  |> check_float "loss scorer is negated" (-0.5);
  Alcotest.(check string)
    "regression scorer name" "neg_mean_absolute_error"
    (Regression_scorer.name Regression_scorer.neg_mean_absolute_error);
  let truth = classification [| 0; 1 |] in
  let labels = classification [| 0; 1 |] in
  let prediction = Binary_prediction.create ~labels () |> get in
  Binary_classification_scorer.score Binary_classification_scorer.accuracy
    ~truth ~prediction ()
  |> get
  |> check_float "label scorer" 1.0;
  Binary_classification_scorer.score
    (Binary_classification_scorer.roc_auc ())
    ~truth ~prediction ()
  |> expect_error is_validation

let test_aggregation () =
  let summary = Score_aggregation.summarize [| 1.0; 2.0; 3.0 |] |> get in
  Alcotest.(check int) "count" 3 summary.Score_aggregation.count;
  check_float "mean" 2.0 summary.Score_aggregation.mean;
  check_float "population standard deviation"
    (Float.sqrt (2.0 /. 3.0))
    summary.Score_aggregation.standard_deviation;
  check_float "minimum" 1.0 summary.Score_aggregation.minimum;
  check_float "maximum" 3.0 summary.Score_aggregation.maximum;
  Score_aggregation.summarize [| 1.0; Float.nan |] |> expect_error is_validation;
  let fallback =
    Score_aggregation.summarize ~undefined:Undefined_metric_policy.Use_fallback
      [| 1.0; Float.nan |]
    |> get
  in
  check_float "NaN aggregation fallback" 0.5 fallback.Score_aggregation.mean

let test_validation () =
  let empty = regression [||] in
  Regression_metrics.mean_absolute_error ~truth:empty ~prediction:empty ()
  |> expect_error is_validation;
  let truth = regression [| 1.0; 2.0 |] in
  let short = regression [| 1.0 |] in
  Regression_metrics.mean_squared_error ~truth ~prediction:short ()
  |> expect_error is_shape;
  let labels = classification [| 0; 1 |] in
  Binary_classification_metrics.log_loss ~truth:labels
    ~positive_probabilities:(Vector.of_array [| 0.2; 1.1 |])
    ()
  |> expect_error is_validation;
  let multiclass = classification [| 0; 1; 2 |] in
  Binary_classification_metrics.accuracy ~truth:multiclass
    ~prediction:multiclass ()
  |> expect_error is_validation;
  Binary_prediction.create () |> expect_error is_validation

let () =
  Alcotest.run "metrics"
    [
      ( "regression",
        [
          Alcotest.test_case "metrics and residuals" `Quick
            test_regression_metrics;
          Alcotest.test_case "undefined R-squared" `Quick
            test_regression_undefined;
        ] );
      ( "binary classification",
        [
          Alcotest.test_case "label metrics" `Quick test_label_metrics;
          Alcotest.test_case "probability metrics and curves" `Quick
            test_probability_metrics_and_curves;
          Alcotest.test_case "undefined metrics" `Quick
            test_undefined_classification_metrics;
        ] );
      ( "workflow",
        [
          Alcotest.test_case "scorers" `Quick test_scorers;
          Alcotest.test_case "aggregation" `Quick test_aggregation;
          Alcotest.test_case "validation" `Quick test_validation;
        ] );
    ]
