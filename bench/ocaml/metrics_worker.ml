open Modelkit

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let () =
  if Array.length Sys.argv <> 2 then fail "usage: metrics_worker SAMPLES";
  let samples = int_of_string Sys.argv.(1) in
  let allocated_before = Gc.allocated_bytes () in
  let regression_truth_values =
    Array.init samples (fun index -> Float.of_int (index mod 1000) /. 10.0)
  in
  let regression_prediction_values =
    Array.mapi
      (fun index truth -> truth +. (Float.of_int ((index mod 7) - 3) *. 0.01))
      regression_truth_values
  in
  let classification_truth_values =
    Array.init samples (fun index -> index mod 2)
  in
  let classification_prediction_values =
    Array.mapi
      (fun index truth -> if index mod 11 = 0 then 1 - truth else truth)
      classification_truth_values
  in
  let probability_values =
    Array.mapi
      (fun index truth ->
        let adjustment = Float.of_int (index mod 17) *. 0.02 in
        if truth = 1 then 0.55 +. adjustment else 0.45 -. adjustment)
      classification_truth_values
  in
  let weight_values =
    Array.init samples (fun index ->
        1.0 +. (Float.of_int (index mod 5) *. 0.25))
  in
  let regression_truth =
    Target.regression (Vector.of_array regression_truth_values) |> get_data
  in
  let regression_prediction =
    Target.regression (Vector.of_array regression_prediction_values) |> get_data
  in
  let classification_truth =
    Target.classification classification_truth_values
  in
  let classification_prediction =
    Target.classification classification_prediction_values
  in
  let positive_probabilities = Vector.of_array probability_values in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples weight_values |> get_data
  in
  let regression metric =
    metric ?sample_weight:(Some sample_weight) ~truth:regression_truth
      ~prediction:regression_prediction ()
    |> get
  in
  let labels metric =
    metric ?sample_weight:(Some sample_weight) ~truth:classification_truth
      ~prediction:classification_prediction ()
    |> get
  in
  let probabilities metric =
    metric ?sample_weight:(Some sample_weight) ~truth:classification_truth
      ~positive_probabilities ()
    |> get
  in
  let roc =
    Binary_classification_metrics.roc_curve ~sample_weight
      ~truth:classification_truth ~positive_probabilities ()
    |> get
  in
  let precision_recall =
    Binary_classification_metrics.precision_recall_curve ~sample_weight
      ~truth:classification_truth ~positive_probabilities ()
    |> get
  in
  let scalar =
    [|
      regression Regression_metrics.mean_absolute_error;
      regression Regression_metrics.mean_squared_error;
      regression Regression_metrics.root_mean_squared_error;
      regression (fun ?sample_weight ~truth ~prediction () ->
          Regression_metrics.r2 ?sample_weight ~truth ~prediction ());
      labels Binary_classification_metrics.accuracy;
      labels (fun ?sample_weight ~truth ~prediction () ->
          Binary_classification_metrics.balanced_accuracy ?sample_weight ~truth
            ~prediction ());
      labels (fun ?sample_weight ~truth ~prediction () ->
          Binary_classification_metrics.precision ?sample_weight ~truth
            ~prediction ());
      labels (fun ?sample_weight ~truth ~prediction () ->
          Binary_classification_metrics.recall ?sample_weight ~truth ~prediction
            ());
      labels (fun ?sample_weight ~truth ~prediction () ->
          Binary_classification_metrics.f1 ?sample_weight ~truth ~prediction ());
      probabilities (fun ?sample_weight ~truth ~positive_probabilities () ->
          Binary_classification_metrics.log_loss ?sample_weight ~truth
            ~positive_probabilities ());
      probabilities (fun ?sample_weight ~truth ~positive_probabilities () ->
          Binary_classification_metrics.roc_auc ?sample_weight ~truth
            ~positive_probabilities ());
    |]
  in
  let aggregate = Score_aggregation.summarize scalar |> get in
  let signature =
    Array.append scalar
      [|
        Float.of_int
          (Vector.length roc.Binary_classification_metrics.thresholds);
        Float.of_int
          (Vector.length
             precision_recall.Binary_classification_metrics.decision_thresholds);
        aggregate.Score_aggregation.mean;
        aggregate.Score_aggregation.standard_deviation;
      |]
  in
  let signature_text =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"ocaml":%S,"operations":["regression_metrics","classification_metrics","ranking_curves","score_aggregation"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words signature_text Sys.ocaml_version samples signature_text
