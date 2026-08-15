open Modelkit

let read_fixture path =
  let values = Hashtbl.create 32 in
  In_channel.with_open_text path (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if String.length line > 0 && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ name; encoded ] ->
                let parsed =
                  if String.equal encoded "" then [||]
                  else
                    encoded |> String.split_on_char ','
                    |> List.map float_of_string |> Array.of_list
                in
                Hashtbl.replace values name parsed
            | fields ->
                Alcotest.failf "invalid fixture row with %d fields"
                  (List.length fields)));
  values

let vector fixture name =
  match Hashtbl.find_opt fixture name with
  | Some values -> values
  | None -> Alcotest.failf "fixture vector %S is missing" name

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let agrees expected observed =
  if Float.is_infinite expected then expected = observed
  else
    let scale = Float.max (Float.abs expected) (Float.abs observed) in
    Float.abs (expected -. observed) <= 1e-7 *. Float.max 1.0 scale

let check_float label expected observed =
  Alcotest.(check bool) label true (agrees expected observed)

let check_vector fixture name observed =
  let expected = vector fixture name in
  Alcotest.(check int)
    (name ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      check_float (Format.sprintf "%s[%d]" name index) expected observed.(index))
    expected

let setup_regression fixture =
  let target name =
    vector fixture name |> Vector.of_array |> Target.regression |> get_data
  in
  let truth = target "regression_truth" in
  let prediction = target "regression_prediction" in
  let values = vector fixture "regression_weight" in
  let sample_weight =
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  (truth, prediction, sample_weight)

let setup_classification fixture =
  let labels name = vector fixture name |> Array.map int_of_float in
  let truth = labels "classification_truth" |> Target.classification in
  let prediction =
    labels "classification_prediction" |> Target.classification
  in
  let probabilities =
    vector fixture "positive_probability" |> Vector.of_array
  in
  let values = vector fixture "classification_weight" in
  let sample_weight =
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  (truth, prediction, probabilities, sample_weight)

let test_regression fixture () =
  let truth, prediction, sample_weight = setup_regression fixture in
  let check name result =
    check_float name (vector fixture name).(0) (get result)
  in
  check "mean_absolute_error"
    (Regression_metrics.mean_absolute_error ~sample_weight ~truth ~prediction ());
  check "mean_squared_error"
    (Regression_metrics.mean_squared_error ~sample_weight ~truth ~prediction ());
  check "root_mean_squared_error"
    (Regression_metrics.root_mean_squared_error ~sample_weight ~truth
       ~prediction ());
  check "r2" (Regression_metrics.r2 ~sample_weight ~truth ~prediction ())

let test_labels fixture () =
  let truth, prediction, _, sample_weight = setup_classification fixture in
  let check name result =
    check_float name (vector fixture name).(0) (get result)
  in
  check "accuracy"
    (Binary_classification_metrics.accuracy ~sample_weight ~truth ~prediction ());
  check "balanced_accuracy"
    (Binary_classification_metrics.balanced_accuracy ~sample_weight ~truth
       ~prediction ());
  check "precision"
    (Binary_classification_metrics.precision ~sample_weight ~truth ~prediction
       ());
  check "recall"
    (Binary_classification_metrics.recall ~sample_weight ~truth ~prediction ());
  check "f1"
    (Binary_classification_metrics.f1 ~sample_weight ~truth ~prediction ())

let test_probabilities fixture () =
  let truth, _, positive_probabilities, sample_weight =
    setup_classification fixture
  in
  check_float "log_loss"
    (vector fixture "log_loss").(0)
    (Binary_classification_metrics.log_loss ~sample_weight ~truth
       ~positive_probabilities ()
    |> get);
  check_float "roc_auc"
    (vector fixture "roc_auc").(0)
    (Binary_classification_metrics.roc_auc ~sample_weight ~truth
       ~positive_probabilities ()
    |> get)

let test_curves fixture () =
  let truth, _, positive_probabilities, sample_weight =
    setup_classification fixture
  in
  let roc =
    Binary_classification_metrics.roc_curve ~sample_weight ~truth
      ~positive_probabilities ()
    |> get
  in
  check_vector fixture "roc_thresholds"
    (Vector.to_array roc.Binary_classification_metrics.thresholds);
  check_vector fixture "false_positive_rates"
    (Vector.to_array roc.Binary_classification_metrics.false_positive_rates);
  check_vector fixture "true_positive_rates"
    (Vector.to_array roc.Binary_classification_metrics.true_positive_rates);
  let precision_recall =
    Binary_classification_metrics.precision_recall_curve ~sample_weight ~truth
      ~positive_probabilities ()
    |> get
  in
  check_vector fixture "precision_recall_thresholds"
    (Vector.to_array
       precision_recall.Binary_classification_metrics.decision_thresholds);
  check_vector fixture "precision_curve"
    (Vector.to_array precision_recall.Binary_classification_metrics.precisions);
  check_vector fixture "recall_curve"
    (Vector.to_array precision_recall.Binary_classification_metrics.recalls)

let () =
  let path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_METRICS_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_METRICS_FIXTURE is not set"
  in
  let fixture = read_fixture path in
  Alcotest.run "sklearn metric fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "regression" `Quick (test_regression fixture);
          Alcotest.test_case "binary labels" `Quick (test_labels fixture);
          Alcotest.test_case "probability metrics" `Quick
            (test_probabilities fixture);
          Alcotest.test_case "curve data" `Quick (test_curves fixture);
        ] );
    ]
