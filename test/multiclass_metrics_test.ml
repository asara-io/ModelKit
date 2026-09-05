open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_shape = function
  | Error.Shape_mismatch _ -> true
  | _ -> false

let check_float message expected observed =
  Alcotest.check (Alcotest.float 1e-12) message expected observed

let classification values = Target.classification values

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let truth = classification [| 0; 0; 1; 1; 2; 2 |]
let prediction = classification [| 0; 1; 1; 1; 2; 0 |]

let test_confusion_and_averages () =
  let confusion =
    Multiclass_classification_metrics.confusion_matrix ~truth ~prediction ()
    |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.0)))
    "counts"
    [| [| 1.0; 1.0; 0.0 |]; [| 0.0; 2.0; 0.0 |]; [| 1.0; 0.0; 1.0 |] |]
    (Matrix.to_arrays confusion.Multiclass_classification_metrics.counts);
  let open Multiclass_classification_metrics in
  check_float "accuracy" (4.0 /. 6.0) (accuracy ~truth ~prediction () |> get);
  check_float "micro precision equals accuracy" (4.0 /. 6.0)
    (precision ~average:Micro ~truth ~prediction () |> get);
  check_float "macro precision"
    ((0.5 +. (2.0 /. 3.0) +. 1.0) /. 3.0)
    (precision ~average:Macro ~truth ~prediction () |> get);
  check_float "macro recall equals balanced accuracy"
    (balanced_accuracy ~truth ~prediction () |> get)
    (recall ~average:Macro ~truth ~prediction () |> get);
  check_float "weighted recall equals accuracy" (4.0 /. 6.0)
    (recall ~average:Weighted ~truth ~prediction () |> get);
  let scores = class_scores ~truth ~prediction () |> get in
  Alcotest.(check (array int)) "class order" [| 0; 1; 2 |] scores.class_labels;
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "supports" [| 2.0; 2.0; 2.0 |]
    (Vector.to_array scores.supports);
  let sample_weight = weights [| 1.0; 1.0; 1.0; 1.0; 1.0; 0.0 |] in
  check_float "zero-weight rows vanish" (4.0 /. 5.0)
    (accuracy ~sample_weight ~truth ~prediction () |> get);
  let doubled_truth = classification [| 0; 0; 1; 1; 2; 2; 0; 0 |] in
  let doubled_prediction = classification [| 0; 1; 1; 1; 2; 0; 0; 1 |] in
  let doubled = weights [| 2.0; 2.0; 1.0; 1.0; 1.0; 1.0 |] in
  check_float "integer weights match row replication"
    (f1 ~average:Weighted ~truth:doubled_truth ~prediction:doubled_prediction ()
    |> get)
    (f1 ~average:Weighted ~sample_weight:doubled ~truth ~prediction () |> get)

let test_labels_and_policies () =
  let open Multiclass_classification_metrics in
  let truth = classification [| 0; 0; 1; 1 |] in
  let prediction = classification [| 0; 0; 2; 1 |] in
  precision ~average:Macro ~truth ~prediction () |> expect_error is_validation;
  check_float "fallback treats an unpredicted class as zero" (2.0 /. 3.0)
    (precision ~undefined:Undefined_metric_policy.Use_fallback ~average:Macro
       ~truth ~prediction ()
    |> get);
  Alcotest.(check bool)
    "Return_nan propagates through macro" true
    (Float.is_nan
       (recall ~undefined:Undefined_metric_policy.Return_nan ~average:Macro
          ~truth ~prediction ()
       |> get));
  check_float "explicit labels restrict averaged precision" 1.0
    (precision ~labels:[| 0; 1 |] ~average:Macro ~truth ~prediction () |> get);
  check_float "explicit labels keep truth support for recall" 0.75
    (recall ~labels:[| 0; 1 |] ~average:Macro ~truth ~prediction () |> get);
  check_float "balanced accuracy ignores unsupported classes" 0.75
    (balanced_accuracy ~truth ~prediction () |> get);
  let restricted =
    confusion_matrix ~labels:[| 1 |] ~truth ~prediction () |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.0)))
    "rows outside the label set are dropped" [| [| 1.0 |] |]
    (Matrix.to_arrays restricted.counts);
  confusion_matrix ~labels:[||] ~truth ~prediction ()
  |> expect_error is_validation;
  confusion_matrix ~labels:[| 1; 1 |] ~truth ~prediction ()
  |> expect_error is_validation;
  accuracy ~truth ~prediction:(classification [| 0 |]) ()
  |> expect_error is_shape;
  accuracy ~truth:(classification [||]) ~prediction:(classification [||]) ()
  |> expect_error is_validation;
  balanced_accuracy
    ~truth:(classification [| 5; 5 |])
    ~prediction:(classification [| 6; 6 |])
    ()
  |> fun result ->
  check_float "balanced accuracy over one supported class" 0.0 (get result)

let test_log_loss_and_prediction () =
  let open Multiclass_classification_metrics in
  let truth = classification [| 0; 2; 1 |] in
  let classes = [| 0; 1; 2 |] in
  let probabilities =
    Matrix.of_arrays
      [| [| 0.5; 0.25; 0.25 |]; [| 0.2; 0.2; 0.6 |]; [| 0.1; 0.8; 0.1 |] |]
    |> get_data
  in
  check_float "log loss"
    (-.(Float.log 0.5 +. Float.log 0.6 +. Float.log 0.8) /. 3.0)
    (log_loss ~truth ~classes ~probabilities () |> get);
  let permuted =
    Matrix.of_arrays
      [| [| 0.25; 0.5; 0.25 |]; [| 0.6; 0.2; 0.2 |]; [| 0.1; 0.1; 0.8 |] |]
    |> get_data
  in
  check_float "class order is honoured"
    (log_loss ~truth ~classes ~probabilities () |> get)
    (log_loss ~truth ~classes:[| 2; 0; 1 |] ~probabilities:permuted () |> get);
  let certain =
    Matrix.of_arrays
      [| [| 1.0; 0.0; 0.0 |]; [| 0.0; 0.0; 1.0 |]; [| 0.0; 1.0; 0.0 |] |]
    |> get_data
  in
  Alcotest.(check bool)
    "certain correct predictions clip to a finite loss" true
    (let value = log_loss ~truth ~classes ~probabilities:certain () |> get in
     Float.is_finite value && value >= 0.0);
  log_loss ~truth ~classes:[| 0; 1 |]
    ~probabilities:
      (Matrix.of_arrays [| [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 1.0; 0.0 |] |]
      |> get_data)
    ()
  |> expect_error is_validation;
  let unnormalized =
    Matrix.of_arrays
      [| [| 0.5; 0.5; 0.5 |]; [| 0.2; 0.2; 0.6 |]; [| 0.1; 0.8; 0.1 |] |]
    |> get_data
  in
  log_loss ~truth ~classes ~probabilities:unnormalized ()
  |> expect_error is_validation;
  Multiclass_prediction.create () |> expect_error is_validation;
  Multiclass_prediction.create ~classes ~probabilities:unnormalized ()
  |> expect_error is_validation;
  Multiclass_prediction.create ~probabilities () |> expect_error is_validation;
  Multiclass_prediction.create ~labels:(classification [| 0 |]) ~classes
    ~probabilities ()
  |> expect_error is_shape;
  let both =
    Multiclass_prediction.create ~labels:truth ~classes ~probabilities () |> get
  in
  Alcotest.(check int) "prediction length" 3 (Multiclass_prediction.length both);
  Alcotest.(check (option (array int)))
    "classes are exposed" (Some classes)
    (Multiclass_prediction.classes both)

let test_scorers () =
  let open Multiclass_classification_scorer in
  let names =
    [
      (accuracy, "accuracy");
      (balanced_accuracy (), "balanced_accuracy");
      (precision (), "precision_macro");
      ( recall ~average:Multiclass_classification_metrics.Micro (),
        "recall_micro" );
      (f1 ~average:Multiclass_classification_metrics.Weighted (), "f1_weighted");
      (neg_log_loss, "neg_log_loss");
    ]
  in
  List.iter
    (fun (scorer, expected) ->
      Alcotest.(check string) expected expected (name scorer))
    names;
  Alcotest.(check bool) "label response" true (response accuracy = Labels);
  Alcotest.(check bool)
    "probability response" true
    (response neg_log_loss = Class_probabilities);
  let prediction = Multiclass_prediction.create ~labels:prediction () |> get in
  check_float "scorer accuracy" (4.0 /. 6.0)
    (score accuracy ~truth ~prediction () |> get);
  score neg_log_loss ~truth ~prediction () |> expect_error is_validation

let () =
  Alcotest.run "multiclass metrics"
    [
      ( "metrics",
        [
          Alcotest.test_case "confusion and averages" `Quick
            test_confusion_and_averages;
          Alcotest.test_case "labels and policies" `Quick
            test_labels_and_policies;
          Alcotest.test_case "log loss and predictions" `Quick
            test_log_loss_and_prediction;
        ] );
      ( "scorers",
        [ Alcotest.test_case "names and dispatch" `Quick test_scorers ] );
    ]
