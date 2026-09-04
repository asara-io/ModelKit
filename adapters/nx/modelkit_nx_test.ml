open Modelkit
open Modelkit_nx

let fail message = Alcotest.fail message

let get_ok = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let expect_kind predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> fail "expected an error"

type error_category = Validation | Shape | Data | Other

let error_category = function
  | Error.Validation _ -> Validation
  | Error.Shape_mismatch _ -> Shape
  | Error.Data _ -> Data
  | Error.Feature_schema_mismatch _ | Error.Numerical _ | Error.Convergence _
  | Error.Compatibility _ | Error.Artifact _ | Error.Cancelled ->
      Other

let has_category expected kind = error_category kind = expected

let test_features_preserve_names_and_null_identity () =
  let names = [| "temperature"; "pressure" |] in
  let x = Nx.create Nx.float64 [| 2; 2 |] [| 1.0; infinity; Float.nan; 4.0 |] in
  let mask = Nx.create Nx.bool [| 2; 2 |] [| false; true; false; false |] in
  let admitted = Modelkit_nx.features ~names ~null_mask:mask x |> get_ok in
  names.(0) <- "changed";
  Nx.set_item [ 0; 0 ] 99.0 x;
  Nx.set_item [ 0; 1 ] false mask;
  Alcotest.check
    (Alcotest.pair Alcotest.int Alcotest.int)
    "shape" (2, 2)
    (Matrix.shape admitted.matrix);
  Alcotest.check (Alcotest.float 0.0) "copied value" 1.0
    (Matrix.get admitted.matrix 0 0);
  Alcotest.check Alcotest.bool "explicit null" true
    (Null_mask.get (Option.get admitted.null_mask) 0 1);
  Alcotest.check Alcotest.bool "genuine NaN is not a null" false
    (Null_mask.get (Option.get admitted.null_mask) 1 0);
  Alcotest.check Alcotest.bool "null stored as NaN" true
    (Float.is_nan (Matrix.get admitted.matrix 0 1));
  Alcotest.check Alcotest.bool "genuine NaN retained" true
    (Float.is_nan (Matrix.get admitted.matrix 1 0));
  let feature_names = Feature_schema.names admitted.schema |> Option.get in
  Alcotest.check
    (Alcotest.array Alcotest.string)
    "ordered names"
    [| "temperature"; "pressure" |]
    (Feature_names.to_array feature_names);
  Alcotest.check Alcotest.int "value and mask reports" 2
    (List.length admitted.feature_reports);
  let value_report = List.hd admitted.feature_reports in
  Alcotest.check Alcotest.string "report dtype" "float64"
    (Conversion_report.source_dtype value_report);
  Alcotest.check Alcotest.int64 "retained float payload" 32L
    (Conversion_report.retained_payload_bytes value_report)

let test_strided_features () =
  let source = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let transposed = Nx.transpose source in
  let admitted = Modelkit_nx.features transposed |> get_ok in
  Alcotest.check
    (Alcotest.pair Alcotest.int Alcotest.int)
    "logical shape" (2, 3)
    (Matrix.shape admitted.matrix);
  Alcotest.check (Alcotest.float 0.0) "logical indexing" 5.0
    (Matrix.get admitted.matrix 0 2);
  let report = List.hd admitted.feature_reports in
  Alcotest.check
    (Alcotest.option Alcotest.bool)
    "reported strided source" (Some false)
    (Conversion_report.source_contiguous report)

let test_feature_validation () =
  let vector = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |] in
  expect_kind (has_category Validation) (Modelkit_nx.features vector);
  let x = Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |] in
  let wrong_mask = Nx.create Nx.bool [| 1; 2 |] [| false; false |] in
  expect_kind (has_category Shape)
    (Modelkit_nx.features ~null_mask:wrong_mask x);
  let infinity = Nx.create Nx.float64 [| 1; 1 |] [| infinity |] in
  expect_kind (has_category Data) (Modelkit_nx.features infinity);
  expect_kind (has_category Data) (Modelkit_nx.features ~names:[| "" |] x)

let test_targets_and_metadata () =
  let regression =
    Nx.create Nx.float64 [| 2 |] [| 1.5; 2.5 |]
    |> Modelkit_nx.regression_target |> get_ok
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "regression values" [| 1.5; 2.5 |]
    (Target.regression_values regression.value |> Vector.to_array);
  let labels = Nx.create Nx.int64 [| 3 |] [| 3L; -2L; 3L |] in
  let classification = Modelkit_nx.classification_target labels |> get_ok in
  Nx.set_item [ 0 ] 9L labels;
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "classification values" [| 3; -2; 3 |]
    (Target.classification_values classification.value);
  let int_payload = Int64.of_int (Sys.word_size / 8 * 3) in
  Alcotest.check Alcotest.int64 "classification staging payload" int_payload
    (Conversion_report.temporary_payload_bytes classification.report);
  Alcotest.check Alcotest.int64 "classification retained payload" int_payload
    (Conversion_report.retained_payload_bytes classification.report);
  let overflow = Nx.create Nx.int64 [| 1 |] [| Int64.max_int |] in
  expect_kind (has_category Validation)
    (Modelkit_nx.classification_target overflow);
  let negative_weights = Nx.create Nx.float64 [| 2 |] [| 1.; -1. |] in
  expect_kind (has_category Data) (Modelkit_nx.sample_weight negative_weights)

let test_dataset_admission () =
  let x = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 2.; 3. |] in
  let y = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
  let mask = Nx.create Nx.bool [| 2; 2 |] [| false; true; false; false |] in
  let weights = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |] in
  let groups = Nx.create Nx.int64 [| 2 |] [| 10L; 20L |] in
  let admitted =
    Modelkit_nx.classification_dataset ~names:[| "a"; "b" |]
      ~feature_null_mask:mask ~sample_weight:weights ~groups ~x ~y ()
    |> get_ok
  in
  Alcotest.check Alcotest.int "samples" 2
    (Dataset.sample_count admitted.dataset);
  Alcotest.check Alcotest.int "features" 2
    (Dataset.feature_count admitted.dataset);
  Alcotest.check Alcotest.int "reports" 5 (List.length admitted.dataset_reports);
  Alcotest.check Alcotest.int "null count" 1
    (Null_mask.null_count (Option.get admitted.feature_null_mask));
  let admitted_weights = Dataset.sample_weight admitted.dataset |> Option.get in
  Alcotest.check (Alcotest.float 0.0) "weight" 2.0
    (Sample_weight.get admitted_weights 1);
  let admitted_groups = Dataset.groups admitted.dataset |> Option.get in
  Alcotest.check Alcotest.int "group" 20 (Groups.get admitted_groups 1);
  let short_y = Nx.create Nx.int64 [| 1 |] [| 0L |] in
  expect_kind (has_category Data)
    (Modelkit_nx.classification_dataset ~x ~y:short_y ())

let test_regression_dataset () =
  let x = Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |] in
  let y = Nx.create Nx.float64 [| 2 |] [| 3.; 4. |] in
  let admitted = Modelkit_nx.regression_dataset ~x ~y () |> get_ok in
  let values =
    Dataset.target admitted.dataset
    |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "regression dataset target" [| 3.; 4. |] values

let () =
  Alcotest.run "modelkit-nx"
    [
      ( "admission",
        [
          Alcotest.test_case "features preserve identity" `Quick
            test_features_preserve_names_and_null_identity;
          Alcotest.test_case "strided features" `Quick test_strided_features;
          Alcotest.test_case "feature validation" `Quick test_feature_validation;
          Alcotest.test_case "targets and metadata" `Quick
            test_targets_and_metadata;
          Alcotest.test_case "dataset" `Quick test_dataset_admission;
          Alcotest.test_case "regression dataset" `Quick test_regression_dataset;
        ] );
    ]
