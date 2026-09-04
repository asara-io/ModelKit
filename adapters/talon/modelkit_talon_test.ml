open Modelkit
open Modelkit_talon

let fail message = Alcotest.fail message

let get_ok = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let expect_kind predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> fail "expected an error"

type error_category = Validation | Data | Other

let error_category = function
  | Error.Validation _ -> Validation
  | Error.Data _ -> Data
  | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _ | Error.Numerical _
  | Error.Convergence _ | Error.Compatibility _ | Error.Artifact _
  | Error.Cancelled ->
      Other

let has_category expected kind = error_category kind = expected

let contains fragment text =
  let pattern = Str.regexp_string fragment in
  match Str.search_forward pattern text 0 with
  | _ -> true
  | exception Not_found -> false

let validation_reason_contains fragment kind =
  match kind with
  | Error.Validation { reason; _ } -> contains fragment reason
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let frame () =
  Talon.create
    [
      ("id", Talon.Col.int64 [| 7L; 8L; 9L |]);
      ("temperature", Talon.Col.float64 [| 1.0; Float.nan; 3.0 |]);
      ("pressure", Talon.Col.float64_opt [| Some 10.0; Some 20.0; None |]);
      ("label", Talon.Col.int64 [| 0L; 1L; 0L |]);
      ("weight", Talon.Col.float64 [| 1.0; 2.0; 0.5 |]);
      ("group", Talon.Col.int64 [| 100L; 100L; 200L |]);
      ("city", Talon.Col.string [| "a"; "b"; "c" |]);
      ("sparse_label", Talon.Col.int64_opt [| Some 0L; None; Some 1L |]);
      ("bad_weight", Talon.Col.float64 [| 1.0; -1.0; 1.0 |]);
      ("unbounded", Talon.Col.float64 [| 1.0; infinity; 1.0 |]);
      ("narrow", Talon.Col.float32 [| 1.0; 2.0; 3.0 |]);
    ]

let test_features_follow_selection_order () =
  let frame = frame () in
  let admitted =
    Modelkit_talon.features frame [ "pressure"; "temperature" ] |> get_ok
  in
  Alcotest.check
    (Alcotest.pair Alcotest.int Alcotest.int)
    "shape" (3, 2)
    (Matrix.shape admitted.matrix);
  let names = Feature_schema.names admitted.schema |> Option.get in
  Alcotest.check
    (Alcotest.array Alcotest.string)
    "selection order becomes feature order"
    [| "pressure"; "temperature" |]
    (Feature_names.to_array names);
  Alcotest.check (Alcotest.float 0.0) "pressure first" 20.0
    (Matrix.get admitted.matrix 1 0);
  Alcotest.check (Alcotest.float 0.0) "temperature second" 3.0
    (Matrix.get admitted.matrix 2 1);
  let mask = Option.get admitted.null_mask in
  Alcotest.check Alcotest.bool "explicit null" true (Null_mask.get mask 2 0);
  Alcotest.check Alcotest.bool "null stored as NaN" true
    (Float.is_nan (Matrix.get admitted.matrix 2 0));
  Alcotest.check Alcotest.bool "genuine NaN is not a null" false
    (Null_mask.get mask 1 1);
  Alcotest.check Alcotest.bool "genuine NaN retained" true
    (Float.is_nan (Matrix.get admitted.matrix 1 1));
  Alcotest.check Alcotest.int "one null overall" 1 (Null_mask.null_count mask);
  Alcotest.check Alcotest.int "value and mask reports" 2
    (List.length admitted.feature_reports);
  let value_report = List.hd admitted.feature_reports in
  Alcotest.check Alcotest.string "report dtype" "float64"
    (Conversion_report.source_dtype value_report);
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "report shape" [| 3; 2 |]
    (Conversion_report.source_shape value_report);
  Alcotest.check Alcotest.int64 "retained float payload" 48L
    (Conversion_report.retained_payload_bytes value_report);
  Alcotest.check Alcotest.int64 "no staging payload" 0L
    (Conversion_report.temporary_payload_bytes value_report);
  Alcotest.check
    (Alcotest.option Alcotest.bool)
    "column assembly has no single-source contiguity" None
    (Conversion_report.source_contiguous value_report);
  let mask_report = List.nth admitted.feature_reports 1 in
  Alcotest.check Alcotest.string "mask dtype" "bool"
    (Conversion_report.source_dtype mask_report);
  Alcotest.check Alcotest.int64 "mask payload"
    (Int64.of_int (Sys.word_size / 8 * 6))
    (Conversion_report.retained_payload_bytes mask_report)

let test_features_copy_and_unmasked_columns () =
  let frame = frame () in
  let source =
    Talon.get_column_exn frame "temperature"
    |> Talon.Col.to_tensor Nx.float64
    |> Option.get
  in
  let admitted = Modelkit_talon.features frame [ "temperature" ] |> get_ok in
  Nx.set_item [ 0 ] 99.0 source;
  Alcotest.check (Alcotest.float 0.0) "admitted storage is independent" 1.0
    (Matrix.get admitted.matrix 0 0);
  Alcotest.check Alcotest.bool "no mask without nullable columns" true
    (Option.is_none admitted.null_mask);
  Alcotest.check Alcotest.int "single report" 1
    (List.length admitted.feature_reports)

let test_feature_validation () =
  let frame = frame () in
  expect_kind (has_category Validation) (Modelkit_talon.features frame []);
  expect_kind
    (validation_reason_contains "\"missing\"")
    (Modelkit_talon.features frame [ "missing" ]);
  expect_kind
    (validation_reason_contains "selected more than once")
    (Modelkit_talon.features frame [ "temperature"; "temperature" ]);
  expect_kind
    (validation_reason_contains "has int64 but float64 is required")
    (Modelkit_talon.features frame [ "temperature"; "id" ]);
  expect_kind
    (validation_reason_contains "has float32")
    (Modelkit_talon.features frame [ "narrow" ]);
  expect_kind
    (validation_reason_contains "has string")
    (Modelkit_talon.features frame [ "city" ]);
  expect_kind (has_category Data)
    (Modelkit_talon.features frame [ "unbounded" ])

let test_targets_and_metadata () =
  let frame = frame () in
  let regression = Modelkit_talon.regression_target frame "weight" |> get_ok in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "regression values" [| 1.0; 2.0; 0.5 |]
    (Target.regression_values regression.value |> Vector.to_array);
  Alcotest.check
    (Alcotest.option Alcotest.bool)
    "column tensor contiguity reported" (Some true)
    (Conversion_report.source_contiguous regression.report);
  Alcotest.check Alcotest.int64 "regression payload" 24L
    (Conversion_report.retained_payload_bytes regression.report);
  let classification =
    Modelkit_talon.classification_target frame "label" |> get_ok
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "classification values" [| 0; 1; 0 |]
    (Target.classification_values classification.value);
  let int_payload = Int64.of_int (Sys.word_size / 8 * 3) in
  Alcotest.check Alcotest.int64 "classification staging payload" int_payload
    (Conversion_report.temporary_payload_bytes classification.report);
  let groups = Modelkit_talon.groups frame "group" |> get_ok in
  Alcotest.check Alcotest.int "group value" 200 (Groups.get groups.value 2);
  let weights = Modelkit_talon.sample_weight frame "weight" |> get_ok in
  Alcotest.check (Alcotest.float 0.0) "weight value" 0.5
    (Sample_weight.get weights.value 2);
  expect_kind
    (validation_reason_contains "contains 1 null values")
    (Modelkit_talon.classification_target frame "sparse_label");
  expect_kind
    (validation_reason_contains "contains 1 null values")
    (Modelkit_talon.regression_target frame "pressure");
  expect_kind
    (validation_reason_contains "contains 1 null values")
    (Modelkit_talon.sample_weight frame "pressure");
  expect_kind
    (validation_reason_contains "contains 1 null values")
    (Modelkit_talon.groups frame "sparse_label");
  expect_kind (has_category Data)
    (Modelkit_talon.regression_target frame "temperature");
  expect_kind (has_category Data)
    (Modelkit_talon.sample_weight frame "bad_weight");
  expect_kind
    (validation_reason_contains "has float64 but int64 is required")
    (Modelkit_talon.classification_target frame "weight");
  expect_kind
    (validation_reason_contains "has int64 but float64 is required")
    (Modelkit_talon.sample_weight frame "group");
  let overflow =
    Talon.create [ ("label", Talon.Col.int64 [| Int64.max_int |]) ]
  in
  expect_kind (has_category Validation)
    (Modelkit_talon.classification_target overflow "label")

let test_dataset_admission () =
  let frame = frame () in
  let admitted =
    Modelkit_talon.classification_dataset ~sample_weight:"weight"
      ~groups:"group"
      ~features:[ "temperature"; "pressure" ]
      ~target:"label" frame
    |> get_ok
  in
  Alcotest.check Alcotest.int "samples" 3
    (Dataset.sample_count admitted.dataset);
  Alcotest.check Alcotest.int "features" 2
    (Dataset.feature_count admitted.dataset);
  Alcotest.check Alcotest.int "reports" 5 (List.length admitted.dataset_reports);
  Alcotest.check Alcotest.int "null count" 1
    (Null_mask.null_count (Option.get admitted.feature_null_mask));
  let names =
    Dataset.feature_schema admitted.dataset
    |> Feature_schema.names |> Option.get
  in
  Alcotest.check
    (Alcotest.array Alcotest.string)
    "dataset names"
    [| "temperature"; "pressure" |]
    (Feature_names.to_array names);
  let weights = Dataset.sample_weight admitted.dataset |> Option.get in
  Alcotest.check (Alcotest.float 0.0) "weight" 2.0 (Sample_weight.get weights 1);
  let groups = Dataset.groups admitted.dataset |> Option.get in
  Alcotest.check Alcotest.int "group" 100 (Groups.get groups 1);
  expect_kind
    (validation_reason_contains "selected as both features and target")
    (Modelkit_talon.classification_dataset ~features:[ "temperature"; "label" ]
       ~target:"label" frame);
  expect_kind
    (validation_reason_contains "selected as both target and sample weights")
    (Modelkit_talon.regression_dataset ~sample_weight:"weight"
       ~features:[ "temperature" ] ~target:"weight" frame);
  expect_kind
    (validation_reason_contains "null")
    (Modelkit_talon.classification_dataset ~features:[ "temperature" ]
       ~target:"sparse_label" frame)

let test_regression_dataset () =
  let frame = frame () in
  let admitted =
    Modelkit_talon.regression_dataset ~features:[ "temperature" ]
      ~target:"weight" frame
    |> get_ok
  in
  let values =
    Dataset.target admitted.dataset
    |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "regression dataset target" [| 1.0; 2.0; 0.5 |] values;
  Alcotest.check Alcotest.bool "no mask" true
    (Option.is_none admitted.feature_null_mask);
  Alcotest.check Alcotest.int "reports" 2 (List.length admitted.dataset_reports)

module Conformance = Adapter_conformance.Make (struct
  let name = "modelkit-talon"

  let feature_columns ?null_mask ~names values =
    Array.to_list
      (Array.mapi
         (fun column name ->
           let column_values = Array.map (fun row -> row.(column)) values in
           match null_mask with
           | None -> (name, Talon.Col.float64 column_values)
           | Some null_mask ->
               ( name,
                 Talon.Col.float64_opt
                   (Array.mapi
                      (fun row value ->
                        if null_mask.(row).(column) then None else Some value)
                      column_values) ))
         names)

  let features ?null_mask ~names values =
    Modelkit_talon.features
      (Talon.create (feature_columns ?null_mask ~names values))
      (Array.to_list names)

  let single name column = Talon.create [ (name, column) ]

  let regression_target values =
    Modelkit_talon.regression_target
      (single "target" (Talon.Col.float64 values))
      "target"

  let classification_target values =
    Modelkit_talon.classification_target
      (single "target" (Talon.Col.int64 values))
      "target"

  let sample_weight values =
    Modelkit_talon.sample_weight
      (single "weight" (Talon.Col.float64 values))
      "weight"

  let groups values =
    Modelkit_talon.groups (single "group" (Talon.Col.int64 values)) "group"

  let frame ?null_mask ?sample_weight ?groups ~names ~x target =
    Talon.create
      (feature_columns ?null_mask ~names x
      @ [ ("target", target) ]
      @ Option.fold ~none:[]
          ~some:(fun values -> [ ("weight", Talon.Col.float64 values) ])
          sample_weight
      @ Option.fold ~none:[]
          ~some:(fun values -> [ ("group", Talon.Col.int64 values) ])
          groups)

  let classification_dataset ?null_mask ?sample_weight ?groups ~names ~x ~y () =
    Modelkit_talon.classification_dataset
      ?sample_weight:(Option.map (fun _ -> "weight") sample_weight)
      ?groups:(Option.map (fun _ -> "group") groups)
      ~features:(Array.to_list names) ~target:"target"
      (frame ?null_mask ?sample_weight ?groups ~names ~x (Talon.Col.int64 y))

  let regression_dataset ?null_mask ?sample_weight ?groups ~names ~x ~y () =
    Modelkit_talon.regression_dataset
      ?sample_weight:(Option.map (fun _ -> "weight") sample_weight)
      ?groups:(Option.map (fun _ -> "group") groups)
      ~features:(Array.to_list names) ~target:"target"
      (frame ?null_mask ?sample_weight ?groups ~names ~x (Talon.Col.float64 y))
end)

let () =
  Alcotest.run "modelkit-talon"
    [
      ( "admission",
        [
          Alcotest.test_case "features follow selection order" `Quick
            test_features_follow_selection_order;
          Alcotest.test_case "features copy and unmasked columns" `Quick
            test_features_copy_and_unmasked_columns;
          Alcotest.test_case "feature validation" `Quick test_feature_validation;
          Alcotest.test_case "targets and metadata" `Quick
            test_targets_and_metadata;
          Alcotest.test_case "dataset" `Quick test_dataset_admission;
          Alcotest.test_case "regression dataset" `Quick test_regression_dataset;
        ] );
      ("conformance", Conformance.tests);
    ]
