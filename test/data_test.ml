open Modelkit

let fail message = raise (Failure message)
let check condition message = if not condition then fail message

let get_ok = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

type error_kind =
  | Negative_dimension
  | Ragged_matrix
  | Length_mismatch
  | Index_out_of_bounds
  | Non_finite
  | Negative_weight
  | All_zero_weights
  | Empty_feature_name
  | Duplicate_feature_name

let error_kind = function
  | Data_error.Negative_dimension _ -> Negative_dimension
  | Data_error.Ragged_matrix _ -> Ragged_matrix
  | Data_error.Length_mismatch _ -> Length_mismatch
  | Data_error.Index_out_of_bounds _ -> Index_out_of_bounds
  | Data_error.Non_finite _ -> Non_finite
  | Data_error.Negative_weight _ -> Negative_weight
  | Data_error.All_zero_weights -> All_zero_weights
  | Data_error.Empty_feature_name _ -> Empty_feature_name
  | Data_error.Duplicate_feature_name _ -> Duplicate_feature_name

let expect_error expected = function
  | Error error when error_kind error = expected -> ()
  | Error error -> fail ("unexpected error: " ^ Data_error.to_string error)
  | Ok _ -> fail "expected an error"

let test_vector_ownership () =
  let source = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 2 in
  Bigarray.Array1.set source 0 1.0;
  Bigarray.Array1.set source 1 2.0;
  let vector = Vector.of_bigarray source in
  Bigarray.Array1.set source 0 9.0;
  check (Vector.get vector 0 = 1.0) "vector retained caller mutation";
  let exported = Vector.to_bigarray vector in
  Bigarray.Array1.set exported 1 9.0;
  check (Vector.get vector 1 = 2.0) "vector export exposed internal storage"

let test_matrix () =
  let matrix =
    get_ok (Matrix.of_arrays [| [| 1.0; Float.nan |]; [| 3.0; 4.0 |] |])
  in
  check (Matrix.shape matrix = (2, 2)) "matrix shape is incorrect";
  check (Float.is_nan (Matrix.get matrix 0 1)) "matrix did not preserve NaN";
  let second_row = Matrix.row matrix 1 in
  check (Vector.to_array second_row = [| 3.0; 4.0 |]) "matrix row is incorrect";
  expect_error Ragged_matrix (Matrix.of_arrays [| [| 1.0 |]; [| 2.0; 3.0 |] |])

let test_matrix_ownership () =
  let source = Bigarray.Array2.create Bigarray.float64 Bigarray.c_layout 1 1 in
  Bigarray.Array2.set source 0 0 3.0;
  let matrix = Matrix.of_bigarray source in
  Bigarray.Array2.set source 0 0 9.0;
  check (Matrix.get matrix 0 0 = 3.0) "matrix retained caller mutation";
  let exported = Matrix.to_bigarray matrix in
  Bigarray.Array2.set exported 0 0 9.0;
  check (Matrix.get matrix 0 0 = 3.0) "matrix export exposed internal storage"

let test_row_view () =
  let indices = [| 2; 0; 2 |] in
  let view = get_ok (Row_view.create ~source_size:3 indices) in
  indices.(0) <- 1;
  check (Row_view.indices view = [| 2; 0; 2 |]) "row view did not copy indices";
  expect_error Index_out_of_bounds (Row_view.create ~source_size:3 [| 3 |])

let test_targets () =
  expect_error Non_finite
    (Target.regression (Vector.of_array [| 1.0; Float.infinity |]));
  let regression =
    get_ok (Target.regression (Vector.of_array [| 1.5; 2.5; 3.5 |]))
  in
  let regression_view = get_ok (Row_view.create ~source_size:3 [| 1; 1 |]) in
  let selected_regression = get_ok (Target.select regression regression_view) in
  check
    (Vector.to_array (Target.regression_values selected_regression)
    = [| 2.5; 2.5 |])
    "regression target selection is incorrect";
  let labels = [| 4; 7; 9 |] in
  let target = Target.classification labels in
  labels.(0) <- 99;
  let view = get_ok (Row_view.create ~source_size:3 [| 2; 0 |]) in
  let selected = get_ok (Target.select target view) in
  check
    (Target.classification_values selected = [| 9; 4 |])
    "classification target selection is incorrect";
  let incompatible_view = get_ok (Row_view.create ~source_size:2 [| 0 |]) in
  expect_error Length_mismatch (Target.select target incompatible_view)

let test_feature_names () =
  let source = [| "height"; "weight" |] in
  let names = get_ok (Feature_names.create ~expected_count:2 source) in
  source.(0) <- "changed";
  check
    (Feature_names.to_array names = [| "height"; "weight" |])
    "feature names did not copy input";
  expect_error Empty_feature_name
    (Feature_names.create ~expected_count:2 [| "height"; "" |]);
  expect_error Duplicate_feature_name
    (Feature_names.create ~expected_count:2 [| "height"; "height" |]);
  expect_error Empty_feature_name (Feature_name.create "");
  expect_error Length_mismatch
    (Feature_names.create ~expected_count:3 [| "height"; "weight" |])

let test_sample_weights () =
  let weights =
    get_ok (Sample_weight.of_array ~expected_length:3 [| 0.0; 1.5; 2.0 |])
  in
  check (Sample_weight.get weights 1 = 1.5) "sample weight is incorrect";
  expect_error Negative_weight
    (Sample_weight.of_array ~expected_length:2 [| 1.0; -1.0 |]);
  expect_error All_zero_weights
    (Sample_weight.of_array ~expected_length:2 [| 0.0; 0.0 |]);
  expect_error Non_finite
    (Sample_weight.of_array ~expected_length:1 [| Float.nan |])

let test_groups () =
  let source = [| 10; 10; 20 |] in
  let groups = get_ok (Groups.create ~expected_length:3 source) in
  source.(0) <- 30;
  check (Groups.distinct_count groups = 2) "group count is incorrect";
  let view = get_ok (Row_view.create ~source_size:3 [| 2; 0 |]) in
  let selected = get_ok (Groups.select groups view) in
  check (Groups.to_array selected = [| 20; 10 |]) "group selection is incorrect";
  expect_error Length_mismatch (Groups.create ~expected_length:3 [| 1; 2 |])

let test_negative_dimensions () =
  expect_error Negative_dimension (Vector.create ~length:(-1) 0.0);
  expect_error Negative_dimension (Matrix.create ~rows:1 ~columns:(-1) 0.0);
  expect_error Negative_dimension (Row_view.all ~source_size:(-1))

let dataset_inputs () =
  let x =
    get_ok
      (Matrix.of_arrays
         [| [| 1.0; Float.nan |]; [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |])
  in
  let y = Target.classification [| 10; 20; 30 |] in
  let feature_names =
    get_ok (Feature_names.create ~expected_count:2 [| "height"; "weight" |])
  in
  let sample_weight =
    get_ok (Sample_weight.of_array ~expected_length:3 [| 1.0; 2.0; 3.0 |])
  in
  let groups = get_ok (Groups.create ~expected_length:3 [| 7; 7; 9 |]) in
  (x, y, feature_names, sample_weight, groups)

let test_dataset_admission () =
  let x, y, feature_names, sample_weight, groups = dataset_inputs () in
  expect_error Non_finite
    (Dataset.create ~finiteness:Dataset.Require_finite ~x ~y ());
  let dataset =
    get_ok
      (Dataset.create ~finiteness:Dataset.Allow_nan ~feature_names
         ~sample_weight ~groups ~x ~y ())
  in
  check (Dataset.sample_count dataset = 3) "dataset sample count is wrong";
  check (Dataset.feature_count dataset = 2) "dataset feature count is wrong";
  check
    (Dataset.finiteness dataset = Dataset.Allow_nan)
    "dataset finiteness policy changed";
  let report = Dataset.access_report dataset in
  check (report.Dataset.feature_access = Dataset.View) "features were copied";
  check (report.Dataset.target_access = Dataset.View) "target was copied";
  check
    (report.Dataset.sample_weight_access = Some Dataset.View)
    "sample weights were copied";
  check (report.Dataset.group_access = Some Dataset.View) "groups were copied";
  let short_target = Target.classification [| 10; 20 |] in
  expect_error Length_mismatch
    (Dataset.create ~finiteness:Dataset.Allow_nan ~x ~y:short_target ());
  let short_groups = get_ok (Groups.create ~expected_length:2 [| 1; 2 |]) in
  expect_error Length_mismatch
    (Dataset.create ~finiteness:Dataset.Allow_nan ~groups:short_groups ~x ~y ());
  let short_weights =
    get_ok (Sample_weight.of_array ~expected_length:2 [| 1.0; 2.0 |])
  in
  expect_error Length_mismatch
    (Dataset.create ~finiteness:Dataset.Allow_nan ~sample_weight:short_weights
       ~x ~y ());
  let short_names =
    get_ok (Feature_names.create ~expected_count:1 [| "height" |])
  in
  expect_error Length_mismatch
    (Dataset.create ~finiteness:Dataset.Allow_nan ~feature_names:short_names ~x
       ~y ());
  let infinity =
    get_ok (Matrix.of_arrays [| [| Float.infinity; Float.neg_infinity |] |])
  in
  expect_error Non_finite
    (Dataset.create ~finiteness:Dataset.Allow_nan ~x:infinity
       ~y:(Target.classification [| 1 |])
       ())

let test_schema_fingerprints () =
  let _, _, feature_names, _, _ = dataset_inputs () in
  let same_names =
    get_ok (Feature_names.create ~expected_count:2 [| "height"; "weight" |])
  in
  let reversed_names =
    get_ok (Feature_names.create ~expected_count:2 [| "weight"; "height" |])
  in
  let fingerprint =
    Feature_schema.fingerprint (Feature_schema.named feature_names)
  in
  let same = Feature_schema.fingerprint (Feature_schema.named same_names) in
  let reversed =
    Feature_schema.fingerprint (Feature_schema.named reversed_names)
  in
  let anonymous =
    Feature_schema.fingerprint
      (get_ok (Feature_schema.anonymous ~feature_count:2))
  in
  check
    (Schema_fingerprint.equal fingerprint same)
    "equivalent schemas have different fingerprints";
  check
    (not (Schema_fingerprint.equal fingerprint reversed))
    "feature order is absent from the fingerprint";
  check
    (not (Schema_fingerprint.equal fingerprint anonymous))
    "named and anonymous schemas have the same fingerprint";
  check
    (String.starts_with ~prefix:"modelkit-schema-v1:"
       (Schema_fingerprint.to_string fingerprint))
    "schema fingerprint is not versioned";
  let encoded = Schema_fingerprint.to_string fingerprint in
  check
    (encoded = "modelkit-schema-v1:35e180681ce2f0abfd546e900cbfc1f3")
    "schema fingerprint changed"

let test_regression_dataset_view () =
  let x = get_ok (Matrix.of_arrays [| [| 1.0 |]; [| 2.0 |] |]) in
  let y = get_ok (Target.regression (Vector.of_array [| 10.5; 20.5 |])) in
  let dataset =
    get_ok (Dataset.create ~finiteness:Dataset.Require_finite ~x ~y ())
  in
  let view =
    Dataset.view dataset (get_ok (Row_view.create ~source_size:2 [| 1 |]))
    |> get_ok
  in
  check
    (Dataset.regression_target view 0 = 20.5)
    "regression view target is wrong";
  let report = Dataset.access_report dataset in
  check
    (report.Dataset.sample_weight_access = None
    && report.Dataset.group_access = None)
    "absent metadata was reported as present"

let test_dataset_row_view () =
  let x, y, feature_names, sample_weight, groups = dataset_inputs () in
  let dataset =
    get_ok
      (Dataset.create ~finiteness:Dataset.Allow_nan ~feature_names
         ~sample_weight ~groups ~x ~y ())
  in
  let rows = get_ok (Row_view.create ~source_size:3 [| 2; 0; 2 |]) in
  let view = get_ok (Dataset.view dataset rows) in
  check (Dataset.view_sample_count view = 3) "dataset view length is wrong";
  check (Dataset.source_row view 1 = 0) "dataset view source row is wrong";
  check (Dataset.feature view ~row:0 ~column:1 = 6.0) "view feature is wrong";
  check (Dataset.classification_target view 1 = 10) "view target is wrong";
  check
    (Dataset.sample_weight_value view 2 = Some 3.0)
    "view sample weight is wrong";
  check (Dataset.group view 0 = Some 9) "view group is wrong";
  check
    ((Dataset.view_access_report view).Dataset.feature_access = Dataset.View)
    "row view materialized features";
  let materialized = get_ok (Dataset.materialize view) in
  let materialized_x = Dataset.features materialized in
  check
    (Matrix.shape materialized_x = (3, 2)
    && Matrix.get materialized_x 0 0 = 5.0
    && Matrix.get materialized_x 0 1 = 6.0
    && Matrix.get materialized_x 1 0 = 1.0
    && Float.is_nan (Matrix.get materialized_x 1 1)
    && Matrix.get materialized_x 2 0 = 5.0
    && Matrix.get materialized_x 2 1 = 6.0)
    "materialized view features are wrong";
  check
    (Target.classification_values (Dataset.target materialized)
    = [| 30; 10; 30 |])
    "materialized view target is wrong";
  let report = Dataset.access_report materialized in
  check
    (report.Dataset.feature_access = Dataset.Copy)
    "features were not copied";
  check (report.Dataset.target_access = Dataset.Copy) "target was not copied";
  check
    (report.Dataset.sample_weight_access = Some Dataset.Copy
    && report.Dataset.group_access = Some Dataset.Copy)
    "materialized metadata was not copied";
  check
    (Schema_fingerprint.equal
       (Dataset.schema_fingerprint dataset)
       (Dataset.schema_fingerprint materialized))
    "materialization changed the schema fingerprint";
  let incompatible = get_ok (Row_view.create ~source_size:2 [| 0 |]) in
  expect_error Length_mismatch (Dataset.view dataset incompatible)

let () =
  Alcotest.run "unit"
    [
      ( "data contracts",
        [
          Alcotest.test_case "vector ownership" `Quick test_vector_ownership;
          Alcotest.test_case "matrix values" `Quick test_matrix;
          Alcotest.test_case "matrix ownership" `Quick test_matrix_ownership;
          Alcotest.test_case "row views" `Quick test_row_view;
          Alcotest.test_case "targets" `Quick test_targets;
          Alcotest.test_case "feature names" `Quick test_feature_names;
          Alcotest.test_case "sample weights" `Quick test_sample_weights;
          Alcotest.test_case "groups" `Quick test_groups;
          Alcotest.test_case "negative dimensions" `Quick
            test_negative_dimensions;
          Alcotest.test_case "dataset admission" `Quick test_dataset_admission;
          Alcotest.test_case "schema fingerprints" `Quick
            test_schema_fingerprints;
          Alcotest.test_case "regression dataset view" `Quick
            test_regression_dataset_view;
          Alcotest.test_case "dataset row views" `Quick test_dataset_row_view;
        ] );
    ]
