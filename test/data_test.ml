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
  let source =
    Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 2
  in
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
  expect_error Ragged_matrix
    (Matrix.of_arrays [| [| 1.0 |]; [| 2.0; 3.0 |] |])

let test_matrix_ownership () =
  let source =
    Bigarray.Array2.create Bigarray.float64 Bigarray.c_layout 1 1
  in
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
  expect_error Index_out_of_bounds
    (Row_view.create ~source_size:3 [| 3 |])

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
  expect_error Length_mismatch
    (Groups.create ~expected_length:3 [| 1; 2 |])

let test_negative_dimensions () =
  expect_error Negative_dimension (Vector.create ~length:(-1) 0.0);
  expect_error Negative_dimension (Matrix.create ~rows:1 ~columns:(-1) 0.0);
  expect_error Negative_dimension (Row_view.all ~source_size:(-1))

let () =
  test_vector_ownership ();
  test_matrix ();
  test_matrix_ownership ();
  test_row_view ();
  test_targets ();
  test_feature_names ();
  test_sample_weights ();
  test_groups ();
  test_negative_dimensions ()
