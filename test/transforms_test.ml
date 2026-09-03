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

let[@warning "-4"] is_numerical = function
  | Error.Numerical _ -> true
  | _ -> false

let[@warning "-4"] is_non_finite = function
  | Error.Data (Data_error.Non_finite _) -> true
  | _ -> false

let[@warning "-4"] is_index_out_of_bounds = function
  | Error.Data (Data_error.Index_out_of_bounds _) -> true
  | _ -> false

let close expected observed =
  Float.abs (expected -. observed)
  <= 1e-12 *. Float.max 1.0 (Float.abs expected)

let check_vector label expected observed =
  let observed = Vector.to_array observed in
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      Alcotest.(check bool)
        (Format.sprintf "%s[%d]" label index)
        true
        (close expected observed.(index)))
    expected

let check_matrix label expected observed =
  let rows, columns = Matrix.shape observed in
  Alcotest.(check int) (label ^ " rows") (Array.length expected) rows;
  let expected_columns =
    if Array.length expected = 0 then 0 else Array.length expected.(0)
  in
  Alcotest.(check int) (label ^ " columns") expected_columns columns;
  Array.iteri
    (fun row expected_row ->
      Array.iteri
        (fun column expected ->
          Alcotest.(check bool)
            (Format.sprintf "%s[%d,%d]" label row column)
            true
            (close expected (Matrix.get observed row column)))
        expected_row)
    expected

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let rng () = Rng.create (Seed.of_int 1729)
let matrix values = Matrix.of_arrays values |> get_data

let test_min_max_scaler () =
  let x = matrix [| [| -1.0; 2.0 |]; [| 1.0; 2.0 |]; [| 3.0; 6.0 |] |] in
  let schema = named_schema [| "a"; "constant_then_change" |] in
  let specification = Min_max_scaler.create () |> get in
  let fitted =
    Min_max_scaler.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:None ()
    |> get
  in
  check_vector "minimum" [| -1.0; 2.0 |] (Min_max_scaler.data_min fitted);
  check_vector "maximum" [| 3.0; 6.0 |] (Min_max_scaler.data_max fitted);
  check_vector "range" [| 4.0; 4.0 |] (Min_max_scaler.data_range fitted);
  check_matrix "min-max output"
    [| [| 0.0; 0.0 |]; [| 0.5; 0.0 |]; [| 1.0; 1.0 |] |]
    (Min_max_scaler.transform fitted ~feature_schema:schema ~x |> get);
  let constant = matrix [| [| 7.0 |]; [| 7.0 |] |] in
  let constant_schema = named_schema [| "constant" |] in
  let clipped =
    Min_max_scaler.create ~feature_range:(-1.0, 1.0) ~clip:true () |> get
  in
  let fitted =
    Min_max_scaler.fit clipped ~rng:(rng ()) ~feature_schema:constant_schema
      ~x:constant ~y:None ()
    |> get
  in
  check_matrix "constant min-max"
    [| [| -1.0 |]; [| -1.0 |] |]
    (Min_max_scaler.transform fitted ~feature_schema:constant_schema ~x:constant
    |> get);
  check_matrix "clipped min-max" [| [| 1.0 |] |]
    (Min_max_scaler.transform fitted ~feature_schema:constant_schema
       ~x:(matrix [| [| 100.0 |] |])
    |> get);
  expect_error is_validation
    (Min_max_scaler.create ~feature_range:(1.0, 1.0) ());
  let extreme = matrix [| [| Float.max_float |]; [| -.Float.max_float |] |] in
  expect_error is_numerical
    (Min_max_scaler.fit
       (Min_max_scaler.create () |> get)
       ~rng:(rng ()) ~feature_schema:constant_schema ~x:extreme ~y:None ())

let test_max_abs_and_robust_scalers () =
  let x = matrix [| [| -2.0; 1.0 |]; [| 1.0; 2.0 |]; [| 4.0; 100.0 |] |] in
  let schema = named_schema [| "a"; "outlier" |] in
  let max_abs =
    Max_abs_scaler.fit (Max_abs_scaler.create ()) ~rng:(rng ())
      ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_vector "max abs" [| 4.0; 100.0 |] (Max_abs_scaler.max_abs max_abs);
  check_matrix "max abs output"
    [| [| -0.5; 0.01 |]; [| 0.25; 0.02 |]; [| 1.0; 1.0 |] |]
    (Max_abs_scaler.transform max_abs ~feature_schema:schema ~x |> get);
  let robust =
    Robust_scaler.fit
      (Robust_scaler.create () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_vector "robust center" [| 1.0; 2.0 |] (Robust_scaler.center robust);
  check_vector "robust scale" [| 3.0; 49.5 |] (Robust_scaler.scale robust);
  check_matrix "robust output"
    [| [| -1.0; -1.0 /. 49.5 |]; [| 0.0; 0.0 |]; [| 1.0; 98.0 /. 49.5 |] |]
    (Robust_scaler.transform robust ~feature_schema:schema ~x |> get);
  expect_error is_validation
    (Robust_scaler.create ~quantile_range:(75.0, 25.0) ())

let test_normalizer () =
  let x = matrix [| [| 3.0; 4.0 |]; [| 0.0; 0.0 |]; [| -2.0; 1.0 |] |] in
  let schema = named_schema [| "a"; "b" |] in
  let fit norm =
    Normalizer.fit
      (Normalizer.create ~norm ())
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "L2 normalization"
    [|
      [| 0.6; 0.8 |];
      [| 0.0; 0.0 |];
      [| -2.0 /. Float.sqrt 5.0; 1.0 /. Float.sqrt 5.0 |];
    |]
    (Normalizer.transform (fit Normalizer.L2) ~feature_schema:schema ~x |> get);
  check_matrix "L1 normalization"
    [|
      [| 3.0 /. 7.0; 4.0 /. 7.0 |];
      [| 0.0; 0.0 |];
      [| -2.0 /. 3.0; 1.0 /. 3.0 |];
    |]
    (Normalizer.transform (fit Normalizer.L1) ~feature_schema:schema ~x |> get);
  check_matrix "max normalization"
    [| [| 0.75; 1.0 |]; [| 0.0; 0.0 |]; [| -1.0; 0.5 |] |]
    (Normalizer.transform (fit Normalizer.Max) ~feature_schema:schema ~x |> get)

let test_one_hot_encoder () =
  let x = matrix [| [| 2.0; 10.0 |]; [| 1.0; 20.0 |]; [| 2.0; 10.0 |] |] in
  let schema = named_schema [| "kind"; "group" |] in
  let specification = One_hot_encoder.create () |> get in
  let fitted =
    One_hot_encoder.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:None ()
    |> get
  in
  let categories = One_hot_encoder.categories fitted in
  check_vector "kind categories" [| 1.0; 2.0 |] categories.(0);
  check_vector "group categories" [| 10.0; 20.0 |] categories.(1);
  let expected =
    [|
      [| 0.0; 1.0; 1.0; 0.0 |];
      [| 1.0; 0.0; 0.0; 1.0 |];
      [| 0.0; 1.0; 1.0; 0.0 |];
    |]
  in
  check_matrix "one-hot output" expected
    (One_hot_encoder.transform fitted ~feature_schema:schema ~x |> get);
  check_matrix "one-hot CSR output" expected
    (One_hot_encoder.transform_csr fitted ~feature_schema:schema ~x
    |> get |> Csr_matrix.to_dense);
  let names =
    One_hot_encoder.output_schema fitted
    |> Feature_schema.names |> Option.get |> Feature_names.to_array
  in
  Alcotest.(check int) "one-hot name count" 4 (Array.length names);
  Alcotest.(check string) "one-hot provenance" "one_hot[0]:kind=1" names.(0);
  let unknown = matrix [| [| 3.0; 10.0 |] |] in
  expect_error is_validation
    (One_hot_encoder.transform fitted ~feature_schema:schema ~x:unknown);
  let permissive =
    One_hot_encoder.fit
      (One_hot_encoder.create ~unknown_category:One_hot_encoder.Ignore () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "ignored one-hot category"
    [| [| 0.0; 0.0; 1.0; 0.0 |] |]
    (One_hot_encoder.transform permissive ~feature_schema:schema ~x:unknown
    |> get);
  expect_error is_validation
    (One_hot_encoder.fit
       (One_hot_encoder.create ~max_output_features:3 () |> get)
       ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ());
  expect_error is_non_finite
    (One_hot_encoder.fit specification ~rng:(rng ()) ~feature_schema:schema
       ~x:(matrix [| [| Float.nan; 1.0 |] |])
       ~y:None ())

let test_ordinal_and_label_encoders () =
  let x = matrix [| [| 2.0; 10.0 |]; [| 1.0; 20.0 |]; [| 2.0; 10.0 |] |] in
  let schema = named_schema [| "kind"; "group" |] in
  let ordinal =
    Ordinal_encoder.fit
      (Ordinal_encoder.create () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "ordinal output"
    [| [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 1.0; 0.0 |] |]
    (Ordinal_encoder.transform ordinal ~feature_schema:schema ~x |> get);
  let permissive =
    Ordinal_encoder.fit
      (Ordinal_encoder.create
         ~unknown_category:(Ordinal_encoder.Use_encoded_value (-1.0)) ()
      |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "unknown ordinal category"
    [| [| -1.0; 0.0 |] |]
    (Ordinal_encoder.transform permissive ~feature_schema:schema
       ~x:(matrix [| [| 3.0; 10.0 |] |])
    |> get);
  expect_error is_validation
    (Ordinal_encoder.fit
       (Ordinal_encoder.create
          ~unknown_category:(Ordinal_encoder.Use_encoded_value 0.0) ()
       |> get)
       ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ());
  let target = Target.classification [| 42; -3; 42; 10 |] in
  let labels = Label_encoder.fit (Label_encoder.create ()) ~y:target |> get in
  Alcotest.(check (array int))
    "label classes" [| -3; 10; 42 |]
    (Label_encoder.classes labels);
  let encoded = Label_encoder.transform labels target |> get in
  Alcotest.(check (array int))
    "encoded labels" [| 2; 0; 2; 1 |]
    (Target.classification_values encoded);
  Alcotest.(check (array int))
    "decoded labels" [| 42; -3; 42; 10 |]
    (Label_encoder.inverse_transform labels encoded
    |> get |> Target.classification_values);
  expect_error is_validation
    (Label_encoder.transform labels (Target.classification [| 99 |]));
  expect_error is_index_out_of_bounds
    (Label_encoder.inverse_transform labels (Target.classification [| 3 |]))

let test_polynomial_features () =
  let x = matrix [| [| 2.0; 3.0 |]; [| -1.0; 4.0 |] |] in
  let schema = named_schema [| "a"; "b" |] in
  let fitted =
    Polynomial_features.fit
      (Polynomial_features.create () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  Alcotest.(check (array (array int)))
    "polynomial terms"
    [| [||]; [| 0 |]; [| 1 |]; [| 0; 0 |]; [| 0; 1 |]; [| 1; 1 |] |]
    (Polynomial_features.terms fitted);
  check_matrix "polynomial output"
    [|
      [| 1.0; 2.0; 3.0; 4.0; 6.0; 9.0 |]; [| 1.0; -1.0; 4.0; 1.0; -4.0; 16.0 |];
    |]
    (Polynomial_features.transform fitted ~feature_schema:schema ~x |> get);
  let interaction =
    Polynomial_features.fit
      (Polynomial_features.create ~interaction_only:true () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "interaction output"
    [| [| 1.0; 2.0; 3.0; 6.0 |]; [| 1.0; -1.0; 4.0; -4.0 |] |]
    (Polynomial_features.transform interaction ~feature_schema:schema ~x |> get);
  expect_error is_validation
    (Polynomial_features.fit
       (Polynomial_features.create ~max_output_features:5 () |> get)
       ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ());
  let extreme_fitted =
    Polynomial_features.fit
      (Polynomial_features.create () |> get)
      ~rng:(rng ()) ~feature_schema:(named_schema [| "x" |])
      ~x:(matrix [| [| Float.max_float |] |])
      ~y:None ()
    |> get
  in
  expect_error is_non_finite
    (Polynomial_features.transform extreme_fitted
       ~feature_schema:(named_schema [| "x" |])
       ~x:(matrix [| [| Float.max_float |] |]))

let test_missing_indicator () =
  let x = matrix [| [| Float.nan; 1.0 |]; [| 2.0; 3.0 |] |] in
  let schema = named_schema [| "sometimes"; "complete" |] in
  let fitted =
    Missing_indicator.fit
      (Missing_indicator.create ())
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  Alcotest.(check (array int))
    "selected missing features" [| 0 |]
    (Missing_indicator.selected_features fitted);
  check_matrix "missing output" [| [| 1.0 |]; [| 0.0 |] |]
    (Missing_indicator.transform fitted ~feature_schema:schema ~x |> get);
  expect_error is_validation
    (Missing_indicator.transform fitted ~feature_schema:schema
       ~x:(matrix [| [| 1.0; Float.nan |] |]));
  let all =
    Missing_indicator.fit
      (Missing_indicator.create ~features:Missing_indicator.All ())
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  check_matrix "all missing output"
    [| [| 1.0; 0.0 |]; [| 0.0; 0.0 |] |]
    (Missing_indicator.transform all ~feature_schema:schema ~x |> get);
  expect_error is_non_finite
    (Missing_indicator.fit
       (Missing_indicator.create ())
       ~rng:(rng ()) ~feature_schema:schema
       ~x:(matrix [| [| Float.infinity; 1.0 |] |])
       ~y:None ())

let () =
  Alcotest.run "additional preprocessing"
    [
      ( "numeric scaling",
        [
          Alcotest.test_case "min-max" `Quick test_min_max_scaler;
          Alcotest.test_case "max-absolute and robust" `Quick
            test_max_abs_and_robust_scalers;
          Alcotest.test_case "normalizer" `Quick test_normalizer;
        ] );
      ( "encoding",
        [
          Alcotest.test_case "one-hot" `Quick test_one_hot_encoder;
          Alcotest.test_case "ordinal and label" `Quick
            test_ordinal_and_label_encoders;
        ] );
      ( "feature generation",
        [
          Alcotest.test_case "polynomial" `Quick test_polynomial_features;
          Alcotest.test_case "missing indicator" `Quick test_missing_indicator;
        ] );
    ]
