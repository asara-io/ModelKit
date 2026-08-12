open Modelkit

let fail message = raise (Failure message)
let check condition message = if not condition then fail message

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> fail "expected an error"

let is_validation = function
  | Error.Validation _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_numerical = function
  | Error.Numerical _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Validation _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_data_non_finite = function
  | Error.Data (Data_error.Non_finite _) -> true
  | Error.Data
      ( Data_error.Negative_dimension _ | Data_error.Ragged_matrix _
      | Data_error.Length_mismatch _ | Data_error.Index_out_of_bounds _
      | Data_error.Negative_weight _ | Data_error.All_zero_weights
      | Data_error.Empty_feature_name _ | Data_error.Duplicate_feature_name _ )
  | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Validation _ | Error.Numerical _ | Error.Convergence _
  | Error.Compatibility _ | Error.Artifact _ | Error.Cancelled ->
      false

let is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let close expected observed = Float.abs (expected -. observed) <= 1e-12

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let rng () = Rng.create (Seed.of_int 1729)

let fit_imputer specification schema x =
  Simple_imputer.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
    ~y:None ()
  |> get

let fit_scaler specification schema x =
  Standard_scaler.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
    ~y:None ()
  |> get

let fit_variance_threshold specification schema x =
  Variance_threshold.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
    ~y:None ()
  |> get

let test_mean_imputer () =
  let schema = named_schema [| "a"; "b" |] in
  let training =
    Matrix.of_arrays [| [| 1.0; Float.nan |]; [| 3.0; 4.0 |]; [| 5.0; 8.0 |] |]
    |> get_data
  in
  let fitted = fit_imputer (Simple_imputer.mean ()) schema training in
  check
    (Vector.to_array (Simple_imputer.statistics fitted) = [| 3.0; 6.0 |])
    "mean imputer statistics are wrong";
  check
    (Feature_schema.equal (Simple_imputer.output_schema fitted) schema)
    "mean imputer changed feature names";
  let production =
    Matrix.of_arrays [| [| Float.nan; 10.0 |]; [| 100.0; Float.nan |] |]
    |> get_data
  in
  let transformed =
    Simple_imputer.transform fitted ~feature_schema:schema ~x:production |> get
  in
  check
    (Matrix.to_arrays transformed = [| [| 3.0; 10.0 |]; [| 100.0; 6.0 |] |])
    "mean imputer used transform data while filling values";
  let extreme =
    Matrix.of_arrays [| [| Float.max_float |]; [| -.Float.max_float |] |]
    |> get_data
  in
  let extreme_schema = named_schema [| "extreme" |] in
  let extreme_fitted =
    fit_imputer (Simple_imputer.mean ()) extreme_schema extreme
  in
  check
    (Vector.get (Simple_imputer.statistics extreme_fitted) 0 = 0.0)
    "mean imputer overflowed a representable mean"

let test_median_and_constant_imputers () =
  let schema = named_schema [| "value" |] in
  let training =
    Matrix.of_arrays
      [| [| 1.0 |]; [| 3.0 |]; [| Float.nan |]; [| 10.0 |]; [| 100.0 |] |]
    |> get_data
  in
  let median = fit_imputer (Simple_imputer.median ()) schema training in
  check
    (Vector.get (Simple_imputer.statistics median) 0 = 6.5)
    "median imputer statistic is wrong";
  let tiny =
    Matrix.of_arrays [| [| Float.min_float |]; [| Float.min_float |] |]
    |> get_data
  in
  let tiny_median = fit_imputer (Simple_imputer.median ()) schema tiny in
  check
    (Vector.get (Simple_imputer.statistics tiny_median) 0 = Float.min_float)
    "median imputer underflowed an equal-value midpoint";
  let all_missing = Matrix.of_arrays [| [| Float.nan |] |] |> get_data in
  expect_error is_validation
    (Simple_imputer.fit (Simple_imputer.median ()) ~rng:(rng ())
       ~feature_schema:schema ~x:all_missing ~y:None ());
  let constant = Simple_imputer.constant (-2.0) |> get in
  let fitted = fit_imputer constant schema all_missing in
  let transformed =
    Simple_imputer.transform fitted ~feature_schema:schema ~x:all_missing |> get
  in
  check (Matrix.get transformed 0 0 = -2.0) "constant imputation failed";
  expect_error is_validation (Simple_imputer.constant Float.nan);
  expect_error is_validation (Simple_imputer.constant Float.infinity)

let test_imputer_errors () =
  let schema = named_schema [| "value" |] in
  let infinity = Matrix.of_arrays [| [| Float.infinity |] |] |> get_data in
  expect_error is_data_non_finite
    (Simple_imputer.fit (Simple_imputer.mean ()) ~rng:(rng ())
       ~feature_schema:schema ~x:infinity ~y:None ());
  let training = Matrix.of_arrays [| [| 1.0 |] |] |> get_data in
  let fitted = fit_imputer (Simple_imputer.mean ()) schema training in
  expect_error is_schema_mismatch
    (Simple_imputer.transform fitted
       ~feature_schema:(named_schema [| "other" |])
       ~x:training);
  let weights =
    Sample_weight.of_array ~expected_length:1 [| 1.0 |] |> get_data
  in
  expect_error is_validation
    (Simple_imputer.fit (Simple_imputer.mean ()) ~sample_weight:weights
       ~rng:(rng ()) ~feature_schema:schema ~x:training ~y:None ())

let test_standard_scaler () =
  let schema = named_schema [| "a"; "b"; "constant" |] in
  let training =
    Matrix.of_arrays
      [| [| 1.0; 2.0; 7.0 |]; [| 3.0; 4.0; 7.0 |]; [| 5.0; 6.0; 7.0 |] |]
    |> get_data
  in
  let fitted = fit_scaler (Standard_scaler.create ()) schema training in
  let expected_variance = 8.0 /. 3.0 in
  check
    (Vector.to_array (Standard_scaler.mean fitted) = [| 3.0; 4.0; 7.0 |])
    "standard scaler means are wrong";
  check
    (close expected_variance (Vector.get (Standard_scaler.variance fitted) 0)
    && close expected_variance (Vector.get (Standard_scaler.variance fitted) 1)
    && Vector.get (Standard_scaler.variance fitted) 2 = 0.0)
    "standard scaler population variances are wrong";
  check
    (Vector.get (Standard_scaler.scale fitted) 2 = 1.0)
    "constant feature scale is not one";
  let transformed =
    Standard_scaler.transform fitted ~feature_schema:schema ~x:training |> get
  in
  let expected = 2.0 /. Float.sqrt expected_variance in
  check
    (close (-.expected) (Matrix.get transformed 0 0)
    && Matrix.get transformed 1 0 = 0.0
    && close expected (Matrix.get transformed 2 0)
    && Matrix.get transformed 0 2 = 0.0)
    "standard scaling is wrong";
  check
    (Feature_schema.equal (Standard_scaler.output_schema fitted) schema)
    "standard scaler changed feature names"

let test_standard_scaler_options_and_errors () =
  let schema = named_schema [| "value" |] in
  let training = Matrix.of_arrays [| [| 1.0 |]; [| 3.0 |] |] |> get_data in
  let specification = Standard_scaler.create ~with_mean:false () in
  let fitted = fit_scaler specification schema training in
  let transformed =
    Standard_scaler.transform fitted ~feature_schema:schema ~x:training |> get
  in
  check
    (close (1.0 /. 1.0) (Matrix.get transformed 0 0)
    && close (3.0 /. 1.0) (Matrix.get transformed 1 0))
    "with_mean=false was ignored";
  let missing = Matrix.of_arrays [| [| Float.nan |] |] |> get_data in
  expect_error is_data_non_finite
    (Standard_scaler.fit
       (Standard_scaler.create ())
       ~rng:(rng ()) ~feature_schema:schema ~x:missing ~y:None ());
  let empty = Matrix.create ~rows:0 ~columns:1 0.0 |> get_data in
  expect_error is_validation
    (Standard_scaler.fit
       (Standard_scaler.create ())
       ~rng:(rng ()) ~feature_schema:schema ~x:empty ~y:None ());
  let extreme =
    Matrix.of_arrays [| [| Float.max_float |]; [| -.Float.max_float |] |]
    |> get_data
  in
  expect_error is_numerical
    (Standard_scaler.fit
       (Standard_scaler.create ())
       ~rng:(rng ()) ~feature_schema:schema ~x:extreme ~y:None ());
  let weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 1.0 |] |> get_data
  in
  expect_error is_validation
    (Standard_scaler.fit
       (Standard_scaler.create ())
       ~sample_weight:weights ~rng:(rng ()) ~feature_schema:schema ~x:training
       ~y:None ())

let test_variance_threshold () =
  let schema = named_schema [| "constant"; "signal"; "other" |] in
  let training =
    Matrix.of_arrays
      [| [| 0.0; 1.0; 5.0 |]; [| 0.0; 3.0; 7.0 |]; [| 0.0; 5.0; 9.0 |] |]
    |> get_data
  in
  let specification = Variance_threshold.create () |> get in
  let fitted = fit_variance_threshold specification schema training in
  check
    (Variance_threshold.selected_indices fitted = [| 1; 2 |])
    "variance threshold selected the wrong columns";
  check
    (Vector.get (Variance_threshold.variances fitted) 0 = 0.0)
    "constant feature variance is not zero";
  let output_names =
    Variance_threshold.output_schema fitted
    |> Feature_schema.names |> Option.get |> Feature_names.to_array
  in
  check
    (output_names = [| "signal"; "other" |])
    "variance threshold did not propagate selected names";
  let transformed =
    Variance_threshold.transform fitted ~feature_schema:schema ~x:training
    |> get
  in
  check
    (Matrix.to_arrays transformed
    = [| [| 1.0; 5.0 |]; [| 3.0; 7.0 |]; [| 5.0; 9.0 |] |])
    "variance threshold transform is wrong"

let test_variance_threshold_errors () =
  expect_error is_validation (Variance_threshold.create ~threshold:(-1.0) ());
  expect_error is_validation (Variance_threshold.create ~threshold:Float.nan ());
  let schema = named_schema [| "constant" |] in
  let constant = Matrix.of_arrays [| [| 2.0 |]; [| 2.0 |] |] |> get_data in
  let specification = Variance_threshold.create () |> get in
  expect_error is_validation
    (Variance_threshold.fit specification ~rng:(rng ()) ~feature_schema:schema
       ~x:constant ~y:None ());
  let missing = Matrix.of_arrays [| [| Float.nan |] |] |> get_data in
  expect_error is_data_non_finite
    (Variance_threshold.fit specification ~rng:(rng ()) ~feature_schema:schema
       ~x:missing ~y:None ());
  let weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 1.0 |] |> get_data
  in
  expect_error is_validation
    (Variance_threshold.fit specification ~sample_weight:weights ~rng:(rng ())
       ~feature_schema:schema ~x:constant ~y:None ())

let () =
  Alcotest.run "preprocessing"
    [
      ( "simple imputer",
        [
          Alcotest.test_case "mean" `Quick test_mean_imputer;
          Alcotest.test_case "median and constant" `Quick
            test_median_and_constant_imputers;
          Alcotest.test_case "typed errors" `Quick test_imputer_errors;
        ] );
      ( "standard scaler",
        [
          Alcotest.test_case "scale" `Quick test_standard_scaler;
          Alcotest.test_case "options and errors" `Quick
            test_standard_scaler_options_and_errors;
        ] );
      ( "variance threshold",
        [
          Alcotest.test_case "select and transform" `Quick
            test_variance_threshold;
          Alcotest.test_case "typed errors" `Quick
            test_variance_threshold_errors;
        ] );
    ]
