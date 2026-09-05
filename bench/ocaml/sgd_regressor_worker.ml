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

let feature ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.0
  -. 5.0

let target x row =
  2.0
  +. (1.5 *. Matrix.get x row 0)
  -. (0.8 *. Matrix.get x row 1)
  +. (0.3 *. Matrix.get x row 2)
  +. (Float.of_int ((row mod 7) - 3) *. 0.02)

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 6 then
    fail "usage: sgd_regressor_worker SAMPLES FEATURES SEED ETA0 EPOCHS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let eta0 = float_of_string Sys.argv.(4) in
  let epochs = int_of_string Sys.argv.(5) in
  if features < 3 then fail "SGD benchmark requires at least 3 features";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (feature ~seed) |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let y =
    Target.regression (Vector.init ~length:samples (target x) |> get_data)
    |> get_data
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples
      (Array.init samples (fun row -> 1.0 +. (Float.of_int (row mod 5) *. 0.25)))
    |> get_data
  in
  let specification =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
      ~learning_rate:Sgd_regressor.Constant ~eta0 ~max_epochs:epochs
      ~shuffle:false ()
    |> get
  in
  let rng = Rng.create (Seed.of_int seed) in
  let fitted =
    Sgd_regressor.fit specification ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_regressor.partial_fit checkpoint ~sample_weight ~feature_schema ~x ~y
        ()
      |> get
      |> train (remaining - 1)
  in
  let incremental =
    Sgd_regressor.start specification ~rng ~feature_schema
    |> train epochs |> Sgd_regressor.to_fitted |> get
  in
  let predictions =
    Sgd_regressor.predict fitted ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let incremental_predictions =
    Sgd_regressor.predict incremental ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let last = samples - 1 in
  let signature =
    [|
      Vector.get (Sgd_regressor.coefficients fitted) 0;
      Sgd_regressor.intercept fitted;
      Vector.get predictions 0;
      Vector.get predictions last;
      Vector.get (Sgd_regressor.coefficients incremental) 0;
      Sgd_regressor.intercept incremental;
      Vector.get incremental_predictions 0;
      Vector.get incremental_predictions last;
      Float.of_int (Sgd_regressor.report fitted).Sgd_regressor.updates;
      Float.of_int (Sgd_regressor.report incremental).Sgd_regressor.updates;
    |]
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["weighted_sgd_regressor_fit_predict","weighted_sgd_regressor_partial_fit_predict"],"samples":%d,"signature":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    signature.(0) signature.(1) signature.(2) signature.(3) signature.(4)
    signature.(5) signature.(6) signature.(7) signature.(8) signature.(9)
