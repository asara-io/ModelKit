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
  let raw =
    0.2
    +. (0.08 *. Matrix.get x row 0)
    -. (0.04 *. Matrix.get x row 1)
    +. (0.03 *. Matrix.get x row 2)
  in
  Float.exp raw *. (0.8 +. (Float.of_int (row mod 5) *. 0.1))

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 8 then
    fail
      "usage: glm_worker SAMPLES FEATURES SEED ALPHA POWER TOLERANCE \
       MAX_ITERATIONS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let alpha = float_of_string Sys.argv.(4) in
  let power = float_of_string Sys.argv.(5) in
  let tolerance = float_of_string Sys.argv.(6) in
  let max_iterations = int_of_string Sys.argv.(7) in
  if features < 3 then fail "GLM benchmark requires at least 3 features";
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
  let rng = Rng.create (Seed.of_int seed) in
  let poisson =
    Poisson_regression.fit
      (Poisson_regression.create ~alpha ~tolerance ~max_iterations () |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let tweedie =
    Tweedie_regression.fit
      (Tweedie_regression.create ~power ~alpha ~link:Tweedie_regression.Log
         ~tolerance ~max_iterations ()
      |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let poisson_predictions =
    Poisson_regression.predict poisson ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let tweedie_predictions =
    Tweedie_regression.predict tweedie ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let last = samples - 1 in
  let signature =
    [|
      Vector.get (Poisson_regression.coefficients poisson) 0;
      Poisson_regression.intercept poisson;
      Vector.get poisson_predictions 0;
      Vector.get poisson_predictions last;
      Vector.get (Tweedie_regression.coefficients tweedie) 0;
      Tweedie_regression.intercept tweedie;
      Vector.get tweedie_predictions 0;
      Vector.get tweedie_predictions last;
    |]
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["weighted_poisson_fit_predict","weighted_tweedie_fit_predict"],"samples":%d,"signature":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    signature.(0) signature.(1) signature.(2) signature.(3) signature.(4)
    signature.(5) signature.(6) signature.(7)
