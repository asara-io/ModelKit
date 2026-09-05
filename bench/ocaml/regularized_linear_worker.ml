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

let parse_alphas value =
  value |> String.split_on_char ',' |> List.map float_of_string |> Array.of_list
  |> Vector.of_array

let () =
  if Array.length Sys.argv <> 9 then
    fail
      "usage: regularized_linear_worker SAMPLES FEATURES SEED ALPHA L1_RATIO \
       TOLERANCE MAX_ITERATIONS PATH_ALPHAS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let alpha = float_of_string Sys.argv.(4) in
  let l1_ratio = float_of_string Sys.argv.(5) in
  let tolerance = float_of_string Sys.argv.(6) in
  let max_iterations = int_of_string Sys.argv.(7) in
  let path_alphas = parse_alphas Sys.argv.(8) in
  if features < 3 then
    fail "regularized-linear benchmark requires at least 3 features";
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
  let lasso =
    Lasso_regression.fit
      (Lasso_regression.create ~alpha ~tolerance ~max_iterations () |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let elastic =
    Elastic_net_regression.fit
      (Elastic_net_regression.create ~alpha ~l1_ratio ~tolerance ~max_iterations
         ()
      |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let lasso_path =
    Lasso_path.fit
      (Lasso_path.create ~tolerance ~max_iterations () |> get)
      ~alphas:path_alphas ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let elastic_path =
    Elastic_net_path.fit
      (Elastic_net_path.create ~l1_ratio ~tolerance ~max_iterations () |> get)
      ~alphas:path_alphas ~sample_weight ~rng ~feature_schema ~x ~y ()
    |> get
  in
  let lasso_predictions =
    Lasso_regression.predict lasso ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let elastic_predictions =
    Elastic_net_regression.predict elastic ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let last = samples - 1 in
  let last_alpha = Vector.length path_alphas - 1 in
  let signature =
    [|
      Vector.get (Lasso_regression.coefficients lasso) 0;
      Lasso_regression.intercept lasso;
      Vector.get lasso_predictions 0;
      Vector.get lasso_predictions last;
      Vector.get (Elastic_net_regression.coefficients elastic) 0;
      Elastic_net_regression.intercept elastic;
      Vector.get elastic_predictions 0;
      Vector.get elastic_predictions last;
      Matrix.get (Lasso_path.coefficients lasso_path) 0 0;
      Matrix.get (Lasso_path.coefficients lasso_path) last_alpha 0;
      Matrix.get (Elastic_net_path.coefficients elastic_path) 0 0;
      Matrix.get (Elastic_net_path.coefficients elastic_path) last_alpha 0;
    |]
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["weighted_lasso_fit_predict","weighted_elastic_net_fit_predict","weighted_lasso_path","weighted_elastic_net_path"],"samples":%d,"signature":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    signature.(0) signature.(1) signature.(2) signature.(3) signature.(4)
    signature.(5) signature.(6) signature.(7) signature.(8) signature.(9)
    signature.(10) signature.(11)
