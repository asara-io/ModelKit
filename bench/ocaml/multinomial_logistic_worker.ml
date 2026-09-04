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

let classification_target x row =
  if Matrix.get x row 0 +. (0.25 *. Matrix.get x row 1) > 1.0 then 9
  else if Matrix.get x row 2 -. (0.2 *. Matrix.get x row 3) > 0.0 then 2
  else -4

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: multinomial_logistic_worker SAMPLES FEATURES SEED C TOLERANCE \
       MAX_ITERATIONS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let c = float_of_string Sys.argv.(4) in
  let tolerance = float_of_string Sys.argv.(5) in
  let max_iterations = int_of_string Sys.argv.(6) in
  if features < 4 then
    fail "multinomial-logistic benchmark requires at least 4 features";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (feature ~seed) |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let target =
    Target.classification (Array.init samples (classification_target x))
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples
      (Array.init samples (fun row -> 1.0 +. (Float.of_int (row mod 5) *. 0.25)))
    |> get_data
  in
  let fitted =
    Multinomial_logistic_regression.fit
      (Multinomial_logistic_regression.create ~c ~tolerance ~max_iterations ()
      |> get)
      ~sample_weight
      ~rng:(Rng.create (Seed.of_int seed))
      ~feature_schema ~x ~y:target ()
    |> get
  in
  let decisions =
    Multinomial_logistic_regression.decision_function fitted ~feature_schema ~x
    |> get
  in
  let probabilities =
    Multinomial_logistic_regression.predict_proba fitted ~feature_schema ~x
    |> get
  in
  let predictions =
    Multinomial_logistic_regression.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values
  in
  let last = samples - 1 in
  let signature =
    [|
      Matrix.get decisions 0 0;
      Matrix.get decisions 0 1;
      Matrix.get decisions 0 2;
      Matrix.get probabilities 0 0;
      Matrix.get probabilities 0 1;
      Matrix.get probabilities 0 2;
      Matrix.get probabilities last 0;
      Matrix.get probabilities last 1;
      Matrix.get probabilities last 2;
      Float.of_int predictions.(0);
      Float.of_int predictions.(last);
    |]
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["weighted_multinomial_logistic_fit","decision_function","predict_proba","predict"],"samples":%d,"signature":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    signature.(0) signature.(1) signature.(2) signature.(3) signature.(4)
    signature.(5) signature.(6) signature.(7) signature.(8) signature.(9)
    signature.(10)
