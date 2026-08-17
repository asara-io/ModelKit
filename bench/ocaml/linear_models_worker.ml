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

let regression_coefficient column = Float.of_int ((column mod 5) - 2) *. 0.2

let regression_target x row =
  let value = ref 1.25 in
  for column = 0 to Matrix.columns x - 1 do
    value := !value +. (Matrix.get x row column *. regression_coefficient column)
  done;
  let noise = Float.of_int ((((row * 13) + 1729) mod 11) - 5) *. 0.01 in
  !value +. noise

let classification_target x row =
  let score =
    Matrix.get x row 0
    +. (0.25 *. Matrix.get x row 1)
    -. (0.1 *. Matrix.get x row 2)
  in
  if score > 0.0 then 7 else -3

let boundary_index x =
  let selected = ref 0 in
  let smallest = ref Float.infinity in
  for row = 0 to Matrix.rows x - 1 do
    let score =
      Matrix.get x row 0
      +. (0.25 *. Matrix.get x row 1)
      -. (0.1 *. Matrix.get x row 2)
      |> Float.abs
    in
    if score < !smallest then (
      selected := row;
      smallest := score)
  done;
  !selected

let signature linear ridge probabilities prediction boundary =
  let last = Vector.length linear - 1 in
  let next = Int.min last (boundary + 1) in
  [|
    Vector.get linear 0;
    Vector.get linear last;
    Vector.get ridge 0;
    Vector.get ridge last;
    Matrix.get probabilities boundary 1;
    Matrix.get probabilities next 1;
    Float.of_int prediction.(boundary);
    Float.of_int prediction.(next);
  |]

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 8 then
    fail
      "usage: linear_models_worker SAMPLES FEATURES SEED RIDGE_ALPHA \
       LOGISTIC_C LOGISTIC_TOLERANCE LOGISTIC_MAX_ITERATIONS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let ridge_alpha = float_of_string Sys.argv.(4) in
  let logistic_c = float_of_string Sys.argv.(5) in
  let logistic_tolerance = float_of_string Sys.argv.(6) in
  let logistic_max_iterations = int_of_string Sys.argv.(7) in
  if features < 3 then
    fail "linear-model benchmark requires at least 3 features";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (feature ~seed) |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let regression =
    Vector.init ~length:samples (regression_target x)
    |> get_data |> Target.regression |> get_data
  in
  let classification =
    Target.classification (Array.init samples (classification_target x))
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples
      (Array.init samples (fun row -> 1.0 +. (Float.of_int (row mod 5) *. 0.25)))
    |> get_data
  in
  let rng = Rng.create (Seed.of_int seed) in
  let linear =
    Linear_regression.fit
      (Linear_regression.create ())
      ~sample_weight ~rng ~feature_schema ~x ~y:regression ()
    |> get
  in
  let linear_prediction =
    Linear_regression.predict linear ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let ridge =
    Ridge_regression.fit
      (Ridge_regression.create ~alpha:ridge_alpha () |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y:regression ()
    |> get
  in
  let ridge_prediction =
    Ridge_regression.predict ridge ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let logistic =
    Logistic_regression.fit
      (Logistic_regression.create ~c:logistic_c ~tolerance:logistic_tolerance
         ~max_iterations:logistic_max_iterations ()
      |> get)
      ~sample_weight ~rng ~feature_schema ~x ~y:classification ()
    |> get
  in
  let probabilities =
    Logistic_regression.predict_proba logistic ~feature_schema ~x |> get
  in
  let prediction =
    Logistic_regression.predict logistic ~feature_schema ~x
    |> get |> Target.classification_values
  in
  let signature =
    signature linear_prediction ridge_prediction probabilities prediction
      (boundary_index x)
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["ordinary_least_squares","ridge_regression","binary_logistic_regression"],"samples":%d,"signature":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    signature.(0) signature.(1) signature.(2) signature.(3) signature.(4)
    signature.(5) signature.(6) signature.(7)
