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

(* Deterministic feature generator shared with the scikit-learn worker. The
   last [duplicates] columns copy the first columns exactly, which makes the
   design rank-deficient by construction. *)
let feature ~seed ~duplicates ~features row column =
  let column =
    if column >= features - duplicates then column - (features - duplicates)
    else column
  in
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

let binary_target x row =
  let score =
    Matrix.get x row 0
    +. (0.25 *. Matrix.get x row 1)
    -. (0.1 *. Matrix.get x row 2)
  in
  if score > 0.0 then 7 else -3

let multiclass_target x row =
  if Matrix.get x row 0 +. (0.25 *. Matrix.get x row 1) > 1.0 then 9
  else if Matrix.get x row 2 -. (0.2 *. Matrix.get x row 3) > 0.0 then 2
  else -4

type timing = { elapsed_ns : int64; allocated_words : float }

let timed f =
  let allocated_before = Gc.allocated_bytes () in
  let started = Unix.gettimeofday () in
  let result = f () in
  let elapsed = Unix.gettimeofday () -. started in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  (result, { elapsed_ns = Int64.of_float (elapsed *. 1e9); allocated_words })

let fit_json ~name ~timing ~(report : Solver_report.t) =
  Printf.sprintf
    {|%S:{"allocated_words":%.0f,"converged":%b,"elapsed_ns":%Ld,"iterations":%d,"rank":%s}|}
    name timing.allocated_words
    (Solver_report.converged report)
    timing.elapsed_ns
    (Solver_report.iterations report)
    (match Solver_report.rank report with
    | Some rank -> string_of_int rank
    | None -> "null")

type shape = { name : string; samples : int; features : int; duplicates : int }

let parse_shape text =
  match String.split_on_char ':' text with
  | [ name; samples; features; duplicates ] ->
      {
        name;
        samples = int_of_string samples;
        features = int_of_string features;
        duplicates = int_of_string duplicates;
      }
  | _ -> fail ("malformed shape " ^ text)

let run_shape ~seed ~ridge_alpha ~c ~tolerance ~max_iterations shape =
  let { name; samples; features; duplicates } = shape in
  if features < 4 then
    fail "solver-shape benchmark requires at least 4 features";
  let x =
    Matrix.init ~rows:samples ~columns:features
      (feature ~seed ~duplicates ~features)
    |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let regression =
    Vector.init ~length:samples (regression_target x)
    |> get_data |> Target.regression |> get_data
  in
  let binary = Target.classification (Array.init samples (binary_target x)) in
  let multiclass =
    Target.classification (Array.init samples (multiclass_target x))
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples
      (Array.init samples (fun row -> 1.0 +. (Float.of_int (row mod 5) *. 0.25)))
    |> get_data
  in
  let rng = Rng.create (Seed.of_int seed) in
  let linear, linear_timing =
    timed (fun () ->
        Linear_regression.fit
          (Linear_regression.create ())
          ~sample_weight ~rng ~feature_schema ~x ~y:regression ()
        |> get)
  in
  let linear_prediction =
    Linear_regression.predict linear ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let ridge, ridge_timing =
    timed (fun () ->
        Ridge_regression.fit
          (Ridge_regression.create ~alpha:ridge_alpha () |> get)
          ~sample_weight ~rng ~feature_schema ~x ~y:regression ()
        |> get)
  in
  let ridge_prediction =
    Ridge_regression.predict ridge ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let logistic, logistic_timing =
    timed (fun () ->
        Logistic_regression.fit
          (Logistic_regression.create ~c ~tolerance ~max_iterations () |> get)
          ~sample_weight ~rng ~feature_schema ~x ~y:binary ()
        |> get)
  in
  let probabilities =
    Logistic_regression.predict_proba logistic ~feature_schema ~x |> get
  in
  let multinomial, multinomial_timing =
    timed (fun () ->
        Multinomial_logistic_regression.fit
          (Multinomial_logistic_regression.create ~c ~tolerance ~max_iterations
             ()
          |> get)
          ~sample_weight ~rng ~feature_schema ~x ~y:multiclass ()
        |> get)
  in
  let multiclass_probabilities =
    Multinomial_logistic_regression.predict_proba multinomial ~feature_schema ~x
    |> get
  in
  let last = samples - 1 in
  let signature =
    [|
      Vector.get linear_prediction 0;
      Vector.get linear_prediction last;
      Float.of_int
        (Option.value ~default:(-1)
           (Solver_report.rank (Linear_regression.report linear)));
      Vector.get ridge_prediction 0;
      Vector.get ridge_prediction last;
      Matrix.get probabilities 0 1;
      Matrix.get probabilities last 1;
      Matrix.get multiclass_probabilities 0 0;
      Matrix.get multiclass_probabilities 0 2;
      Matrix.get multiclass_probabilities last 1;
    |]
  in
  let json =
    Printf.sprintf
      {|{"duplicates":%d,"features":%d,"fits":{%s,%s,%s,%s},"samples":%d,"shape":%S}|}
      duplicates features
      (fit_json ~name:"ordinary_least_squares" ~timing:linear_timing
         ~report:(Linear_regression.report linear))
      (fit_json ~name:"ridge_regression" ~timing:ridge_timing
         ~report:(Ridge_regression.report ridge))
      (fit_json ~name:"binary_logistic_regression" ~timing:logistic_timing
         ~report:(Logistic_regression.report logistic))
      (fit_json ~name:"multinomial_logistic_regression"
         ~timing:multinomial_timing
         ~report:(Multinomial_logistic_regression.report multinomial))
      samples name
  in
  (signature, json)

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: solver_shapes_worker SHAPES SEED RIDGE_ALPHA C TOLERANCE \
       MAX_ITERATIONS";
  let shapes = String.split_on_char ',' Sys.argv.(1) |> List.map parse_shape in
  let seed = int_of_string Sys.argv.(2) in
  let ridge_alpha = float_of_string Sys.argv.(3) in
  let c = float_of_string Sys.argv.(4) in
  let tolerance = float_of_string Sys.argv.(5) in
  let max_iterations = int_of_string Sys.argv.(6) in
  let allocated_before = Gc.allocated_bytes () in
  let results =
    List.map (run_shape ~seed ~ridge_alpha ~c ~tolerance ~max_iterations) shapes
  in
  let signature = Array.concat (List.map fst results) in
  let formatted =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"ocaml":%S,"operations":["weighted_ols_fit_predict","weighted_ridge_fit_predict","weighted_binary_logistic_fit_proba","weighted_multinomial_logistic_fit_proba"],"shapes":[%s],"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words
    (Digest.to_hex (Digest.string formatted))
    Sys.ocaml_version
    (String.concat "," (List.map snd results))
    formatted
