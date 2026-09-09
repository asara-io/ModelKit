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

let raw_value ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.0
  -. 5.0

let coefficient column = Float.of_int ((column mod 5) - 2) *. 0.2

let target ~seed ~features group =
  let row = group * 2 in
  let value = ref 1.25 in
  for column = 0 to features - 1 do
    value := !value +. (coefficient column *. raw_value ~seed row column)
  done;
  !value

let pipeline alpha =
  let scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let ridge = Ridge_regression.create ~alpha () |> get in
  let estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) ridge |> get
  in
  Pipeline.add_transformer Pipeline.empty scaler |> get |> fun pipeline ->
  Pipeline.set_estimator pipeline estimator |> get

let signature report =
  Array.concat
    [
      [| Permutation_test.observed_score report |];
      Permutation_test.permutation_scores report;
      [| Permutation_test.p_value report |];
    ]

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: permutation_test_worker SAMPLES FEATURES SEED FOLDS PERMUTATIONS \
       RIDGE_ALPHA";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let folds = int_of_string Sys.argv.(4) in
  let permutations = int_of_string Sys.argv.(5) in
  let alpha = float_of_string Sys.argv.(6) in
  if samples mod 2 <> 0 then
    fail "permutation-test benchmark requires an even sample count";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (raw_value ~seed) |> get_data
  in
  let y =
    Target.regression
      (Vector.of_array
         (Array.init samples (fun row -> target ~seed ~features (row / 2))))
    |> get_data
  in
  let groups =
    Groups.create ~expected_length:samples
      (Array.init samples (fun row -> row / 2))
    |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~groups ~x ~y ()
    |> get_data
  in
  let splitter =
    K_fold.create ~folds () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let specification = Permutation_test.create ~permutations () |> get in
  let report =
    Permutation_test.Regression.evaluate ~specification ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error ~seed:(Seed.of_int seed)
      (pipeline alpha) dataset
    |> get
  in
  let signature = signature report in
  let signature_text =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"folds":%d,"ocaml":%S,"operations":["standard_scaling","ridge_regression","permutation_test_score","within_group_shuffling","corrected_p_value"],"permutations":%d,"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features folds Sys.ocaml_version
    permutations samples signature_text
