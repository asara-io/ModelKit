open Modelkit

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let data_value ~seed ~missing_modulus row column =
  if column = 0 then 1.0
  else
    let row = Int64.of_int row in
    let column = Int64.of_int column in
    let seed = Int64.of_int seed in
    let missing_hash =
      Int64.add (Int64.add (Int64.mul row 101L) (Int64.mul column 53L)) seed
    in
    if Int64.rem missing_hash (Int64.of_int missing_modulus) = 0L then Float.nan
    else
      let value_hash =
        Int64.add (Int64.add (Int64.mul row 17L) (Int64.mul column 31L)) seed
      in
      Int64.rem value_hash 1000L |> Int64.to_float |> fun value ->
      value /. 100.0

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: preprocessing_worker SAMPLES FEATURES SEED MISSING_MODULUS \
       VARIANCE_THRESHOLD IMPUTATION_CONSTANT";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let missing_modulus = int_of_string Sys.argv.(4) in
  let threshold = float_of_string Sys.argv.(5) in
  let imputation_constant = float_of_string Sys.argv.(6) in
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features
      (data_value ~seed ~missing_modulus)
    |> function
    | Ok matrix -> matrix
    | Error error -> fail (Data_error.to_string error)
  in
  let schema =
    Feature_schema.of_matrix x |> function
    | Ok schema -> schema
    | Error error -> fail (Data_error.to_string error)
  in
  let rng = Rng.create (Seed.of_int seed) in
  let imputer =
    Simple_imputer.fit (Simple_imputer.mean ()) ~rng ~feature_schema:schema ~x
      ~y:None ()
    |> get
  in
  let complete =
    Simple_imputer.transform imputer ~feature_schema:schema ~x |> get
  in
  let median_imputer =
    Simple_imputer.fit (Simple_imputer.median ()) ~rng ~feature_schema:schema ~x
      ~y:None ()
    |> get
  in
  let median_complete =
    Simple_imputer.transform median_imputer ~feature_schema:schema ~x |> get
  in
  let constant_imputer =
    Simple_imputer.fit
      (Simple_imputer.constant imputation_constant |> get)
      ~rng ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  let constant_complete =
    Simple_imputer.transform constant_imputer ~feature_schema:schema ~x |> get
  in
  let scaler =
    Standard_scaler.fit
      (Standard_scaler.create ())
      ~rng ~feature_schema:schema ~x:complete ~y:None ()
    |> get
  in
  let scaled =
    Standard_scaler.transform scaler ~feature_schema:schema ~x:complete |> get
  in
  let selector =
    Variance_threshold.fit
      (Variance_threshold.create ~threshold () |> get)
      ~rng ~feature_schema:schema ~x:scaled ~y:None ()
    |> get
  in
  let selected =
    Variance_threshold.transform selector ~feature_schema:schema ~x:scaled
    |> get
  in
  let rows, columns = Matrix.shape selected in
  let first =
    if rows = 0 || columns = 0 then 0.0 else Matrix.get selected 0 0
  in
  let last =
    if rows = 0 || columns = 0 then 0.0
    else Matrix.get selected (rows - 1) (columns - 1)
  in
  let median_value = Matrix.get median_complete 0 1 in
  let constant_value = Matrix.get constant_complete 0 1 in
  let checksum =
    Printf.sprintf "%d:%d:%.17g:%.17g:%.17g:%.17g" rows columns first last
      median_value constant_value
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features_out":%d,"ocaml":%S,"operations":["constant_imputation","mean_imputation","median_imputation","standard_scaling","variance_threshold"],"samples":%d,"signature":[%d,%d,%.17g,%.17g,%.17g,%.17g],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words checksum columns Sys.ocaml_version rows rows columns first
    last median_value constant_value
