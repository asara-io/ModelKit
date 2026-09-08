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

let value ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.
  -. 5.

let target ~seed ~features row =
  let signal = ref 1.25 in
  for column = 0 to features - 1 do
    let coefficient = Float.of_int ((column mod 5) - 2) *. 0.2 in
    signal := !signal +. (coefficient *. value ~seed row column)
  done;
  !signal +. (Float.of_int ((((row * 13) + seed) mod 11) - 5) *. 0.01)

let pipeline store =
  let scaler =
    Pipeline.cacheable_transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let ridge = Ridge_regression.create ~alpha:1. () |> get in
  let estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) ridge |> get
  in
  Pipeline.add_transformer Pipeline.empty scaler |> get |> fun builder ->
  Pipeline.set_estimator builder estimator |> get |> fun specification ->
  Pipeline.with_cache specification store

let timed fit =
  let started = Sys.time () in
  let fitted = fit () in
  (fitted, Sys.time () -. started)

let () =
  if Array.length Sys.argv <> 5 then
    fail "usage: transform_cache_worker SAMPLES FEATURES SEED WARM_FITS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let warm_fits = int_of_string Sys.argv.(4) in
  if samples < 2 || features < 1 || warm_fits < 1 then
    fail "transform-cache benchmark dimensions are invalid";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (value ~seed) |> get_data
  in
  let y =
    Array.init samples (target ~seed ~features)
    |> Vector.of_array |> Target.regression |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let raw_cache = Transform_cache.Memory.create () in
  let specification = pipeline (Transform_cache.Store.memory raw_cache) in
  let fit () =
    Pipeline.fit specification
      ~rng:(Rng.create (Seed.of_int seed))
      ~feature_schema ~x ~y ()
    |> get
  in
  let _, cold_seconds = timed fit in
  let warm_seconds = ref 0. in
  let fitted = ref None in
  for _ = 1 to warm_fits do
    let model, elapsed = timed fit in
    fitted := Some model;
    warm_seconds := !warm_seconds +. elapsed
  done;
  let fitted = Option.get !fitted in
  let predictions =
    Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.regression_values
  in
  let signature =
    [| Vector.get predictions 0; Vector.get predictions (samples - 1) |]
  in
  let signature_text =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let cache_stats = Transform_cache.Memory.stats raw_cache in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"cache_entries":%d,"cache_hits":%Ld,"cache_misses":%Ld,"checksum":%S,"cold_seconds":%.9g,"features":%d,"ocaml":%S,"operations":["cold_pipeline_fit","warm_pipeline_fit","standard_scaler_cache_decode","ridge_fit"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}],"warm_fits":%d,"warm_seconds_per_fit":%.9g}|}
    allocated_words cache_stats.Transform_cache.Memory.entries
    cache_stats.Transform_cache.Memory.hits
    cache_stats.Transform_cache.Memory.misses signature_text cold_seconds
    features Sys.ocaml_version samples signature_text warm_fits
    (!warm_seconds /. Float.of_int warm_fits)
