open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let data ?(offset = 0.) () =
  let x =
    Array.init 12 (fun row ->
        let value = Float.of_int row +. offset in
        [| value; (value *. value) +. 1. |])
    |> matrix
  in
  let y =
    Array.init 12 (fun row ->
        let value = Float.of_int row +. offset in
        (2. *. value) -. 3.)
    |> regression
  in
  (x, y)

let memory () =
  let raw = Transform_cache.Memory.create () in
  (raw, Transform_cache.Store.memory raw)

let estimator () =
  Pipeline.estimator ~name:"linear"
    (module Linear_regression)
    (Linear_regression.create ())
  |> get

let scaler ?(route_sample_weight = false) ?(with_mean = true) ?(with_std = true)
    () =
  Pipeline.cacheable_transformer ~route_sample_weight ~name:"scale"
    (module Standard_scaler)
    (Standard_scaler.create ~with_mean ~with_std ())
  |> get

let pipeline_of_transformer store transformer =
  Pipeline.add_transformer Pipeline.empty transformer |> get |> fun builder ->
  Pipeline.set_estimator builder (estimator ()) |> get |> fun pipeline ->
  Pipeline.with_cache pipeline store

let pipeline ?route_sample_weight ?with_mean ?with_std store =
  pipeline_of_transformer store
    (scaler ?route_sample_weight ?with_mean ?with_std ())

let fit ?sample_weight ?(seed = 17) specification x y =
  Pipeline.fit specification ?sample_weight
    ~rng:(Rng.create (Seed.of_int seed))
    ~feature_schema:(schema x) ~x ~y ()
  |> get

let stats raw = Transform_cache.Memory.stats raw

let check_stats raw ~entries ~hits ~misses =
  let observed = stats raw in
  Alcotest.(check int)
    "cache entries" entries observed.Transform_cache.Memory.entries;
  Alcotest.(check int64) "cache hits" hits observed.Transform_cache.Memory.hits;
  Alcotest.(check int64)
    "cache misses" misses observed.Transform_cache.Memory.misses

let predictions fitted x =
  Pipeline.predict fitted ~feature_schema:(schema x) ~x
  |> get |> Target.regression_values |> Vector.to_array

let test_builtin_round_trips () =
  let x, _ = data () in
  let feature_schema = schema x in
  let rng = Rng.create (Seed.of_int 1) in
  let check_imputer specification =
    let fitted =
      Simple_imputer.fit specification ~rng ~feature_schema ~x ~y:None () |> get
    in
    let decoded =
      Simple_imputer.encode_fitted fitted
      |> get |> Simple_imputer.decode_fitted |> get
    in
    Alcotest.(check bool)
      "imputer schema" true
      (Feature_schema.equal feature_schema
         (Simple_imputer.input_schema decoded))
  in
  check_imputer (Simple_imputer.mean ());
  check_imputer (Simple_imputer.median ());
  check_imputer (Simple_imputer.constant 4. |> get);
  let specification = Standard_scaler.create () in
  let fitted =
    Standard_scaler.fit specification ~rng ~feature_schema ~x ~y:None () |> get
  in
  let decoded =
    Standard_scaler.encode_fitted fitted
    |> get |> Standard_scaler.decode_fitted |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "scaler mean"
    (Standard_scaler.mean fitted |> Vector.to_array)
    (Standard_scaler.mean decoded |> Vector.to_array);
  match Standard_scaler.decode_fitted (Bytes.of_string "invalid") with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "invalid cache payload was accepted"

let test_pipeline_keys_and_reuse () =
  let x, y = data () in
  let raw, store = memory () in
  let specification = pipeline store in
  Alcotest.(check bool)
    "cache enabled" true
    (Pipeline.cache_enabled specification);
  Alcotest.(check bool)
    "cache disabled" false
    (Pipeline.cache_enabled (Pipeline.without_cache specification));
  let first = fit specification x y in
  let second = fit specification x y in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "warm prediction" (predictions first x) (predictions second x);
  check_stats raw ~entries:1 ~hits:1L ~misses:1L;
  let changed_x, changed_y = data ~offset:0.5 () in
  ignore (fit specification changed_x changed_y);
  ignore (fit ~seed:18 specification x y);
  ignore (fit (pipeline ~with_mean:false store) x y);
  check_stats raw ~entries:4 ~hits:1L ~misses:4L;
  let weighted_raw, weighted_store = memory () in
  let weighted = pipeline ~route_sample_weight:true weighted_store in
  let weights values =
    Sample_weight.create ~expected_length:(Array.length values)
      (Vector.of_array values)
    |> get_data
  in
  let first_weights = weights (Array.make 12 1.) in
  let second_weights =
    weights (Array.init 12 (fun index -> if index = 0 then 2. else 1.))
  in
  ignore (fit ~sample_weight:first_weights weighted x y);
  ignore (fit ~sample_weight:first_weights weighted x y);
  ignore (fit ~sample_weight:second_weights weighted x y);
  check_stats weighted_raw ~entries:2 ~hits:1L ~misses:2L;
  let ignored_raw, ignored_store = memory () in
  let ignored = pipeline ~route_sample_weight:false ignored_store in
  ignore (fit ~sample_weight:first_weights ignored x y);
  ignore (fit ~sample_weight:second_weights ignored x y);
  check_stats ignored_raw ~entries:1 ~hits:1L ~misses:1L

let test_artifact_stage_cache () =
  let x, y = data () in
  let raw, store = memory () in
  let transformer =
    Artifact.standard_scaler_stage ~name:"scale" (Standard_scaler.create ())
    |> get
  in
  let specification = pipeline_of_transformer store transformer in
  ignore (fit specification x y);
  ignore (fit specification x y);
  check_stats raw ~entries:1 ~hits:1L ~misses:1L

let scaler_key x seed =
  let feature_schema = schema x in
  let training_data =
    Transform_cache.Content_id.combine ~domain:"pipeline-training-input-v1"
      [|
        Transform_cache.Content_id.of_string
          (Schema_fingerprint.to_string
             (Feature_schema.fingerprint feature_schema));
        Transform_cache.Content_id.of_matrix x;
      |]
    |> get
  in
  let routed_metadata =
    Transform_cache.Content_id.combine ~domain:"pipeline-routed-metadata-v1"
      [| Transform_cache.Content_id.of_string "no-sample-weight" |]
    |> get
  in
  let stage_seed =
    Seed.derive (Seed.of_int seed) ~operation:"pipeline-transformer:scale"
      ~index:0
  in
  Transform_cache.Key.create ~component:Standard_scaler.cache_component
    ~configuration:
      (Standard_scaler.cache_configuration (Standard_scaler.create ()))
    ~training_data ~target:None ~routed_metadata ~seed:stage_seed

let test_invalid_cached_state_refits () =
  let x, y = data () in
  let raw, store = memory () in
  Transform_cache.Memory.put raw (scaler_key x 17) (Bytes.of_string "invalid")
  |> get;
  let specification = pipeline store in
  ignore (fit specification x y);
  ignore (fit specification x y);
  check_stats raw ~entries:1 ~hits:2L ~misses:0L

let remove_if_present path = try Sys.remove path with Sys_error _ -> ()

let clean_directory path =
  (try
     Array.iter
       (fun name -> remove_if_present (Filename.concat path name))
       (Sys.readdir path)
   with Sys_error _ -> ());
  try Sys.rmdir path with Sys_error _ -> ()

let test_persistent_workflow () =
  let root = Filename.temp_file "modelkit-pipeline-cache-" "" in
  Sys.remove root;
  Sys.mkdir root 0o700;
  Fun.protect
    ~finally:(fun () -> clean_directory root)
    (fun () ->
      let x, y = data () in
      let first_store =
        Transform_cache.Persistent.create ~root ()
        |> get |> Transform_cache.Store.persistent
      in
      let first = fit (pipeline first_store) x y in
      let second_store =
        Transform_cache.Persistent.create ~root ()
        |> get |> Transform_cache.Store.persistent
      in
      let second = fit (pipeline second_store) x y in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "persistent prediction" (predictions first x) (predictions second x);
      Alcotest.(check int)
        "one persistent entry" 1
        (Sys.readdir root |> Array.to_list
        |> List.filter (String.ends_with ~suffix:".mkcache")
        |> List.length))

module Target_shift = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type fitted = { shift : float; schema : Feature_schema.t }
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y () =
    match y with
    | Some y ->
        let values = Target.regression_values y in
        let total = ref 0. in
        for index = 0 to Vector.length values - 1 do
          total := !total +. Vector.get values index
        done;
        Ok
          {
            shift = !total /. Float.of_int (Vector.length values);
            schema = feature_schema;
          }
    | None ->
        Error
          (Error.make ~remediation:"provide regression targets"
             (Error.Validation
                { name = "target shift"; reason = "targets are required" }))

  let transform fitted ~feature_schema:_ ~x =
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
      (fun row column -> Matrix.get x row column +. fitted.shift)
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"use representable matrix dimensions"
          error)

  let fitted_params _ = ()
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema

  let cache_component =
    Transform_cache.Component.create ~package:"modelkit-tests"
      ~name:"target-shift" ~version:1
    |> get

  let cache_configuration () = Transform_cache.Content_id.of_string "unit"

  let encode_fitted fitted =
    Ok (Bytes.of_string (Int64.bits_of_float fitted.shift |> Int64.to_string))

  let decode_fitted payload =
    match Int64.of_string_opt (Bytes.to_string payload) with
    | Some bits ->
        let shift = Int64.float_of_bits bits in
        if Float.is_finite shift then
          Feature_schema.anonymous ~feature_count:2
          |> Result.map (fun schema -> { shift; schema })
          |> Result.map_error (fun error ->
              Error.of_data_error ~remediation:"discard the cache entry" error)
        else
          Error
            (Error.make ~remediation:"discard the cache entry"
               (Error.Compatibility
                  { component = "target shift"; reason = "non-finite shift" }))
    | None ->
        Error
          (Error.make ~remediation:"discard the cache entry"
             (Error.Compatibility
                { component = "target shift"; reason = "invalid payload" }))
end

let test_target_key_and_preflight () =
  let x, y = data () in
  let raw, store = memory () in
  let stage =
    Pipeline.Supervised.cacheable_transformer ~name:"target"
      (module Target_shift)
      ()
    |> get
  in
  let specification =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty stage |> get
    |> fun builder ->
    Pipeline.Supervised.set_estimator builder (estimator ()) |> get
    |> fun pipeline -> Pipeline.with_cache pipeline store
  in
  ignore (fit specification x y);
  ignore (fit specification x y);
  let changed =
    Target.regression
      (Target.regression_values y |> Vector.to_array
      |> Array.map (fun value -> value +. 1.)
      |> Vector.of_array)
    |> get_data
  in
  ignore (fit specification x changed);
  check_stats raw ~entries:2 ~hits:1L ~misses:2L;
  let unsupported_raw, unsupported_store = memory () in
  let ordinary =
    Pipeline.transformer ~name:"ordinary"
      (module Variance_threshold)
      (Variance_threshold.create () |> get)
    |> get
  in
  let builder =
    Pipeline.add_transformer Pipeline.empty (scaler ()) |> get |> fun builder ->
    Pipeline.add_transformer builder ordinary |> get
  in
  let unsupported =
    Pipeline.set_estimator builder (estimator ()) |> get |> fun pipeline ->
    Pipeline.with_cache pipeline unsupported_store
  in
  (match
     Pipeline.fit unsupported
       ~rng:(Rng.create (Seed.of_int 17))
       ~feature_schema:(schema x) ~x ~y ()
   with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "unsupported cache stage was accepted");
  check_stats unsupported_raw ~entries:0 ~hits:0L ~misses:0L

let check_nested make_stage =
  let x, y = data () in
  let raw, store = memory () in
  let specification = pipeline_of_transformer store (make_stage ()) in
  ignore (fit specification x y);
  ignore (fit specification x y);
  check_stats raw ~entries:1 ~hits:1L ~misses:1L

let test_nested_composition () =
  check_nested (fun () ->
      Transformer_pipeline.create [| scaler () |]
      |> get
      |> Transformer_pipeline.stage ~name:"chain"
      |> get);
  check_nested (fun () ->
      Feature_union.create [| Feature_union.transformer (scaler ()) |]
      |> get
      |> Feature_union.stage ~name:"union"
      |> get);
  check_nested (fun () ->
      Column_transformer.create
        [|
          Column_transformer.transformer ~columns:Column_selector.all
            (scaler ());
        |]
      |> get
      |> Column_transformer.stage ~name:"columns"
      |> get);
  let x, y = data () in
  let raw, store = memory () in
  let child =
    Pipeline.Supervised.cacheable_transformer ~name:"target"
      (module Target_shift)
      ()
    |> get
  in
  let nested =
    Transformer_pipeline.Supervised.create [| child |]
    |> get
    |> Transformer_pipeline.Supervised.stage ~name:"supervised-chain"
    |> get
  in
  let specification =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty nested |> get
    |> fun builder ->
    Pipeline.Supervised.set_estimator builder (estimator ()) |> get
    |> fun pipeline -> Pipeline.with_cache pipeline store
  in
  ignore (fit specification x y);
  ignore (fit specification x y);
  check_stats raw ~entries:1 ~hits:1L ~misses:1L

let dataset () =
  let x, y = data () in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let splitter () =
  K_fold.create ~folds:3 () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let test_cross_validation_and_search () =
  let raw, store = memory () in
  let specification = pipeline store in
  let run_cv () =
    Cross_validation.Regression.cross_validate ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 41) specification (dataset ())
    |> get |> ignore
  in
  run_cv ();
  check_stats raw ~entries:3 ~hits:0L ~misses:3L;
  run_cv ();
  check_stats raw ~entries:3 ~hits:3L ~misses:3L;
  let search_raw, search_store = memory () in
  let axis =
    Grid_search.axis ~name:"alpha" ~values:[| 0.; 1. |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ alpha -> Ok alpha)
    |> get
  in
  let grid =
    Grid_search.create ~base:0.
      ~build:(fun alpha ->
        let ridge = Ridge_regression.create ~alpha () |> get in
        let estimator =
          Pipeline.estimator ~name:"ridge" (module Ridge_regression) ridge
          |> get
        in
        Pipeline.add_transformer Pipeline.empty (scaler ()) |> get
        |> fun builder ->
        Pipeline.set_estimator builder estimator
        |> Result.map (fun pipeline ->
            Pipeline.with_cache pipeline search_store))
      [| axis |]
    |> get
  in
  let run_search () =
    Grid_search.Regression.search_with_policy ~grid ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~policy:Grid_search.No_refit ~seed:(Seed.of_int 53) (dataset ())
    |> get |> ignore
  in
  run_search ();
  check_stats search_raw ~entries:6 ~hits:0L ~misses:6L;
  run_search ();
  check_stats search_raw ~entries:6 ~hits:6L ~misses:6L

let () =
  Alcotest.run "Pipeline transform cache"
    [
      ( "integration",
        [
          Alcotest.test_case "built-in codecs" `Quick test_builtin_round_trips;
          Alcotest.test_case "complete keys and warm reuse" `Quick
            test_pipeline_keys_and_reuse;
          Alcotest.test_case "artifact stage cache" `Quick
            test_artifact_stage_cache;
          Alcotest.test_case "invalid state refits" `Quick
            test_invalid_cached_state_refits;
          Alcotest.test_case "persistent workflow" `Quick
            test_persistent_workflow;
          Alcotest.test_case "target key and preflight" `Quick
            test_target_key_and_preflight;
          Alcotest.test_case "nested composition" `Quick test_nested_composition;
          Alcotest.test_case "cross-validation and search" `Quick
            test_cross_validation_and_search;
        ] );
    ]
