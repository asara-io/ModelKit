open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let component name =
  Transform_cache.Component.create ~package:"example" ~name ~version:1 |> get

let key ?(component = component "scale") ?(configuration = "configuration")
    ?(training_data = "training-data") ?target ?(routed_metadata = "metadata")
    ?(seed = 7) () =
  Transform_cache.Key.create ~component
    ~configuration:(Transform_cache.Content_id.of_string configuration)
    ~training_data:(Transform_cache.Content_id.of_string training_data)
    ~target:(Option.map Transform_cache.Content_id.of_string target)
    ~routed_metadata:(Transform_cache.Content_id.of_string routed_metadata)
    ~seed:(Seed.of_int seed)

let test_identities_and_keys () =
  let identity = component "standard-scaler" in
  Alcotest.(check string)
    "component rendering" "example:standard-scaler@1"
    (Transform_cache.Component.to_string identity);
  expect_error "blank package"
    (Transform_cache.Component.create ~package:" " ~name:"scale" ~version:1);
  expect_error "blank name"
    (Transform_cache.Component.create ~package:"example" ~name:"" ~version:1);
  expect_error "invalid version"
    (Transform_cache.Component.create ~package:"example" ~name:"scale"
       ~version:0);
  expect_error "blank content domain"
    (Transform_cache.Content_id.combine ~domain:" " [||]);
  let left = Transform_cache.Content_id.of_string "left" in
  let right = Transform_cache.Content_id.of_string "right" in
  let combined =
    Transform_cache.Content_id.combine ~domain:"pair" [| left; right |] |> get
  in
  let reversed =
    Transform_cache.Content_id.combine ~domain:"pair" [| right; left |] |> get
  in
  Alcotest.(check bool)
    "ordered combination" false
    (Transform_cache.Content_id.equal combined reversed);
  let base = key ~target:"target" () in
  let variants =
    [|
      key ~component:(component "other") ~target:"target" ();
      key ~configuration:"other" ~target:"target" ();
      key ~training_data:"other" ~target:"target" ();
      key ~target:"other" ();
      key ();
      key ~target:"target" ~routed_metadata:"other" ();
      key ~target:"target" ~seed:8 ();
    |]
  in
  Array.iter
    (fun variant ->
      Alcotest.(check bool)
        "key field isolation" false
        (Transform_cache.Key.equal base variant))
    variants;
  Alcotest.(check string)
    "stable key" "f23190fff78087fb719b1c4e7652399f"
    (Transform_cache.Key.to_hex base)

let cache_component = component "external-transformer"

module External_cacheable_transformer = struct
  type t = int
  type params = int
  type target = unit
  type fitted = int
  type rng = Rng.t

  let clone value = value
  let params value = value

  let fit value ?sample_weight:_ ~rng:_ ~feature_schema:_ ~x:_ ~y:_ () =
    Ok value

  let transform _ ~feature_schema:_ ~x = Ok x
  let fitted_params value = value

  let input_schema _ =
    Feature_schema.anonymous ~feature_count:1 |> Result.get_ok

  let output_schema = input_schema
  let cache_component = cache_component

  let cache_configuration value =
    string_of_int value |> Transform_cache.Content_id.of_string

  let encode_fitted value = Ok (Bytes.of_string (string_of_int value))

  let decode_fitted payload =
    match int_of_string_opt (Bytes.to_string payload) with
    | Some value -> Ok value
    | None ->
        Error
          (Error.make ~remediation:"use a payload produced by this cache codec"
             (Error.Compatibility
                {
                  component = "external-transformer cache codec";
                  reason = "invalid fitted integer payload";
                }))
end

let test_codec_contract () =
  let codec =
    Transform_cache.Codec.of_module (module External_cacheable_transformer)
  in
  Alcotest.(check bool)
    "component retained" true
    (Transform_cache.Component.equal cache_component
       (Transform_cache.Codec.component codec));
  let configuration = Transform_cache.Codec.configuration codec 42 in
  Alcotest.(check bool)
    "configuration identity" true
    (Transform_cache.Content_id.equal configuration
       (Transform_cache.Content_id.of_string "42"));
  let encoded = Transform_cache.Codec.encode codec 42 |> get in
  Alcotest.(check int)
    "fitted round trip" 42
    (Transform_cache.Codec.decode codec encoded |> get);
  expect_error "invalid fitted payload"
    (Transform_cache.Codec.decode codec (Bytes.of_string "invalid"));
  let supported = Transform_cache.Codec.Supported codec in
  ignore
    (Transform_cache.Codec.require ~component:"external-transformer" supported
    |> get);
  expect_error "unsupported codec"
    (Transform_cache.Codec.require ~component:"ordinary-transformer"
       Transform_cache.Codec.Unsupported)

let test_bounded_memory () =
  expect_error "zero entry limit"
    (Transform_cache.Memory.limits ~max_entries:0 ~max_bytes:10L);
  expect_error "zero byte limit"
    (Transform_cache.Memory.limits ~max_entries:1 ~max_bytes:0L);
  let limits =
    Transform_cache.Memory.limits ~max_entries:2 ~max_bytes:5L |> get
  in
  let cache = Transform_cache.Memory.create ~limits () in
  let first = key ~configuration:"first" () in
  let second = key ~configuration:"second" () in
  let third = key ~configuration:"third" () in
  Transform_cache.Memory.put cache first (Bytes.of_string "aa") |> get;
  Transform_cache.Memory.put cache second (Bytes.of_string "bbb") |> get;
  let copy = Transform_cache.Memory.get cache first |> Option.get in
  Bytes.set copy 0 'z';
  Alcotest.(check string)
    "get returns a copy" "aa"
    (Transform_cache.Memory.get cache first |> Option.get |> Bytes.to_string);
  Transform_cache.Memory.put cache third (Bytes.of_string "c") |> get;
  Alcotest.(check bool)
    "oldest write evicted" true
    (Option.is_none (Transform_cache.Memory.get cache first));
  let before = Transform_cache.Memory.stats cache in
  expect_error "oversized payload"
    (Transform_cache.Memory.put cache first (Bytes.of_string "123456"));
  let after = Transform_cache.Memory.stats cache in
  Alcotest.(check int)
    "oversize leaves entries unchanged" before.Transform_cache.Memory.entries
    after.Transform_cache.Memory.entries;
  Alcotest.(check int64)
    "oversize leaves bytes unchanged"
    before.Transform_cache.Memory.payload_bytes
    after.Transform_cache.Memory.payload_bytes;
  Alcotest.(check bool)
    "remove existing" true
    (Transform_cache.Memory.remove cache second);
  Alcotest.(check bool)
    "remove absent" false
    (Transform_cache.Memory.remove cache second);
  Transform_cache.Memory.clear cache;
  let cleared = Transform_cache.Memory.stats cache in
  Alcotest.(check int) "clear entries" 0 cleared.Transform_cache.Memory.entries;
  Alcotest.(check int64)
    "clear bytes" 0L cleared.Transform_cache.Memory.payload_bytes;
  Transform_cache.Memory.put cache first Bytes.empty |> get;
  Alcotest.(check bool)
    "remove zero-byte payload" true
    (Transform_cache.Memory.remove cache first)

let test_concurrent_memory () =
  let limits =
    Transform_cache.Memory.limits ~max_entries:400 ~max_bytes:10_000L |> get
  in
  let cache = Transform_cache.Memory.create ~limits () in
  let worker domain =
    for index = 0 to 99 do
      let key = key ~configuration:(Format.sprintf "%d:%d" domain index) () in
      Transform_cache.Memory.put cache key (Bytes.of_string "value") |> get;
      match Transform_cache.Memory.get cache key with
      | Some value when Bytes.equal value (Bytes.of_string "value") -> ()
      | None | Some _ -> Alcotest.fail "concurrent cache read"
    done
  in
  let domains =
    Array.init 4 (fun domain -> Domain.spawn (fun () -> worker domain))
  in
  Array.iter Domain.join domains;
  let stats = Transform_cache.Memory.stats cache in
  Alcotest.(check int)
    "all concurrent entries" 400 stats.Transform_cache.Memory.entries;
  Alcotest.(check int64)
    "all concurrent hits" 400L stats.Transform_cache.Memory.hits

let () =
  Alcotest.run "Transform cache foundation"
    [
      ( "contracts",
        [
          Alcotest.test_case "identities and keys" `Quick
            test_identities_and_keys;
          Alcotest.test_case "codec contract" `Quick test_codec_contract;
          Alcotest.test_case "bounded memory" `Quick test_bounded_memory;
          Alcotest.test_case "concurrent memory" `Quick test_concurrent_memory;
        ] );
    ]
