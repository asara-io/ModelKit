open Modelkit
module Persistent = Transform_cache.Persistent

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let key name =
  let component =
    Transform_cache.Component.create ~package:"example" ~name:"persisted"
      ~version:1
    |> get
  in
  Transform_cache.Key.create ~component
    ~configuration:(Transform_cache.Content_id.of_string name)
    ~training_data:(Transform_cache.Content_id.of_string "training")
    ~target:None
    ~routed_metadata:(Transform_cache.Content_id.of_string "metadata")
    ~seed:(Seed.of_int 11)

let remove_if_present path = try Sys.remove path with Sys_error _ -> ()

let clean_directory path =
  (try
     Array.iter
       (fun name -> remove_if_present (Filename.concat path name))
       (Sys.readdir path)
   with Sys_error _ -> ());
  try Sys.rmdir path with Sys_error _ -> ()

let with_temp_directory test =
  let path = Filename.temp_file "modelkit-cache-test-" "" in
  Sys.remove path;
  Sys.mkdir path 0o700;
  Fun.protect ~finally:(fun () -> clean_directory path) (fun () -> test path)

let cache_file root =
  match
    Sys.readdir root |> Array.to_list
    |> List.filter (String.ends_with ~suffix:".mkcache")
  with
  | [ name ] -> Filename.concat root name
  | _ -> Alcotest.fail "expected one persistent cache entry"

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let length = in_channel_length channel in
      let bytes = Bytes.create length in
      really_input channel bytes 0 length;
      bytes)

let write_file path bytes =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_bytes channel bytes)

let expect_hit expected = function
  | Persistent.Hit payload ->
      Alcotest.(check bytes) "cached payload" expected payload
  | Persistent.Miss -> Alcotest.fail "expected a cache hit, observed a miss"
  | Persistent.Corrupt error -> Alcotest.fail (Error.to_string error)

let test_root_and_limits () =
  expect_error "zero payload limit"
    (Transform_cache.Persistent.limits ~max_payload_bytes:0);
  expect_error "payload limit leaves no framing space"
    (Transform_cache.Persistent.limits ~max_payload_bytes:max_int);
  expect_error "blank cache root"
    (Transform_cache.Persistent.create ~root:" " ());
  let file = Filename.temp_file "modelkit-cache-root-" "" in
  Fun.protect
    ~finally:(fun () -> remove_if_present file)
    (fun () ->
      expect_error "file cannot be a cache root"
        (Transform_cache.Persistent.create ~root:file ()));
  let parent = Filename.temp_file "modelkit-cache-parent-" "" in
  Sys.remove parent;
  Sys.mkdir parent 0o700;
  let root = Filename.concat parent "cache" in
  Fun.protect
    ~finally:(fun () ->
      clean_directory root;
      clean_directory parent)
    (fun () ->
      let cache = Transform_cache.Persistent.create ~root () |> get in
      Alcotest.(check bool) "created root" true (Sys.is_directory root);
      Alcotest.(check bool)
        "absolute retained root" false
        (Filename.is_relative (Transform_cache.Persistent.root cache)))

let test_round_trip_and_conflict () =
  with_temp_directory (fun root ->
      let cache = Transform_cache.Persistent.create ~root () |> get in
      let key = key "round-trip" in
      let payload = Bytes.of_string "fitted-state" in
      Alcotest.(check bool)
        "first publication" true
        (match Transform_cache.Persistent.put cache key payload |> get with
        | Persistent.Published -> true
        | Persistent.Already_present -> false);
      Bytes.set payload 0 'x';
      Transform_cache.Persistent.get cache key
      |> get
      |> expect_hit (Bytes.of_string "fitted-state");
      Alcotest.(check bool)
        "idempotent publication" true
        (match
           Transform_cache.Persistent.put cache key
             (Bytes.of_string "fitted-state")
           |> get
         with
        | Persistent.Already_present -> true
        | Persistent.Published -> false);
      expect_error "conflicting valid payload"
        (Transform_cache.Persistent.put cache key (Bytes.of_string "different"));
      Alcotest.(check bool)
        "remove existing" true
        (Transform_cache.Persistent.remove cache key |> get);
      Alcotest.(check bool)
        "remove absent" false
        (Transform_cache.Persistent.remove cache key |> get);
      Alcotest.(check bool)
        "miss after removal" true
        (match Transform_cache.Persistent.get cache key |> get with
        | Persistent.Miss -> true
        | Persistent.Hit _ | Persistent.Corrupt _ -> false))

let test_corruption_and_recovery () =
  with_temp_directory (fun root ->
      let cache = Transform_cache.Persistent.create ~root () |> get in
      let key = key "corruption" in
      let payload = Bytes.of_string "trusted-fitted-state" in
      ignore (Transform_cache.Persistent.put cache key payload |> get);
      let path = cache_file root in
      let entry = read_file path in
      Bytes.set entry
        (Bytes.length entry - 1)
        (if Bytes.get entry (Bytes.length entry - 1) = 'x' then 'y' else 'x');
      write_file path entry;
      (match Transform_cache.Persistent.get cache key |> get with
      | Persistent.Corrupt _ -> ()
      | Persistent.Miss | Persistent.Hit _ ->
          Alcotest.fail "corrupt payload was not detected");
      Alcotest.(check bool)
        "corrupt entry replaced" true
        (match Transform_cache.Persistent.put cache key payload |> get with
        | Persistent.Published -> true
        | Persistent.Already_present -> false);
      Transform_cache.Persistent.get cache key |> get |> expect_hit payload;
      write_file path (Bytes.of_string "truncated");
      Alcotest.(check bool)
        "truncation detected" true
        (match Transform_cache.Persistent.get cache key |> get with
        | Persistent.Corrupt _ -> true
        | Persistent.Miss | Persistent.Hit _ -> false))

let test_bounded_reader () =
  with_temp_directory (fun root ->
      let bounded_key = key "bounded" in
      let writer = Transform_cache.Persistent.create ~root () |> get in
      ignore
        (Transform_cache.Persistent.put writer bounded_key
           (Bytes.of_string "12345")
        |> get);
      let limits =
        Transform_cache.Persistent.limits ~max_payload_bytes:4 |> get
      in
      let reader = Transform_cache.Persistent.create ~limits ~root () |> get in
      Alcotest.(check bool)
        "oversized entry rejected before payload allocation" true
        (match Transform_cache.Persistent.get reader bounded_key |> get with
        | Persistent.Corrupt _ -> true
        | Persistent.Miss | Persistent.Hit _ -> false);
      expect_error "oversized write"
        (Transform_cache.Persistent.put reader (key "write-bound")
           (Bytes.of_string "12345")))

let test_concurrent_publication () =
  with_temp_directory (fun root ->
      let cache = Transform_cache.Persistent.create ~root () |> get in
      let key = key "concurrent" in
      let payload = Bytes.of_string "same-fitted-state" in
      let writers =
        Array.init 8 (fun _ ->
            Domain.spawn (fun () ->
                match Transform_cache.Persistent.put cache key payload with
                | Ok Persistent.Published | Ok Persistent.Already_present -> ()
                | Error error -> Alcotest.fail (Error.to_string error)))
      in
      Array.iter Domain.join writers;
      Transform_cache.Persistent.get cache key |> get |> expect_hit payload;
      Alcotest.(check int)
        "one published entry" 1
        (Sys.readdir root |> Array.to_list
        |> List.filter (String.ends_with ~suffix:".mkcache")
        |> List.length))

let () =
  Alcotest.run "Persistent transform cache"
    [
      ( "contracts",
        [
          Alcotest.test_case "root and limits" `Quick test_root_and_limits;
          Alcotest.test_case "round trip and conflict" `Quick
            test_round_trip_and_conflict;
          Alcotest.test_case "corruption and recovery" `Quick
            test_corruption_and_recovery;
          Alcotest.test_case "bounded reader" `Quick test_bounded_reader;
          Alcotest.test_case "concurrent publication" `Quick
            test_concurrent_publication;
        ] );
    ]
