open Modelkit
module Persistent = Transform_cache.Persistent

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let key () =
  let component =
    Transform_cache.Component.create ~package:"example" ~name:"process-writer"
      ~version:1
    |> get
  in
  Transform_cache.Key.create ~component
    ~configuration:(Transform_cache.Content_id.of_string "same-configuration")
    ~training_data:(Transform_cache.Content_id.of_string "same-training-data")
    ~target:None
    ~routed_metadata:(Transform_cache.Content_id.of_string "same-metadata")
    ~seed:(Seed.of_int 91)

let payload = Bytes.of_string "same-process-independent-fitted-state"
let remove_if_present path = try Sys.remove path with Sys_error _ -> ()

let clean_directory path =
  Array.iter
    (fun name -> remove_if_present (Filename.concat path name))
    (Sys.readdir path);
  Sys.rmdir path

let publish root =
  let cache = Persistent.create ~root () |> get in
  match Persistent.put cache (key ()) payload with
  | Ok Persistent.Published | Ok Persistent.Already_present -> ()
  | Error error -> Alcotest.fail (Error.to_string error)

let spawn root =
  let executable = Sys.executable_name in
  Unix.create_process executable
    [| executable; "publish"; root |]
    Unix.stdin Unix.stdout Unix.stderr

let join process =
  match snd (Unix.waitpid [] process) with
  | Unix.WEXITED 0 -> ()
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ ->
      Alcotest.fail "persistent-cache publisher process failed"

let test_process_publication () =
  let root = Filename.temp_file "modelkit-cache-process-" "" in
  Sys.remove root;
  Sys.mkdir root 0o700;
  Fun.protect
    ~finally:(fun () -> clean_directory root)
    (fun () ->
      Array.init 4 (fun _ -> spawn root) |> Array.iter join;
      let cache = Persistent.create ~root () |> get in
      match Persistent.get cache (key ()) |> get with
      | Persistent.Hit actual ->
          Alcotest.(check bytes) "published payload" payload actual
      | Persistent.Miss -> Alcotest.fail "publisher processes produced no entry"
      | Persistent.Corrupt error -> Alcotest.fail (Error.to_string error))

let () =
  match Array.to_list Sys.argv with
  | [ _; "publish"; root ] -> publish root
  | _ ->
      Alcotest.run "Persistent cache process publication"
        [
          ( "concurrency",
            [
              Alcotest.test_case "independent writers" `Quick
                test_process_publication;
            ] );
        ]
