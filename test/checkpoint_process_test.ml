let write path bytes =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out channel)
    (fun () -> output_bytes channel bytes)

let read path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in channel)
    (fun () ->
      really_input_string channel (in_channel_length channel) |> Bytes.of_string)

let child mode path =
  let executable = Sys.executable_name in
  let pid =
    Unix.create_process executable
      [| executable; mode; path |]
      Unix.stdin Unix.stdout Unix.stderr
  in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ ->
      Alcotest.fail "checkpoint child failed"

let test_restart () =
  let path = Filename.temp_file "modelkit-search-checkpoint" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      child "produce" path;
      child "resume" path)

let () =
  match Array.to_list Sys.argv with
  | [ _; "produce"; path ] -> write path (Checkpoint_support.produce ())
  | [ _; "resume"; path ] -> Checkpoint_support.resume (read path)
  | _ ->
      Alcotest.run "Search process restart"
        [ ("restart", [ ("independent processes", `Quick, test_restart) ]) ]
