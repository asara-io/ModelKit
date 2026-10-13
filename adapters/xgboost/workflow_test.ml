let check_startup_profiles () =
  let executable =
    if Filename.is_relative Sys.executable_name then
      Filename.concat (Sys.getcwd ()) Sys.executable_name
    else Sys.executable_name
  in
  let inherited =
    Unix.environment () |> Array.to_list
    |> List.filter (fun entry ->
        not
          (String.starts_with ~prefix:"OMP_NUM_THREADS=" entry
          || String.starts_with ~prefix:"OMP_THREAD_LIMIT=" entry))
  in
  List.iter
    (fun (name, settings, expected) ->
      let environment = Array.of_list (settings @ inherited) in
      let sink = Unix.openfile Filename.null [ Unix.O_WRONLY ] 0 in
      let status =
        Fun.protect
          ~finally:(fun () -> Unix.close sink)
          (fun () ->
            let pid =
              Unix.create_process_env executable
                [| executable; "--check-thread-contract" |]
                environment Unix.stdin sink sink
            in
            snd (Unix.waitpid [] pid))
      in
      Alcotest.check Alcotest.bool name true (status = Unix.WEXITED expected))
    [
      ( "bounded startup accepted",
        [ "OMP_NUM_THREADS=1"; "OMP_THREAD_LIMIT=1" ],
        0 );
      ( "two-thread startup rejected",
        [ "OMP_NUM_THREADS=2"; "OMP_THREAD_LIMIT=2" ],
        1 );
      ("uncapped startup rejected", [], 1);
    ]

let () =
  if Array.length Sys.argv = 2 && Sys.argv.(1) = "--check-thread-contract" then (
    try Workflow_support.check_thread_contract ()
    with Failure reason ->
      prerr_endline reason;
      exit 1)
  else
    Alcotest.run "XGBoost workflow integration"
      [
        ( "workflows",
          ("fresh-process OpenMP profiles", `Quick, check_startup_profiles)
          :: Workflow_support.tests );
      ]
