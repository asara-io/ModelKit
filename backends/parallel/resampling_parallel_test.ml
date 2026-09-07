let () =
  Alcotest.run "Parallel resampling"
    [
      ( "CV",
        List.map
          (fun domains ->
            ( string_of_int domains ^ " domains",
              `Quick,
              fun () ->
                Modelkit_parallel.create ~inner_threads:1 ~domains ()
                |> Evaluation_metadata_support.get
                |> Modelkit_parallel.execution
                |> Resampling_support.check_execution ))
          [ 1; 2; 4 ] );
    ]
