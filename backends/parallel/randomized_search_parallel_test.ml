let () =
  Alcotest.run "Parallel randomized search"
    [
      ( "search",
        List.map
          (fun domains ->
            ( string_of_int domains ^ " domains",
              `Quick,
              fun () ->
                Modelkit_parallel.create ~inner_threads:1 ~domains ()
                |> Evaluation_metadata_support.get
                |> Modelkit_parallel.execution
                |> Randomized_search_support.check_execution ))
          [ 1; 2; 4 ] );
    ]
