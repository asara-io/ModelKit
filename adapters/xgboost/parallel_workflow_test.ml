let () =
  Alcotest.run "XGBoost parallel workflow integration"
    [
      ( "execution",
        List.map
          (fun domains ->
            ( string_of_int domains ^ " domains",
              `Quick,
              fun () ->
                let backend =
                  Modelkit_parallel.create ~inner_threads:1 ~domains ()
                  |> Result.get_ok
                in
                Workflow_support.check_execution
                  (Modelkit_parallel.execution backend) ))
          [ 1; 2; 4 ] );
    ]
