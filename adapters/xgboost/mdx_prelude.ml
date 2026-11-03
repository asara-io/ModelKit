List.iter
  (fun name ->
    if Sys.getenv_opt name <> Some "1" then
      failwith ("XGBoost documentation tests require " ^ name ^ "=1 at startup"))
  [ "OMP_NUM_THREADS"; "OMP_THREAD_LIMIT" ]
