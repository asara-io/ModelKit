open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let dataset () =
  let x =
    Array.init 30 (fun row ->
        let value = Float.of_int row in
        [| value; Float.sin value |])
    |> Matrix.of_arrays |> get_data
  in
  let y =
    Array.init 30 (fun row -> (3. *. Float.of_int row) +. 2.)
    |> Vector.of_array |> Target.regression |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let splitter () =
  K_fold.create ~folds:6 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let pipeline store =
  let scale =
    Pipeline.cacheable_transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.add_transformer Pipeline.empty scale |> get |> fun builder ->
  Pipeline.set_estimator builder estimator |> get |> fun pipeline ->
  Pipeline.with_cache pipeline store

let score report =
  Cross_validation.folds report
  |> Array.map (fun fold ->
      match fold.Cross_validation.scores.(0).Cross_validation.test_score with
      | Some (Ok value) -> value
      | Some (Error error) -> Alcotest.fail (Error.to_string error)
      | None -> Alcotest.fail "missing cache test score")

let run execution store =
  Cross_validation.Regression.cross_validate ~execution ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~seed:(Seed.of_int 2027) (pipeline store) (dataset ())
  |> get

let execution domains =
  Modelkit_parallel.create ~inner_threads:1 ~domains ()
  |> get |> Modelkit_parallel.execution

let test_schedule_independence () =
  let baseline_raw = Transform_cache.Memory.create () in
  let baseline_store = Transform_cache.Store.memory baseline_raw in
  let expected = run Execution.sequential baseline_store |> score in
  let baseline_stats = Transform_cache.Memory.stats baseline_raw in
  Alcotest.(check int)
    "sequential fold entries" 6 baseline_stats.Transform_cache.Memory.entries;
  List.iter
    (fun domains ->
      let raw = Transform_cache.Memory.create () in
      let store = Transform_cache.Store.memory raw in
      let cold = run (execution domains) store |> score in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "cold scores" expected cold;
      let cold_stats = Transform_cache.Memory.stats raw in
      Alcotest.(check int)
        "cold entries" 6 cold_stats.Transform_cache.Memory.entries;
      Alcotest.(check int64)
        "cold misses" 6L cold_stats.Transform_cache.Memory.misses;
      let warm = run (execution domains) store |> score in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "warm scores" expected warm;
      let warm_stats = Transform_cache.Memory.stats raw in
      Alcotest.(check int64)
        "warm hits" 6L warm_stats.Transform_cache.Memory.hits)
    [ 1; 2; 4 ]

let () =
  Alcotest.run "Parallel pipeline cache"
    [
      ( "determinism",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_schedule_independence;
        ] );
    ]
