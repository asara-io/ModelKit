open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

type random_fitted = { value : float; schema : Feature_schema.t }

module Random_regressor :
  REGRESSOR
    with type t = unit
     and type params = unit
     and type fitted = random_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = random_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng ~feature_schema ~x:_ ~y:_ () =
    let value, _ = Rng.next_float rng in
    Ok { value; schema = feature_schema }

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make ~remediation:"use the fitted feature schema"
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      Target.regression
        (Vector.of_array (Array.make (Matrix.rows x) fitted.value))
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"produce finite predictions" error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

let dataset () =
  let x = Array.init 24 (fun row -> [| Float.of_int row |]) |> matrix in
  let y = Array.init 24 (fun row -> Float.of_int (row mod 7)) |> regression in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let pipeline () =
  let estimator =
    Pipeline.estimator ~name:"random" (module Random_regressor) () |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let splitter () =
  K_fold.create ~folds:6 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let run execution =
  Cross_validation.Regression.cross_validate ~return_train_score:true
    ~return_models:true ~return_indices:true ~execution ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~seed:(Seed.of_int 2026) (pipeline ()) (dataset ())
  |> get

let score = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (Error.to_string error)
  | None -> Alcotest.fail "expected a score"

let model_value fold =
  let fitted = Option.get fold.Cross_validation.model in
  let source = dataset () in
  Pipeline.predict fitted
    ~feature_schema:(Dataset.feature_schema source)
    ~x:(Dataset.features source)
  |> get |> Target.regression_values
  |> fun values -> Vector.get values 0

let check_report expected observed =
  let expected = Cross_validation.folds expected in
  let observed = Cross_validation.folds observed in
  Alcotest.(check int)
    "fold count" (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected_fold ->
      let observed_fold = observed.(index) in
      Alcotest.(check int)
        "logical fold index" expected_fold.Cross_validation.fold_index
        observed_fold.Cross_validation.fold_index;
      Alcotest.(check (array int))
        "training indices"
        (Option.get expected_fold.Cross_validation.train_indices)
        (Option.get observed_fold.Cross_validation.train_indices);
      Alcotest.(check (array int))
        "test indices"
        (Option.get expected_fold.Cross_validation.test_indices)
        (Option.get observed_fold.Cross_validation.test_indices);
      Alcotest.check (Alcotest.float 0.0) "training score"
        (score
           expected_fold.Cross_validation.scores.(0)
             .Cross_validation.train_score)
        (score
           observed_fold.Cross_validation.scores.(0)
             .Cross_validation.train_score);
      Alcotest.check (Alcotest.float 0.0) "test score"
        (score
           expected_fold.Cross_validation.scores.(0).Cross_validation.test_score)
        (score
           observed_fold.Cross_validation.scores.(0).Cross_validation.test_score);
      Alcotest.check (Alcotest.float 0.0) "fitted random value"
        (model_value expected_fold)
        (model_value observed_fold);
      Alcotest.(check int)
        "failure count"
        (Array.length expected_fold.Cross_validation.failures)
        (Array.length observed_fold.Cross_validation.failures);
      Alcotest.(check bool)
        "nonnegative fit timing" true
        (observed_fold.Cross_validation.fit_time >= 0.0);
      Alcotest.(check bool)
        "nonnegative score timing" true
        (observed_fold.Cross_validation.score_time >= 0.0))
    expected

let test_domain_count_invariance () =
  let expected = run Execution.sequential in
  List.iter
    (fun domains ->
      let configuration =
        Modelkit_parallel.create ~inner_threads:1 ~domains () |> get
      in
      Alcotest.(check int)
        "reported concurrency" domains
        (Modelkit_parallel.concurrency configuration);
      let execution = Modelkit_parallel.execution configuration in
      Alcotest.(check int)
        "packaged concurrency" domains
        (Execution.concurrency execution);
      check_report expected (run execution))
    [ 1; 2; 4 ]

let update_maximum maximum value =
  let rec update () =
    let observed = Atomic.get maximum in
    if value <= observed then ()
    else if not (Atomic.compare_and_set maximum observed value) then update ()
  in
  update ()

let test_bounded_ordered_map () =
  let configuration =
    Modelkit_parallel.create ~inner_threads:1 ~domains:3 () |> get
  in
  let active = Atomic.make 0 in
  let maximum = Atomic.make 0 in
  let outputs =
    Modelkit_parallel.map configuration
      ~f:(fun ~index value ->
        let running = Atomic.fetch_and_add active 1 + 1 in
        update_maximum maximum running;
        for _ = 1 to 10_000 do
          Domain.cpu_relax ()
        done;
        Atomic.decr active;
        Ok (index, value * value))
      [| 3; 4; 5; 6; 7; 8 |]
    |> get
  in
  Alcotest.(check (array (pair int int)))
    "stable result order"
    [| (0, 9); (1, 16); (2, 25); (3, 36); (4, 49); (5, 64) |]
    outputs;
  Alcotest.(check bool) "bounded active tasks" true (Atomic.get maximum <= 3)

let test_lowest_failure () =
  let configuration =
    Modelkit_parallel.create ~inner_threads:1 ~domains:4 () |> get
  in
  match
    Modelkit_parallel.map configuration
      ~f:(fun ~index value ->
        if index = 2 then (
          for _ = 1 to 50_000 do
            Domain.cpu_relax ()
          done;
          Error "two")
        else if index = 5 then Error "five"
        else Ok value)
      [| 0; 1; 2; 3; 4; 5; 6; 7 |]
  with
  | Ok _ -> Alcotest.fail "parallel execution ignored task failures"
  | Error error -> Alcotest.(check string) "lowest failure" "two" error

let test_diagnostics_and_validation () =
  (match Modelkit_parallel.create ~domains:0 () with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "zero domains were accepted");
  (match Modelkit_parallel.create ~inner_threads:0 ~domains:1 () with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "zero inner threads were accepted");
  let configuration =
    Modelkit_parallel.create ~inner_threads:2 ~domains:2 () |> get
  in
  let diagnostics = Modelkit_parallel.diagnostics configuration in
  Alcotest.(check int)
    "fold domains" 2 diagnostics.Modelkit_parallel.fold_domains;
  Alcotest.(check int)
    "inner threads" 2 diagnostics.Modelkit_parallel.inner_threads;
  Alcotest.(check int)
    "estimated runnable threads" 4
    diagnostics.Modelkit_parallel.estimated_runnable_threads;
  Alcotest.(check bool)
    "nested parallelism warning" true
    (Array.exists
       (function
         | Modelkit_parallel.Nested_parallelism _ -> true
         | Modelkit_parallel.Invalid_environment_limit _
         | Modelkit_parallel.Conflicting_environment_limits _
         | Modelkit_parallel.Domains_exceed_recommended _
         | Modelkit_parallel.Potential_oversubscription _ ->
             false)
       diagnostics.Modelkit_parallel.warnings);
  let saturated =
    Modelkit_parallel.create ~inner_threads:max_int ~domains:2 ()
    |> get |> Modelkit_parallel.diagnostics
  in
  Alcotest.(check int)
    "saturated runnable estimate" max_int
    saturated.Modelkit_parallel.estimated_runnable_threads

let test_programmer_exception () =
  let configuration =
    Modelkit_parallel.create ~inner_threads:1 ~domains:2 () |> get
  in
  let propagated =
    try
      ignore
        (Modelkit_parallel.map configuration
           ~f:(fun ~index value -> if index = 1 then raise Exit else Ok value)
           [| 0; 1; 2 |]);
      false
    with Exit -> true
  in
  Alcotest.(check bool) "programmer exception propagated" true propagated

let () =
  Alcotest.run "parallel execution"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
          Alcotest.test_case "bounded ordered map" `Quick
            test_bounded_ordered_map;
          Alcotest.test_case "lowest failure" `Quick test_lowest_failure;
          Alcotest.test_case "programmer exception" `Quick
            test_programmer_exception;
        ] );
      ( "diagnostics",
        [
          Alcotest.test_case "validation and warnings" `Quick
            test_diagnostics_and_validation;
        ] );
    ]
