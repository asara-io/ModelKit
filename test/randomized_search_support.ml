open Modelkit
module Support = Evaluation_metadata_support
module Callback = Modelkit.Callback

let get = Support.get
let data = Support.data
let ( let* ) = Result.bind
let rng seed = Rng.create (Seed.of_int seed)
let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let axis distribution =
  Randomized_search.axis ~name:"offset" ~distribution
    ~encode:(fun value -> Grid_search.Float value)
    ~set:(fun _ value -> Ok value)
  |> get

let space ?(iterations = 3) distribution =
  Randomized_search.create ~iterations ~base:0.
    ~build:(fun offset -> Ok (Support.pipeline ~offset ()))
    [| axis distribution |]
  |> get

let choices () = Parameter_distribution.choice [| 0.; 2.; 4. |] |> get
let sampled space = Randomized_search.sample ~seed:(Seed.of_int 19) space

let parameters samples =
  Array.map (fun sample -> sample.Randomized_search.sampled_parameters) samples

let run ?execution ?metadata ?(failure_policy = Cross_validation.Record) policy
    space =
  Randomized_search.Regression.search_with_policy ?execution ?metadata
    ~failure_policy ~space ~splitter:(Support.splitter ())
    ~scorers:
      [|
        Regression_scorer.neg_mean_squared_error;
        Regression_scorer.neg_mean_absolute_error;
      |]
    ~policy ~seed:(Seed.of_int 19) (Support.dataset ())

let summary_mean summary = (get summary.Grid_search.test).Score_aggregation.mean

let test_distributions () =
  let distributions =
    [
      ( Parameter_distribution.uniform ~low:(-.Float.max_float)
          ~high:Float.max_float ()
        |> get,
        -.Float.max_float,
        Float.max_float );
      ( Parameter_distribution.log_uniform ~low:1e-300 ~high:1e300 () |> get,
        1e-300,
        1e300 );
      ( Parameter_distribution.uniform ~low:1.
          ~high:(Float.next_after 1. infinity)
          ()
        |> get,
        1.,
        Float.next_after 1. infinity );
    ]
  in
  List.iter
    (fun (distribution, low, high) ->
      for seed = 0 to 100 do
        let value =
          Parameter_distribution.sample ~rng:(rng seed) distribution |> get
        in
        Alcotest.(check bool)
          "finite half-open support" true
          (Float.is_finite value && value >= low && value < high);
        Alcotest.check (Alcotest.float 0.) "distribution reproducibility" value
          (Parameter_distribution.sample ~rng:(rng seed) distribution |> get)
      done)
    distributions;
  let integers =
    Parameter_distribution.int_uniform ~low:min_int ~high:max_int () |> get
  in
  for seed = 0 to 100 do
    let value = Parameter_distribution.sample ~rng:(rng seed) integers |> get in
    Alcotest.(check bool)
      "wide integer support" true
      (value >= min_int && value < max_int)
  done;
  expect_error "empty choices" (Parameter_distribution.choice [||]);
  List.iter
    (fun (low, high) ->
      expect_error "invalid uniform bounds"
        (Parameter_distribution.uniform ~low ~high ()))
    [ (nan, 1.); (0., infinity); (1., 1.); (2., 1.) ];
  expect_error "nonpositive log bound"
    (Parameter_distribution.log_uniform ~low:0. ~high:1. ());
  expect_error "empty integer range"
    (Parameter_distribution.int_uniform ~low:1 ~high:1 ())

let test_sampling () =
  let values = [| 0.; 2.; 4. |] in
  let distribution = Parameter_distribution.choice values |> get in
  values.(0) <- 999.;
  let finite = space ~iterations:10 distribution in
  Alcotest.(check int)
    "finite count capped" 3
    (Randomized_search.candidate_count finite);
  let samples = sampled finite in
  let configurations =
    Array.map
      (fun sample -> get sample.Randomized_search.sampled_configuration)
      samples
  in
  let sorted = Array.copy configurations in
  Array.sort Float.compare sorted;
  Alcotest.(check (array (Alcotest.float 0.)))
    "without replacement and defensive copy" [| 0.; 2.; 4. |] sorted;
  Alcotest.(check bool)
    "finite prefix" true
    (parameters (sampled (space ~iterations:2 distribution))
    = Array.sub (parameters samples) 0 2);
  let mixed = Parameter_distribution.uniform ~low:0. ~high:4. () |> get in
  Alcotest.(check bool)
    "distribution prefix" true
    (parameters (sampled (space ~iterations:5 mixed))
    = Array.sub (parameters (sampled (space ~iterations:8 mixed))) 0 5);
  let make_axis i =
    Randomized_search.axis
      ~name:("p" ^ string_of_int i)
      ~distribution:(Parameter_distribution.choice [| 0; 1 |] |> get)
      ~encode:(fun x -> Grid_search.Int x)
      ~set:(fun sum value -> Ok (sum + value))
    |> get
  in
  let huge =
    Randomized_search.create ~iterations:4 ~base:0
      ~build:(fun _ -> Ok (Support.pipeline ()))
      (Array.init 24 make_axis)
    |> get
  in
  Alcotest.(check int)
    "large finite space samples without expansion" 4
    (Array.length (sampled huge));
  expect_error "choice product overflow"
    (Randomized_search.create ~base:0
       ~build:(fun _ -> Ok (Support.pipeline ()))
       (Array.init Sys.int_size make_axis));
  let one =
    Randomized_search.create ~base:0.
      ~build:(fun _ -> Ok (Support.pipeline ()))
      [||]
    |> get
  in
  Alcotest.(check int)
    "base-only space" 1
    (Randomized_search.candidate_count one);
  expect_error "duplicate names"
    (Randomized_search.create ~base:0.
       ~build:(fun _ -> Ok (Support.pipeline ()))
       [| axis distribution; axis distribution |]);
  expect_error "zero iterations"
    (Randomized_search.create ~iterations:0 ~base:0.
       ~build:(fun _ -> Ok (Support.pipeline ()))
       [| axis distribution |])

let custom_index candidates =
  let value candidate =
    Array.fold_left
      (fun sum score -> sum +. summary_mean score)
      0. candidate.Grid_search.scores
  in
  let best = ref 0 in
  Array.iteri
    (fun i candidate ->
      if value candidate > value candidates.(!best) then best := i)
    candidates;
  Ok !best

let test_policies () =
  let space = space (choices ()) in
  Atomic.set Support.fits 0;
  let report = run Grid_search.No_refit space |> get in
  Alcotest.(check bool)
    "preview matches evaluated parameters" true
    (parameters (sampled space)
    = Array.map
        (fun candidate -> candidate.Grid_search.parameters)
        (Grid_search.candidates report));
  Alcotest.(check int) "no full-data refit" 12 (Atomic.get Support.fits);
  (match Grid_search.refit_result report |> get with
  | None -> ()
  | Some _ -> Alcotest.fail "unexpected refit");
  expect_error "legacy selection reports disabled refit"
    (Grid_search.selection report);
  Array.iter
    (fun candidate ->
      Alcotest.(check (option int))
        "no implicit ranking metric" None candidate.Grid_search.rank)
    (Grid_search.candidates report);
  let expected = custom_index (Grid_search.candidates report) |> get in
  let mutate_copy candidates =
    let selected = custom_index candidates |> get in
    candidates.(selected).Grid_search.parameters.(0) <-
      {
        Grid_search.parameter_name = "mutated";
        parameter_value = Grid_search.Float nan;
      };
    candidates.(selected).Grid_search.scores.(0) <-
      {
        Grid_search.scorer_name = "mutated";
        train = None;
        test = Error (Error.make Error.Cancelled ~remediation:"test");
      };
    Ok selected
  in
  Atomic.set Support.fits 0;
  let custom = run (Grid_search.Custom mutate_copy) space |> get in
  Alcotest.(check int) "custom refits exactly once" 13 (Atomic.get Support.fits);
  Alcotest.(check int)
    "multi-metric selection" expected
    (Grid_search.selection custom |> get).Grid_search.selected_candidate_index;
  Alcotest.(check string)
    "selector sees copied parameters" "offset"
    (Grid_search.candidates custom).(expected).Grid_search.parameters.(0)
      .Grid_search.parameter_name;
  let grid = Support.grid () in
  Atomic.set Support.fits 0;
  let grid_report =
    Grid_search.Regression.search_with_policy ~grid
      ~splitter:(Support.splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~policy:Grid_search.No_refit ~seed:(Seed.of_int 19) (Support.dataset ())
    |> get
  in
  Alcotest.(check int) "grid shares no-refit policy" 8 (Atomic.get Support.fits);
  (match Grid_search.refit_result grid_report |> get with
  | None -> ()
  | Some _ -> Alcotest.fail "grid refitted");
  let invalid = run (Grid_search.Custom (fun _ -> Ok 99)) space |> get in
  expect_error "invalid selector recorded" (Grid_search.refit_result invalid);
  expect_error "invalid selector aborted"
    (run ~failure_policy:Cross_validation.Abort
       (Grid_search.Custom (fun _ -> Ok (-1)))
       space)

let test_failures () =
  let error =
    Error.make
      (Error.Validation { name = "sample"; reason = "deliberate" })
      ~remediation:"test"
  in
  let bad = space (Parameter_distribution.custom (fun _ -> Error error)) in
  let preview = sampled bad in
  Array.iteri
    (fun index candidate ->
      match candidate.Randomized_search.sampled_configuration with
      | Ok _ -> Alcotest.fail "failed draw succeeded"
      | Error error ->
          Alcotest.(check bool)
            "candidate and axis error context" true
            (Error.context error
            = [ Error.Candidate index; Error.Stage "offset" ]))
    preview;
  let report = run Grid_search.No_refit bad |> get in
  Array.iter
    (fun candidate ->
      Alcotest.(check bool)
        "draw failure recorded" true
        (Option.is_some candidate.Grid_search.build_error))
    (Grid_search.candidates report);
  (match Grid_search.refit_result report |> get with
  | None -> ()
  | Some _ -> Alcotest.fail "failed sampler refit");
  expect_error "sample failure aborts"
    (run ~failure_policy:Cross_validation.Abort Grid_search.No_refit bad);
  expect_error "cannot select failed candidate"
    (run (Grid_search.Custom (fun _ -> Ok 0)) bad
    |> get |> Grid_search.refit_result);
  expect_error "selector control errors always abort"
    (run
       (Grid_search.Custom
          (fun _ -> Error (Error.make Error.Cancelled ~remediation:"test")))
       (space (choices ())));
  let invalid_scorer =
    Randomized_search.Regression.search
      ~space:(space (choices ()))
      ~splitter:(Support.splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"missing" ~seed:(Seed.of_int 19) (Support.dataset ())
  in
  expect_error "invalid named scorer" invalid_scorer

let[@warning "-4"] test_cancellation () =
  let draws = ref 0 in
  let distribution =
    Parameter_distribution.custom (fun _ ->
        incr draws;
        Ok 0.)
  in
  let callback =
    Callback.create (fun event ->
        match
          ( event.Callback.operation,
            event.Callback.status,
            event.Callback.context )
        with
        | Callback.Candidate, Callback.Started, [ Error.Candidate 1 ] ->
            Ok Callback.Cancel
        | _ -> Ok Callback.Continue)
    |> get
  in
  let metadata = Metadata.of_dataset ~callback (Support.dataset ()) in
  expect_error "candidate callback cancellation"
    (run ~metadata Grid_search.No_refit (space distribution));
  Alcotest.(check int)
    "later candidates never sample after cancellation" 1 !draws;
  let cancelled =
    Parameter_distribution.custom (fun _ ->
        Error (Error.make Error.Cancelled ~remediation:"test"))
  in
  let later =
    Randomized_search.axis ~name:"later" ~distribution
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let cancelled_space =
    Randomized_search.create ~base:0.
      ~build:(fun _ -> Ok (Support.pipeline ()))
      [| axis cancelled; later |]
    |> get
  in
  draws := 0;
  expect_error "sampling control error"
    (run Grid_search.No_refit cancelled_space);
  Alcotest.(check int) "control errors stop later axes" 0 !draws;
  let ordinary_error =
    Error.make
      (Error.Validation { name = "first axis"; reason = "deliberate" })
      ~remediation:"test"
  in
  let first =
    Randomized_search.axis ~name:"first"
      ~distribution:
        (Parameter_distribution.custom (fun _ -> Error ordinary_error))
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let precedence =
    Randomized_search.create ~base:0.
      ~build:(fun _ -> Ok (Support.pipeline ()))
      [| first; axis cancelled; later |]
    |> get
  in
  (match run Grid_search.No_refit precedence with
  | Ok _ -> Alcotest.fail "ordinary error masked cancellation"
  | Error error ->
      Alcotest.(check bool)
        "control error takes precedence" true
        (Callback.is_control_error error));
  Alcotest.(check int) "later axis stays undrawn" 0 !draws;
  let exception Sampler_failure in
  let bad =
    space (Parameter_distribution.custom (fun _ -> raise Sampler_failure))
  in
  (try
     ignore (run Grid_search.No_refit bad);
     Alcotest.fail "sampler exception swallowed"
   with Sampler_failure -> ());
  let exception Selector_failure in
  try
    ignore
      (run
         (Grid_search.Custom (fun _ -> raise Selector_failure))
         (space (choices ())));
    Alcotest.fail "selector exception swallowed"
  with Selector_failure -> ()

let check_execution execution =
  let run execution =
    let events = ref [] in
    let owner = Domain.self () in
    let callback =
      Callback.create (fun event ->
          if Domain.self () <> owner then
            failwith "callback delivered off caller domain";
          events := event :: !events;
          Ok Callback.Continue)
      |> get
    in
    let source = Support.dataset () in
    let metadata = Metadata.of_dataset ~callback source in
    let report =
      run ~execution ~metadata (Grid_search.Custom custom_index)
        (space (choices ()))
      |> get
    in
    let selected = Grid_search.selection report |> get in
    let candidates =
      Grid_search.candidates report
      |> Array.map (fun candidate ->
          ( candidate.Grid_search.parameters,
            Array.map summary_mean candidate.Grid_search.scores ))
    in
    ( List.rev !events,
      candidates,
      selected.Grid_search.selected_candidate_index,
      Support.predictions
        (Metadata.of_dataset source)
        source selected.Grid_search.selected_model )
  in
  Alcotest.(check bool)
    "samples, scores, selection, refit and events match" true
    (run execution = run Execution.sequential)

let test_classification () =
  List.iter
    (fun classes ->
      let count = classes * 8 in
      let y =
        Target.classification (Array.init count (fun i -> i mod classes))
      in
      let x =
        Matrix.init ~rows:count ~columns:classes (fun row column ->
            if row mod classes = column then 1. else 0.)
        |> data
      in
      let source =
        Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> data
      in
      let build c =
        let* estimator =
          if classes = 2 then
            let* specification = Logistic_regression.create ~c () in
            Pipeline.classifier ~name:"classifier"
              (module Logistic_regression)
              specification
          else
            let* specification = Multinomial_logistic_regression.create ~c () in
            Pipeline.classifier ~name:"classifier"
              (module Multinomial_logistic_regression)
              specification
        in
        Pipeline.set_estimator Pipeline.empty estimator
      in
      let axis =
        Randomized_search.axis ~name:"c"
          ~distribution:(Parameter_distribution.choice [| 0.1; 1. |] |> get)
          ~encode:(fun c -> Grid_search.Float c)
          ~set:(fun _ c -> Ok c)
        |> get
      in
      let space = Randomized_search.create ~base:1. ~build [| axis |] |> get in
      let splitter =
        Cross_validation.target_aware_splitter
          (module Stratified_k_fold)
          (Stratified_k_fold.create ~folds:4 () |> get)
      in
      let report =
        if classes = 2 then
          Randomized_search.Binary_classification.search ~space ~splitter
            ~scorers:[| Binary_classification_scorer.accuracy |]
            ~refit:"accuracy" ~seed:(Seed.of_int 19) source
          |> get
        else
          Randomized_search.Multiclass_classification.search_with_policy ~space
            ~splitter
            ~scorers:[| Multiclass_classification_scorer.accuracy |]
            ~policy:
              (Grid_search.Custom
                 (fun candidates -> Ok (Array.length candidates - 1)))
            ~seed:(Seed.of_int 19) source
          |> get
      in
      let fitted =
        (Grid_search.selection report |> get).Grid_search.selected_model
      in
      let prediction =
        Pipeline.predict fitted
          ~feature_schema:(Dataset.feature_schema source)
          ~x
        |> get
      in
      Alcotest.(check (array int))
        "classification search full refit"
        (Target.classification_values y)
        (Target.classification_values prediction))
    [ 2; 3 ]

let tests =
  [
    ("binary and multiclass search", `Quick, test_classification);
    ("distribution bounds and reproducibility", `Quick, test_distributions);
    ("finite and distribution sampling", `Quick, test_sampling);
    ("no-refit and multi-metric policies", `Quick, test_policies);
    ("sampling and selection failures", `Quick, test_failures);
    ("cancellation and user exceptions", `Quick, test_cancellation);
    ( "metadata and execution integration",
      `Quick,
      fun () -> check_execution Execution.sequential );
  ]
