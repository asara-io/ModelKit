open Modelkit
module Support = Evaluation_metadata_support
module Callback = Modelkit.Callback

let get = Support.get
let data = Support.data
let ( let* ) = Result.bind

let error =
  Error.make
    (Error.Validation { name = "test"; reason = "deliberate" })
    ~remediation:"test"

let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let budget ?max_fits () =
  Successive_halving.budget ?max_fits ~min_samples:2 ~max_samples:12 ~factor:2
    ()
  |> get

let offsets = [| 1000.; 0.; 0.; 1000.; 1000. |]

let grid ?(build = fun offset -> Ok (Support.pipeline ~offset ())) values =
  Grid_search.create ~base:0. ~build
    [|
      Grid_search.axis ~name:"offset" ~values
        ~encode:(fun value -> Grid_search.Float value)
        ~set:(fun _ value -> Ok value)
      |> get;
    |]
  |> get |> Successive_halving.of_grid

let run ?execution ?metadata ?(failure_policy = Cross_validation.Record)
    ?(budget = budget ()) ?(source = Support.dataset ())
    ?(splitter = Support.splitter ()) policy candidates =
  Successive_halving.Regression.search_with_policy ?execution ?metadata
    ~failure_policy ~budget ~candidates ~splitter
    ~scorers:
      [|
        Regression_scorer.neg_mean_squared_error;
        Regression_scorer.neg_mean_absolute_error;
      |]
    ~promotion_score:"neg_mean_squared_error" ~policy ~seed:(Seed.of_int 19)
    source

let best = Grid_search.Best_score "neg_mean_squared_error"

let ids candidates =
  Array.map (fun candidate -> candidate.Grid_search.candidate_index) candidates

let folds candidate =
  Cross_validation.folds (Option.get candidate.Grid_search.evaluation)

let rows fold =
  ( Option.get fold.Cross_validation.train_indices,
    Option.get fold.Cross_validation.test_indices )

let offset candidate =
  match candidate.Grid_search.parameters.(0).Grid_search.parameter_value with
  | Grid_search.Float value -> value
  | Grid_search.Bool _ | Grid_search.Int _ | Grid_search.String _ ->
      assert false

let snapshot report =
  Successive_halving.rounds report
  |> Array.map (fun round ->
      ( round.Successive_halving.training_samples,
        round.Successive_halving.promoted_candidate_indices,
        Array.map
          (fun candidate ->
            ( candidate.Grid_search.candidate_index,
              candidate.Grid_search.rank,
              candidate.Grid_search.parameters,
              Array.map
                (fun fold -> (rows fold, Support.score fold))
                (folds candidate) ))
          round.Successive_halving.candidates ))

let check_rows_and_scores report =
  let source = Support.dataset () in
  let y = Target.regression_values (Dataset.target source) in
  let weights = Option.get (Dataset.sample_weight source) in
  let groups = Option.get (Dataset.groups source) in
  let previous = ref None in
  Successive_halving.rounds report
  |> Array.iter (fun round ->
      let candidates = round.Successive_halving.candidates in
      let reference = Array.map rows (folds candidates.(0)) in
      Array.iter
        (fun candidate ->
          Array.iteri
            (fun index fold ->
              let train, test = rows fold in
              Alcotest.(check int)
                "declared training rows"
                round.Successive_halving.training_samples (Array.length train);
              Alcotest.(check bool)
                "fair allocation" true
                ((train, test) = reference.(index));
              Array.iter
                (fun row ->
                  Array.iter
                    (fun held_out ->
                      if Groups.get groups row = Groups.get groups held_out then
                        Alcotest.fail "group leakage")
                    test)
                train;
              let mean = Support.mean source train +. offset candidate in
              let sum, total =
                Array.fold_left
                  (fun (sum, total) row ->
                    let weight = Sample_weight.get weights row in
                    let prediction =
                      mean
                      +. (0.01 *. Float.of_int (Groups.get groups row))
                      +. (0.001 *. weight)
                    in
                    let residual = prediction -. Vector.get y row in
                    (sum +. (weight *. residual *. residual), total +. weight))
                  (0., 0.) test
              in
              Alcotest.check (Alcotest.float 1e-7)
                "fresh fit uses only current training rows" (-.sum /. total)
                (Support.score fold))
            (folds candidate))
        candidates;
      (match !previous with
      | None -> ()
      | Some earlier ->
          Array.iteri
            (fun i (train, test) ->
              let old_train, old_test = earlier.(i) in
              Alcotest.(check (array int))
                "nested training prefix" old_train
                (Array.sub train 0 (Array.length old_train));
              Alcotest.(check (array int)) "fixed validation rows" old_test test)
            reference);
      previous := Some reference)

let test_schedule () =
  let specification = budget () in
  let schedule = Successive_halving.resources specification in
  Alcotest.(check (array int))
    "capped geometric resources" [| 2; 4; 8; 12 |] schedule;
  schedule.(0) <- 0;
  Alcotest.(check int)
    "copied resource schedule" 2
    (Successive_halving.resources specification).(0);
  let huge =
    Successive_halving.budget ~min_samples:2 ~max_samples:max_int
      ~factor:max_int ()
    |> get
  in
  Alcotest.(check (array int))
    "overflow-safe schedule" [| 2; max_int |]
    (Successive_halving.resources huge);
  List.iter
    (fun (low, high, factor) ->
      expect_error "invalid budget"
        (Successive_halving.budget ~min_samples:low ~max_samples:high ~factor ()))
    [ (0, 12, 2); (4, 2, 2); (2, 12, 1) ];
  expect_error "zero fit cap"
    (Successive_halving.budget ~max_fits:0 ~min_samples:2 ~max_samples:12
       ~factor:2 ());
  let builds = ref 0 in
  let candidates =
    grid
      ~build:(fun value ->
        incr builds;
        Ok (Support.pipeline ~offset:value ()))
      offsets
  in
  expect_error "fit cap preflight"
    (run ~budget:(budget ~max_fits:44 ()) best candidates);
  Alcotest.(check int) "no builds before cap check" 0 !builds;
  Atomic.set Support.fits 0;
  ignore
    (run ~budget:(budget ~max_fits:44 ()) Grid_search.No_refit candidates |> get);
  Alcotest.(check int) "exact no-refit fit budget" 44 (Atomic.get Support.fits);
  expect_error "fold too small"
    (run
       ~budget:
         (Successive_halving.budget ~min_samples:2 ~max_samples:13 ~factor:2 ()
         |> get)
       best candidates);
  let splitter =
    Cross_validation.target_independent_splitter
      (module Predefined_split)
      (Predefined_split.create
         ~test_folds:
           [| 0; 0; 0; 0; -1; -1; -1; -1; -1; -1; -1; -1; -1; -1; -1; -1 |]
         ()
      |> get)
  in
  let holdout = run ~splitter Grid_search.No_refit candidates |> get in
  Alcotest.(check int)
    "single holdout fold supported" 1
    (Array.length
       (folds
          (Successive_halving.rounds holdout).(0).Successive_halving.candidates.(
          0)))

let test_promotion () =
  Atomic.set Support.fits 0;
  let report =
    run ~budget:(budget ~max_fits:45 ()) best (grid offsets) |> get
  in
  let rounds = Successive_halving.rounds report in
  Alcotest.(check (array int))
    "allocation shrinks fairly" [| 5; 3; 2; 1 |]
    (Array.map
       (fun round -> Array.length round.Successive_halving.candidates)
       rounds);
  Alcotest.(check (array int))
    "tie promotion uses original indices" [| 1; 2; 0 |]
    rounds.(0).Successive_halving.promoted_candidate_indices;
  Alcotest.(check (array int))
    "survivors evaluated in original order" [| 0; 1; 2 |]
    (ids rounds.(1).Successive_halving.candidates);
  let selected = Successive_halving.selection report |> get in
  Alcotest.(check int)
    "sparse original winner index" 1
    selected.Grid_search.selected_candidate_index;
  Alcotest.(check int)
    "fresh fold fits and one full refit" 45 (Atomic.get Support.fits);
  check_rows_and_scores report;
  let source = Support.dataset () in
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-10))
    "refit uses full original dataset"
    (Support.expected source
       (Metadata.of_dataset source)
       (Array.init 16 Fun.id))
    (Support.predictions
       (Metadata.of_dataset source)
       source selected.Grid_search.selected_model);
  let before = snapshot report in
  rounds.(0).Successive_halving.promoted_candidate_indices.(0) <- 99;
  rounds.(0).Successive_halving.candidates.(0).Grid_search.parameters.(0) <-
    {
      Grid_search.parameter_name = "broken";
      parameter_value = Grid_search.Bool true;
    };
  rounds.(0).Successive_halving.candidates.(0) <-
    rounds.(0).Successive_halving.candidates.(1);
  Alcotest.(check bool) "report arrays isolated" true (snapshot report = before);
  let calls = ref 0 in
  let custom =
    Grid_search.Custom
      (fun candidates ->
        incr calls;
        Alcotest.(check (array int))
          "selector sees only final survivors with stable IDs" [| 1 |]
          (ids candidates);
        Ok 0)
  in
  let custom_report = run custom (grid offsets) |> get in
  Alcotest.(check int) "selector called once" 1 !calls;
  Alcotest.(check int)
    "custom selection maps position to original ID" 1
    (Successive_halving.selection custom_report |> get)
      .Grid_search.selected_candidate_index;
  let no_refit = run Grid_search.No_refit (grid offsets) |> get in
  Alcotest.(check bool)
    "no-refit result" true
    (Successive_halving.refit_result no_refit = Ok None);
  expect_error "no-refit selection unavailable"
    (Successive_halving.selection no_refit)

let test_sampling () =
  let draws = ref 0 and builds = ref 0 in
  let distribution =
    Parameter_distribution.custom (fun _ ->
        let index = !draws in
        incr draws;
        Ok offsets.(index))
  in
  let axis =
    Randomized_search.axis ~name:"offset" ~distribution
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let space =
    Randomized_search.create ~iterations:5 ~base:0.
      ~build:(fun offset ->
        incr builds;
        Ok (Support.pipeline ~offset ()))
      [| axis |]
    |> get
  in
  let report = run best (Successive_halving.of_randomized space) |> get in
  Alcotest.(check int) "one draw per original candidate" 5 !draws;
  Alcotest.(check int) "fresh build in each round" 11 !builds;
  Alcotest.(check bool)
    "same sampled parameters and results as grid" true
    (snapshot report = snapshot (run best (grid offsets) |> get))

let test_failures () =
  let candidates =
    grid
      ~build:(fun value ->
        if value < 0. then Error error
        else Ok (Support.pipeline ~offset:value ()))
      [| -1.; 0.; 0. |]
  in
  let report = run best candidates |> get in
  let first = (Successive_halving.rounds report).(0) in
  Alcotest.(check (array int))
    "failed candidate cannot promote" [| 1; 2 |]
    first.Successive_halving.promoted_candidate_indices;
  Alcotest.(check bool)
    "failed candidate retained" true
    (Option.is_some
       first.Successive_halving.candidates.(0).Grid_search.build_error);
  let all = grid ~build:(fun _ -> Error error) [| 0.; 1. |] in
  let report = run Grid_search.No_refit all |> get in
  Alcotest.(check int)
    "stop when no promotion possible" 1
    (Array.length (Successive_halving.rounds report));
  expect_error "no survivors reported" (Successive_halving.refit_result report);
  expect_error "abort on ordinary failure"
    (run ~failure_policy:Cross_validation.Abort best all);
  let control =
    grid
      ~build:(fun _ -> Error (Error.make Error.Cancelled ~remediation:"test"))
      [| 0. |]
  in
  expect_error "control failure always aborts" (run best control);
  let invalid_selector =
    run (Grid_search.Custom (fun _ -> Ok 1)) (grid offsets) |> get
  in
  expect_error "selector uses position, not candidate ID"
    (Successive_halving.selection invalid_selector);
  let source = Support.dataset () in
  let weights =
    Sample_weight.of_array ~expected_length:16
      (Array.init 16 (fun i -> if i = 0 then 1. else 0.))
    |> data
  in
  let weighted =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights
      ~groups:(Option.get (Dataset.groups source))
      ~x:(Dataset.features source) ~y:(Dataset.target source) ()
    |> data
  in
  Atomic.set Support.fits 0;
  expect_error "zero weight subset rejected before fitting"
    (run ~source:weighted best (grid offsets));
  Alcotest.(check int)
    "no fit on infeasible weights" 0 (Atomic.get Support.fits)

let check_execution execution =
  let evaluate execution =
    let events = ref [] in
    let owner = Domain.self () in
    let callback =
      Callback.create (fun event ->
          if Domain.self () <> owner then failwith "callback off caller domain";
          events := event :: !events;
          Ok Callback.Continue)
      |> get
    in
    let source = Support.dataset () in
    let metadata = Metadata.of_dataset ~callback source in
    let report = run ~execution ~metadata best (grid offsets) |> get in
    check_rows_and_scores report;
    let selected = Successive_halving.selection report |> get in
    ( List.rev !events,
      snapshot report,
      selected.Grid_search.selected_candidate_index,
      Support.predictions
        (Metadata.of_dataset source)
        source selected.Grid_search.selected_model )
  in
  Alcotest.(check bool)
    "execution preserves events, rows, scores, promotion and refit" true
    (evaluate execution = evaluate Execution.sequential)

let test_cancellation () =
  let builds = ref 0 and starts = ref 0 in
  let callback =
    Callback.create (fun[@warning "-4"] event ->
        match (event.Callback.operation, event.Callback.status) with
        | Callback.Search, Callback.Started ->
            incr starts;
            Ok Callback.Continue
        | Callback.Search, Callback.Progress { completed = 1; _ } ->
            Ok Callback.Cancel
        | _ -> Ok Callback.Continue)
    |> get
  in
  let metadata = Metadata.of_dataset ~callback (Support.dataset ()) in
  let candidates =
    grid
      ~build:(fun offset ->
        incr builds;
        Ok (Support.pipeline ~offset ()))
      offsets
  in
  expect_error "round progress cancellation" (run ~metadata best candidates);
  Alcotest.(check int) "single outer search lifecycle" 1 !starts;
  Alcotest.(check int) "no subsequent round builds" 5 !builds

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
      let budget =
        Successive_halving.budget ~min_samples:classes
          ~max_samples:(classes * 6) ~factor:2 ()
        |> get
      in
      let candidates = Successive_halving.of_randomized space in
      let report =
        if classes = 2 then
          Successive_halving.Binary_classification.search ~budget ~candidates
            ~splitter
            ~scorers:[| Binary_classification_scorer.accuracy |]
            ~refit:"accuracy" ~seed:(Seed.of_int 19) source
          |> get
        else
          Successive_halving.Multiclass_classification.search_with_policy
            ~budget ~candidates ~promotion_score:"accuracy" ~splitter
            ~scorers:[| Multiclass_classification_scorer.accuracy |]
            ~policy:
              (Grid_search.Custom
                 (fun candidates -> Ok (Array.length candidates - 1)))
            ~seed:(Seed.of_int 19) source
          |> get
      in
      Successive_halving.rounds report
      |> Array.iter (fun round ->
          Array.iter
            (fun candidate ->
              Array.iter
                (fun fold ->
                  let train, _ = rows fold in
                  let observed =
                    Array.to_list train
                    |> List.map (fun row -> row mod classes)
                    |> List.sort_uniq Int.compare
                  in
                  Alcotest.(check int)
                    "every class in each training prefix" classes
                    (List.length observed))
                (folds candidate))
            round.Successive_halving.candidates);
      let too_small =
        Successive_halving.budget ~min_samples:2 ~max_samples:(classes * 6)
          ~factor:2 ()
        |> get
      in
      if classes = 3 then
        expect_error "class budget infeasible"
          (Successive_halving.Multiclass_classification.search ~budget:too_small
             ~candidates ~splitter
             ~scorers:[| Multiclass_classification_scorer.accuracy |]
             ~refit:"accuracy" ~seed:(Seed.of_int 19) source);
      let fitted =
        (Successive_halving.selection report |> get).Grid_search.selected_model
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
    ("resource and fit budgets", `Quick, test_schedule);
    ("promotion, fresh fits, and refit policies", `Quick, test_promotion);
    ("sample once and rebuild survivors", `Quick, test_sampling);
    ("failures and weight feasibility", `Quick, test_failures);
    ("classification class coverage", `Quick, test_classification);
    ("cancellation between rounds", `Quick, test_cancellation);
    ( "metadata and execution",
      `Quick,
      fun () -> check_execution Execution.sequential );
  ]
