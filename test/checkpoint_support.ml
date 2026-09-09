open Modelkit
module Support = Evaluation_metadata_support
module Callback = Modelkit.Callback

let get = Support.get
let data = Support.data
let values = [| 1000.; 0.; 2.; 4. |]
let continue = Callback.create (fun _ -> Ok Callback.Continue) |> get

let default_metadata () =
  Metadata.of_dataset ~callback:continue (Support.dataset ())

let session ?resume ?(specification_id = "weighted-search-v1")
    ?(configuration_id = Float.to_string) () =
  Search_checkpoint.create ?resume ~specification_id ~configuration_id () |> get

let grid ?(offsets = values)
    ?(build = fun offset -> Ok (Support.pipeline ~offset ())) () =
  let axis =
    Grid_search.axis ~name:"offset" ~values:offsets
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  Grid_search.create ~base:0. ~build [| axis |] |> get

let space () =
  let distribution = Parameter_distribution.choice values |> get in
  let axis =
    Randomized_search.axis ~name:"offset" ~distribution
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  Randomized_search.create ~iterations:4 ~base:0.
    ~build:(fun offset -> Ok (Support.pipeline ~offset ()))
    [| axis |]
  |> get

let best = Grid_search.Best_score "neg_mean_squared_error"

let run ?checkpoint ?execution ?metadata ?(seed = Seed.of_int 19)
    ?(source = Support.dataset ()) ?(splitter = Support.splitter ())
    ?(policy = best) ?(failure_policy = Cross_validation.Record)
    ?(return_train_score = true) grid =
  let metadata =
    Option.value metadata
      ~default:(Metadata.of_dataset ~callback:continue source)
  in
  Grid_search.Regression.search_with_policy ?checkpoint ?execution ~metadata
    ~return_train_score ~failure_policy ~grid ~splitter
    ~scorers:
      [|
        Regression_scorer.neg_mean_squared_error;
        Regression_scorer.neg_mean_absolute_error;
      |]
    ~policy ~seed source

let halving ?checkpoint ?execution ?metadata () =
  let metadata = Option.value metadata ~default:(default_metadata ()) in
  Successive_halving.Regression.search ?checkpoint ?execution ~metadata
    ~return_train_score:true
    ~budget:
      (Successive_halving.budget ~min_samples:3 ~max_samples:12 ~factor:2 ()
      |> get)
    ~candidates:(Successive_halving.of_grid (grid ()))
    ~splitter:(Support.splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19) (Support.dataset ())

let random ?checkpoint ?execution ?metadata () =
  let metadata = Option.value metadata ~default:(default_metadata ()) in
  Randomized_search.Regression.search ?checkpoint ?execution ~metadata
    ~return_train_score:true ~space:(space ()) ~splitter:(Support.splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19) (Support.dataset ())

let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let cancelled label = function
  | Ok _ -> Alcotest.fail label
  | Error error -> (
      (match Error.kind error with
      | Error.Cancelled -> ()
      | _ -> Alcotest.fail (Error.to_string error))
      [@warning "-4"])

let cancel predicate =
  Callback.create (fun event ->
      Ok (if predicate event then Callback.Cancel else Callback.Continue))
  |> get

let candidate_two event =
  event.Callback.operation = Callback.Candidate
  && event.Callback.status = Callback.Started
  && List.mem (Error.Candidate 2) event.Callback.context

let metadata callback = Metadata.of_dataset ~callback (Support.dataset ())

let roundtrip checkpoint =
  Search_checkpoint.snapshot checkpoint
  |> Search_checkpoint.encode |> get |> Search_checkpoint.decode |> get

let scores candidates =
  Array.map
    (fun candidate ->
      ( candidate.Grid_search.candidate_index,
        candidate.Grid_search.parameters,
        candidate.Grid_search.rank,
        Array.map
          (fun summary ->
            ( summary.Grid_search.scorer_name,
              summary.Grid_search.train,
              summary.Grid_search.test ))
          candidate.Grid_search.scores ))
    candidates

let predictions model =
  let source = Support.dataset () in
  Support.predictions (Metadata.of_dataset source) source model

let compare_grid expected observed =
  Alcotest.(check bool)
    "candidate scores and ranks" true
    (scores (Grid_search.candidates expected)
    = scores (Grid_search.candidates observed));
  let expected = Grid_search.selection expected |> get
  and observed = Grid_search.selection observed |> get in
  Alcotest.(check int)
    "selected original index" expected.Grid_search.selected_candidate_index
    observed.Grid_search.selected_candidate_index;
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "full refit prediction"
    (predictions expected.Grid_search.selected_model)
    (predictions observed.Grid_search.selected_model)

let produce () =
  let checkpoint = session () in
  cancelled "candidate cancellation"
    (run ~checkpoint ~metadata:(metadata (cancel candidate_two)) (grid ()));
  let state = Search_checkpoint.snapshot checkpoint in
  Alcotest.(check int)
    "two completed candidates" 2
    (Array.length (Search_checkpoint.completed state));
  Search_checkpoint.encode state |> get

let resume ?(execution = Execution.sequential) bytes =
  let snapshot = Search_checkpoint.decode bytes |> get in
  let checkpoint = session ~resume:snapshot () in
  Atomic.set Support.fits 0;
  let actual = run ~checkpoint ~execution (grid ()) |> get in
  Alcotest.(check int)
    "resume skips eight committed fold fits" 9 (Atomic.get Support.fits);
  compare_grid (run (grid ()) |> get) actual;
  Alcotest.(check int)
    "all candidates committed after resume" 4
    (Search_checkpoint.snapshot checkpoint
    |> Search_checkpoint.completed |> Array.length)

let check_execution execution = resume ~execution (produce ())

let test_algorithms () =
  List.iter
    (fun execute ->
      let checkpoint = session () in
      cancelled "random search cancelled"
        (execute ~checkpoint ~metadata:(metadata (cancel candidate_two)) ());
      let restored = session ~resume:(roundtrip checkpoint) () in
      Atomic.set Support.fits 0;
      let actual =
        execute ~checkpoint:restored ~metadata:(default_metadata ()) () |> get
      in
      Alcotest.(check int)
        "random resume skips committed fits" 9 (Atomic.get Support.fits);
      compare_grid
        (execute ~checkpoint:(session ()) ~metadata:(default_metadata ()) ()
        |> get)
        actual)
    [ (fun ~checkpoint ~metadata () -> random ~checkpoint ~metadata ()) ];
  let checkpoint = session () in
  let callback =
    cancel (fun[@warning "-4"] event ->
        match (event.Callback.operation, event.Callback.status) with
        | Callback.Search, Callback.Progress { completed = 1; _ } -> true
        | _ -> false)
  in
  cancelled "halving round cancelled"
    (halving ~checkpoint ~metadata:(metadata callback) ());
  let restored = session ~resume:(roundtrip checkpoint) () in
  Atomic.set Support.fits 0;
  let actual = halving ~checkpoint:restored () |> get in
  Alcotest.(check int)
    "halving resume skips first round" 13 (Atomic.get Support.fits);
  let expected = halving () |> get in
  let rounds report =
    Successive_halving.rounds report
    |> Array.map (fun round ->
        ( round.Successive_halving.training_samples,
          round.Successive_halving.promoted_candidate_indices,
          scores round.Successive_halving.candidates ))
  in
  Alcotest.(check bool)
    "resumed promotion and scores" true
    (rounds expected = rounds actual);
  let selected report = Successive_halving.selection report |> get in
  Alcotest.(check int)
    "resumed halving selection"
    (selected expected).Grid_search.selected_candidate_index
    (selected actual).Grid_search.selected_candidate_index;
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "halving full refit"
    (predictions (selected expected).Grid_search.selected_model)
    (predictions (selected actual).Grid_search.selected_model)

let test_identity () =
  let saved = produce () |> Search_checkpoint.decode |> get in
  let verify label f =
    let checkpoint = session ~resume:saved () in
    Atomic.set Support.fits 0;
    expect_error label (f checkpoint);
    Alcotest.(check int)
      "identity mismatch before fit" 0 (Atomic.get Support.fits)
  in
  verify "seed" (fun checkpoint ->
      run ~checkpoint ~seed:(Seed.of_int 20) (grid ()));
  verify "parameters" (fun checkpoint ->
      run ~checkpoint (grid ~offsets:[| 1000.; 0.; 2.; 5. |] ()));
  verify "future candidate count" (fun checkpoint ->
      run ~checkpoint (grid ~offsets:[| 1000.; 0.; 2. |] ()));
  verify "options" (fun checkpoint ->
      run ~checkpoint ~return_train_score:false (grid ()));
  verify "policy" (fun checkpoint ->
      run ~checkpoint ~policy:Grid_search.No_refit (grid ()));
  verify "failure policy" (fun checkpoint ->
      run ~checkpoint ~failure_policy:Cross_validation.Abort (grid ()));
  verify "split membership" (fun checkpoint ->
      run ~checkpoint
        ~splitter:
          (Cross_validation.target_independent_splitter
             (module K_fold)
             (K_fold.create ~folds:4 () |> get))
        (grid ()));
  let source = Support.dataset () in
  let changed ?(x = Dataset.features source) ?(y = Dataset.target source)
      ?(sample_weight = Option.get (Dataset.sample_weight source))
      ?(groups = Option.get (Dataset.groups source)) ?feature_names () =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight ~groups
      ?feature_names ~x ~y ()
    |> data
  in
  let x = Matrix.to_arrays (Dataset.features source) in
  x.(15).(0) <- 99.;
  let targets =
    Target.regression_values (Dataset.target source) |> Vector.to_array
  in
  targets.(15) <- 99.;
  List.iter
    (fun (name, source) ->
      verify name (fun checkpoint -> run ~checkpoint ~source (grid ())))
    [
      ("features", changed ~x:(Matrix.of_arrays x |> data) ());
      ("targets", changed ~y:(Support.regression targets) ());
      ( "weights",
        changed
          ~sample_weight:
            (Sample_weight.of_array ~expected_length:16 (Array.make 16 1.)
            |> data)
          () );
      ( "groups",
        changed
          ~groups:
            (Groups.create ~expected_length:16
               (Array.init 16 (fun i -> 200 + (i / 2)))
            |> data)
          () );
      ( "schema",
        changed
          ~feature_names:
            (Feature_names.create ~expected_count:2 [| "a"; "b" |] |> data)
          () );
    ];
  verify "callback presence" (fun checkpoint ->
      run ~checkpoint ~metadata:(Metadata.of_dataset source) (grid ()));
  verify "explicit metadata" (fun checkpoint ->
      run ~checkpoint ~metadata:(Metadata.create ()) (grid ()));
  expect_error "specification ID"
    (Search_checkpoint.create ~resume:saved ~specification_id:"v2"
       ~configuration_id:Float.to_string ());
  let checkpoint =
    session ~resume:saved
      ~configuration_id:(fun value -> "v2:" ^ Float.to_string value)
      ()
  in
  expect_error "configuration ID" (run ~checkpoint (grid ()))

let test_boundaries () =
  let checkpoint = session () in
  let callback =
    cancel (fun[@warning "-4"] event ->
        event.Callback.operation = Callback.Fold
        && List.mem (Error.Candidate 1) event.Callback.context
        &&
        match event.Callback.status with
        | Callback.Finished _ -> true
        | _ -> false)
  in
  cancelled "fold cancellation"
    (run ~checkpoint ~metadata:(metadata callback) (grid ()));
  Alcotest.(check int)
    "interrupted candidate not committed" 1
    (Search_checkpoint.snapshot checkpoint
    |> Search_checkpoint.completed |> Array.length);
  Atomic.set Support.fits 0;
  let actual =
    run ~checkpoint:(session ~resume:(roundtrip checkpoint) ()) (grid ()) |> get
  in
  Alcotest.(check int)
    "interrupted candidate repeats whole evaluation" 13
    (Atomic.get Support.fits);
  compare_grid (run (grid ()) |> get) actual;
  let checkpoint = session () in
  cancelled "refit cancellation"
    (run ~checkpoint
       ~metadata:
         (metadata
            (cancel (fun event ->
                 event.Callback.operation = Callback.Refit
                 && event.Callback.status = Callback.Started)))
       (grid ()));
  Atomic.set Support.fits 0;
  ignore
    (run ~checkpoint:(session ~resume:(roundtrip checkpoint) ()) (grid ())
    |> get);
  Alcotest.(check int) "only full refit remains" 1 (Atomic.get Support.fits);
  let checkpoint = session () in
  let reentered = ref false in
  let callback =
    Callback.create (fun event ->
        if
          event.Callback.operation = Callback.Search
          && event.Callback.status = Callback.Started
        then (
          expect_error "same session cannot reenter" (run ~checkpoint (grid ()));
          reentered := true);
        Ok Callback.Continue)
    |> get
  in
  ignore (run ~checkpoint ~metadata:(metadata callback) (grid ()) |> get);
  Alcotest.(check bool) "reentrancy checked" true !reentered

let test_codec () =
  let source = Support.dataset () in
  let schema = Dataset.feature_schema source in
  let named =
    Feature_names.create ~expected_count:2 [| "a"; "b" |]
    |> data |> Feature_schema.named
  in
  let kinds =
    [|
      Error.Data
        (Data_error.Length_mismatch { name = "x"; expected = 2; observed = 1 });
      Error.Shape_mismatch
        { name = "x"; expected = [ 2; 3 ]; observed = [ 1; 3 ] };
      Error.Feature_schema_mismatch { expected = schema; observed = named };
      Error.Validation { name = "x"; reason = "bad" };
      Error.Numerical { operation = "fit"; reason = "bad" };
      Error.Convergence { algorithm = "fit"; reason = "bad" };
      Error.Compatibility { component = "fit"; reason = "bad" };
      Error.Artifact { operation = "fit"; reason = "bad" };
    |]
  in
  let failures =
    Array.mapi
      (fun i kind ->
        Error.make
          ~context:
            [
              Error.Stage "nested";
              Error.Candidate i;
              Error.Fold 1;
              Error.Feature (Feature_name.create "x" |> data);
            ]
          ~remediation:"try again" kind)
      kinds
  in
  let checkpoint = session () in
  let bad =
    grid
      ~offsets:(Array.init (Array.length failures) Float.of_int)
      ~build:(fun index -> Error failures.(int_of_float index))
      ()
  in
  ignore (run ~checkpoint ~policy:Grid_search.No_refit bad |> get);
  let bytes =
    Search_checkpoint.snapshot checkpoint |> Search_checkpoint.encode |> get
  in
  let restored = Search_checkpoint.decode bytes |> get in
  Search_checkpoint.completed restored
  |> Array.iteri (fun i entry ->
      match entry.Search_checkpoint.evaluation with
      | Ok _ -> Alcotest.fail "missing build failure"
      | Error error ->
          Alcotest.(check string)
            "typed failure roundtrip"
            (Error.to_string
               (Error.with_context (Error.Candidate i) failures.(i)))
            (Error.to_string error));
  let calls = ref 0 in
  let replay =
    grid
      ~offsets:(Array.init (Array.length failures) Float.of_int)
      ~build:(fun _ ->
        incr calls;
        failwith "cached failure rebuilt")
      ()
  in
  ignore
    (run
       ~checkpoint:(session ~resume:restored ())
       ~policy:Grid_search.No_refit replay
    |> get);
  Alcotest.(check int) "recorded failures reused" 0 !calls;
  let corrupt = Bytes.copy bytes in
  Bytes.set corrupt (Bytes.length corrupt / 2) '?';
  List.iter
    (fun bytes ->
      expect_error "invalid checkpoint bytes" (Search_checkpoint.decode bytes))
    [
      Bytes.empty;
      Bytes.of_string "9999999999:";
      Bytes.sub bytes 0 (Bytes.length bytes - 1);
      Bytes.cat bytes (Bytes.of_string "x");
      corrupt;
    ];
  let checkpoint = session () in
  ignore (run ~checkpoint (grid ()) |> get);
  let state = Search_checkpoint.snapshot checkpoint in
  let entries = Search_checkpoint.completed state in
  let report = get entries.(0).Search_checkpoint.evaluation in
  let folds = Cross_validation.folds report in
  folds.(0).Cross_validation.scores.(0) <-
    { Cross_validation.name = "changed"; train_score = None; test_score = None };
  Alcotest.(check bool)
    "reports expose defensive copies" true
    (Search_checkpoint.encode state
    |> get
    = (Search_checkpoint.snapshot checkpoint |> Search_checkpoint.encode |> get)
    )

let test_classification () =
  List.iter
    (fun classes ->
      let count = classes * 8 in
      let source =
        Dataset.create ~finiteness:Dataset.Require_finite
          ~x:
            (Matrix.init ~rows:count ~columns:classes (fun row column ->
                 if row mod classes = column then 1. else 0.)
            |> data)
          ~y:
            (Target.classification
               (Array.init count (fun row -> row mod classes)))
          ()
        |> data
      in
      let build c =
        let ( let* ) = Result.bind in
        let* estimator =
          if classes = 2 then
            let* spec = Logistic_regression.create ~c () in
            Pipeline.classifier ~name:"classifier"
              (module Logistic_regression)
              spec
          else
            let* spec = Multinomial_logistic_regression.create ~c () in
            Pipeline.classifier ~name:"classifier"
              (module Multinomial_logistic_regression)
              spec
        in
        Pipeline.set_estimator Pipeline.empty estimator
      in
      let grid =
        Grid_search.create ~base:1. ~build
          [|
            Grid_search.axis ~name:"c" ~values:[| 0.1; 1. |]
              ~encode:(fun c -> Grid_search.Float c)
              ~set:(fun _ c -> Ok c)
            |> get;
          |]
        |> get
      in
      let space =
        Randomized_search.create ~base:1. ~build
          [|
            Randomized_search.axis ~name:"c"
              ~distribution:(Parameter_distribution.choice [| 0.1; 1. |] |> get)
              ~encode:(fun c -> Grid_search.Float c)
              ~set:(fun _ c -> Ok c)
            |> get;
          |]
        |> get
      in
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
      let seed = Seed.of_int 19 in
      let selected selection report =
        let selected = selection report |> get in
        ( selected.Grid_search.selected_candidate_index,
          Pipeline.predict selected.Grid_search.selected_model
            ~feature_schema:(Dataset.feature_schema source)
            ~x:(Dataset.features source)
          |> get |> Target.classification_values )
      in
      let execute algorithm checkpoint metadata =
        match (algorithm, classes) with
        | 0, 2 ->
            Grid_search.Binary_classification.search ~checkpoint ~metadata ~grid
              ~splitter
              ~scorers:[| Binary_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Grid_search.selection)
        | 0, _ ->
            Grid_search.Multiclass_classification.search ~checkpoint ~metadata
              ~grid ~splitter
              ~scorers:[| Multiclass_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Grid_search.selection)
        | 1, 2 ->
            Randomized_search.Binary_classification.search ~checkpoint ~metadata
              ~space ~splitter
              ~scorers:[| Binary_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Grid_search.selection)
        | 1, _ ->
            Randomized_search.Multiclass_classification.search ~checkpoint
              ~metadata ~space ~splitter
              ~scorers:[| Multiclass_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Grid_search.selection)
        | _, 2 ->
            Successive_halving.Binary_classification.search ~checkpoint
              ~metadata ~budget
              ~candidates:(Successive_halving.of_grid grid)
              ~splitter
              ~scorers:[| Binary_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Successive_halving.selection)
        | _, _ ->
            Successive_halving.Multiclass_classification.search ~checkpoint
              ~metadata ~budget
              ~candidates:(Successive_halving.of_grid grid)
              ~splitter
              ~scorers:[| Multiclass_classification_scorer.accuracy |]
              ~refit:"accuracy" ~seed source
            |> Result.map (selected Successive_halving.selection)
      in
      List.iter
        (fun algorithm ->
          let checkpoint = session () in
          let callback =
            cancel (fun event ->
                event.Callback.operation = Callback.Candidate
                && event.Callback.status = Callback.Started
                && List.mem (Error.Candidate 1) event.Callback.context)
          in
          cancelled "classification interruption"
            (execute algorithm checkpoint
               (Metadata.of_dataset ~callback source));
          let restored = session ~resume:(roundtrip checkpoint) () in
          let metadata = Metadata.of_dataset ~callback:continue source in
          let actual = execute algorithm restored metadata |> get in
          let expected = execute algorithm (session ()) metadata |> get in
          Alcotest.(check bool)
            "classification resume matches uninterrupted" true
            (expected = actual))
        [ 0; 1; 2 ])
    [ 2; 3 ]

let test_policies_and_fold_failures () =
  let calls = ref 0 in
  let policy =
    Grid_search.Custom
      (fun candidates ->
        incr calls;
        Ok (Array.length candidates - 1))
  in
  let checkpoint = session () in
  cancelled "custom selection interrupted"
    (run ~checkpoint ~policy
       ~metadata:(metadata (cancel candidate_two))
       (grid ()));
  Alcotest.(check int)
    "selector not called before evaluation completes" 0 !calls;
  let actual =
    run
      ~checkpoint:(session ~resume:(roundtrip checkpoint) ())
      ~policy (grid ())
    |> get
  in
  Alcotest.(check int) "restored selector called once" 1 !calls;
  compare_grid (run ~policy (grid ()) |> get) actual;
  let checkpoint = session () in
  ignore (run ~checkpoint ~policy:Grid_search.No_refit (grid ()) |> get);
  Atomic.set Support.fits 0;
  let report =
    run
      ~checkpoint:(session ~resume:(roundtrip checkpoint) ())
      ~policy:Grid_search.No_refit (grid ())
    |> get
  in
  Alcotest.(check int)
    "completed no-refit performs no fits" 0 (Atomic.get Support.fits);
  Alcotest.(check bool)
    "no-refit preserved" true
    (Grid_search.refit_result report = Ok None);
  let bad =
    grid ~offsets:[| 0. |]
      ~build:(fun _ ->
        Ok (Support.pipeline ~callbacks:Metadata.Request.Reject ()))
      ()
  in
  let checkpoint = session () in
  let original = run ~checkpoint ~policy:Grid_search.No_refit bad |> get in
  let state = roundtrip checkpoint in
  let partial =
    (Search_checkpoint.completed state).(0).Search_checkpoint.evaluation |> get
    |> Cross_validation.folds
  in
  Alcotest.(check int)
    "recorded fold failures survive codec" 4
    (Array.fold_left
       (fun count fold -> count + Array.length fold.Cross_validation.failures)
       0 partial);
  let report =
    run ~checkpoint:(session ~resume:state ()) ~policy:Grid_search.No_refit bad
    |> get
  in
  Alcotest.(check bool)
    "failure aggregates preserved" true
    (scores (Grid_search.candidates original)
    = scores (Grid_search.candidates report))

let test_preparation_cancellation () =
  let draws = ref 0 in
  let distribution =
    Parameter_distribution.custom (fun _ ->
        incr draws;
        Error (Error.make Error.Cancelled ~remediation:"test"))
  in
  let axis =
    Randomized_search.axis ~name:"offset" ~distribution
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let space =
    Randomized_search.create ~iterations:4 ~base:0.
      ~build:(fun offset -> Ok (Support.pipeline ~offset ()))
      [| axis |]
    |> get
  in
  let checkpoint = session () in
  cancelled "preparation control error"
    (Randomized_search.Regression.search ~checkpoint
       ~metadata:(default_metadata ()) ~space ~splitter:(Support.splitter ())
       ~scorers:[| Regression_scorer.neg_mean_squared_error |]
       ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19)
       (Support.dataset ()));
  Alcotest.(check int) "control errors stop later draws" 1 !draws;
  Alcotest.(check int)
    "preparation commits no evaluation" 0
    (Search_checkpoint.snapshot checkpoint
    |> Search_checkpoint.completed |> Array.length)

let tests =
  [
    ("preparation cancellation", `Quick, test_preparation_cancellation);
    ("classification checkpoint integration", `Quick, test_classification);
    ( "custom selection, no-refit, and fold failures",
      `Quick,
      test_policies_and_fold_failures );
    ( "grid restart and fit reuse",
      `Quick,
      fun () -> check_execution Execution.sequential );
    ("randomized and halving restart", `Quick, test_algorithms);
    ("identity validation", `Quick, test_identity);
    ("interruption and refit boundaries", `Quick, test_boundaries);
    ("bounded codec and typed failures", `Quick, test_codec);
  ]
