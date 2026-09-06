open Modelkit
open Evaluation_metadata_support
module Callback = Modelkit.Callback

let test_alignment () =
  let source = dataset () in
  let metadata = Metadata.of_dataset source in
  let evaluation = run (pipeline ()) source |> get in
  Alcotest.(check int)
    "all folds fit" 4
    (Cross_validation.successful_fold_count evaluation);
  Array.iter
    (fun fold ->
      let rows = Option.get fold.Cross_validation.train_indices in
      let observed =
        predictions metadata source (Option.get fold.Cross_validation.model)
      in
      Alcotest.(check (array (Alcotest.float 1e-12)))
        "fit and inference metadata alignment"
        (expected source metadata rows)
        observed;
      let test = Option.get fold.Cross_validation.test_indices in
      let train_groups =
        Array.map (Groups.get (Option.get (Dataset.groups source))) rows
      in
      Array.iter
        (fun row ->
          Alcotest.(check bool)
            "splitter groups stay excluded" false
            (Array.mem
               (Groups.get (Option.get (Dataset.groups source)) row)
               train_groups))
        test;
      let weights = Dataset.sample_weight source |> Option.get in
      let targets = Target.regression_values (Dataset.target source) in
      let total, weight =
        Array.fold_left
          (fun (sum, total) row ->
            let w = Sample_weight.get weights row
            and residual = Vector.get targets row -. observed.(row) in
            (sum +. (w *. residual *. residual), total +. w))
          (0., 0.) test
      in
      Alcotest.check (Alcotest.float 1e-10) "scoring retains dataset weights"
        (-.total /. weight) (score fold))
    (Cross_validation.folds evaluation);
  let reordered = Row_view.create ~source_size:16 [| 8; 1; 5 |] |> data in
  let selected = Metadata.select metadata reordered |> get in
  Alcotest.(check (array int))
    "metadata selection preserves arbitrary row order" [| 104; 100; 102 |]
    (Metadata.groups selected |> Option.get |> Groups.to_array)

let test_leakage () =
  let source = dataset () in
  let metadata = Metadata.of_dataset source in
  let specification = pipeline ~align:false () in
  let baseline = run ~metadata specification source |> get in
  let fold = (Cross_validation.folds baseline).(0) in
  let held_out = Option.get fold.Cross_validation.test_indices in
  let weights =
    Metadata.sample_weight metadata
    |> Option.get |> Sample_weight.to_vector |> Vector.to_array
  in
  let groups = Metadata.groups metadata |> Option.get |> Groups.to_array in
  Array.iter
    (fun row ->
      weights.(row) <- 1000.;
      groups.(row) <- -200)
    held_out;
  let changed =
    Metadata.create
      ~sample_weight:(Sample_weight.of_array ~expected_length:16 weights |> data)
      ~groups:(Groups.create ~expected_length:16 groups |> data)
      ()
  in
  let perturbed = run ~metadata:changed specification source |> get in
  let after = (Cross_validation.folds perturbed).(0) in
  Alcotest.(check (array int))
    "explicit consumer groups do not replace splitter groups" held_out
    (Option.get after.Cross_validation.test_indices);
  Alcotest.(check (array (Alcotest.float 0.)))
    "held-out metadata cannot affect fitted state"
    (predictions metadata source (Option.get fold.Cross_validation.model))
    (predictions metadata source (Option.get after.Cross_validation.model));
  Alcotest.(check bool)
    "held-out metadata reaches inference" true
    (score fold <> score after)

let test_refit () =
  let source = dataset () in
  Atomic.set fits 0;
  let result = search source |> get in
  Alcotest.(check int) "candidate folds and full refit" 9 (Atomic.get fits);
  let selected = Grid_search.selection result |> get in
  let offset =
    Float.of_int (2 * selected.Grid_search.selected_candidate_index)
  in
  let expected =
    expected source (Metadata.of_dataset source) (Array.init 16 Fun.id)
    |> Array.map (( +. ) offset)
  in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "refit receives complete aligned metadata" expected
    (predictions
       (Metadata.of_dataset source)
       source selected.Grid_search.selected_model)

let[@warning "-4"] check_control kind context result =
  match result with
  | Ok _ -> Alcotest.fail "expected callback control error"
  | Error error ->
      Alcotest.(check bool)
        "typed control error" true
        (match (kind, Error.kind error) with
        | `Cancel, Error.Cancelled | `Failure, Error.Callback_failure _ -> true
        | _ -> false);
      Alcotest.(check bool)
        "control context" true
        (Error.context error = context)

let[@warning "-4"] test_callbacks () =
  let source = dataset () in
  let owner = Domain.self () and events = ref [] in
  let callback =
    Callback.create (fun event ->
        Alcotest.(check bool)
          "caller domain owns callbacks" true
          (Domain.self () = owner);
        events := event :: !events;
        Ok Callback.Continue)
    |> get
  in
  let metadata = Metadata.of_dataset ~callback source in
  ignore (run ~metadata (pipeline ()) source |> get);
  let ordered = List.rev !events in
  Alcotest.(check bool)
    "CV starts first" true
    ((List.hd ordered).Callback.operation = Callback.Cross_validation
    && (List.hd ordered).Callback.status = Callback.Started);
  let fold_starts =
    List.filter_map
      (fun event ->
        match
          ( event.Callback.operation,
            event.Callback.status,
            event.Callback.context )
        with
        | Callback.Fold, Callback.Started, [ Error.Fold index ] -> Some index
        | _ -> None)
      ordered
  in
  Alcotest.(check (list int)) "fold event order" [ 0; 1; 2; 3 ] fold_starts;
  Alcotest.(check bool)
    "consumer progress keeps full nested path" true
    (List.exists
       (fun event ->
         event.Callback.operation = Callback.Fit
         && event.Callback.status
            = Callback.Progress { completed = 1; total = Some 1 }
         && event.Callback.context
            = [
                Error.Fold 0;
                Error.Stage "columns";
                Error.Stage "union";
                Error.Stage "chain";
                Error.Stage "consumer";
              ])
       ordered);
  events := [];
  ignore (search ~metadata source |> get);
  let ordered = List.rev !events in
  let candidates =
    List.filter_map
      (fun event ->
        match
          ( event.Callback.operation,
            event.Callback.status,
            event.Callback.context )
        with
        | Callback.Candidate, Callback.Started, [ Error.Candidate index ] ->
            Some index
        | _ -> None)
      ordered
  in
  Alcotest.(check (list int)) "candidate event order" [ 0; 1 ] candidates;
  Alcotest.(check int)
    "one refit event" 1
    (List.filter
       (fun event ->
         event.Callback.operation = Callback.Refit
         && event.Callback.status = Callback.Started)
       ordered
    |> List.length)

let test_control () =
  let source = dataset () in
  let callback = Callback.create (fun _ -> Ok Callback.Cancel) |> get in
  Atomic.set fits 0;
  run ~metadata:(Metadata.of_dataset ~callback source) (pipeline ()) source
  |> check_control `Cancel [];
  Alcotest.(check int) "cancel before any work" 0 (Atomic.get fits);
  let callback =
    Callback.create (fun event ->
        if
          event.Callback.operation = Callback.Fold
          && event.Callback.status = Callback.Started
        then Ok Callback.Cancel
        else Ok Callback.Continue)
    |> get
  in
  run ~metadata:(Metadata.of_dataset ~callback source) (pipeline ()) source
  |> check_control `Cancel [ Error.Fold 0 ];
  Alcotest.(check int)
    "sequential cancellation stops later folds" 1 (Atomic.get fits);
  let callback =
    Callback.create (fun event ->
        if event.Callback.operation = Callback.Fit then Error "handler failed"
        else Ok Callback.Continue)
    |> get
  in
  search ~metadata:(Metadata.of_dataset ~callback source) source
  |> check_control `Failure
       [
         Error.Candidate 0;
         Error.Fold 0;
         Error.Stage "columns";
         Error.Stage "union";
         Error.Stage "chain";
         Error.Stage "consumer";
       ];
  let callback =
    Callback.create ~max_buffered_events:1 (fun _ -> Ok Callback.Continue)
    |> get
  in
  run ~metadata:(Metadata.of_dataset ~callback source) (pipeline ()) source
  |> check_control `Failure
       [
         Error.Fold 0;
         Error.Stage "columns";
         Error.Stage "union";
         Error.Stage "chain";
         Error.Stage "consumer";
       ];
  let callback = Callback.create (fun _ -> raise Exit) |> get in
  let raised =
    try
      ignore
        (run
           ~metadata:(Metadata.of_dataset ~callback source)
           (pipeline ()) source);
      false
    with Exit -> true
  in
  Alcotest.(check bool) "handler exceptions propagate" true raised

let test_preflight_and_direct () =
  let source = dataset () in
  Atomic.set fits 0;
  let calls = ref 0 in
  let callback =
    Callback.create (fun _ ->
        incr calls;
        Ok Callback.Continue)
    |> get
  in
  let metadata =
    Metadata.create ~callback
      ~groups:(Groups.create ~expected_length:1 [| 1 |] |> data)
      ()
  in
  (match run ~metadata (pipeline ()) source with
  | Ok _ -> Alcotest.fail "accepted wrong row count"
  | Error _ -> ());
  Alcotest.(check int) "metadata preflight before callbacks" 0 !calls;
  Alcotest.(check int) "metadata preflight before fitting" 0 (Atomic.get fits);
  let callback =
    Callback.create (fun event ->
        match event.Callback.status with
        | Callback.Progress _ -> Ok Callback.Cancel
        | Callback.Started | Callback.Finished _ -> Ok Callback.Continue)
    |> get
  in
  Pipeline.fit_with_metadata (pipeline ())
    ~metadata:(Metadata.of_dataset ~callback source)
    ~rng:(Rng.create (Seed.of_int 42))
    ~feature_schema:(Dataset.feature_schema source)
    ~x:(Dataset.features source) ~y:(Dataset.target source) ()
  |> check_control `Cancel
       [
         Error.Stage "columns";
         Error.Stage "union";
         Error.Stage "chain";
         Error.Stage "consumer";
       ]

module Class_consumer = struct
  include Consumer

  type target = Target.classification Target.t

  let fit specification ~metadata ~rng ~feature_schema ~x ~y () =
    let y =
      Option.map
        (fun y ->
          Target.classification_values y |> Array.map Float.of_int |> regression)
        y
    in
    Consumer.fit specification ~metadata ~rng ~feature_schema ~x ~y ()
end

let test_classification () =
  let regression_source = dataset () in
  List.iter
    (fun classes ->
      let source =
        Dataset.create ~finiteness:Dataset.Require_finite
          ?sample_weight:(Dataset.sample_weight regression_source)
          ?groups:(Dataset.groups regression_source)
          ~x:(Dataset.features regression_source)
          ~y:
            (Target.classification (Array.init 16 (fun row -> row mod classes)))
          ()
        |> data
      in
      let stage =
        Pipeline.Supervised.metadata_transformer ~name:"consumer"
          (module Class_consumer)
          Consumer.
            {
              align = false;
              offset = 0.;
              callbacks = Metadata.Request.Optional;
            }
        |> get |> nest
      in
      let builder =
        Pipeline.Supervised.add_transformer Pipeline.Supervised.empty stage
        |> get
      in
      let terminal =
        if classes = 2 then
          Pipeline.classifier ~name:"classifier"
            (module Logistic_regression)
            ~predict_proba:Logistic_regression.predict_proba
            ~classes:Logistic_regression.classes
            (Logistic_regression.create () |> get)
          |> get
        else
          Pipeline.classifier ~name:"classifier"
            (module Multinomial_logistic_regression)
            ~predict_proba:Multinomial_logistic_regression.predict_proba
            ~classes:Multinomial_logistic_regression.classes
            (Multinomial_logistic_regression.create () |> get)
          |> get
      in
      let specification =
        Pipeline.Supervised.set_estimator builder terminal |> get
      in
      let folds =
        if classes = 2 then
          Cross_validation.Binary_classification.cross_validate
            ~return_train_score:true ~splitter:(splitter ())
            ~scorers:
              [|
                Binary_classification_scorer.accuracy;
                Binary_classification_scorer.neg_log_loss ();
              |]
            ~seed:(Seed.of_int 42) specification source
          |> get |> Cross_validation.successful_fold_count
        else
          Cross_validation.Multiclass_classification.cross_validate
            ~return_train_score:true ~splitter:(splitter ())
            ~scorers:
              [|
                Multiclass_classification_scorer.accuracy;
                Multiclass_classification_scorer.neg_log_loss;
              |]
            ~seed:(Seed.of_int 42) specification source
          |> get |> Cross_validation.successful_fold_count
      in
      Alcotest.(check int)
        "labels and probabilities receive inference metadata" 4 folds;
      let grid =
        Grid_search.create ~base:() ~build:(fun () -> Ok specification) [||]
        |> get
      in
      let refitted =
        if classes = 2 then
          Grid_search.Binary_classification.search ~grid ~splitter:(splitter ())
            ~scorers:[| Binary_classification_scorer.neg_log_loss () |]
            ~refit:"neg_log_loss" ~seed:(Seed.of_int 42) source
          |> get |> Grid_search.selection |> get
        else
          Grid_search.Multiclass_classification.search ~grid
            ~splitter:(splitter ())
            ~scorers:[| Multiclass_classification_scorer.neg_log_loss |]
            ~refit:"neg_log_loss" ~seed:(Seed.of_int 42) source
          |> get |> Grid_search.selection |> get
      in
      ignore
        (Pipeline.predict_proba_with_metadata
           refitted.Grid_search.selected_model
           ~metadata:(Metadata.of_dataset source)
           ~feature_schema:(Dataset.feature_schema source)
           ~x:(Dataset.features source)
        |> get))
    [ 2; 3 ]

let test_callback_requests () =
  let source = dataset () in
  let calls = ref 0 in
  let callback =
    Callback.create (fun _ ->
        incr calls;
        Ok Callback.Continue)
    |> get
  in
  let metadata = Metadata.of_dataset ~callback source in
  List.iter
    (fun policy ->
      let request = Metadata.Request.create ~callback:policy () in
      match policy with
      | Metadata.Request.Ignore ->
          Alcotest.(check bool)
            "callback ignored" true
            (Metadata.route request metadata
            |> get |> Metadata.callback |> Option.is_none)
      | Metadata.Request.Optional | Metadata.Request.Required ->
          Alcotest.(check bool)
            "callback delivered" true
            (Metadata.route request metadata
            |> get |> Metadata.callback |> Option.is_some)
      | Metadata.Request.Reject -> (
          match Metadata.route request metadata with
          | Ok _ -> Alcotest.fail "callback not rejected"
          | Error _ -> ()))
    [
      Metadata.Request.Ignore;
      Metadata.Request.Optional;
      Metadata.Request.Required;
      Metadata.Request.Reject;
    ];
  Alcotest.(check int) "routing does not invoke callback" 0 !calls;
  Atomic.set fits 0;
  let report =
    run (pipeline ~callbacks:Metadata.Request.Required ()) source |> get
  in
  Alcotest.(check int)
    "required callback absence fails folds" 0
    (Cross_validation.successful_fold_count report);
  Alcotest.(check int) "absence rejected before fitting" 0 (Atomic.get fits)

let[@warning "-4"] test_recorded_failures_and_refit_cancel () =
  let source = dataset () in
  let events = ref [] in
  let callback =
    Callback.create (fun event ->
        events := event :: !events;
        Ok Callback.Continue)
    |> get
  in
  let result =
    run
      ~metadata:(Metadata.of_dataset ~callback source)
      (pipeline ~callbacks:Metadata.Request.Reject ())
      source
    |> get
  in
  Alcotest.(check int)
    "recorded failures remain available" 0
    (Cross_validation.successful_fold_count result);
  Alcotest.(check int)
    "failed fold lifecycle events" 4
    (List.filter
       (fun event ->
         match (event.Callback.operation, event.Callback.status) with
         | Callback.Fold, Callback.Finished (Callback.Failed _) -> true
         | _ -> false)
       !events
    |> List.length);
  (match (List.hd !events).Callback.status with
  | Callback.Finished (Callback.Failed _) -> ()
  | _ -> Alcotest.fail "CV omitted recorded failure status");
  Atomic.set fits 0;
  let callback =
    Callback.create (fun event ->
        if
          event.Callback.operation = Callback.Refit
          && event.Callback.status = Callback.Started
        then Ok Callback.Cancel
        else Ok Callback.Continue)
    |> get
  in
  (match search ~metadata:(Metadata.of_dataset ~callback source) source with
  | Error error ->
      Alcotest.(check bool)
        "refit cancellation aborts Record search" true
        (Error.kind error = Error.Cancelled)
  | Ok _ ->
      Alcotest.fail "refit cancellation was recorded as an ordinary failure");
  Alcotest.(check int) "cancelled refit does no fitting" 8 (Atomic.get fits);
  match
    Callback.create ~max_buffered_events:0 (fun _ -> Ok Callback.Continue)
  with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "non-positive event bound accepted"

let () =
  Alcotest.run "evaluation metadata and callbacks"
    [
      ( "contracts",
        [
          Alcotest.test_case "recorded failure events and refit cancellation"
            `Quick test_recorded_failures_and_refit_cancel;
          Alcotest.test_case "binary and multiclass metadata delivery" `Quick
            test_classification;
          Alcotest.test_case "callback requests and absence" `Quick
            test_callback_requests;
          Alcotest.test_case
            "fold fit, inference, scoring and splitter alignment" `Quick
            test_alignment;
          Alcotest.test_case "held-out metadata leakage" `Quick test_leakage;
          Alcotest.test_case "search full-data refit" `Quick test_refit;
          Alcotest.test_case "callback lifecycle and nested contexts" `Quick
            test_callbacks;
          Alcotest.test_case "cancellation, failures, bounds and exceptions"
            `Quick test_control;
          Alcotest.test_case "preflight and synchronous direct cancellation"
            `Quick test_preflight_and_direct;
        ] );
    ]
