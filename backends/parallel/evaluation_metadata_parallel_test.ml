open Modelkit
module Support = Evaluation_metadata_support
module Callback = Modelkit.Callback

let get = Support.get

let execute domains =
  Modelkit_parallel.create ~inner_threads:1 ~domains ()
  |> get |> Modelkit_parallel.execution

let run execution =
  let source = Support.dataset () in
  let owner = Domain.self () and events = ref [] in
  let callback =
    Callback.create (fun event ->
        Alcotest.(check bool)
          "serialized caller-domain delivery" true
          (Domain.self () = owner);
        events := event :: !events;
        Ok Callback.Continue)
    |> get
  in
  let metadata = Metadata.of_dataset ~callback source in
  let evaluation =
    Support.run ~metadata ~execution (Support.pipeline ()) source |> get
  in
  let summaries =
    Cross_validation.folds evaluation
    |> Array.map (fun fold ->
        ( Option.get fold.Cross_validation.train_indices,
          Option.get fold.Cross_validation.test_indices,
          Support.score fold,
          Support.predictions
            (Metadata.of_dataset source)
            source
            (Option.get fold.Cross_validation.model) ))
  in
  (List.rev !events, summaries)

let test_delivery () =
  let expected_events, expected_summaries = run Execution.sequential in
  List.iter
    (fun domains ->
      let events, summaries = run (execute domains) in
      Alcotest.(check bool)
        "same event trace across domain counts" true (events = expected_events);
      Alcotest.(check bool)
        "same indices, scores and metadata-dependent predictions" true
        (summaries = expected_summaries))
    [ 1; 2; 4 ]

let[@warning "-4"] test_cancellation () =
  List.iter
    (fun domains ->
      let source = Support.dataset () in
      Atomic.set Support.fits 0;
      let fold_events = ref [] in
      let callback =
        Callback.create (fun event ->
            match
              ( event.Callback.operation,
                event.Callback.status,
                event.Callback.context )
            with
            | Callback.Fold, Callback.Started, [ Error.Fold index ] ->
                fold_events := index :: !fold_events;
                Ok Callback.Cancel
            | _ -> Ok Callback.Continue)
        |> get
      in
      (match
         Support.run
           ~metadata:(Metadata.of_dataset ~callback source)
           ~execution:(execute domains) (Support.pipeline ()) source
       with
      | Ok _ -> Alcotest.fail "cancellation ignored"
      | Error error ->
          Alcotest.(check bool)
            "typed cancellation" true
            (Error.kind error = Error.Cancelled));
      Alcotest.(check (list int))
        "no events delivered after cancellation" [ 0 ] !fold_events;
      Alcotest.(check int)
        "only the in-flight batch may finish" domains (Atomic.get Support.fits))
    [ 1; 2; 4 ]

let test_search () =
  let source = Support.dataset () in
  let run execution =
    let events = ref [] and owner = Domain.self () in
    let callback =
      Callback.create (fun event ->
          Alcotest.(check bool)
            "search caller owns callbacks" true
            (Domain.self () = owner);
          events := event :: !events;
          Ok Callback.Continue)
      |> get
    in
    let report =
      Support.search ~execution
        ~metadata:(Metadata.of_dataset ~callback source)
        source
      |> get
    in
    let selection = Grid_search.selection report |> get in
    ( List.rev !events,
      selection.Grid_search.selected_candidate_index,
      Support.predictions
        (Metadata.of_dataset source)
        source selection.Grid_search.selected_model )
  in
  let expected = run Execution.sequential in
  List.iter
    (fun domains ->
      Alcotest.(check bool)
        "search event order and refit are invariant" true
        (run (execute domains) = expected))
    [ 1; 2; 4 ]

let () =
  Alcotest.run "parallel metadata callbacks"
    [
      ( "contracts",
        [
          Alcotest.test_case "deterministic metadata and callback delivery"
            `Quick test_delivery;
          Alcotest.test_case "bounded batch cancellation" `Quick
            test_cancellation;
          Alcotest.test_case "search lifecycle and refit" `Quick test_search;
        ] );
    ]
