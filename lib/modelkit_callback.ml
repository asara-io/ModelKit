open Modelkit_data

module Callback = struct
  type operation =
    | Fit
    | Transform
    | Cross_validation
    | Fold
    | Search
    | Candidate
    | Refit

  type outcome = Succeeded | Failed of Error.t

  type status =
    | Started
    | Progress of { completed : int; total : int option }
    | Finished of outcome

  type event = {
    operation : operation;
    context : Error.context list;
    status : status;
  }

  type decision = Continue | Cancel

  type t = {
    deliver : event -> (decision, string) result;
    current_operation : operation;
    current_context : Error.context list;
    max_buffered_events : int;
  }

  let ( let* ) = Result.bind

  let create ?(max_buffered_events = 10_000) deliver =
    if max_buffered_events < 1 then
      Error
        (Error.make ~remediation:"use a positive callback event bound"
           (Error.Validation
              { name = "callback event limit"; reason = "must be positive" }))
    else
      Ok
        {
          deliver;
          current_operation = Fit;
          current_context = [];
          max_buffered_events;
        }

  let scope context callback =
    { callback with current_context = callback.current_context @ [ context ] }

  let for_operation operation callback =
    { callback with current_operation = operation }

  let failure reason =
    Error.make ~remediation:"inspect the callback handler and event bound"
      (Error.Callback_failure { reason })

  let dispatch callback event =
    match callback.deliver event with
    | Ok Continue -> Ok ()
    | Ok Cancel ->
        Error
          (Error.make
             ~remediation:"start a new operation when ready to continue"
             Error.Cancelled)
    | Error reason -> Error (failure reason)

  let emit callback status =
    dispatch callback
      {
        operation = callback.current_operation;
        context = callback.current_context;
        status;
      }

  let progress callback ~completed ?total () =
    if
      completed < 0
      || Option.fold ~none:false ~some:(fun total -> total < completed) total
    then Error (failure "progress requires 0 <= completed <= total")
    else emit callback (Progress { completed; total })

  let[@warning "-4"] is_control_error error =
    match Error.kind error with
    | Error.Cancelled | Error.Callback_failure _ -> true
    | _ -> false

  let run ?(outcome = fun _ -> Succeeded) callback ~operation f =
    match callback with
    | None -> f ()
    | Some callback -> (
        let callback = for_operation operation callback in
        let* () = emit callback Started in
        let result = f () in
        match result with
        | Error error when is_control_error error -> result
        | Ok _ | Error _ ->
            let outcome =
              match result with
              | Ok value -> outcome value
              | Error error -> Failed error
            in
            let* () = emit callback (Finished outcome) in
            result)

  let buffer callback =
    let state = Atomic.make (0, []) in
    let rec deliver event =
      let ((count, reversed) as before) = Atomic.get state in
      if count >= callback.max_buffered_events then
        Error "callback event buffer limit exceeded"
      else if Atomic.compare_and_set state before (count + 1, event :: reversed)
      then Ok Continue
      else deliver event
    in
    let flush () =
      let _, reversed = Atomic.exchange state (0, []) in
      let events = List.rev reversed in
      let rec loop = function
        | [] -> Ok ()
        | event :: remaining ->
            let* () =
              dispatch callback event
              |> Result.map_error (fun error ->
                  List.fold_right Error.with_context event.context error)
            in
            loop remaining
      in
      loop events
    in
    ({ callback with deliver }, flush)
end
