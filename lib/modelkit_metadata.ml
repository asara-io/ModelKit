open Modelkit_data
module Callback = Modelkit_callback.Callback

module Metadata = struct
  type t = {
    sample_weight : Sample_weight.t option;
    groups : Groups.t option;
    callback : Callback.t option;
  }

  let create ?sample_weight ?groups ?callback () =
    { sample_weight; groups; callback }

  let empty = create ()
  let sample_weight metadata = metadata.sample_weight
  let groups metadata = metadata.groups
  let callback metadata = metadata.callback
  let with_callback metadata callback = { metadata with callback }

  let scope context metadata =
    with_callback metadata
      (Option.map (Callback.scope context) metadata.callback)

  let of_dataset ?callback dataset =
    create ?callback
      ?sample_weight:(Dataset.sample_weight dataset)
      ?groups:(Dataset.groups dataset) ()

  let ( let* ) = Result.bind

  let validate ~rows metadata =
    let* () =
      if rows >= 0 then Ok ()
      else
        Error
          (Error.make ~remediation:"provide a non-negative row count"
             (Error.Validation
                { name = "metadata rows"; reason = "negative row count" }))
    in
    let length name observed =
      if observed = rows then Ok ()
      else
        Error
          (Error.of_data_error
             ~remediation:"provide one metadata value per input row"
             (Data_error.Length_mismatch { name; expected = rows; observed }))
    in
    let* () =
      match metadata.sample_weight with
      | None -> Ok ()
      | Some weights ->
          length "metadata sample weights" (Sample_weight.length weights)
    in
    match metadata.groups with
    | None -> Ok ()
    | Some groups -> length "metadata groups" (Groups.length groups)

  module Request = struct
    type policy = Ignore | Optional | Required | Reject
    type t = { sample_weight : policy; groups : policy; callback : policy }

    let create ?(sample_weight = Ignore) ?(groups = Ignore) ?(callback = Ignore)
        () : t =
      { sample_weight; groups; callback }

    let none = create ()
    let sample_weight (request : t) = request.sample_weight
    let groups (request : t) = request.groups
    let callback (request : t) = request.callback

    let check name policy present =
      match (policy, present) with
      | Required, false | Reject, true ->
          let reason =
            if present then "metadata is rejected by this consumer"
            else "required metadata is absent"
          in
          Error
            (Error.make
               ~remediation:
                 "supply metadata matching the consumer's declared request"
               (Error.Validation { name; reason }))
      | Ignore, _ | Optional, _ | Required, true | Reject, false -> Ok ()
  end

  let validate_request request metadata =
    let* () =
      Request.check "sample weights"
        (Request.sample_weight request)
        (Option.is_some metadata.sample_weight)
    in
    let* () =
      Request.check "groups" (Request.groups request)
        (Option.is_some metadata.groups)
    in
    Request.check "callback" (Request.callback request)
      (Option.is_some metadata.callback)

  let route request metadata =
    let* () = validate_request request metadata in
    let select policy value =
      match policy with
      | Request.Ignore | Request.Reject -> None
      | Request.Optional | Request.Required -> value
    in
    Ok
      {
        sample_weight =
          select (Request.sample_weight request) metadata.sample_weight;
        groups = select (Request.groups request) metadata.groups;
        callback = select (Request.callback request) metadata.callback;
      }

  let select metadata rows =
    let* () = validate ~rows:(Row_view.source_size rows) metadata in
    let select value f =
      match value with
      | None -> Ok None
      | Some value ->
          f value rows |> Result.map Option.some
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"select metadata from aligned source rows" error)
    in
    let* sample_weight = select metadata.sample_weight Sample_weight.select in
    let* groups = select metadata.groups Groups.select in
    Ok { sample_weight; groups; callback = metadata.callback }

  let consume ~name ~operation request metadata f =
    let* metadata = route request metadata in
    let callback =
      Option.map
        (fun callback ->
          Callback.scope (Error.Stage name) callback
          |> Callback.for_operation operation)
        metadata.callback
    in
    let metadata = with_callback metadata callback in
    Callback.run callback ~operation (fun () -> f metadata)
end
