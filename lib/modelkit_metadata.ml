open Modelkit_data

module Metadata = struct
  type t = { sample_weight : Sample_weight.t option; groups : Groups.t option }

  let create ?sample_weight ?groups () = { sample_weight; groups }
  let empty = create ()
  let sample_weight metadata = metadata.sample_weight
  let groups metadata = metadata.groups
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
    type t = { sample_weight : policy; groups : policy }

    let create ?(sample_weight = Ignore) ?(groups = Ignore) () : t =
      { sample_weight; groups }

    let none = create ()
    let sample_weight (request : t) = request.sample_weight
    let groups (request : t) = request.groups

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
    Request.check "groups" (Request.groups request)
      (Option.is_some metadata.groups)

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
      }
end
