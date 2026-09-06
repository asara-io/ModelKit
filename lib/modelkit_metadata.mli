open Modelkit_data
module Callback = Modelkit_callback.Callback

(** Immutable, typed, row-aligned inputs for metadata-aware consumers. Metadata
    is supplied independently for each operation; fitted pipelines do not retain
    fit metadata for later inference. *)
module Metadata : sig
  type t

  val create :
    ?sample_weight:Sample_weight.t ->
    ?groups:Groups.t ->
    ?callback:Callback.t ->
    unit ->
    t

  val empty : t
  val sample_weight : t -> Sample_weight.t option
  val groups : t -> Groups.t option
  val callback : t -> Callback.t option
  val of_dataset : ?callback:Callback.t -> _ Dataset.t -> t

  val select : t -> Row_view.t -> (t, Error.t) result
  (** Selects weights and groups in exactly the row-view order; the callback is
      shared, not sliced. Source lengths are checked before selection. *)

  val validate : rows:int -> t -> (unit, Error.t) result
  (** Checks every supplied field, including fields ignored by consumers. *)

  module Request : sig
    (** [Ignore] never delivers the field; [Optional] delivers it when supplied;
        [Required] rejects absence; [Reject] rejects presence. Requests are
        independent for each method and each consumer. *)
    type policy = Ignore | Optional | Required | Reject

    type t

    val create :
      ?sample_weight:policy -> ?groups:policy -> ?callback:policy -> unit -> t
    (** All policies default to [Ignore]. *)

    val none : t
    val sample_weight : t -> policy
    val groups : t -> policy
    val callback : t -> policy
  end

  val validate_request : Request.t -> t -> (unit, Error.t) result

  val route : Request.t -> t -> (t, Error.t) result
  (** Checks presence policies and returns only requested fields, sharing the
      immutable values. This does not check row lengths; use [validate] at the
      operation boundary. Ignored fields may still reach requesting siblings. *)

  val with_callback : t -> Callback.t option -> t
  val scope : Error.context -> t -> t

  val consume :
    name:string ->
    operation:Callback.operation ->
    Request.t ->
    t ->
    (t -> ('a, Error.t) result) ->
    ('a, Error.t) result
end
