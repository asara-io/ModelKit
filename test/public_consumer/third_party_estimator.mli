open Modelkit

module Weighted_mean_regressor : sig
  type params = { offset : float }
  type t = params
  type fitted

  val create : ?offset:float -> unit -> (t, Error.t) result

  include
    METADATA_ESTIMATOR
      with type t := t
       and type params := params
       and type target = Target.regression Target.t
       and type prediction = Target.regression Target.t
       and type fitted := fitted
       and type rng = Rng.t
end

val dataset : unit -> (Target.regression Dataset.t, Error.t) result

val pipeline :
  ?nested:bool ->
  Weighted_mean_regressor.t ->
  ( (Target.regression Target.t, Target.regression Target.t) Pipeline.t,
    Error.t )
  result

val conformance :
  Weighted_mean_regressor.t -> (Conformance.report, Error.t) result
