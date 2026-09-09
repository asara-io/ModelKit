open Modelkit_data
open Modelkit_metadata
open Modelkit_protocols
open Modelkit_pipeline

(** Regression with a learned, invertible transformation of scalar targets. *)
module Transformed_target_regressor : sig
  (** Implementations preserve target length and row order, fit only on supplied
      training targets, and keep specifications immutable. Transform and inverse
      use fitted state alone. Do not retain training metadata for inference. *)
  module type TRANSFORMER = sig
    include SPECIFICATION

    type fitted

    val fit_request : t -> Metadata.Request.t

    val fit :
      t ->
      metadata:Metadata.t ->
      rng:Rng.t ->
      y:Target.regression Target.t ->
      (fitted, Error.t) result

    val transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result

    val inverse_transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result
  end

  type transformer

  val transformer :
    (module TRANSFORMER with type t = 'specification and type fitted = 'fitted) ->
    'specification ->
    transformer
  (** Captures the fit request once and clones the specification for every fit.
      Weights, groups and callbacks follow the same policies as other consumers.
  *)

  val functions :
    transform:(float -> float) ->
    inverse_transform:(float -> float) ->
    transformer
  (** Pure, deterministic scalar functions, for example [log1p] and [expm1].
      Non-finite results are typed errors; exceptions from user code propagate.
  *)

  val create :
    ?rtol:float ->
    ?atol:float ->
    name:string ->
    transformer:transformer ->
    regressor:
      ( Target.regression Target.t,
        Target.regression Target.t )
      Pipeline.estimator ->
    unit ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result
  (** Packages a terminal regressor for ordinary or supervised pipelines, CV and
      search. Every fit learns the target transformation on that training
      partition, checks [inverse_transform (transform y)] against every training
      target, then fits the regressor on transformed targets. Defaults are
      [rtol=1e-7] and [atol=1e-9]; both must be finite and nonnegative. The
      check uses [abs (restored - y) <= atol + rtol * abs y], evaluated without
      overflowing the tolerance calculation. There is no opt-out.

      Predictions are inverse-transformed before scoring, so scorers always
      receive original-space targets. Target and prediction lengths are checked;
      finite values are guaranteed by {!val:Target.regression}. Row order and
      invertibility away from training targets remain implementer obligations.
      Separate deterministic RNG streams fit the transformer and regressor. Fit
      metadata is independently routed to both consumers; inverse prediction
      requires no metadata. Child errors carry stage context. Target transforms
      have no artifact codec, so saving this wrapper returns a typed error. *)
end
