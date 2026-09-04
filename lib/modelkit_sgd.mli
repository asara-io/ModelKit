open Modelkit_data
open Modelkit_protocols

module Sgd_regressor : sig
  type penalty = No_penalty | L1 | L2 | Elastic_net
  type learning_rate = Constant | Inverse_scaling of { power_t : float }
  type stopping_reason = Epoch_limit | Step_tolerance | Partial_fit

  type report = {
    converged : bool;
    batches_processed : int;
    updates : int;
    objective : float;
    stopping_reason : stopping_reason;
  }

  type params = {
    penalty : penalty;
    alpha : float;
    l1_ratio : float;
    fit_intercept : bool;
    learning_rate : learning_rate;
    eta0 : float;
    max_epochs : int;
    tolerance : float option;
    shuffle : bool;
  }

  type t
  type fitted
  type checkpoint

  val create :
    ?penalty:penalty ->
    ?alpha:float ->
    ?l1_ratio:float ->
    ?fit_intercept:bool ->
    ?learning_rate:learning_rate ->
    ?eta0:float ->
    ?max_epochs:int ->
    ?tolerance:float ->
    ?shuffle:bool ->
    unit ->
    (t, Error.t) result

  val start : t -> rng:Rng.t -> feature_schema:Feature_schema.t -> checkpoint

  val partial_fit :
    checkpoint ->
    ?sample_weight:Sample_weight.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:Target.regression Target.t ->
    unit ->
    (checkpoint, Error.t) result

  val to_fitted : checkpoint -> (fitted, Error.t) result
  val checkpoint : fitted -> checkpoint
  val checkpoint_updates : checkpoint -> int
  val checkpoint_batches_processed : checkpoint -> int
  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val report : fitted -> report

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end
