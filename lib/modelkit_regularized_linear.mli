open Modelkit_data
open Modelkit_protocols

module Lasso_regression : sig
  type params = {
    alpha : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

  val create :
    ?alpha:float ->
    ?fit_intercept:bool ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val report : fitted -> Modelkit_linear_models.Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

module Elastic_net_regression : sig
  type params = {
    alpha : float;
    l1_ratio : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

  val create :
    ?alpha:float ->
    ?l1_ratio:float ->
    ?fit_intercept:bool ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val report : fitted -> Modelkit_linear_models.Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

module Lasso_path : sig
  type params = {
    fit_intercept : bool;
    epsilon : float;
    count : int;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

  val create :
    ?fit_intercept:bool ->
    ?epsilon:float ->
    ?count:int ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val fit :
    t ->
    ?alphas:Vector.t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:Target.regression Target.t ->
    unit ->
    (fitted, Error.t) result

  val params : t -> params
  val alphas : fitted -> Vector.t
  val coefficients : fitted -> Matrix.t
  val intercepts : fitted -> Vector.t
  val reports : fitted -> Modelkit_linear_models.Solver_report.t array
  val model : fitted -> index:int -> (Lasso_regression.fitted, Error.t) result
end

module Elastic_net_path : sig
  type params = {
    l1_ratio : float;
    fit_intercept : bool;
    epsilon : float;
    count : int;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

  val create :
    ?l1_ratio:float ->
    ?fit_intercept:bool ->
    ?epsilon:float ->
    ?count:int ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val fit :
    t ->
    ?alphas:Vector.t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:Target.regression Target.t ->
    unit ->
    (fitted, Error.t) result

  val params : t -> params
  val alphas : fitted -> Vector.t
  val coefficients : fitted -> Matrix.t
  val intercepts : fitted -> Vector.t
  val reports : fitted -> Modelkit_linear_models.Solver_report.t array

  val model :
    fitted -> index:int -> (Elastic_net_regression.fitted, Error.t) result
end
