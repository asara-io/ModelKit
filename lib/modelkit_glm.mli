open Modelkit_data
open Modelkit_protocols

module Tweedie_regression : sig
  type link = Auto | Identity | Log

  type params = {
    power : float;
    alpha : float;
    fit_intercept : bool;
    link : link;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    tweedie_params : params;
    tweedie_coefficients : float array;
    tweedie_intercept : float;
    tweedie_link : link;
    tweedie_schema : Feature_schema.t;
    tweedie_report : Modelkit_linear_models.Solver_report.t;
  }

  val create :
    ?power:float ->
    ?alpha:float ->
    ?fit_intercept:bool ->
    ?link:link ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val resolved_link : fitted -> link
  val report : fitted -> Modelkit_linear_models.Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Modelkit_protocols.Rng.t
end

module Poisson_regression : sig
  type params = {
    alpha : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    poisson_params : params;
    poisson_coefficients : float array;
    poisson_intercept : float;
    poisson_schema : Feature_schema.t;
    poisson_report : Modelkit_linear_models.Solver_report.t;
  }

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
       and type rng = Modelkit_protocols.Rng.t
end
