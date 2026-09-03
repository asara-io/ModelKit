open Modelkit_data
open Modelkit_protocols

module Solver_report : sig
  type stopping_reason = Direct_solution | Gradient_tolerance | Step_tolerance

  type t = {
    converged : bool;
    iterations : int;
    objective : float;
    stopping_reason : stopping_reason;
    rank : int option;
  }

  val converged : t -> bool
  val iterations : t -> int
  val objective : t -> float
  val stopping_reason : t -> stopping_reason
  val rank : t -> int option

  val create :
    iterations:int ->
    objective:float ->
    stopping_reason:stopping_reason ->
    rank:int option ->
    t
end

module Linear_model_internal : sig
  val validation : name:string -> reason:string -> remediation:string -> Error.t

  val numerical :
    operation:string -> reason:string -> remediation:string -> Error.t

  val validate_matrix :
    ?require_samples:bool ->
    Feature_schema.t ->
    Matrix.t ->
    (unit, Error.t) result

  val validate_prediction_input :
    schema:Feature_schema.t ->
    Feature_schema.t ->
    Matrix.t ->
    (unit, Error.t) result

  val validate_target_length : Matrix.t -> int -> (unit, Error.t) result

  val validate_sample_weight :
    Matrix.t -> Sample_weight.t option -> (unit, Error.t) result

  val weight : Sample_weight.t option -> int -> float

  val weighted_means :
    Matrix.t -> Vector.t -> Sample_weight.t option -> float array * float

  val regression_prediction :
    operation:string ->
    schema:Feature_schema.t ->
    coefficients:float array ->
    intercept:float ->
    Feature_schema.t ->
    Matrix.t ->
    (Target.regression Target.t, Error.t) result
end

module Linear_regression : sig
  type params = { fit_intercept : bool }
  type t = params

  type fitted = {
    linear_params : params;
    linear_coefficients : float array;
    linear_intercept : float;
    linear_schema : Feature_schema.t;
    linear_report : Solver_report.t;
  }

  val create : ?fit_intercept:bool -> unit -> t
  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

module Ridge_regression : sig
  type params = { alpha : float; fit_intercept : bool }
  type t = params

  type fitted = {
    ridge_params : params;
    ridge_coefficients : float array;
    ridge_intercept : float;
    ridge_schema : Feature_schema.t;
    ridge_report : Solver_report.t;
  }

  val create :
    ?alpha:float -> ?fit_intercept:bool -> unit -> (t, Error.t) result

  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

module Logistic_regression : sig
  type params = {
    c : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    logistic_params : params;
    logistic_coefficients : float array;
    logistic_intercept : float;
    logistic_classes : int * int;
    logistic_schema : Feature_schema.t;
    logistic_report : Solver_report.t;
  }

  val create :
    ?c:float ->
    ?fit_intercept:bool ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val coefficients : fitted -> Vector.t
  val intercept : fitted -> float
  val classes : fitted -> int array
  val report : fitted -> Solver_report.t

  val decision_function :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Vector.t, Error.t) result

  val predict_proba :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  include
    CLASSIFIER
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end
