open Modelkit_data
open Modelkit_protocols

module Ridge_classifier : sig
  type params = { alpha : float; fit_intercept : bool }
  type t
  type fitted

  val create :
    ?alpha:float -> ?fit_intercept:bool -> unit -> (t, Error.t) result

  val coefficients : fitted -> Matrix.t
  val intercepts : fitted -> Vector.t
  val classes : fitted -> int array
  val reports : fitted -> Modelkit_linear_models.Solver_report.t array

  val decision_function :
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
