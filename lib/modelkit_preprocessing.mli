open Modelkit_data
open Modelkit_protocols

module Preprocessing_internal : sig
  val reject_sample_weight :
    string -> Sample_weight.t option -> (unit, Error.t) result

  val validate_sample_weight :
    string -> Matrix.t -> Sample_weight.t option -> (unit, Error.t) result

  val validate_fit_input :
    operation:string ->
    allow_nan:bool ->
    Feature_schema.t ->
    Matrix.t ->
    (unit, Error.t) result

  val validate_transform_input :
    operation:string ->
    allow_nan:bool ->
    expected_schema:Feature_schema.t ->
    Feature_schema.t ->
    Matrix.t ->
    (unit, Error.t) result

  val validate_values :
    operation:string -> allow_nan:bool -> Matrix.t -> (unit, Error.t) result

  val no_observations : operation:string -> Feature_schema.t -> int -> Error.t

  val numerical_error :
    operation:string -> Feature_schema.t -> int -> string -> Error.t

  val matrix :
    rows:int ->
    columns:int ->
    (int -> int -> float) ->
    (Matrix.t, Error.t) result

  val subset_schema :
    Feature_schema.t -> int array -> (Feature_schema.t, Error.t) result
end

module Simple_imputer : sig
  type strategy = Mean | Median | Constant of float
  type params = { strategy : strategy }
  type t = params

  type fitted = {
    params : params;
    statistics : Vector.t;
    schema : Feature_schema.t;
  }

  val mean : unit -> t
  val median : unit -> t
  val constant : float -> (t, Error.t) result
  val statistics : fitted -> Vector.t

  include
    Modelkit_transform_cache.Transform_cache.CACHEABLE_TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Standard_scaler : sig
  type params = { with_mean : bool; with_std : bool }
  type t = params

  type fitted = {
    params : params;
    mean : Vector.t;
    variance : Vector.t;
    scale : Vector.t;
    schema : Feature_schema.t;
  }

  val create : ?with_mean:bool -> ?with_std:bool -> unit -> t
  val mean : fitted -> Vector.t
  val variance : fitted -> Vector.t
  val scale : fitted -> Vector.t

  include
    Modelkit_transform_cache.Transform_cache.CACHEABLE_TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Variance_threshold : sig
  type params = { threshold : float }
  type t = params

  type fitted = {
    params : params;
    variances : Vector.t;
    selected : int array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  val create : ?threshold:float -> unit -> (t, Error.t) result
  val variances : fitted -> Vector.t
  val selected_indices : fitted -> int array

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end
