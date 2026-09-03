open Modelkit_data
open Modelkit_protocols

module Min_max_scaler : sig
  type params = { feature_range : float * float; clip : bool }
  type t = params
  type fitted

  val create :
    ?feature_range:float * float -> ?clip:bool -> unit -> (t, Error.t) result

  val data_min : fitted -> Vector.t
  val data_max : fitted -> Vector.t
  val data_range : fitted -> Vector.t
  val scale : fitted -> Vector.t
  val offset : fitted -> Vector.t

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Max_abs_scaler : sig
  type params = unit
  type t = params
  type fitted

  val create : unit -> t
  val max_abs : fitted -> Vector.t
  val scale : fitted -> Vector.t

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Robust_scaler : sig
  type params = {
    with_centering : bool;
    with_scaling : bool;
    quantile_range : float * float;
  }

  type t = params
  type fitted

  val create :
    ?with_centering:bool ->
    ?with_scaling:bool ->
    ?quantile_range:float * float ->
    unit ->
    (t, Error.t) result

  val center : fitted -> Vector.t
  val scale : fitted -> Vector.t

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Normalizer : sig
  type norm = L1 | L2 | Max
  type params = { norm : norm }
  type t = params
  type fitted

  val create : ?norm:norm -> unit -> t

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module One_hot_encoder : sig
  type unknown_category = Reject | Ignore

  type params = {
    unknown_category : unknown_category;
    max_output_features : int;
  }

  type t = params
  type fitted

  val create :
    ?unknown_category:unknown_category ->
    ?max_output_features:int ->
    unit ->
    (t, Error.t) result

  val categories : fitted -> Vector.t array

  val transform_csr :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Csr_matrix.t, Error.t) result

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Ordinal_encoder : sig
  type unknown_category = Reject | Use_encoded_value of float
  type params = { unknown_category : unknown_category }
  type t = params
  type fitted

  val create : ?unknown_category:unknown_category -> unit -> (t, Error.t) result
  val categories : fitted -> Vector.t array

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Label_encoder : sig
  type t
  type fitted

  val create : unit -> t
  val fit : t -> y:Target.classification Target.t -> (fitted, Error.t) result

  val transform :
    fitted ->
    Target.classification Target.t ->
    (Target.classification Target.t, Error.t) result

  val inverse_transform :
    fitted ->
    Target.classification Target.t ->
    (Target.classification Target.t, Error.t) result

  val classes : fitted -> int array
end

module Polynomial_features : sig
  type params = {
    degree : int;
    include_bias : bool;
    interaction_only : bool;
    max_output_features : int;
  }

  type t = params
  type fitted

  val create :
    ?degree:int ->
    ?include_bias:bool ->
    ?interaction_only:bool ->
    ?max_output_features:int ->
    unit ->
    (t, Error.t) result

  val terms : fitted -> int array array

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

module Missing_indicator : sig
  type features = Missing_only | All
  type params = { features : features; error_on_new : bool }
  type t = params
  type fitted

  val create : ?features:features -> ?error_on_new:bool -> unit -> t
  val selected_features : fitted -> int array

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end
