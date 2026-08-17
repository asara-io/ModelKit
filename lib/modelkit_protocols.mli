open Modelkit_data

module type SPECIFICATION = sig
  type t
  type params

  val clone : t -> t
  val params : t -> params
end

(** Common contract for immutable estimator specifications.

    [t] is a training specification and [fitted] is the value produced by
    fitting it. Implementations receive RNG state explicitly. *)
module type ESTIMATOR = sig
  include SPECIFICATION

  type target
  type prediction
  type fitted
  type rng

  val fit :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:rng ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:target ->
    unit ->
    (fitted, Error.t) result

  val predict :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (prediction, Error.t) result

  val fitted_params : fitted -> params
  val feature_schema : fitted -> Feature_schema.t
end

(** Estimator whose targets and predictions are integer class labels. *)
module type CLASSIFIER = sig
  include
    ESTIMATOR
      with type target = Target.classification Target.t
       and type prediction = Target.classification Target.t
end

(** Estimator whose targets and predictions are float64 regression values. *)
module type REGRESSOR = sig
  include
    ESTIMATOR
      with type target = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

(** Contract for a learned matrix-to-matrix transformation.

    [fit] receives only the training partition. [y] is optional because some
    transformations are supervised while others depend only on features. *)
module type TRANSFORMER = sig
  include SPECIFICATION

  type target
  type fitted
  type rng

  val fit :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:rng ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:target option ->
    unit ->
    (fitted, Error.t) result

  val transform :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val fitted_params : fitted -> params
  val input_schema : fitted -> Feature_schema.t
  val output_schema : fitted -> Feature_schema.t
end

(** A named scoring rule over observed and predicted values. *)
module type SCORER = sig
  include SPECIFICATION

  type truth
  type prediction

  val name : t -> string

  val score :
    t ->
    ?sample_weight:Sample_weight.t ->
    truth:truth ->
    prediction:prediction ->
    unit ->
    (float, Error.t) result
end

(** Contract for deterministic materialization of train/test row selections. *)
module type SPLITTER = sig
  include SPECIFICATION

  type target
  type rng

  val split :
    t ->
    rng:rng ->
    ?groups:Groups.t ->
    x:Matrix.t ->
    y:target option ->
    unit ->
    ((Row_view.t * Row_view.t) array, Error.t) result
end

(** Bounded execution with output positions matching input positions.

    [map] passes the logical input index to each task. When tasks fail, an
    implementation cancels work that is no longer needed and returns the error
    belonging to the lowest failing input index. *)
module type EXECUTION = sig
  type t

  val concurrency : t -> int

  val map :
    t ->
    f:(index:int -> 'input -> ('output, 'error) result) ->
    'input array ->
    ('output array, 'error) result
end

(** Functional random-number generation with schedule-independent child seeds.
*)
module type RNG = sig
  type seed
  type t

  val create : seed -> t

  val derive : seed -> operation:string -> index:int -> seed
  (** [derive seed ~operation ~index] identifies a logical child operation;
      callers must not use worker or completion order as [index]. *)

  val next_int64 : t -> int64 * t

  val next_float : t -> float * t
  (** [next_float state] returns a value in the half-open interval from [0.]
      inclusive to [1.] exclusive, together with the successor state. *)
end

(** Portable numerical primitives shared by reference and accelerated backends.
*)
module type NUMERICAL_BACKEND = sig
  val name : string
  val sum : Vector.t -> float
  val dot : Vector.t -> Vector.t -> (float, Error.t) result
  val matrix_vector_product : Matrix.t -> Vector.t -> (Vector.t, Error.t) result

  val transposed_matrix_vector_product :
    Matrix.t -> Vector.t -> (Vector.t, Error.t) result
end

(** Stable, platform-independent seed values.

    Derivation hashes the bytes of [operation] and the logical [index] with a
    fixed algorithm. It is independent of domain scheduling and OCaml's runtime
    hash implementation, to ensure cross-platform and cross-runtime version
    values are stable and reproducible. *)
module Seed : sig
  type t

  val of_int : int -> t
  val of_int64 : int64 -> t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val derive : t -> operation:string -> index:int -> t
  val pp : Format.formatter -> t -> unit
  val to_string : t -> string
end

(** Pure portable SplitMix64 random-number generation. *)
module Rng : sig
  include RNG with type seed = Seed.t

  val to_seed : t -> Seed.t
  (** [to_seed rng] identifies the current stream state so composite operations
      can derive child streams. *)
end

(** Always-available sequential execution in ascending logical-index order. *)
module Sequential_execution : sig
  type t

  val default : t

  include EXECUTION with type t := t
end

(** A packaged execution backend.

    The portable default is {!Sequential_execution}. Optional packages can
    provide bounded parallel implementations without becoming dependencies of
    [modelkit]. *)
module Execution : sig
  type t

  val of_backend :
    (module EXECUTION with type t = 'configuration) -> 'configuration -> t

  val sequential : t
  val concurrency : t -> int

  val map :
    t ->
    f:(index:int -> 'input -> ('output, 'error) result) ->
    'input array ->
    ('output array, 'error) result
end

(** Native OCaml float64 kernels with stable fixed-order reductions. *)
module Reference_backend : sig
  include NUMERICAL_BACKEND

  module Accumulator : sig
    type t

    val create : unit -> t
    val add : t -> float -> unit
    val value : t -> float
  end
end
