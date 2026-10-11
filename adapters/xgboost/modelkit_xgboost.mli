(** Optional XGBoost input and CPU histogram-tree specifications.

    This package depends on the OCaml [xgboost] binding; the portable [modelkit]
    package does not. The supported binding is 0.2.1 with native libxgboost 3.x.
    Native input conversion copies float64 values to row-major float32 storage.
    {!Native} provides scoped interoperability with upstream handles. *)

module Specification : sig
  type params = {
    rounds : int;
    max_depth : int;
    max_bin : int;
    learning_rate : float;
    min_child_weight : float;
    min_split_loss : float;
    reg_alpha : float;
    reg_lambda : float;
    subsample : float;
    colsample_bytree : float;
    threads : int;
  }

  type t

  val create :
    ?rounds:int ->
    ?max_depth:int ->
    ?max_bin:int ->
    ?learning_rate:float ->
    ?min_child_weight:float ->
    ?min_split_loss:float ->
    ?reg_alpha:float ->
    ?reg_lambda:float ->
    ?subsample:float ->
    ?colsample_bytree:float ->
    ?threads:int ->
    unit ->
    (t, Modelkit.Error.t) result
  (** Defaults are 100 rounds, depth 6, 256 bins, learning rate 0.3, minimum
      child weight 1, split loss 0, L1 0, L2 1, full row/column sampling, and
      one native booster thread. Integers must fit positive int32 values;
      [max_bin] is at least two. Learning rate is between zero and one
      inclusive; sampling fractions remain positive after float32 rounding and
      are at most one. Other floats are finite, nonnegative, and no greater than
      the largest finite float32 value. *)

  val clone : t -> t
  val params : t -> params
end

module Input : sig
  type report

  val source_shape : report -> int array

  val payload_bytes : report -> int64
  (** Float32 buffer payload allocated by this conversion, excluding upstream
      DMatrix/booster storage, OCaml headers, and allocator overhead. *)

  val temporary_payload_bytes : report -> int64
  (** Full-size staging payload discarded after conversion. Classification
      target export requires an OCaml-int array; other conversions do not. *)

  val rounded_values : report -> int
  (** Finite values changed by float32 rounding, including underflow to zero.
      Feature NaNs are excluded. *)

  type features
  type target
  type weights

  val features :
    feature_schema:Modelkit.Feature_schema.t ->
    Modelkit.Matrix.t ->
    (features, Modelkit.Error.t) result
  (** Validates the supplied schema and nonempty shape, then copies values. NaNs
      remain missing values; infinities and finite values rounding to infinity
      fail with a typed error naming the row and column. Finite underflow is
      permitted and counted. No mutable buffer is exposed. *)

  val regression_target :
    Modelkit.Target.regression Modelkit.Target.t ->
    (target, Modelkit.Error.t) result
  (** Copies finite targets; float32 overflow is rejected. *)

  val encoded_classification_target :
    class_count:int ->
    Modelkit.Target.classification Modelkit.Target.t ->
    (target, Modelkit.Error.t) result
  (** Converts already encoded labels from zero inclusive to [class_count]
      exclusive. This operation does not learn a mapping from original class
      labels. [class_count] is at least two and at most 16,777,217, so every
      encoded label is exactly representable in float32. *)

  val sample_weight :
    Modelkit.Sample_weight.t -> (weights, Modelkit.Error.t) result
  (** Copies weights. Overflow and positive weights rounding to zero are
      rejected, preserving which rows participate in fitting. *)

  val feature_schema : features -> Modelkit.Feature_schema.t
  val feature_report : features -> report
  val target_report : target -> report
  val weight_report : weights -> report
  val feature_value : features -> row:int -> column:int -> float
  val target_value : target -> int -> float

  val weight_value : weights -> int -> float
  (** Read-only access to the converted values. Out-of-range indices raise
      [Invalid_argument]. *)
end

module Native : sig
  val version : unit -> (int * int * int, Modelkit.Error.t) result
  (** Returns the linked native library version; unsupported majors are typed
      compatibility failures. Only major version three is admitted. *)

  val with_dmatrix :
    ?target:Input.target ->
    ?sample_weight:Input.weights ->
    Input.features ->
    (Xgboost.DMatrix.t -> ('a, Modelkit.Error.t) result) ->
    ('a, Modelkit.Error.t) result
  (** Validates target/weight row alignment before native allocation. The
      callback borrows a fresh DMatrix, released on success, typed error, or
      exception. Do not free or retain the borrowed handle beyond the callback.
      Float32 buffers remain alive for its lifetime. Native-library allocations
      are additional to {!Input.payload_bytes} and cannot be measured here.

      Binding errors become typed ModelKit errors; unrelated callback exceptions
      propagate after cleanup. The released binding has no constructor thread
      argument: configure OpenMP before process startup when construction must
      have bounded native concurrency. No process-global settings are changed.
  *)

  val with_booster :
    ?cache:Xgboost.DMatrix.t list ->
    specification:Specification.t ->
    rng:Modelkit.Rng.t ->
    (Xgboost.Booster.t -> ('a, Modelkit.Error.t) result) ->
    ('a, Modelkit.Error.t) result
  (** Borrows a fresh CPU histogram-tree booster configured from the immutable
      specification and explicit RNG. The training objective is set by the
      callback. [rounds] is an OCaml training-loop parameter, not a native
      booster parameter. Booster [nthread] is set explicitly; DMatrix thread
      limits are separate. Cached DMatrices must remain live for the entire
      callback. Seed derivation is independent of worker identity. Handles are
      released under the same callback/error contract as {!with_dmatrix}. *)
end
