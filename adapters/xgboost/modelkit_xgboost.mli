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
  (** Retained float32 buffer payload allocated by this conversion, excluding
      upstream DMatrix/booster storage, OCaml headers, and allocator overhead.
  *)

  val temporary_payload_bytes : report -> int64
  (** Full-size staging payload discarded after conversion. Classification
      target export requires an OCaml-int array. Classifier fitting additionally
      reports row-selection indices in its target report and full float32
      feature/weight staging when zero-weight rows are removed. *)

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
      have bounded native concurrency. The tested single-thread contract sets
      both [OMP_NUM_THREADS=1] and [OMP_THREAD_LIMIT=1] before startup, with
      [Specification.threads=1] for boosters. Backend [inner_threads]
      diagnostics do not configure native execution. No process-global settings
      are changed. *)

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

module Regressor : sig
  type t = Specification.t
  type params = Specification.params
  type fitted

  type report = {
    rounds : int;
    native_version : int * int * int;
    native_seed : int64;
    feature_conversion : Input.report;
    target_conversion : Input.report;
    weight_conversion : Input.report option;
  }

  val capabilities : Modelkit.Capability.estimator

  val report : fitted -> report
  (** Fit diagnostics contain conversion reports and native version/seed, but no
      training observations. Payload accounting excludes native allocations and
      the temporary model buffer used to detach the fitted training cache. *)

  (** Squared-error CPU histogram trees with a fixed training-round count.
      Specifications are constructed with {!Specification.create}. Fitting
      validates row alignment, schema, numeric conversion, and weights before
      native training. Each fit creates independent native state. Native NaNs
      route as missing values; targets and weights must fit the float32 policy.

      Fitted state owns a GC-managed inference booster detached from training
      buffers. Prediction validates exact schema identity and matrix width,
      returns finite float64 values in row order, and accepts zero prediction
      rows. Calls sharing one fitted value serialize native output copying;
      distinct fitted values do not share a lock. Training and prediction do not
      mutate the specification or expose native handles. Numerical results are
      deterministic for fixed data, parameters, RNG, native version, and thread
      configuration; cross-version equivalence is limited to verified fixtures.
      Native allocations are not included in OCaml heap accounting. *)
  include
    Modelkit.REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Modelkit.Rng.t

  val predict_with_report :
    fitted ->
    feature_schema:Modelkit.Feature_schema.t ->
    x:Modelkit.Matrix.t ->
    ( Modelkit.Target.regression Modelkit.Target.t * Input.report,
      Modelkit.Error.t )
    result
  (** Prediction plus its float32 input conversion report. The report excludes
      native prediction storage and the returned float64 prediction vector. *)

  val estimator :
    ?name:string ->
    t ->
    ( ( Modelkit.Target.regression Modelkit.Target.t,
        Modelkit.Target.regression Modelkit.Target.t )
      Modelkit.Pipeline.estimator,
      Modelkit.Error.t )
    result
  (** Packages the public regressor with package/version provenance. The
      estimator receives routed sample weights and can enter ordinary or
      supervised pipelines. No artifact codec is attached yet; portable encoding
      returns a typed unsupported error. *)
end

(** Binary and multiclass CPU histogram-tree classification. *)
module Classifier : sig
  type t = Specification.t
  type params = Specification.params
  type fitted

  type report = {
    rounds : int;
    native_version : int * int * int;
    native_seed : int64;
    class_count : int;
    training_rows : int;
    effective_rows : int;
    feature_conversion : Input.report;
    target_conversion : Input.report;
    weight_conversion : Input.report option;
  }

  val capabilities : Modelkit.Capability.estimator

  val report : fitted -> report
  (** Records original/effective row counts and conversion allocations. Shapes
      describe the original inputs; retained buffers contain positively weighted
      rows only. Full-input float32 staging and row-index arrays are reported
      separately. Mapping tables, class storage, native allocations, and the
      temporary cache-detachment model buffer are excluded. *)

  val classes : fitted -> int array
  (** A fresh copy of the sorted original integer labels. Only classes with
      positive sample weight enter the fitted vocabulary. Labels are mapped to
      contiguous native indices, preserving values beyond float32 precision. *)

  (** Specifications are created with {!Specification.create}. At least two
      positively weighted classes are required; empty, single-class, and
      single-effective-class targets return typed validation errors. Two
      effective classes use binary logistic loss; three or more use multiclass
      softmax cross-entropy. Class count is bounded by exact float32 index
      representation. Zero-weight rows are removed before native training;
      numeric admission still validates their features. Native feature NaNs
      remain missing values.

      Prediction returns original labels in row order. Probability ties select
      the lowest label, including the binary probability-0.5 tie. Schema, native
      lifetime, concurrency, version-dependent reproducibility, and thread
      contracts match {!Regressor}. Empty prediction batches are supported. *)
  include
    Modelkit.CLASSIFIER
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Modelkit.Rng.t

  val predict_proba :
    fitted ->
    feature_schema:Modelkit.Feature_schema.t ->
    x:Modelkit.Matrix.t ->
    (Modelkit.Matrix.t, Modelkit.Error.t) result
  (** Finite float64 probabilities with one column per entry in {!classes}.
      Binary columns are one-minus-positive-probability and positive probability
      for the lower and higher labels. Multiclass output is checked to sum to
      one within 1e-6, then normalized in float64 to remove float32 roundoff.
      Invalid native probabilities are typed numerical failures. *)

  val predict_proba_with_report :
    fitted ->
    feature_schema:Modelkit.Feature_schema.t ->
    x:Modelkit.Matrix.t ->
    (Modelkit.Matrix.t * Input.report, Modelkit.Error.t) result
  (** Includes the float32 feature-conversion report. Native output buffers and
      returned probabilities are excluded from that report. *)

  val estimator :
    ?name:string ->
    t ->
    ( ( Modelkit.Target.classification Modelkit.Target.t,
        Modelkit.Target.classification Modelkit.Target.t )
      Modelkit.Pipeline.estimator,
      Modelkit.Error.t )
    result
  (** Public estimator packaging with class/probability dispatch, sample-weight
      routing, and package/version provenance. Artifact encoding remains
      explicitly unsupported until the reviewed pipeline codec is available. *)
end
