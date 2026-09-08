(** Portable classical machine learning workflows for OCaml.

    This module is the complete supported public namespace. Physical
    implementation units are private and are not compatibility targets. *)

(** Typed failures raised while admitting and aligning data. *)
module Data_error : sig
  type t =
    | Negative_dimension of { name : string; value : int }
    | Ragged_matrix of {
        row : int;
        expected_columns : int;
        observed_columns : int;
      }
    | Length_mismatch of { name : string; expected : int; observed : int }
    | Index_out_of_bounds of { name : string; index : int; upper_bound : int }
    | Non_finite of { name : string; index : int; value : float }
    | Negative_weight of { index : int; value : float }
    | All_zero_weights
    | Empty_feature_name of { index : int }
    | Duplicate_feature_name of {
        name : string;
        first_index : int;
        duplicate_index : int;
      }
    | Csr_row_offset_mismatch of {
        position : int;
        expected : int;
        observed : int;
      }
    | Invalid_csr_row_offset of {
        position : int;
        previous : int;
        observed : int;
        nonzero_count : int;
      }
    | Invalid_csr_column_order of { row : int; previous : int; observed : int }

  val pp : Format.formatter -> t -> unit
  val to_string : t -> string
end

(** Immutable one-dimensional float64 data.

    Values use C-layout Bigarray storage. Admission from and export to a
    Bigarray copy the buffer so mutation outside ModelKit cannot change an
    admitted value. *)
module Vector : sig
  type t

  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  val create : length:int -> float -> (t, Data_error.t) result
  val init : length:int -> (int -> float) -> (t, Data_error.t) result
  val of_array : float array -> t
  val of_bigarray : bigarray -> t
  val length : t -> int

  val get : t -> int -> float
  (** [get vector index] returns the value at [index].

      Raises [Invalid_argument] if [index] is outside the vector. *)

  val to_array : t -> float array
  val to_bigarray : t -> bigarray
end

(** Immutable two-dimensional float64 data in row-major C layout. *)
module Matrix : sig
  type t

  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array2.t

  val create : rows:int -> columns:int -> float -> (t, Data_error.t) result

  val init :
    rows:int -> columns:int -> (int -> int -> float) -> (t, Data_error.t) result

  val of_arrays : float array array -> (t, Data_error.t) result
  val of_bigarray : bigarray -> t
  val rows : t -> int
  val columns : t -> int
  val shape : t -> int * int

  val get : t -> int -> int -> float
  (** [get matrix row column] returns one matrix element.

      Raises [Invalid_argument] if either index is outside the matrix. *)

  val row : t -> int -> Vector.t
  (** [row matrix index] returns a zero-copy immutable view of one row.

      Raises [Invalid_argument] if [index] is outside the matrix. *)

  val to_arrays : t -> float array array
  val to_bigarray : t -> bigarray
end

(** Immutable rank-two missing-value identity aligned to a feature matrix.

    [true] identifies a source null. The mask is kept separately from feature
    values so adapters can preserve the distinction between an explicit null and
    a genuine IEEE NaN. *)
module Null_mask : sig
  type t

  val init :
    rows:int -> columns:int -> (int -> int -> bool) -> (t, Data_error.t) result

  val of_arrays : bool array array -> (t, Data_error.t) result
  val rows : t -> int
  val columns : t -> int
  val shape : t -> int * int
  val get : t -> int -> int -> bool
  val null_count : t -> int
  val to_arrays : t -> bool array array
end

(** An immutable ordered selection of rows from an aligned source.

    Construction copies and validates the indices. Order and duplicates are
    preserved to support deterministic resampling. *)
module Row_view : sig
  type t

  val create : source_size:int -> int array -> (t, Data_error.t) result
  val all : source_size:int -> (t, Data_error.t) result
  val source_size : t -> int
  val length : t -> int

  val get : t -> int -> int
  (** [get view position] returns the source row at [position].

      Raises [Invalid_argument] if [position] is outside the view. *)

  val indices : t -> int array
end

(** Payload memory used by dense or CSR matrix storage.

    Counts exclude OCaml and Bigarray headers and allocator overhead. The dense
    equivalent is [None] only when its byte count exceeds [Int64.max_int]. *)
module Matrix_memory : sig
  type t = {
    value_bytes : int64;
    column_index_bytes : int64;
    row_offset_bytes : int64;
    total_bytes : int64;
    dense_equivalent_bytes : int64 option;
  }
end

(** Immutable checked compressed sparse row storage.

    Row offsets must begin at zero, end at the stored-value count, and be
    nondecreasing. Column indices must be in bounds and strictly increasing
    within each row. Admission copies index arrays; [of_arrays] also copies
    values. Explicit stored zeroes are retained. *)
module Csr_matrix : sig
  type t
  type view

  type view_memory = {
    allocated_bytes : int64;
    shared_bytes : int64;
    materialized_bytes : int64;
  }

  val create :
    rows:int ->
    columns:int ->
    row_offsets:int array ->
    column_indices:int array ->
    values:Vector.t ->
    (t, Data_error.t) result

  val of_arrays :
    rows:int ->
    columns:int ->
    row_offsets:int array ->
    column_indices:int array ->
    values:float array ->
    (t, Data_error.t) result

  val of_dense : Matrix.t -> t
  val to_dense : t -> Matrix.t
  val rows : t -> int
  val columns : t -> int
  val shape : t -> int * int
  val nonzero_count : t -> int
  val row_offsets : t -> int array
  val column_indices : t -> int array
  val values : t -> Vector.t

  val get : t -> int -> int -> float
  (** Raises [Invalid_argument] if either index is outside the matrix. *)

  val iter_row : t -> row:int -> f:(column:int -> value:float -> unit) -> unit
  (** Iterates stored entries in ascending column order without allocating.
      Raises [Invalid_argument] if [row] is outside the matrix. *)

  val memory : t -> Matrix_memory.t
  val all : t -> view
  val view : t -> Row_view.t -> (view, Data_error.t) result
  val view_rows : view -> int
  val view_columns : view -> int
  val view_nonzero_count : view -> int
  val row_view : view -> Row_view.t
  val source_row : view -> int -> int
  val view_get : view -> row:int -> column:int -> float
  val materialize : view -> t

  val view_memory : view -> view_memory
  (** Reports bytes allocated for row indices, bytes shared with the source, and
      the payload bytes that explicit materialization would require. *)
end

(** A dense or CSR feature matrix selected explicitly at an API boundary. *)
module Feature_matrix : sig
  type format = Dense | Csr
  type t = Dense_matrix of Matrix.t | Csr_matrix of Csr_matrix.t

  val dense : Matrix.t -> t
  val csr : Csr_matrix.t -> t
  val format : t -> format
  val rows : t -> int
  val columns : t -> int
  val shape : t -> int * int
  val get : t -> int -> int -> float
  val memory : t -> Matrix_memory.t
end

(** Regression or classification targets aligned by sample.

    The phantom type distinguishes target kinds so estimators can reject the
    wrong target kind at compile time. *)
module Target : sig
  type regression
  type classification
  type _ t

  val regression : Vector.t -> (regression t, Data_error.t) result
  (** Regression targets must contain only finite values. *)

  val classification : int array -> classification t
  (** Classification labels are copied on admission. *)

  val length : _ t -> int
  val regression_values : regression t -> Vector.t
  val classification_values : classification t -> int array
  val select : 'kind t -> Row_view.t -> ('kind t, Data_error.t) result
end

(** A non-empty feature name. No normalization is performed. *)
module Feature_name : sig
  type t

  val create : string -> (t, Data_error.t) result
  val to_string : t -> string
end

(** Ordered, unique feature names aligned to a matrix width. *)
module Feature_names : sig
  type t

  val create : expected_count:int -> string array -> (t, Data_error.t) result
  val length : t -> int

  val get : t -> int -> Feature_name.t
  (** Raises [Invalid_argument] if the index is outside the collection. *)

  val to_array : t -> string array
end

(** Finite, non-negative sample weights aligned to samples.

    At least one weight must be strictly positive. *)
module Sample_weight : sig
  type t

  val create : expected_length:int -> Vector.t -> (t, Data_error.t) result
  val of_array : expected_length:int -> float array -> (t, Data_error.t) result
  val length : t -> int
  val get : t -> int -> float
  val to_vector : t -> Vector.t
  val select : t -> Row_view.t -> (t, Data_error.t) result
end

(** Integer group labels aligned to samples. *)
module Groups : sig
  type t

  val create : expected_length:int -> int array -> (t, Data_error.t) result
  val length : t -> int

  val get : t -> int -> int
  (** Raises [Invalid_argument] if the index is outside the collection. *)

  val to_array : t -> int array
  val distinct_count : t -> int
  val select : t -> Row_view.t -> (t, Data_error.t) result
end

(** Stable, versioned identity for a feature schema.

    Fingerprints are deterministic across supported platforms and OCaml
    versions, and used to identify compatibility. *)
module Schema_fingerprint : sig
  type t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
  val to_string : t -> string
end

(** Immutable feature identity and width expected by a model boundary.

    Anonymous schemas validate width only. Named schemas additionally make
    feature presence and order compatibility requirements. *)
module Feature_schema : sig
  type t

  val anonymous : feature_count:int -> (t, Data_error.t) result
  val named : Feature_names.t -> t
  val of_matrix : ?names:Feature_names.t -> Matrix.t -> (t, Data_error.t) result
  val feature_count : t -> int
  val names : t -> Feature_names.t option
  val equal : t -> t -> bool
  val fingerprint : t -> Schema_fingerprint.t
  val validate_matrix : t -> Matrix.t -> (unit, Data_error.t) result
  val pp : Format.formatter -> t -> unit
  val to_string : t -> string
end

(** An admitted immutable dense dataset and its zero-copy row selections.

    Admission validates alignment and the declared feature finiteness policy.
    [create] reuses already-immutable ModelKit values. [view] stores only row
    indices; [materialize] packs a view into new row-aligned buffers. *)
module Dataset : sig
  type finiteness = Require_finite | Allow_nan
  type data_access = Copy | View

  type access_report = {
    feature_access : data_access;
    target_access : data_access;
    sample_weight_access : data_access option;
    group_access : data_access option;
  }

  type 'kind t
  type 'kind view

  val create :
    finiteness:finiteness ->
    ?feature_names:Feature_names.t ->
    ?sample_weight:Sample_weight.t ->
    ?groups:Groups.t ->
    x:Matrix.t ->
    y:'kind Target.t ->
    unit ->
    ('kind t, Data_error.t) result

  val sample_count : _ t -> int
  val feature_count : _ t -> int
  val features : _ t -> Matrix.t
  val target : 'kind t -> 'kind Target.t
  val sample_weight : _ t -> Sample_weight.t option
  val groups : _ t -> Groups.t option
  val feature_schema : _ t -> Feature_schema.t
  val schema_fingerprint : _ t -> Schema_fingerprint.t
  val finiteness : _ t -> finiteness
  val access_report : _ t -> access_report
  val all : 'kind t -> 'kind view
  val view : 'kind t -> Row_view.t -> ('kind view, Data_error.t) result
  val view_sample_count : _ view -> int
  val row_view : _ view -> Row_view.t
  val view_access_report : _ view -> access_report

  val source_row : _ view -> int -> int
  (** Raises [Invalid_argument] if the view position is out of bounds. *)

  val feature : _ view -> row:int -> column:int -> float
  (** [feature view ~row ~column] addresses logical rows in the view. Raises
      [Invalid_argument] if either index is out of bounds. *)

  val regression_target : Target.regression view -> int -> float
  val classification_target : Target.classification view -> int -> int
  val sample_weight_value : _ view -> int -> float option
  val group : _ view -> int -> int option

  val materialize : 'kind view -> ('kind t, Data_error.t) result
  (** [materialize view] copies selected row-aligned buffers. Feature names and
      the immutable schema are reused. *)
end

(** Structured failures for public machine learning operations.

    Routine data, validation, numerical, convergence, compatibility, and
    artifact failures use this type. Exceptions are reserved for programmer
    defects such as violating a documented bounds precondition. *)
module Error : sig
  type context =
    | Stage of string
    | Fold of int
    | Candidate of int
    | Feature of Feature_name.t

  type kind =
    | Data of Data_error.t
    | Shape_mismatch of {
        name : string;
        expected : int list;
        observed : int list;
      }
    | Feature_schema_mismatch of {
        expected : Feature_schema.t;
        observed : Feature_schema.t;
      }
    | Validation of { name : string; reason : string }
    | Numerical of { operation : string; reason : string }
    | Convergence of { algorithm : string; reason : string }
    | Compatibility of { component : string; reason : string }
    | Artifact of { operation : string; reason : string }
    | Callback_failure of { reason : string }
    | Cancelled

  type t

  val make : ?context:context list -> remediation:string -> kind -> t

  val of_data_error :
    ?context:context list -> remediation:string -> Data_error.t -> t

  val kind : t -> kind
  val context : t -> context list
  val remediation : t -> string

  val with_context : context -> t -> t
  (** [with_context outer error] records [outer] before existing context. *)

  val pp : Format.formatter -> t -> unit
  val to_string : t -> string
end

(** Numeric payload allocation performed by one adapter conversion.

    Byte counts exclude OCaml headers, allocator metadata, names, and the
    adapter result record. [temporary_payload_bytes] counts full-size staging
    payloads discarded after admission; [retained_payload_bytes] counts the
    immutable payload retained by ModelKit. *)
module Conversion_report : sig
  type t

  val create :
    source:string ->
    source_dtype:string ->
    source_shape:int array ->
    source_contiguous:bool option ->
    temporary_payload_bytes:int64 ->
    retained_payload_bytes:int64 ->
    (t, Error.t) result

  val source : t -> string
  val source_dtype : t -> string
  val source_shape : t -> int array
  val source_contiguous : t -> bool option
  val temporary_payload_bytes : t -> int64
  val retained_payload_bytes : t -> int64
  val allocated_payload_bytes : t -> int64
end

(** Adapter-neutral admission results.

    Every ModelKit adapter returns these records so that conformance tests,
    allocation benchmarks, and application code can treat admitted data
    uniformly regardless of the source library. Each [conversion] pairs an
    immutable ModelKit value with the {!Conversion_report.t} describing the
    payload allocated to produce it. [features] carries the admitted matrix, its
    schema, an explicit null mask when the source supplied one, and the reports
    for the matrix and mask. [dataset] carries a complete {!Dataset.t} together
    with the feature null mask and every report produced while admitting
    features, target, weights, and groups. *)
module Admission : sig
  type 'a conversion = { value : 'a; report : Conversion_report.t }

  type features = {
    matrix : Matrix.t;
    schema : Feature_schema.t;
    null_mask : Null_mask.t option;
    feature_reports : Conversion_report.t list;
  }

  type 'kind dataset = {
    dataset : 'kind Dataset.t;
    feature_null_mask : Null_mask.t option;
    dataset_reports : Conversion_report.t list;
  }

  val retained_payload_bytes : Conversion_report.t list -> int64
  (** Total retained payload across a list of reports. *)

  val temporary_payload_bytes : Conversion_report.t list -> int64
  (** Total discarded staging payload across a list of reports. *)

  val allocated_payload_bytes : Conversion_report.t list -> int64
  (** Total allocated payload (retained plus temporary) across a list of
      reports. *)
end

(** Typed progress and lifecycle notifications. Handlers may continue, cancel,
    or return an explanatory error. Exceptions raised by a handler propagate. *)
module Callback : sig
  (** Lifecycle events enclose evaluation, candidates, folds, refit, and
      requesting consumers' fit/transform methods. [Finished (Failed error)]
      reports ordinary failures, including recorded fold failures. Cancellation,
      callback failure, or an exception may leave a started operation
      unfinished. A handler error becomes [Error.Callback_failure]; [Cancel]
      becomes [Error.Cancelled]. Both abort evaluation even under [Record].
      Handler failures take precedence over an operation failure delivered to
      them. Events after the first handler failure or cancellation are
      discarded.

      CV admits at most [Execution.concurrency] folds per batch. Their events
      are delivered after that batch finishes, in fold order and then emission
      order within each fold. Cancellation prevents subsequent batches,
      candidates, or refit; already-running work may finish. Events are bounded
      per fold, and overflow aborts with [Error.Callback_failure]. Timing is not
      part of an event. Custom consumers with concurrent emitters determine
      their own emission order. No callbacks run while handling an exception. *)
  type operation =
    | Fit
    | Transform
    | Cross_validation
    | Fold
    | Search
    | Candidate
    | Refit

  type outcome = Succeeded | Failed of Error.t

  type status =
    | Started
    | Progress of { completed : int; total : int option }
    | Finished of outcome

  type event = {
    operation : operation;
    context : Error.context list;
    status : status;
  }

  type decision = Continue | Cancel
  type t

  val create :
    ?max_buffered_events:int ->
    (event -> (decision, string) result) ->
    (t, Error.t) result
  (** The positive event bound defaults to [10_000] per buffered fold. Direct
      pipeline calls deliver synchronously; CV buffers fold events and delivers
      them serially on the caller domain in logical fold order, one bounded
      batch at a time. Callbacks need not synchronize their own mutable state
      within one evaluation. Sharing a handler across independent concurrent
      evaluations requires caller synchronization. *)

  val progress :
    t -> completed:int -> ?total:int -> unit -> (unit, Error.t) result
  (** Consumers report progress through the callback delivered in metadata.
      Counts must satisfy [0 <= completed <= total] when a total is given. The
      library supplies the enclosing operation and nested context. Consumers
      must propagate errors from this call and must not retain the delivered
      callback after their operation returns. *)

  val is_control_error : Error.t -> bool
  (** Recognizes [Error.Cancelled] and [Error.Callback_failure]. Evaluators
      always abort on these errors, including under [Record] failure policy. *)
end

(** Immutable, typed, row-aligned inputs for metadata-aware consumers. Metadata
    is supplied independently for each operation; fitted pipelines do not retain
    fit metadata for later inference. *)
module Metadata : sig
  type t

  val create :
    ?sample_weight:Sample_weight.t ->
    ?groups:Groups.t ->
    ?callback:Callback.t ->
    unit ->
    t

  val empty : t
  val sample_weight : t -> Sample_weight.t option
  val groups : t -> Groups.t option
  val callback : t -> Callback.t option
  val of_dataset : ?callback:Callback.t -> _ Dataset.t -> t

  val select : t -> Row_view.t -> (t, Error.t) result
  (** Selects weights and groups in exactly the row-view order; the callback is
      shared, not sliced. Source lengths are checked before selection. *)

  val validate : rows:int -> t -> (unit, Error.t) result
  (** Checks every supplied field, including fields ignored by consumers. *)

  module Request : sig
    (** [Ignore] never delivers the field; [Optional] delivers it when supplied;
        [Required] rejects absence; [Reject] rejects presence. Requests are
        independent for each method and each consumer. *)
    type policy = Ignore | Optional | Required | Reject

    type t

    val create :
      ?sample_weight:policy -> ?groups:policy -> ?callback:policy -> unit -> t
    (** All policies default to [Ignore]. *)

    val none : t
    val sample_weight : t -> policy
    val groups : t -> policy
    val callback : t -> policy
  end

  val validate_request : Request.t -> t -> (unit, Error.t) result

  val route : Request.t -> t -> (t, Error.t) result
  (** Checks presence policies and returns only requested fields, sharing the
      immutable values. This does not check row lengths; use [validate] at the
      operation boundary. Ignored fields may still reach requesting siblings. *)
end

(** Shared convention for immutable configured components.

    Concrete modules expose [params] as a public typed value. [clone] returns an
    equivalent unfitted specification and may return the same value because
    specifications contain no mutable fitted state. *)
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

(** Transformer with explicit per-method metadata requests. Requests are read
    from the specification when packaged and remain fixed for its fitted
    lifetime. Both fit and transform requests are validated before training
    begins, since fitting also transforms the training rows. The implementation
    must preserve row count and order and must not retain metadata solely to
    substitute it for future inference inputs. *)
module type METADATA_TRANSFORMER = sig
  include SPECIFICATION

  type target
  type fitted
  type rng

  val fit_request : t -> Metadata.Request.t
  val transform_request : t -> Metadata.Request.t

  val fit :
    t ->
    metadata:Metadata.t ->
    rng:rng ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:target option ->
    unit ->
    (fitted, Error.t) result

  val transform :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val fitted_params : fitted -> params
  val input_schema : fitted -> Feature_schema.t
  val output_schema : fitted -> Feature_schema.t
end

(** Estimator with a declared fit-metadata request. Prediction uses fitted state
    and features; pipeline preprocessing may separately request inference
    metadata. *)
module type METADATA_ESTIMATOR = sig
  include SPECIFICATION

  type target
  type prediction
  type fitted
  type rng

  val fit_request : t -> Metadata.Request.t

  val fit :
    t ->
    metadata:Metadata.t ->
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

(** Published descriptions of optional behavior implemented by a component.

    Required protocol behavior is not a capability: determinism, immutable
    specifications, typed failures, row alignment, and schema validation remain
    mandatory. These values describe only behavior that a generic consumer may
    need to select before invoking a component. *)
module Capability : sig
  type support = Supported | Unsupported

  type prediction =
    | Direct
    | Labels
    | Positive_probabilities of int
    | Class_probabilities

  type estimator = {
    estimator_sample_weight : support;
    estimator_fit_metadata : support;
    estimator_decision_function : support;
    estimator_predict_proba : support;
  }

  type transformer = {
    transformer_target : support;
    transformer_sample_weight : support;
    transformer_fit_metadata : support;
    transformer_transform_metadata : support;
  }

  type scorer = {
    scorer_sample_weight : support;
    scorer_prediction : prediction;
  }

  val estimator :
    ?sample_weight:support ->
    ?fit_metadata:support ->
    ?decision_function:support ->
    ?predict_proba:support ->
    unit ->
    estimator

  val transformer :
    ?target:support ->
    ?sample_weight:support ->
    ?fit_metadata:support ->
    ?transform_metadata:support ->
    unit ->
    transformer

  val scorer : ?sample_weight:support -> prediction:prediction -> unit -> scorer
end

(** A type-safe, first-class scorer supplied by application or third-party code.
    [of_module] snapshots the immutable specification with [clone]. Names and
    capability compatibility are validated by consuming evaluation APIs before
    fitting begins. *)
module Scorer : sig
  type ('truth, 'prediction) t

  val of_module :
    capabilities:Capability.scorer ->
    (module SCORER
       with type t = 'specification
        and type params = 'params
        and type truth = 'truth
        and type prediction = 'prediction) ->
    'specification ->
    ('truth, 'prediction) t

  val name : ('truth, 'prediction) t -> string
  val capabilities : ('truth, 'prediction) t -> Capability.scorer

  val score :
    ('truth, 'prediction) t ->
    ?sample_weight:Sample_weight.t ->
    truth:'truth ->
    prediction:'prediction ->
    unit ->
    (float, Error.t) result
end

(** Framework-neutral protocol checks for third-party components.

    Reports are ordinary values so package authors can use them from Alcotest,
    OUnit, expect tests, or their own build tooling without adding a ModelKit
    test-framework dependency. *)
module Conformance : sig
  type issue =
    | Protocol_error of Error.t
    | Violation of string
    | Raised of string

  type outcome = Passed | Failed of issue
  type check = { name : string; outcome : outcome }
  type report

  val issue_to_string : issue -> string
  val checks : report -> check array
  val passed : report -> bool
  val failures : report -> check array

  module Estimator : sig
    type ('specification, 'params, 'target, 'prediction, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
      prediction_length : 'prediction -> int;
      equal_prediction : 'prediction -> 'prediction -> bool;
    }

    val check :
      (module ESTIMATOR
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type prediction = 'prediction
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'params, 'target, 'prediction, 'fitted, 'rng) fixture ->
      report
  end

  module Transformer : sig
    type ('specification, 'params, 'target, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target option;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    val check :
      (module TRANSFORMER
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'params, 'target, 'fitted, 'rng) fixture ->
      report
  end

  module Scorer : sig
    type ('specification, 'params, 'truth, 'prediction) fixture = {
      specification : 'specification;
      capabilities : Capability.scorer;
      truth : 'truth;
      prediction : 'prediction;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    val check :
      (module SCORER
         with type t = 'specification
          and type params = 'params
          and type truth = 'truth
          and type prediction = 'prediction) ->
      ('specification, 'params, 'truth, 'prediction) fixture ->
      report
  end
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

  val feature_matrix_vector_product :
    Feature_matrix.t -> Vector.t -> (Vector.t, Error.t) result
  (** Dispatches to dense or CSR storage without densifying sparse input. CSR
      kernels visit stored entries only; dense and CSR results agree for finite
      operands representing the same matrix. *)

  val transposed_feature_matrix_vector_product :
    Feature_matrix.t -> Vector.t -> (Vector.t, Error.t) result
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

(** Content-addressed transform-cache foundations.

    Stores are explicit values with caller-owned lifetimes; ModelKit does not
    install a process-global cache. Cache payloads are data only and are copied
    at the storage boundary. *)
module Transform_cache : sig
  module Component : sig
    type t

    val create :
      package:string -> name:string -> version:int -> (t, Error.t) result
    (** Names must be nonblank and [version] must be positive. Package-qualified
        identities prevent unrelated extensions from sharing entries. *)

    val package : t -> string
    val name : t -> string
    val version : t -> int
    val equal : t -> t -> bool
    val to_string : t -> string
  end

  module Content_id : sig
    type t

    val of_bytes : bytes -> t
    val of_string : string -> t
    val of_matrix : Matrix.t -> t
    val of_sample_weight : Sample_weight.t -> t
    val of_groups : Groups.t -> t

    val combine : domain:string -> t array -> (t, Error.t) result
    (** Combines an ordered array under a nonblank domain. Domain and length
        framing distinguish structurally different material. Content IDs are
        deterministic cache identities, not cryptographic authentication. *)

    val equal : t -> t -> bool
    val to_hex : t -> string
  end

  module Key : sig
    type t

    val create :
      component:Component.t ->
      configuration:Content_id.t ->
      training_data:Content_id.t ->
      target:Content_id.t option ->
      routed_metadata:Content_id.t ->
      seed:Seed.t ->
      t
    (** Canonically frames every field. [None] is an explicit no-target marker,
        not the identity of an empty target. *)

    val equal : t -> t -> bool
    val to_hex : t -> string
  end

  (** A stable fitted-state codec for a transformer. The cache format is
      independent of the public model-artifact schema, and decoders must
      validate payloads with typed failures. *)
  module type CACHEABLE_TRANSFORMER = sig
    include TRANSFORMER

    val cache_component : Component.t
    val cache_configuration : t -> Content_id.t
    val encode_fitted : fitted -> (bytes, Error.t) result
    val decode_fitted : bytes -> (fitted, Error.t) result
  end

  module Codec : sig
    type ('specification, 'fitted) t

    type ('specification, 'fitted) support =
      | Unsupported
      | Supported of ('specification, 'fitted) t

    val of_module :
      (module CACHEABLE_TRANSFORMER
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'fitted) t

    val component : ('specification, 'fitted) t -> Component.t

    val configuration :
      ('specification, 'fitted) t -> 'specification -> Content_id.t

    val encode :
      ('specification, 'fitted) t -> 'fitted -> (bytes, Error.t) result

    val decode :
      ('specification, 'fitted) t -> bytes -> ('fitted, Error.t) result

    val require :
      component:string ->
      ('specification, 'fitted) support ->
      (('specification, 'fitted) t, Error.t) result
    (** Returns a typed compatibility error when caching is requested for a
        transformer without a codec. *)
  end

  module Memory : sig
    type limits

    val limits : max_entries:int -> max_bytes:int64 -> (limits, Error.t) result
    (** Both limits must be positive. [max_bytes] counts payload bytes,
        excluding keys and OCaml allocation headers. *)

    val default_limits : limits

    type stats = {
      entries : int;
      payload_bytes : int64;
      hits : int64;
      misses : int64;
      evictions : int64;
    }

    type t

    val create : ?limits:limits -> unit -> t

    val get : t -> Key.t -> bytes option
    (** Returns a copy and updates hit or miss counters. *)

    val put : t -> Key.t -> bytes -> (unit, Error.t) result
    (** Copies the payload. Oversized entries fail without changing the store;
        least-recently-written entries are evicted until both bounds hold. *)

    val remove : t -> Key.t -> bool
    val clear : t -> unit
    val stats : t -> stats
  end

  (** Portable directory-backed storage for immutable cache entries.

      Entries and temporary publication files contain plaintext fitted state.
      ModelKit requests restrictive permissions for newly created paths but does
      not provide encryption, authenticate content, or override the host
      filesystem's permission semantics. Protect the root, backups, and
      retention policy before caching state derived from secret training data.
  *)
  module Persistent : sig
    type limits

    val limits : max_payload_bytes:int -> (limits, Error.t) result
    (** The positive limit bounds allocation before reading a payload. *)

    val default_limits : limits

    (** Corrupt entries are never returned as hits. A subsequent [put] replaces
        a corrupt entry, allowing callers to refit safely. *)
    type lookup = Miss | Hit of bytes | Corrupt of Error.t

    type publication = Published | Already_present
    type t

    val create : ?limits:limits -> root:string -> unit -> (t, Error.t) result
    (** Creates [root] with restrictive requested permissions when absent. Its
        parent must already exist; existing roots must be directories. *)

    val root : t -> string
    val get : t -> Key.t -> (lookup, Error.t) result

    val put : t -> Key.t -> bytes -> (publication, Error.t) result
    (** Publishes a complete entry by atomic rename. Concurrent writers for one
        key must encode identical payloads. *)

    val remove : t -> Key.t -> (bool, Error.t) result
  end

  (** A cache backend packaged behind one workflow-facing interface. *)
  module Store : sig
    type lookup = Miss | Hit of bytes | Corrupt of Error.t
    type t

    val memory : Memory.t -> t
    val persistent : Persistent.t -> t
    val get : t -> Key.t -> (lookup, Error.t) result
    val put : t -> Key.t -> bytes -> (unit, Error.t) result
    val remove : t -> Key.t -> (bool, Error.t) result
  end
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
end

module Reference_backend : NUMERICAL_BACKEND
(** Native OCaml float64 kernels with stable fixed-order reductions. *)

(** Column-wise replacement of NaN missing values.

    Mean and median fitting fail when a training feature contains no observed
    value. Constant values must be finite. Transform rejects infinities and
    preserves the input feature schema. Sample weights are rejected. Mean
    fitting is [O(rows * columns)]; median fitting is
    [O(columns * rows * log rows)] with one temporary column allocation. *)
module Simple_imputer : sig
  type strategy = Mean | Median | Constant of float
  type params = { strategy : strategy }
  type t
  type fitted

  val mean : unit -> t
  val median : unit -> t
  val constant : float -> (t, Error.t) result
  val statistics : fitted -> Vector.t

  include
    Transform_cache.CACHEABLE_TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Column-wise centering and population-standard-deviation scaling.

    Fitting and transformation require finite inputs. Constant features use a
    scale of one, so centering maps them to zero without division by zero.
    Optional sample weights give weighted means and weighted population
    variances over positively weighted rows; an all-zero weight vector is a
    typed error. Fit and transform are [O(rows * columns)] and transform
    allocates one dense output matrix. *)
module Standard_scaler : sig
  type params = { with_mean : bool; with_std : bool }
  type t
  type fitted

  val create : ?with_mean:bool -> ?with_std:bool -> unit -> t
  val mean : fitted -> Vector.t
  val variance : fitted -> Vector.t
  val scale : fitted -> Vector.t

  include
    Transform_cache.CACHEABLE_TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Removal of features whose population variance is not above a threshold.

    The threshold must be finite and non-negative. Fitting fails when no feature
    survives. Named schemas are filtered in original column order. Sample
    weights are rejected. Fit is [O(rows * columns)]; transform allocates only
    the selected dense columns. *)
module Variance_threshold : sig
  type params = { threshold : float }
  type t
  type fitted

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

(** Dense target-aware feature selection by an association score.

    Count selection retains exactly [k] features. Percentile selection retains
    [floor (input_width * percentile / 100)] features and fails at fit time if
    that is zero. Higher scores rank first; a score tie prefers the lower
    original column index. Selected output columns always retain original input
    order and named schemas are filtered accordingly. *)
module Univariate_selection : sig
  type selection = Count of int | Percentile of float

  (** Squared Pearson-correlation F ranking for scalar regression targets.

      This release provides scores for feature ranking rather than inferential
      p-values. It accepts finite, unweighted dense inputs with at least three
      rows. Constant features or targets score zero, and perfect correlation
      scores [Float.max_float]. *)
  module Regression : sig
    type params = { selection : selection }
    type t
    type fitted

    val create : selection -> (t, Error.t) result
    val scores : fitted -> Vector.t
    val selected_indices : fitted -> int array

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Target.regression Target.t
         and type fitted := fitted
         and type rng = Rng.t
  end

  (** One-way ANOVA F ranking for integer classification targets.

      This release provides scores for feature ranking rather than inferential
      p-values. It accepts finite, unweighted dense inputs with at least two
      classes and one residual degree of freedom. Constant features score zero;
      nonzero between-class variance with zero within-class variance scores
      [Float.max_float]. *)
  module Classification : sig
    type params = { selection : selection }
    type t
    type fitted

    val create : selection -> (t, Error.t) result
    val scores : fitted -> Vector.t
    val selected_indices : fitted -> int array

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Target.classification Target.t
         and type fitted := fitted
         and type rng = Rng.t
  end
end

(** Per-feature affine scaling into a configured finite range.

    Fitting learns finite minima, maxima, scales, and offsets. Constant features
    use a unit denominator and therefore map to the range's lower bound. When
    [clip] is true, values transformed outside the training range are clipped to
    the configured bounds. Sample weights are rejected. *)
module Min_max_scaler : sig
  type params = { feature_range : float * float; clip : bool }
  type t
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

(** Per-feature scaling by the largest absolute training value.

    Zero-valued features use a scale of one and remain zero. Input values must
    be finite, fitted schemas are checked during transform, and sample weights
    are rejected. *)
module Max_abs_scaler : sig
  type params = unit
  type t
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

(** Median centering and percentile-range scaling.

    Quantiles use linear interpolation over sorted training values. A zero
    percentile range is replaced by one. Centering and scaling can be disabled
    independently; input values and quantile bounds must be finite, and sample
    weights are rejected. *)
module Robust_scaler : sig
  type params = {
    with_centering : bool;
    with_scaling : bool;
    quantile_range : float * float;
  }

  type t
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

(** Independent L1, L2, or maximum-norm scaling of each sample.

    This transform learns only the fitted input schema. Each finite row is
    divided by its selected norm, while a zero-norm row remains unchanged.
    Sample weights are rejected. *)
module Normalizer : sig
  type norm = L1 | L2 | Max
  type params = { norm : norm }
  type t
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

(** Encoding of finite float64 categories as dense or CSR indicator columns.

    Categories are learned independently per feature and sorted ascending.
    Output columns follow input-feature order, then category order. [Reject]
    reports a category absent during fitting; [Ignore] emits an all-zero group
    for that feature. [max_output_features] bounds the fitted output width, and
    sample weights are rejected. *)
module One_hot_encoder : sig
  type unknown_category = Reject | Ignore

  type params = {
    unknown_category : unknown_category;
    max_output_features : int;
  }

  type t
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
  (** Applies the fitted encoder directly into canonical checked CSR storage
      without allocating the equivalent dense indicator matrix. *)

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Encoding of finite float64 categories as ordered integer-valued columns.

    Each feature's categories are sorted ascending and encoded from zero.
    Unknown values either fail or map to a caller-selected finite value that
    must not collide with a learned integer code. Sample weights are rejected.
*)
module Ordinal_encoder : sig
  type unknown_category = Reject | Use_encoded_value of float
  type params = { unknown_category : unknown_category }
  type t
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

(** Reversible sorted encoding of integer classification targets.

    Fitting records ascending distinct class labels. [transform] maps them to
    contiguous codes starting at zero; [inverse_transform] rejects invalid codes
    and restores the original labels. This target-specific utility is
    deliberately separate from the matrix transformer protocol. *)
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

(** Deterministically ordered polynomial and interaction feature expansion.

    Terms follow scikit-learn's degree-major combinations-with-replacement
    order, or strictly distinct combinations when [interaction_only] is true.
    [include_bias] controls the degree-zero constant column and
    [max_output_features] bounds allocation. Finite input is required and sample
    weights are rejected. *)
module Polynomial_features : sig
  type params = {
    degree : int;
    include_bias : bool;
    interaction_only : bool;
    max_output_features : int;
  }

  type t
  type fitted

  val create :
    ?degree:int ->
    ?include_bias:bool ->
    ?interaction_only:bool ->
    ?max_output_features:int ->
    unit ->
    (t, Error.t) result

  val terms : fitted -> int array array
  (** Returns one source-feature index sequence for every output column. *)

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Binary indicators for NaN missing-value markers.

    [Missing_only] learns columns containing NaN during fitting; [All] emits one
    indicator per input feature. With [error_on_new], transformation fails when
    NaN appears in a previously complete, unselected column. Infinities are
    rejected and sample weights are not accepted. *)
module Missing_indicator : sig
  type features = Missing_only | All
  type params = { features : features; error_on_new : bool }
  type t
  type fitted

  val create : ?features:features -> ?error_on_new:bool -> unit -> t

  val selected_features : fitted -> int array
  (** Returns source-column indices in output order. *)

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Class-weight specifications resolved into per-row sample weights.

    [Balanced] weights every class by [total / (classes * class_total)] over
    weighted class frequencies, so rarer classes receive larger weights and the
    weighted total is preserved. [Explicit] assigns listed labels their weight
    and every other label one. Resolution multiplies the class weight into the
    supplied sample weight, or into one when no sample weight is given; rows
    with zero weight stay zero and classes with no positive weight are absent
    from {!Class_weight.class_weights}. Labels listed by [Explicit] but absent
    from the rows are ignored rather than rejected, so fold-local training
    subsets that miss a rare class still resolve.

    {!Pipeline.classifier} resolves a class weight on each fit's own rows, which
    keeps balanced weights fold-local under cross-validation. Resolution is
    [O(rows)] time and space. *)
module Class_weight : sig
  type t = Balanced | Explicit of (int * float) array

  val balanced : t

  val explicit : (int * float) list -> (t, Error.t) result
  (** Validates distinct labels and finite non-negative weights. *)

  val class_weights :
    t ->
    ?sample_weight:Sample_weight.t ->
    Target.classification Target.t ->
    ((int * float) array, Error.t) result
  (** Returns the effective weight of each positively weighted class in
      ascending label order. *)

  val resolve :
    t ->
    ?sample_weight:Sample_weight.t ->
    Target.classification Target.t ->
    (Sample_weight.t, Error.t) result
end

(** Immutable sequential composition of fitted preprocessing and an estimator.

    Transformer stages are fitted only from the matrix supplied to [fit]. Their
    fitted values are then reused by [transform], [predict],
    [decision_function], and [predict_proba]. Stage names are non-empty and
    unique across the whole pipeline, and failures carry the responsible
    [Error.Stage] context.

    Unsupervised stages do not receive targets; {!Pipeline.Supervised}
    additionally packages target-aware stages in a builder tied to the target
    kind. Legacy packages route sample weights to the terminal estimator and to
    transformers opting in with [route_sample_weight]. Metadata-aware packages
    use declared per-method requests for weights and groups. Each stage receives
    a child RNG derived from its logical name and position. Fit and inference
    are sequential and allocate one dense matrix per transformer stage.

    Caching is disabled by default. [with_cache] attaches an explicit
    caller-owned store retained by pipeline clones used in cross-validation and
    search. Cache-capable descendants in nested composition receive that same
    store. *)
module Pipeline : sig
  type transformer
  type builder
  type ('target, 'prediction) estimator
  type ('target, 'prediction) t
  type ('target, 'prediction) fitted
  type capabilities = { decision_function : bool; predict_proba : bool }

  val metadata_transformer :
    name:string ->
    (module METADATA_TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result
  (** Packages per-method requests from the specification once. Fit validates
      both the fit request and the training transform request. No artifact codec
      is supplied for this adapter. *)

  val transformer :
    ?route_sample_weight:bool ->
    name:string ->
    (module TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result
  (** Packages an unsupervised transformer specification as a named stage.
      Sample weights reach the stage's [fit] only when [route_sample_weight] is
      true; by default the stage fits unweighted, matching transformers that
      declare no weight support. *)

  val cacheable_transformer :
    ?route_sample_weight:bool ->
    name:string ->
    (module Transform_cache.CACHEABLE_TRANSFORMER
       with type t = 'specification
        and type params = 'params
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result
  (** Packages a stage with an explicit stable fitted-state cache codec. A cache
      hit restores fitted state and transforms the current training matrix; it
      does not cache terminal estimators or transformed matrices. *)

  val metadata_estimator :
    name:string ->
    (module METADATA_ESTIMATOR
       with type t = 'specification
        and type target = 'target
        and type prediction = 'prediction
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    ?decision_function:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result) ->
    ?predict_proba:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result) ->
    ?classes:('fitted -> int array) ->
    'specification ->
    (('target, 'prediction) estimator, Error.t) result
  (** Packages a terminal fit request and optional prediction capabilities.
      Weight delivery follows that request; class-weight resolution, when
      needed, belongs to the consumer. No artifact codec is supplied. *)

  val estimator :
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = 'target
        and type prediction = 'prediction
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    ?decision_function:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result) ->
    ?predict_proba:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result) ->
    ?classes:('fitted -> int array) ->
    'specification ->
    (('target, 'prediction) estimator, Error.t) result
  (** Packages a terminal estimator and its explicitly supported capabilities.
      When supplied, [classes] declares the class label corresponding to each
      [predict_proba] column. *)

  val classifier :
    ?class_weight:Class_weight.t ->
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = Target.classification Target.t
        and type prediction = 'prediction
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    ?decision_function:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result) ->
    ?predict_proba:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result) ->
    ?classes:('fitted -> int array) ->
    'specification ->
    ((Target.classification Target.t, 'prediction) estimator, Error.t) result
  (** Packages a classification terminal like {!val:estimator} and, when
      [class_weight] is supplied, resolves it on each fit's own labels and
      sample weights before the estimator sees them. *)

  val empty : builder
  val add_transformer : builder -> transformer -> (builder, Error.t) result

  val set_estimator :
    builder ->
    ('target, 'prediction) estimator ->
    (('target, 'prediction) t, Error.t) result

  module Supervised : sig
    type 'kind stage
    type 'kind builder

    val metadata_transformer :
      name:string ->
      (module METADATA_TRANSFORMER
         with type t = 'specification
          and type target = 'kind Target.t
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      'specification ->
      ('kind stage, Error.t) result

    val transformer :
      ?route_sample_weight:bool ->
      name:string ->
      (module TRANSFORMER
         with type t = 'specification
          and type target = 'kind Target.t
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      'specification ->
      ('kind stage, Error.t) result
    (** Packages a supervised transformer. Each fit receives [Some y] from the
        pipeline's training rows; weights reach it only when
        [route_sample_weight] is true. Targets and weights must have one entry
        per row. Length errors are rejected before any stage fits. Class weights
        remain a terminal-estimator policy and do not alter the sample weights
        routed to transformers. *)

    val cacheable_transformer :
      ?route_sample_weight:bool ->
      name:string ->
      (module Transform_cache.CACHEABLE_TRANSFORMER
         with type t = 'specification
          and type params = 'params
          and type target = 'kind Target.t
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      'specification ->
      ('kind stage, Error.t) result
    (** Target values become part of the cache key. They are never supplied to
        an unsupervised stage adapted with {!val:unsupervised}. *)

    val unsupervised : transformer -> 'kind stage
    (** Adapts an existing unsupervised stage, retaining its weight-routing and
        artifact-codec policies. Its fit still receives [y:None]. *)

    val empty : 'kind builder

    val add_transformer :
      'kind builder -> 'kind stage -> ('kind builder, Error.t) result

    val set_estimator :
      'kind builder ->
      ('kind Target.t, 'prediction) estimator ->
      (('kind Target.t, 'prediction) t, Error.t) result
    (** The terminal and supervised stages share the same target kind. The
        resulting pipeline uses the ordinary fit, prediction, CV, and search
        APIs. Targets are used only during fitting; inference reuses learned
        transforms without targets. Metadata-aware stages may separately request
        inference weights or groups.

        A supervised stage fits on and transforms the same training rows. This
        is suitable for feature selection; target encoders needing internal
        cross-fitting require a separate fit-transform contract. Supervised
        stages currently have no artifact codec. *)
  end

  val clone : ('target, 'prediction) t -> ('target, 'prediction) t

  val with_cache :
    ('target, 'prediction) t ->
    Transform_cache.Store.t ->
    ('target, 'prediction) t
  (** Returns a specification whose cacheable transformer stages reuse the
      explicitly scoped store. Keys cover ordered feature schema and values,
      targets for supervised stages, routed sample weights, configuration, and
      logical stage seed. Nested unsupported leaves fail before fitting begins.
      CV and search clones retain the store and derive schedule-independent keys
      from logical work identities. *)

  val without_cache : ('target, 'prediction) t -> ('target, 'prediction) t
  val cache_enabled : ('target, 'prediction) t -> bool
  val transformer_names : ('target, 'prediction) t -> string array
  val estimator_name : ('target, 'prediction) t -> string
  val capabilities : ('target, 'prediction) t -> capabilities

  val fit_with_metadata :
    ('target, 'prediction) t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:'target ->
    unit ->
    (('target, 'prediction) fitted, Error.t) result
  (** Metadata-aware operations validate all supplied row lengths and all
      declared requests before any consumer runs. Fit preflight includes the
      transforms of training rows. Requests are structural: configured children
      are checked even when a column selection later proves empty. Metadata
      remains row-aligned through feature transformations; values are neither
      transformed nor implicitly reused from fitting. Existing operations use
      absent metadata, apart from the legacy fit's optional sample weights. CV
      and search can select these inputs from their metadata carrier. *)

  val transform_with_metadata :
    ('target, 'prediction) fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val predict_with_metadata :
    ('target, 'prediction) fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    ('prediction, Error.t) result

  val decision_function_with_metadata :
    ('target, 'prediction) fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Vector.t, Error.t) result

  val predict_proba_with_metadata :
    ('target, 'prediction) fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val fit :
    ('target, 'prediction) t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:'target ->
    unit ->
    (('target, 'prediction) fitted, Error.t) result

  val transform :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val predict :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    ('prediction, Error.t) result

  val decision_function :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Vector.t, Error.t) result

  val predict_proba :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val classes : ('target, 'prediction) fitted -> (int array, Error.t) result
  (** Returns the probability-column class order declared by the terminal
      estimator. *)

  val input_schema : ('target, 'prediction) fitted -> Feature_schema.t
  val output_schema : ('target, 'prediction) fitted -> Feature_schema.t
end

(** Immutable ordered column selections. Duplicate indices or names within a
    selector are rejected; different branches may select the same column. *)
module Column_selector : sig
  type t

  val all : t

  val indices : int array -> (t, Error.t) result
  (** Copies non-negative indices, preserving caller order. Bounds are checked
      when the selector is resolved against an input schema. *)

  val names : string array -> (t, Error.t) result
  (** Copies unique feature names. Non-empty name selection requires a named
      input schema; absent names fail before any branch fits. *)

  val resolve : t -> Feature_schema.t -> (int array, Error.t) result
  (** Returns a fresh array of positions. Empty selections are valid. *)
end

(** Dense column-wise composition of unsupervised transformer stages.

    Branches fit independently on the same training rows and concatenate in
    declaration order. Selectors retain their own order. Unselected columns are
    dropped by default or appended in source order as [remainder]. Columns
    explicitly selected by a dropped branch are excluded from the remainder.
    Empty selections skip fitting and inference, including for custom stages.

    Branch inputs receive named schemas: original feature names when available,
    otherwise [x0], [x1], etc. using original column positions. Output names are
    [branch__feature], using the child's output names or branch-local [x0],
    [x1], etc. for anonymous child outputs. Name collisions are typed errors.
    Inference requires the same complete ordered input schema as fitting.

    Composition validates structure; each child owns its finiteness policy.
    Passthrough preserves values, including NaNs. All-dropped output is a dense
    matrix with the original row count and zero columns; downstream estimators
    may reject it. There is no implicit sparse conversion or artifact codec. *)
module Column_transformer : sig
  type remainder = Drop | Passthrough
  type branch
  type t
  type params = t
  type fitted

  val transformer : columns:Column_selector.t -> Pipeline.transformer -> branch
  (** Uses the packaged stage's name, weight-routing policy, and transform. *)

  val passthrough :
    name:string -> columns:Column_selector.t -> (branch, Error.t) result

  val drop :
    name:string -> columns:Column_selector.t -> (branch, Error.t) result

  val create :
    ?remainder:remainder ->
    ?max_output_features:int ->
    branch array ->
    (t, Error.t) result
  (** Copies the branch array. Names must be non-blank, unique, and different
      from the reserved name [remainder]. [max_output_features] defaults to
      [100_000] and must lie between zero and [Sys.max_array_length]. It bounds
      combined output width before final concatenation; child transforms must
      enforce their own allocation limits. *)

  type branch_info = {
    name : string;
    input_indices : int array;
    output_start : int;
    output_count : int;
  }

  val branches : fitted -> branch_info array
  (** Returns defensive copies of resolved selections and half-open output
      ranges, including dropped/empty branches and any passthrough remainder. *)

  type allocation = { selected_input_bytes : int64; output_bytes : int64 }
  (** Dense payload bytes allocated by the composition operation: selected
      inputs copied for active transform branches, and the final concatenated
      matrix. Passthrough copies directly into the final matrix. Child-owned
      outputs/scratch, schemas, indices, and OCaml allocation overhead are
      excluded; these figures are neither total allocation nor peak memory. *)

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result
  (** Fits each selected child once and reuses its training output. Weights are
      checked for row alignment and supplied to child packages, which route them
      only when explicitly configured. Child RNGs derive from branch names and
      positions, independently of sibling random consumption. *)

  val transform_with_report :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages the composition into a pipeline, reusing training outputs and
      forwarding sample weights to the child packages. This avoids an extra
      transform pass during fitting. Use [Pipeline.Supervised.unsupervised] to
      include it in a target-aware pipeline. *)

  val fit_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted, Error.t) result

  val fit_transform_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result

  val transform_with_metadata :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val transform_with_report_with_metadata :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t
    type 'kind branch

    val transformer :
      columns:Column_selector.t ->
      'kind Pipeline.Supervised.stage ->
      'kind branch

    val passthrough :
      name:string -> columns:Column_selector.t -> ('kind branch, Error.t) result

    val drop :
      name:string -> columns:Column_selector.t -> ('kind branch, Error.t) result

    val create :
      ?remainder:remainder ->
      ?max_output_features:int ->
      'kind branch array ->
      ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Sequential preprocessing that can itself be used as a transformer stage.

    Each child fits only on the output of earlier children from the same
    training partition. Fitted schemas propagate without additional name
    prefixes. The empty chain is the identity, including its input schema.
    Stages preserve row count and order. Child RNGs derive from stage names and
    positions; sample weights retain each child's explicit routing policy. The
    chain allocates no additional numeric buffers beyond its children. *)
module Transformer_pipeline : sig
  type t
  type params = t
  type fitted

  val create : Pipeline.transformer array -> (t, Error.t) result
  (** Copies the stage array and rejects duplicate names. *)

  val stage_names : t -> string array

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t, Error.t) result
  (** Fits and transforms each child once, reusing its output for the next
      child. All children are unsupervised and receive no targets. *)

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages a chain for ordinary pipelines, column transformers, or feature
      unions without an extra training transform pass. Sample weights reach only
      children that request them. Composite artifact codecs are not yet
      supported. *)

  val fit_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted, Error.t) result

  val fit_transform_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t, Error.t) result

  val transform_with_metadata :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t

    val create :
      'kind Pipeline.Supervised.stage array -> ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Dense concatenation of independently fitted unsupervised branches.

    Every active branch receives the same immutable full input and schema,
    without selection copies. Outputs concatenate in declaration order with
    names [branch__feature]. Anonymous child outputs use branch-local [x0],
    [x1], etc. Duplicate generated names are typed failures.

    Dropped branches do no work. Active branches receive zero-column inputs too,
    leaving their admissibility to each transformer. Empty or all-dropped unions
    produce a zero-column matrix retaining the input row count. Branches run
    sequentially; surrounding CV may own bounded parallelism. This API does not
    add sparse output, branch-output weighting, or composite artifact codecs.
    Target-aware branches are available through [Supervised]. *)
module Feature_union : sig
  type branch
  type t
  type params = t
  type fitted

  val transformer : Pipeline.transformer -> branch
  val passthrough : name:string -> (branch, Error.t) result
  val drop : name:string -> (branch, Error.t) result

  val create : ?max_output_features:int -> branch array -> (t, Error.t) result
  (** Copies the branch array and rejects duplicate or blank names.
      [max_output_features] defaults to [100_000] and must be between zero and
      [Sys.max_array_length]. The combined width is checked before generating
      output names and concatenating; each child owns its own allocation bounds.
  *)

  type branch_info = { name : string; output_start : int; output_count : int }

  val branches : fitted -> branch_info array

  type allocation = { output_bytes : int64 }
  (** Payload bytes of the final concatenated matrix only. Input is shared;
      child allocations, metadata, and scratch are excluded. *)

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result
  (** Fits each active branch once, reuses its training output, and derives
      random streams from branch names and positions. Sample weights are checked
      for row alignment before any branch fits, then routed only to children
      explicitly requesting them. *)

  val transform_with_report :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages the union as an ordinary transformer stage, retaining child
      weight routing and reusing branch outputs during fitting. It can nest
      inside a column transformer or transformer pipeline. *)

  val fit_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted, Error.t) result

  val fit_transform_with_metadata :
    t ->
    metadata:Metadata.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result

  val transform_with_metadata :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val transform_with_report_with_metadata :
    fitted ->
    metadata:Metadata.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t
    type 'kind branch

    val transformer : 'kind Pipeline.Supervised.stage -> 'kind branch
    val passthrough : name:string -> ('kind branch, Error.t) result
    val drop : name:string -> ('kind branch, Error.t) result

    val create :
      ?max_output_features:int ->
      'kind branch array ->
      ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Diagnostics retained by fitted numerical estimators.

    [rank] is present when the solver computes a meaningful numerical rank;
    iterative solvers return [None]. *)
module Solver_report : sig
  type stopping_reason = Direct_solution | Gradient_tolerance | Step_tolerance
  type t

  val converged : t -> bool
  val iterations : t -> int
  val objective : t -> float
  val stopping_reason : t -> stopping_reason
  val rank : t -> int option
end

(** Weighted ordinary least squares using column-pivoted Householder QR.

    The solver never forms normal equations. It reports numerical rank and
    returns a deterministic basic least-squares solution for rank-deficient
    inputs. The optional intercept is fitted without regularization. Fit costs
    [O(samples * features squared)] and prediction costs
    [O(samples * features)]. *)
module Linear_regression : sig
  type params = { fit_intercept : bool }
  type t
  type fitted

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

(** Weighted L2-regularized least squares.

    [alpha] must be finite and non-negative. Coefficients, but not the optional
    intercept, receive the penalty. The portable solver uses an augmented
    least-squares system and column-pivoted Householder QR rather than normal
    equations. Fit costs [O(samples * features squared)] and prediction costs
    [O(samples * features)]. *)
module Ridge_regression : sig
  type params = { alpha : float; fit_intercept : bool }
  type t
  type fitted

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

(** Weighted L1-regularized scalar regression.

    The portable solver uses deterministic cyclic coordinate descent to minimize
    weighted mean squared error plus [alpha] times the L1 coefficient norm. The
    optional intercept is not penalized. [alpha] is finite and non-negative;
    [tolerance] and [max_iterations] control checked convergence. Iteration
    exhaustion returns a typed convergence error. For [n] samples and [p]
    features, each coordinate-descent sweep costs [O(n * p)] and fitting uses
    [O(n + p)] working storage. *)
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
  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

(** Weighted scalar regression with combined L1 and L2 regularization.

    The minimized penalty is [alpha * l1_ratio * L1] plus
    [0.5 * alpha * (1 - l1_ratio) * L2 squared]. [l1_ratio] is in [[0, 1]]; one
    is lasso and zero is a pure L2 penalty. The optional intercept remains
    unpenalized. Deterministic cyclic coordinate descent returns a
    {!Solver_report.t} or a typed convergence failure. For [n] samples and [p]
    features, each sweep costs [O(n * p)] and fitting uses [O(n + p)] working
    storage. *)
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
  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

(** A descending lasso regularization path fitted with deterministic warm
    starts.

    Without explicit [alphas], [fit] constructs [count] logarithmically spaced
    values from the smallest alpha producing the all-zero centered solution to
    [epsilon] times that value. Explicit alphas are copied, validated, and
    sorted descending. Coefficient-matrix rows, intercepts, reports, and model
    indices all use this same order. A path of [a] alpha values costs the sum of
    its warm-started coordinate-descent sweeps and stores [O(a * p)] fitted
    coefficients. *)
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
  (** Returns one coefficient row per descending alpha. *)

  val intercepts : fitted -> Vector.t
  val reports : fitted -> Solver_report.t array
  val model : fitted -> index:int -> (Lasso_regression.fitted, Error.t) result
end

(** A descending elastic-net regularization path with deterministic warm starts.

    Path ordering and access follow {!Lasso_path}. Automatic alpha generation
    requires positive [l1_ratio], because a pure L2 penalty has no finite alpha
    at which every coefficient is forced to zero; explicit alphas remain valid
    when [l1_ratio] is zero. A path of [a] alpha values stores [O(a * p)] fitted
    coefficients. *)
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
  (** Returns one coefficient row per descending alpha. *)

  val intercepts : fitted -> Vector.t
  val reports : fitted -> Solver_report.t array

  val model :
    fitted -> index:int -> (Elastic_net_regression.fitted, Error.t) result
end

(** Incremental scalar regression trained by stochastic gradient descent.

    A specification is immutable. [start] creates an opaque zero-initialized
    checkpoint that owns its RNG continuation, and each [partial_fit] call
    processes the supplied non-empty batch exactly once before returning a new
    checkpoint. The input checkpoint remains unchanged. When [shuffle] is false,
    rows retain input order and the RNG is not advanced; when true, each batch
    uses a deterministic Fisher-Yates permutation from the checkpoint's current
    stream and stores the successor stream.

    [fit] implements the common estimator protocol by processing one matrix for
    at most [max_epochs] passes. Omitting [tolerance] requests exactly that
    fixed epoch budget. Supplying it stops when the largest parameter change in
    an epoch is at most [tolerance] times the largest absolute parameter or one;
    exhaustion then returns a typed convergence error. A fitted value produced
    from a checkpoint reports [Partial_fit] and remains resumable through
    [checkpoint].

    The squared-error data gradient is multiplied by the sample weight without
    normalizing individual online updates. The reported objective uses weighted
    mean half-squared error plus the declared coefficient penalty. Intercepts
    are never penalized. [No_penalty], [L1], [L2], and [Elastic_net] use a
    proximal per-sample update; [l1_ratio] controls the L1 share only for
    [Elastic_net].

    One batch costs [O(n * p)] time and [O(n + p)] temporary storage for [n]
    samples and [p] features. Checkpoints and fitted values store [O(p)] data.
    Checkpoints are in-memory training state, not a persistent artifact format.
*)
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
  (** Starts a zero-initialized checkpoint without consuming the RNG. *)

  val partial_fit :
    checkpoint ->
    ?sample_weight:Sample_weight.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:Target.regression Target.t ->
    unit ->
    (checkpoint, Error.t) result

  val to_fitted : checkpoint -> (fitted, Error.t) result
  (** Returns a prediction-ready snapshot after at least one batch. *)

  val checkpoint : fitted -> checkpoint
  (** Returns an independent checkpoint suitable for further training. *)

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

(** Incremental linear classification trained by stochastic gradient descent.

    Penalties, learning-rate schedules, stopping reasons, batch semantics, and
    checkpoint immutability follow {!Sgd_regressor}; the shared variant types
    are re-exported so schedule values interoperate. [start] additionally
    registers the complete class set for the stream: labels are sorted into
    ascending order, must be distinct, and must number at least two. Every
    positively weighted row of a later batch must carry a registered class or
    [partial_fit] returns a typed validation error; zero-weight rows are never
    inspected because they cannot change the parameters. [fit] registers the
    positively weighted classes of its single matrix, matching the other
    built-in classifiers.

    Two registered classes train one model whose positive class is the higher
    label; three or more train one one-versus-rest model per ascending class.
    All models share the update counter, learning rate, and per-batch
    permutation, and each row updates every model before the counter advances.
    [Hinge] uses the margin loss [max 0 (1 - y * score)] with [y] in [-1, +1];
    [Log_loss] uses the stable binomial deviance with gradient
    [sigmoid score - y] for [y] in [0, 1]. Sample weights scale the loss
    gradient without normalizing individual updates; intercepts are never
    penalized. The reported objective is the weighted mean over rows of the
    summed per-model loss plus the declared penalty over all coefficient rows.

    {!Sgd_classifier.coefficients} has one row for binary problems and one row
    per class otherwise, and {!Sgd_classifier.decision_function} returns one
    column per model in the same order. Binary prediction selects the higher
    class when its score is strictly positive; multiclass prediction takes the
    first maximum score, so exact ties select the lowest label.
    {!Sgd_classifier.binary_decision_function} exposes the single binary score
    column as a vector for pipeline dispatch and returns a typed compatibility
    error for multiclass models. {!Sgd_classifier.predict_proba} is available
    only for [Log_loss]: binary probabilities are the sigmoid of the score and
    its complement, and multiclass probabilities normalize the one-versus-rest
    sigmoids, becoming uniform when every sigmoid underflows.

    One batch costs [O(n * m * p)] time and [O(n + m * p)] temporary storage for
    [n] samples, [m] models, and [p] features. Checkpoints and fitted values
    store [O(m * p)] data and are in-memory training state, not a persistent
    artifact format. *)
module Sgd_classifier : sig
  type penalty = Sgd_regressor.penalty = No_penalty | L1 | L2 | Elastic_net

  type learning_rate = Sgd_regressor.learning_rate =
    | Constant
    | Inverse_scaling of { power_t : float }

  type stopping_reason = Sgd_regressor.stopping_reason =
    | Epoch_limit
    | Step_tolerance
    | Partial_fit

  type loss = Hinge | Log_loss

  type report = {
    converged : bool;
    batches_processed : int;
    updates : int;
    objective : float;
    stopping_reason : stopping_reason;
  }

  type params = {
    loss : loss;
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
    ?loss:loss ->
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

  val start :
    t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    classes:int array ->
    (checkpoint, Error.t) result
  (** Registers the distinct class labels and starts a zero-initialized
      checkpoint without consuming the RNG. *)

  val partial_fit :
    checkpoint ->
    ?sample_weight:Sample_weight.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:Target.classification Target.t ->
    unit ->
    (checkpoint, Error.t) result

  val to_fitted : checkpoint -> (fitted, Error.t) result
  (** Returns a prediction-ready snapshot after at least one batch. *)

  val checkpoint : fitted -> checkpoint
  (** Returns an independent checkpoint suitable for further training. *)

  val checkpoint_classes : checkpoint -> int array
  val checkpoint_updates : checkpoint -> int
  val checkpoint_batches_processed : checkpoint -> int

  val classes : fitted -> int array
  (** Returns the registered labels in ascending order. *)

  val coefficients : fitted -> Matrix.t
  (** Returns a [1 * features] matrix for two classes and a [classes * features]
      matrix in {!classes} order otherwise. *)

  val intercepts : fitted -> Vector.t
  val report : fitted -> report

  val decision_function :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result
  (** Returns one score column per model: a single column for two classes and
      one column per ascending class otherwise. *)

  val binary_decision_function :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Vector.t, Error.t) result

  val predict_proba :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result
  (** Returns a [samples * classes] matrix for [Log_loss] and a typed
      compatibility error for [Hinge]. *)

  include
    CLASSIFIER
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

(** Weighted binary and multiclass classification through ridge regression.

    Fitting sorts positively weighted classes and solves one [-1 versus +1]
    ridge problem per class. Coefficients and intercepts have one row or entry
    per ascending class, including for binary classification.
    [decision_function] consequently always returns a [samples * classes]
    matrix. Prediction takes the first maximum score, making an exact tie select
    the lowest class label.

    [alpha] is finite and non-negative, coefficients but not intercepts are
    penalized, and each class fit has its own direct {!Solver_report.t}. At
    least two positively weighted classes are required. For [k] classes, [n]
    samples, and [p] features, fitting costs [O(k * (n * p squared + p cubed))]
    with dense QR solves and stores [O(k * p)] fitted parameters; prediction
    costs [O(n * k * p)]. *)
module Ridge_classifier : sig
  type params = { alpha : float; fit_intercept : bool }
  type t
  type fitted

  val create :
    ?alpha:float -> ?fit_intercept:bool -> unit -> (t, Error.t) result

  val coefficients : fitted -> Matrix.t
  (** Returns a [classes * features] matrix in {!classes} order. *)

  val intercepts : fitted -> Vector.t
  val classes : fitted -> int array
  val reports : fitted -> Solver_report.t array

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

(** Weighted multinomial logistic regression with an L2 coefficient penalty.

    At least three positively weighted integer classes are required and stored
    in ascending order. The solver jointly minimizes stable softmax
    cross-entropy and the coefficient penalty under a sum-to-zero class-score
    constraint; intercepts are not penalized. [c] is the positive inverse
    regularization strength. Deterministic damped Newton iterations stop on
    gradient or step tolerance, and iteration exhaustion is a typed convergence
    failure.

    Coefficient rows, intercept entries, decision columns, and probability
    columns all follow ascending class order. Exact prediction ties select the
    lowest label. For [k] classes, [n] samples, and [p] augmented features,
    fitting costs
    [O(iterations * (n * k squared * p squared + k cubed * p cubed))];
    prediction costs [O(n * k * p)]. *)
module Multinomial_logistic_regression : sig
  type params = {
    c : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

  val create :
    ?c:float ->
    ?fit_intercept:bool ->
    ?tolerance:float ->
    ?max_iterations:int ->
    unit ->
    (t, Error.t) result

  val coefficients : fitted -> Matrix.t
  (** Returns a [classes * features] matrix in {!classes} order. *)

  val intercepts : fitted -> Vector.t
  val classes : fitted -> int array
  val report : fitted -> Solver_report.t

  val decision_function :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

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

(** Weighted binary logistic regression with an L2 coefficient penalty.

    Exactly two positively weighted integer classes are supported and stored in
    ascending order. [c] is the positive inverse regularization strength. Stable
    sigmoid and softplus formulas avoid overflow. Deterministic damped Newton
    iterations stop on gradient or step tolerance; exhausting [max_iterations]
    is a typed convergence failure. Fit costs
    [O(iterations * samples * features squared)] and prediction costs
    [O(samples * features)]. *)
module Logistic_regression : sig
  type params = {
    c : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t
  type fitted

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

(** Weighted Poisson regression with a stable log link.

    Targets must be finite and non-negative, with a positive effective mean when
    fitting an intercept. [alpha] applies an L2 penalty to coefficients but not
    the intercept. Deterministic damped IRLS iterations return typed numerical
    or convergence failures rather than non-finite fitted values. Prediction
    returns finite, strictly positive means or a typed error. For [n] samples
    and [p] features, fitting costs [O(iterations * (n * p squared + p cubed))],
    with [O(p squared)] solver storage; prediction costs [O(n * p)]. *)
module Poisson_regression : sig
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
  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

(** Weighted Tweedie generalized linear regression.

    [power <= 0] accepts real targets, [0 < power < 2] accepts non-negative
    targets, and [power >= 2] requires positive targets. [Auto] selects the
    identity link for nonpositive powers and the log link for positive powers.
    An identity-linked nonzero-power model also requires positive fitted means.
    [alpha] penalizes coefficients but not the intercept. The portable,
    deterministic damped IRLS solver reports checked convergence and prediction
    rejects inverse-link overflow or out-of-domain means. For [n] samples and
    [p] features, fitting costs [O(iterations * (n * p squared + p cubed))],
    with [O(p squared)] solver storage; prediction costs [O(n * p)]. *)
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

  type t
  type fitted

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
  (** Returns [Identity] or [Log]; a fitted model never retains [Auto]. *)

  val report : fitted -> Solver_report.t

  include
    REGRESSOR
      with type t := t
       and type params := params
       and type fitted := fitted
       and type rng = Rng.t
end

(** A validated train/test selection over one aligned source.

    Train and test rows must be non-empty, unique within each partition,
    disjoint, and aligned to the same source size. [materialize] explicitly
    copies both selections into independent aligned datasets; constructing or
    inspecting a split does not copy dataset buffers. *)
module Split : sig
  type t

  val create :
    source_size:int -> train:int array -> test:int array -> (t, Error.t) result

  val of_views : train:Row_view.t -> test:Row_view.t -> (t, Error.t) result
  val train : t -> Row_view.t
  val test : t -> Row_view.t

  val materialize :
    'kind Dataset.t -> t -> ('kind Dataset.t * 'kind Dataset.t, Error.t) result
end

(** Deterministic K-fold partitions.

    Every sample occurs in exactly one test fold. Fold sizes differ by at most
    one, with larger folds first. Optional shuffling changes membership using
    the supplied immutable random stream; emitted train and test views retain
    source-row order. Materializing all views requires [O(folds * samples)]
    indices. *)
module K_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Deterministic class-stratified K-fold partitions.

    Per-class test counts differ by at most one across folds. Optional shuffling
    occurs independently within each class from the supplied random stream;
    emitted row views retain source order. The classification target is required
    and must align with the feature rows. *)
module Stratified_k_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** Deterministic non-overlapping group K-fold partitions.

    Each distinct group occurs in one test fold. Groups are assigned in
    descending sample-count order to the currently smallest fold. Equal-sized
    groups use descending integer-label order and fold ties select the lowest
    fold, matching unshuffled scikit-learn membership. The group vector is
    required and must align with the feature rows. *)
module Group_k_fold : sig
  type params = { folds : int }
  type t

  val create : ?folds:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Expanding-window time-series partitions.

    Training rows are an expanding prefix and test rows are the following
    fixed-size contiguous window. [gap] rows immediately before each test window
    belong to neither partition. When [test_size] is omitted, it is
    [samples / (folds + 1)]. Split generation is chronological and does not use
    the supplied random stream. *)
module Time_series_split : sig
  type params = { folds : int; test_size : int option; gap : int }
  type t

  val create :
    ?folds:int -> ?test_size:int -> ?gap:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Policy for a metric whose denominator or required class support is absent.

    [Error] returns a typed validation failure. [Return_nan] returns IEEE NaN.
    [Use_fallback] uses the documented metric-specific finite value. *)
module Undefined_metric_policy : sig
  type t = Error | Return_nan | Use_fallback
end

(** Regression metrics over aligned finite targets.

    Metrics accept optional non-negative sample weights. Empty inputs, shape
    mismatches, and non-finite numerical results are typed failures. R-squared
    is undefined for a constant truth vector: its finite fallback is [1.] for
    perfect predictions and [0.] otherwise. All operations are [O(samples)] with
    [O(1)] scratch; [residual_curve] additionally allocates its returned
    residual vector. *)
module Regression_metrics : sig
  type residual_curve = { predictions : Vector.t; residuals : Vector.t }

  val mean_absolute_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val mean_squared_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val root_mean_squared_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val r2 :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val residual_curve :
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (residual_curve, Error.t) result
end

(** Aligned binary classifier outputs used by scorer specifications.

    At least one response must be present. Positive-class probabilities are
    finite and lie in [[0., 1.]]. When both responses are supplied, their
    lengths must agree. *)
module Binary_prediction : sig
  type t

  val create :
    ?labels:Target.classification Target.t ->
    ?positive_probabilities:Vector.t ->
    unit ->
    (t, Error.t) result

  val length : t -> int
  val labels : t -> Target.classification Target.t option
  val positive_probabilities : t -> Vector.t option
end

(** Binary classification metrics and plotting-neutral curve data.

    [positive_label] defaults to [1]. Observed labels must contain at most one
    other integer label. Curve thresholds are deterministic: ROC thresholds are
    descending and begin with infinity; precision-recall thresholds are
    ascending. ROC and precision-recall curves require positive and negative
    weighted support. Scalar fallbacks are zero for undefined precision, recall,
    F1, and balanced accuracy, [0.5] for ROC AUC, and zero for average
    precision, which sums precision over recall steps of the precision-recall
    curve without interpolation. Scalar label and loss metrics are [O(samples)]
    with [O(1)] scratch. Ranking curves are [O(samples * log samples)] time and
    [O(samples)] space. *)
module Binary_classification_metrics : sig
  type roc_curve = {
    thresholds : Vector.t;
    false_positive_rates : Vector.t;
    true_positive_rates : Vector.t;
  }

  type precision_recall_curve = {
    decision_thresholds : Vector.t;
    precisions : Vector.t;
    recalls : Vector.t;
  }

  val accuracy :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val balanced_accuracy :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val precision :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val recall :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val f1 :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val log_loss :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (float, Error.t) result

  val roc_auc :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (float, Error.t) result

  val average_precision :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (float, Error.t) result

  val roc_curve :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (roc_curve, Error.t) result

  val precision_recall_curve :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (precision_recall_curve, Error.t) result
end

(** Higher-is-better regression scorer specifications.

    Loss scorers negate their corresponding metric, following scikit-learn's
    selection convention. *)
module Regression_scorer : sig
  type metric =
    | Mean_absolute_error
    | Mean_squared_error
    | Root_mean_squared_error
    | R2

  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t

  val create : ?undefined:Undefined_metric_policy.t -> metric -> t
  val neg_mean_absolute_error : t
  val neg_mean_squared_error : t
  val neg_root_mean_squared_error : t
  val r2 : ?undefined:Undefined_metric_policy.t -> unit -> t

  val as_scorer :
    t -> (Target.regression Target.t, Target.regression Target.t) Scorer.t
  (** Admits a built-in specification through the first-class scorer API. *)

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

(** Higher-is-better binary classification scorer specifications.

    Label metrics request {!Binary_prediction.labels}; log loss, ROC AUC, and
    average precision request positive-class probabilities. Log loss is negated
    for selection. *)
module Binary_classification_scorer : sig
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision
    | Recall
    | F1
    | Log_loss
    | Roc_auc
    | Average_precision

  type response = Labels | Positive_probabilities

  type params = {
    metric : metric;
    positive_label : int;
    undefined : Undefined_metric_policy.t;
  }

  type t

  val create :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> metric -> t

  val response : t -> response
  val accuracy : t

  val balanced_accuracy :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val precision :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val recall :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val f1 :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val neg_log_loss : ?positive_label:int -> unit -> t

  val roc_auc :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val average_precision :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val as_scorer :
    t -> (Target.classification Target.t, Binary_prediction.t) Scorer.t
  (** Admits a built-in specification with its required response capability. *)

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.classification Target.t
       and type prediction = Binary_prediction.t
end

(** Aligned multiclass classifier outputs used by scorer specifications.

    At least one response must be present. Probabilities require their class
    order, must be finite values in [[0., 1.]], and each row must sum to one
    within [1e-6]. When both responses are supplied, their lengths must agree.
*)
module Multiclass_prediction : sig
  type t

  val create :
    ?labels:Target.classification Target.t ->
    ?classes:int array ->
    ?probabilities:Matrix.t ->
    unit ->
    (t, Error.t) result

  val length : t -> int
  val labels : t -> Target.classification Target.t option
  val classes : t -> int array option
  val probabilities : t -> Matrix.t option
end

(** Confusion-matrix and averaged classification metrics for any label count.

    The label set defaults to the ascending union of observed truth and
    prediction labels; an explicit [labels] array fixes the confusion-matrix
    order and drops rows whose labels fall outside it. [Micro] pools weighted
    counts before dividing, [Macro] averages per-class ratios equally, and
    [Weighted] averages them by weighted truth support. A per-class ratio with a
    zero denominator follows the undefined policy, with a fallback of zero,
    matching scikit-learn's default zero-division handling. Balanced accuracy
    averages recall over classes with positive support. Log loss clips
    probabilities to the machine epsilon and requires every truth label to have
    a probability column. Label metrics are [O(samples + classes squared)] with
    [O(classes squared)] scratch. *)
module Multiclass_classification_metrics : sig
  type average = Micro | Macro | Weighted
  type confusion_matrix = { labels : int array; counts : Matrix.t }

  type class_scores = {
    class_labels : int array;
    precisions : Vector.t;
    recalls : Vector.t;
    f1_scores : Vector.t;
    supports : Vector.t;
  }

  val confusion_matrix :
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (confusion_matrix, Error.t) result
  (** Rows are truth labels and columns are predicted labels. *)

  val accuracy :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val balanced_accuracy :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val class_scores :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (class_scores, Error.t) result

  val precision :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val recall :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val f1 :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val log_loss :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    classes:int array ->
    probabilities:Matrix.t ->
    unit ->
    (float, Error.t) result
end

(** Ranking metrics over class-probability matrices.

    One-versus-rest ROC AUC scores every class column against its indicator;
    [Macro] averages classes equally, [Weighted] by weighted truth support, and
    [Micro] pools every row-class indicator into one binary curve.
    One-versus-one ROC AUC averages, over every pair of classes with positive
    weighted support in ascending order, the mean of the two directional AUCs on
    the rows belonging to that pair; [Weighted] uses the pair's weighted
    prevalence, and [Micro] is a typed validation error. A class without
    weighted support follows the undefined policy with a fallback of [0.5].
    scikit-learn refuses sample weights for one-versus-one AUC; ModelKit applies
    them to both the pairwise curves and the prevalences.

    Top-k accuracy counts a row as a hit when fewer than [k] other classes
    outrank the truth class, with exact ties broken toward the higher column
    index as scikit-learn does. [k] must lie in [\[1, classes)]. One-versus-rest
    costs [O(classes * n log n)]; one-versus-one costs
    [O(classes squared * n log n)]; top-k costs [O(n * classes)]. *)
module Multiclass_ranking : sig
  type strategy = One_vs_rest | One_vs_one

  val roc_auc :
    ?strategy:strategy ->
    ?average:Multiclass_classification_metrics.average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    classes:int array ->
    probabilities:Matrix.t ->
    unit ->
    (float, Error.t) result

  val top_k_accuracy :
    k:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    classes:int array ->
    probabilities:Matrix.t ->
    unit ->
    (float, Error.t) result
end

(** Discounted cumulative gain over per-row graded relevance.

    Each row of [relevance] holds finite non-negative gains for the row's items
    and each row of [scores] holds the ranking scores; both matrices share one
    shape with at least two columns. Rank [r] receives discount
    [1 / log2 (r + 2)], and [k] zeroes discounts from rank [k] onward. By
    default tied scores share the mean gain of their group times the group's
    summed discount, following McSherry and Najork; [ignore_ties] instead ranks
    tied items by descending column index as scikit-learn's reversed stable sort
    does. NDCG divides each row by its ideal DCG and scores an all-zero row as
    zero. Both metrics average rows by sample weight and cost
    [O(rows * columns log columns)]. *)
module Ranking_metrics : sig
  val dcg :
    ?k:int ->
    ?ignore_ties:bool ->
    ?sample_weight:Sample_weight.t ->
    relevance:Matrix.t ->
    scores:Matrix.t ->
    unit ->
    (float, Error.t) result

  val ndcg :
    ?k:int ->
    ?ignore_ties:bool ->
    ?sample_weight:Sample_weight.t ->
    relevance:Matrix.t ->
    scores:Matrix.t ->
    unit ->
    (float, Error.t) result
end

(** Higher-is-better multiclass scorer specifications.

    Label metrics request {!Multiclass_prediction.labels}; log loss, ROC AUC,
    and top-k accuracy request class probabilities, and log loss is negated for
    selection. Averaged metrics default to [Macro] and carry the averaging mode
    in their name, for example [f1_weighted]. *)
module Multiclass_classification_scorer : sig
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision of Multiclass_classification_metrics.average
    | Recall of Multiclass_classification_metrics.average
    | F1 of Multiclass_classification_metrics.average
    | Log_loss
    | Roc_auc of {
        strategy : Multiclass_ranking.strategy;
        average : Multiclass_classification_metrics.average;
      }
    | Top_k_accuracy of int

  type response = Labels | Class_probabilities
  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t

  val create : ?undefined:Undefined_metric_policy.t -> metric -> t
  val response : t -> response
  val accuracy : t
  val balanced_accuracy : ?undefined:Undefined_metric_policy.t -> unit -> t

  val precision :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val recall :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val f1 :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val neg_log_loss : t

  val roc_auc :
    ?undefined:Undefined_metric_policy.t ->
    ?strategy:Multiclass_ranking.strategy ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t
  (** Named [roc_auc_ovr] or [roc_auc_ovo] for macro averaging, with a
      [_weighted] or [_micro] suffix otherwise. *)

  val top_k_accuracy : k:int -> t
  (** Named [top_<k>_accuracy] so several cutoffs can share one report. *)

  val as_scorer :
    t -> (Target.classification Target.t, Multiclass_prediction.t) Scorer.t
  (** Admits a built-in specification with its required response capability. *)

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.classification Target.t
       and type prediction = Multiclass_prediction.t
end

(** Stable [O(scores)] population aggregation with [O(1)] scratch.

    Empty arrays and infinities are typed failures. For NaN values, [Error]
    fails, [Return_nan] returns NaN summary statistics, and [Use_fallback]
    substitutes zero. *)
module Score_aggregation : sig
  type t = {
    count : int;
    mean : float;
    standard_deviation : float;
    minimum : float;
    maximum : float;
  }

  val summarize :
    ?undefined:Undefined_metric_policy.t -> float array -> (t, Error.t) result
end

(** Deterministic cross-validation over immutable pipelines.

    Split membership is planned from [seed] before fitting. Each fold receives a
    child seed derived from its logical index and [fit_seed], which defaults to
    [seed]; meta-estimators can therefore vary fit randomness without changing
    split membership. Training and test partitions are materialized explicitly,
    so every preprocessing stage is fitted only from training rows. Fold and
    scorer arrays retain splitter and caller order.

    [fit_time] and [score_time] are portable process CPU seconds measured with
    [Sys.time]; intervals can overlap under parallel execution, so their sum is
    not elapsed wall time. [Abort] returns the lowest-index failure; [Record]
    retains typed failures in the report and continues with later folds. Models
    and indices are retained only when requested. [execution] defaults to
    {!Execution.sequential}; every backend must return outputs and the
    lowest-index failure in logical fold order.

    [Binary_classification] scores with {!Binary_classification_scorer} and
    requires exactly two declared classes for probability scorers.
    [Multiclass_classification] scores with {!Multiclass_classification_scorer},
    accepts any pipeline that declares two or more distinct classes, and passes
    the full probability matrix in declared class order to log-loss scorers.
    Both request predicted labels and probabilities only when a scorer needs
    them; a pipeline without the requested capability records a typed prediction
    failure for the fold.

    Out-of-fold prediction requires test folds to contain every source row
    exactly once and restores successful responses to source row order.
    Classification callers select labels or probabilities. Probabilities use the
    complete dataset's ascending class order; missing fitted-fold classes
    receive zero columns, while unknown or duplicate classes are typed
    compatibility failures. *)
module Cross_validation : sig
  (** [metadata] defaults to {!Metadata.of_dataset}: dataset weights and groups
      are selected with each fold's exact training/test row views, including
      inference. An explicit carrier replaces that default without merging; its
      fields must match the complete dataset's row count. Splitters still use
      dataset groups and scorers still use dataset weights. Search refit
      receives the complete carrier. These inputs are never inferred from a
      previously fitted model.

      A supplied callback receives evaluation lifecycle events and is delivered
      to nested consumers only when their per-method request opts in. Fold
      events are buffered and dispatched on the caller domain in logical order;
      see {!Callback} for bounds, cancellation, and failure semantics.

      Each task-specific [cross_validate] accepts built-in [scorers] plus
      optional first-class [custom_scorers]. Names must be nonblank and unique
      across both arrays. A custom scorer's {!Capability.prediction} is checked
      against the task before fitting, and its declared sample-weight support is
      enforced while scoring. *)

  type failure_policy = Abort | Record
  type partition = Train | Test

  type failure_phase =
    | Materialization
    | Fitting
    | Prediction of partition
    | Scoring of { partition : partition; scorer : string }

  type failure = { phase : failure_phase; error : Error.t }

  type score = {
    name : string;
    train_score : (float, Error.t) result option;
    test_score : (float, Error.t) result option;
  }

  type 'model fold = {
    fold_index : int;
    fit_time : float;
    score_time : float;
    scores : score array;
    model : 'model option;
    train_indices : int array option;
    test_indices : int array option;
    failures : failure array;
  }

  type 'model report
  type classification_response = Labels | Probabilities

  type 'prediction prediction_fold = {
    prediction_fold_index : int;
    prediction_fit_time : float;
    predict_time : float;
    prediction_test_indices : int array;
    prediction_result : ('prediction, failure) result;
  }

  type 'prediction prediction_report
  type 'target splitter

  val target_independent_splitter :
    (module SPLITTER
       with type t = 'specification
        and type target = unit
        and type rng = Rng.t) ->
    'specification ->
    'target splitter
  (** Adapts a target-independent splitter such as {!K_fold}. *)

  val target_aware_splitter :
    (module SPLITTER
       with type t = 'specification
        and type target = 'target
        and type rng = Rng.t) ->
    'specification ->
    'target splitter
  (** Adapts a target-aware splitter such as {!Stratified_k_fold}. *)

  val folds : 'model report -> 'model fold array
  val successful_fold_count : 'model report -> int

  val prediction_folds :
    'prediction prediction_report -> 'prediction prediction_fold array

  val successful_prediction_fold_count : 'prediction prediction_report -> int

  val out_of_fold_predictions :
    'prediction prediction_report -> ('prediction, failure array) result
  (** Returns predictions restored to source row order. Under [Record], any
      failed folds make the assembled value unavailable; their successful peers
      remain inspectable through {!prediction_folds}. *)

  module Regression : sig
    type model =
      (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?custom_scorers:
        (Target.regression Target.t, Target.regression Target.t) Scorer.t array ->
      splitter:Target.regression Target.t splitter ->
      scorers:Regression_scorer.t array ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val cross_val_predict :
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      splitter:Target.regression Target.t splitter ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (Target.regression Target.t prediction_report, Error.t) result
    (** Fits on every training fold and predicts its test fold. *)
  end

  module Binary_classification : sig
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?custom_scorers:
        (Target.classification Target.t, Binary_prediction.t) Scorer.t array ->
      splitter:Target.classification Target.t splitter ->
      scorers:Binary_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val cross_val_predict :
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      response:classification_response ->
      splitter:Target.classification Target.t splitter ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (Multiclass_prediction.t prediction_report, Error.t) result
    (** Produces labels or globally aligned probabilities for binary data. *)
  end

  module Multiclass_classification : sig
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?custom_scorers:
        (Target.classification Target.t, Multiclass_prediction.t) Scorer.t array ->
      splitter:Target.classification Target.t splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val cross_val_predict :
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      response:classification_response ->
      splitter:Target.classification Target.t splitter ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (Multiclass_prediction.t prediction_report, Error.t) result
    (** Produces labels or globally aligned probabilities for multiclass data.
    *)
  end
end

(** Leakage-safe learning curves over nested training-fold prefixes.

    A learning curve measures how train and validation scores change as each
    fold receives more training rows. The splitter runs once. Every requested
    size then uses a prefix of each base training fold while leaving its
    validation fold unchanged, so points are directly comparable and no
    preprocessing or supervised selection is fitted outside a training subset.

    By default, prefixes preserve splitter order. With [shuffle=true], each base
    training fold is shuffled once from its logical fold identity and the
    supplied seed; every size still uses a nested prefix, independently of
    execution scheduling. Curve points run in schedule order. Folds within one
    point use the supplied bounded {!Execution.t}.

    Each point contains an ordinary {!Cross_validation.report} with training
    scores enabled, fitted models omitted, and row indices included only when
    requested. Multiple scorers, weights, metadata routing, callbacks, timings,
    and typed fold failures retain their cross-validation semantics. [Record] is
    the default failure policy so later sizes can still be evaluated; [Abort]
    returns the first error with the current training size in its context. *)
module Learning_curve : sig
  type training_size =
    | Count of int  (** An absolute training-row count in every fold. *)
    | Fraction of float
        (** A fraction of the smallest base training fold, rounded down. *)

  type schedule

  val schedule :
    ?shuffle:bool ->
    ?max_fits:int ->
    training_size array ->
    (schedule, Error.t) result
  (** Builds an immutable schedule. Sizes must be nonempty. Counts are positive;
      fractions are finite, greater than zero, and at most one. Once base folds
      are known, sizes must resolve to a strictly increasing sequence within the
      smallest training fold. [max_fits], when supplied, bounds [sizes * folds]
      before any pipeline is fitted. *)

  val requested_sizes : schedule -> training_size array
  val shuffle : schedule -> bool
  val max_fits : schedule -> int option

  type 'model point = {
    training_samples : int;
        (** The resolved row count used by every training fold. *)
    evaluation : 'model Cross_validation.report;
        (** Per-fold train/test scores, timings, indices, and failures. *)
  }

  type 'model report

  val points : 'model report -> 'model point array
  (** Returns curve points in requested schedule order as a defensive copy. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      schedule:schedule ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      schedule:schedule ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      schedule:schedule ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Candidate-boundary search checkpoints and partial evaluation reports.
    Checkpoints contain data-only reports, including typed failures and timings;
    they contain no fitted models, configurations, functions, or callbacks. *)
module Search_checkpoint : sig
  type snapshot
  type 'configuration t

  type entry = {
    stage : string;
    candidate_index : int;
    configuration_id : string;
    evaluation : (unit Cross_validation.report, Error.t) result;
  }
  (** A completed candidate evaluation, or a recorded build failure. Halving
      stages are zero-based round numbers; other searches use ["candidates"].
      Fold model fields are always [None]. *)

  val create :
    ?resume:snapshot ->
    specification_id:string ->
    configuration_id:('configuration -> string) ->
    unit ->
    ('configuration t, Error.t) result
  (** Supply nonblank stable IDs. [specification_id] must version all behavior
      that cannot be inspected: pipeline builders, configuration defaults,
      samplers, scorers, selectors, and their code/dependency versions.
      [configuration_id] must cover the entire immutable configuration,
      including fields absent from reported parameters. Callers must supply the
      same pure specifications on resume; closure identity cannot be verified
      automatically.

      Search checks feature/target values, schema, dataset and explicit metadata
      weights/groups, actual ordered splits, seed, task, options, encoded
      parameters, and configuration IDs before reusing results. Callback
      presence must match because metadata requests may require or reject it;
      the handler and execution concurrency may change. IDs and encoders must be
      deterministic.

      A session admits one search at a time. Snapshots may be taken from its
      callbacks on the caller domain; other concurrent access is unsupported. *)

  val snapshot : _ t -> snapshot

  val completed : snapshot -> entry array
  (** Defensive copies of committed candidate reports. A candidate is committed
      after its finished callback succeeds. Cancellation leaves earlier entries
      available, and the interrupted candidate restarts in full on resume.
      Ordinary recorded failures are committed; control errors are not. *)

  val encode : snapshot -> (bytes, Error.t) result

  val decode : bytes -> (snapshot, Error.t) result
  (** Versioned, bounded data-only encoding with a 64 MiB limit and an integrity
      checksum. No [Marshal] or executable state is decoded. The checksum
      detects accidental corruption; it does not authenticate untrusted
      producers. The caller owns persistence and should replace checkpoint files
      atomically.

      Resuming reconstructs specifications, candidate reports, promotion, and
      selection, skipping committed candidate fits. Candidate lifecycle
      callbacks may repeat, but cached CV/fit callbacks do not.
      Sampling/configuration IDs are checked before fitting, so checkpointed
      randomized search prepares all initial configurations eagerly. The final
      selector and full-data refit run again; neither fitted state nor their
      completion is checkpointed. Score-based selection agrees with
      uninterrupted execution for deterministic consumers. Timing-based or
      side-effect-dependent selectors cannot provide that guarantee. *)
end

(** Typed exhaustive search over finite immutable configuration grids.

    Axes retain declaration order and their values retain caller order. The
    Cartesian product varies the last axis fastest. Each candidate is evaluated
    on identical split membership, while fitted fold RNGs derive from the
    logical candidate and fold identities. Named refitting ranks by the [refit]
    scorer's mean test score in descending order; equal scores receive equal
    competition ranks and the lowest candidate index wins a tie.

    [Record] keeps failed candidates and selects from candidates whose primary
    test score aggregates successfully. [Abort] returns the first failure in
    candidate order. With refitting enabled, the winning immutable specification
    is fitted once on the complete dataset. [search_with_policy] also supports
    disabled refitting and custom multi-metric selection through
    {!Grid_search.refit_policy}. For [c] candidates, [f] folds, and [s] scorers,
    search performs at most [c * f + 1] fits and retains [O(c * (f + s))] report
    data. An empty axis array evaluates the base configuration once. [execution]
    controls each candidate's fold evaluation and defaults to sequential
    execution; candidates themselves are evaluated in stable sequential order.
*)
module Grid_search : sig
  (** [metadata] defaults to {!Metadata.of_dataset}: dataset weights and groups
      are selected with each fold's exact training/test row views, including
      inference. An explicit carrier replaces that default without merging; its
      fields must match the complete dataset's row count. Splitters still use
      dataset groups and scorers still use dataset weights. Search refit
      receives the complete carrier. These inputs are never inferred from a
      previously fitted model.

      A supplied callback receives evaluation lifecycle events and is delivered
      to nested consumers only when their per-method request opts in. Fold
      events are buffered and dispatched on the caller domain in logical order;
      see {!Callback} for bounds, cancellation, and failure semantics. *)

  type parameter_value =
    | Bool of bool
    | Int of int
    | Float of float
    | String of string

  type parameter = {
    parameter_name : string;
    parameter_value : parameter_value;
  }

  type 'configuration axis

  val axis :
    name:string ->
    values:'value array ->
    encode:('value -> parameter_value) ->
    set:('configuration -> 'value -> ('configuration, Error.t) result) ->
    ('configuration axis, Error.t) result
  (** Creates one non-empty typed axis. [set] must return a new configuration
      without mutating its input. *)

  type ('configuration, 'target, 'prediction) grid

  val create :
    base:'configuration ->
    build:
      ('configuration -> (('target, 'prediction) Pipeline.t, Error.t) result) ->
    'configuration axis array ->
    (('configuration, 'target, 'prediction) grid, Error.t) result

  val candidate_count : ('configuration, 'target, 'prediction) grid -> int

  type score_summary = {
    scorer_name : string;
    train : (Score_aggregation.t, Error.t) result option;
    test : (Score_aggregation.t, Error.t) result;
  }

  type 'model candidate = {
    candidate_index : int;
    parameters : parameter array;
    rank : int option;
    mean_fit_time : float;
    mean_score_time : float;
    scores : score_summary array;
    evaluation : 'model Cross_validation.report option;
    build_error : Error.t option;
  }

  type 'model refit_policy =
    | No_refit
    | Best_score of string
    | Custom of ('model candidate array -> (int, Error.t) result)
        (** [No_refit] evaluates candidates without ranking or fitting a winner.
            [Best_score name] preserves named-scorer ranking and refitting.
            [Custom select] receives copied candidate arrays after evaluation
            and returns an array index. The chosen candidate must have built and
            have at least one successful test-score aggregate; the selector
            decides which metrics it requires. Custom selection leaves ranks
            unset. Selectors must not depend on timings if reproducibility
            across execution backends matters. User exceptions propagate;
            returned errors follow the failure policy, while callback/control
            errors always abort. *)

  type 'model selected = {
    selected_candidate_index : int;
    selected_model : 'model;
  }

  type 'model report

  val candidates : 'model report -> 'model candidate array

  val selection : 'model report -> ('model selected, Error.t) result
  (** Returns the fitted winner; disabled refitting returns a typed error. *)

  val refit_result : 'model report -> ('model selected option, Error.t) result
  (** [Ok None] identifies an intentional no-refit run, including a recorded
      report where all candidates failed. [Error] identifies selection/refit
      failure. Candidate failures remain available independently. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.regression Target.t, Target.regression Target.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        grid ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.regression Target.t, Target.regression Target.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        grid ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Binary_prediction.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Binary_prediction.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Multiclass_prediction.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Multiclass_prediction.t) Scorer.t array ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Portable, versioned persistence for fitted built-in pipelines.

    Artifacts contain validated data only: no closures, marshalled OCaml values,
    commands, or training observations are serialized. The container and each
    component codec are independently versioned. Readers verify a declared MD5
    checksum and enforce configurable bounds before allocating component
    payloads. The checksum detects accidental corruption; it does not
    authenticate or encrypt an artifact. The current 0.x format is experimental;
    released readers remain covered by golden compatibility tests.

    Pipelines intended for persistence must use the artifact-aware stage and
    estimator constructors below. Encoding a fitted pipeline containing a
    component packaged only through [Pipeline.transformer] or
    [Pipeline.estimator] returns a typed artifact error. *)
module Artifact : sig
  type limits
  type metadata
  type 'model loaded

  type regression_model =
    (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

  type binary_classification_model =
    ( Target.classification Target.t,
      Target.classification Target.t )
    Pipeline.fitted

  val default_limits : limits

  val limits :
    ?max_bytes:int ->
    ?max_components:int ->
    ?max_features:int ->
    ?max_string_bytes:int ->
    ?max_metadata_entries:int ->
    unit ->
    (limits, Error.t) result

  val empty_metadata : metadata

  val metadata :
    ?training_rows:int ->
    ?root_seed:Seed.t ->
    ?sample_weighted:bool ->
    ?labels:(string * string) array ->
    unit ->
    (metadata, Error.t) result

  val model : 'model loaded -> 'model
  val metadata_of_loaded : _ loaded -> metadata
  val producer_version : _ loaded -> string
  val producer_ocaml_version : _ loaded -> string option
  val training_rows : metadata -> int option
  val root_seed : metadata -> Seed.t option
  val sample_weighted : metadata -> bool option
  val labels : metadata -> (string * string) array

  val simple_imputer_stage :
    name:string -> Simple_imputer.t -> (Pipeline.transformer, Error.t) result

  val standard_scaler_stage :
    ?route_sample_weight:bool ->
    name:string ->
    Standard_scaler.t ->
    (Pipeline.transformer, Error.t) result

  val variance_threshold_stage :
    name:string ->
    Variance_threshold.t ->
    (Pipeline.transformer, Error.t) result

  val linear_regression_estimator :
    name:string ->
    Linear_regression.t ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result

  val ridge_regression_estimator :
    name:string ->
    Ridge_regression.t ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result

  val logistic_regression_estimator :
    ?class_weight:Class_weight.t ->
    name:string ->
    Logistic_regression.t ->
    ( ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.estimator,
      Error.t )
    result

  val encode_regression :
    ?metadata:metadata ->
    ?limits:limits ->
    regression_model ->
    (bytes, Error.t) result

  val encode_binary_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    binary_classification_model ->
    (bytes, Error.t) result

  val decode_regression :
    ?limits:limits -> bytes -> (regression_model loaded, Error.t) result

  val decode_binary_classification :
    ?limits:limits ->
    bytes ->
    (binary_classification_model loaded, Error.t) result

  val save_regression :
    ?metadata:metadata ->
    ?limits:limits ->
    path:string ->
    regression_model ->
    (unit, Error.t) result
  (** Encodes completely before opening [path], then writes the destination
      directly. Atomic replacement, signing, and encryption are transport-level
      responsibilities. *)

  val save_binary_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    path:string ->
    binary_classification_model ->
    (unit, Error.t) result
  (** The binary-classification counterpart to [save_regression]. *)

  val load_regression :
    ?limits:limits ->
    path:string ->
    unit ->
    (regression_model loaded, Error.t) result

  val load_binary_classification :
    ?limits:limits ->
    path:string ->
    unit ->
    (binary_classification_model loaded, Error.t) result
end

(** Regression with a learned, invertible transformation of scalar targets. *)
module Transformed_target_regressor : sig
  (** Implementations preserve target length and row order, fit only on supplied
      training targets, and keep specifications immutable. Transform and inverse
      use fitted state alone. Do not retain training metadata for inference. *)
  module type TRANSFORMER = sig
    include SPECIFICATION

    type fitted

    val fit_request : t -> Metadata.Request.t

    val fit :
      t ->
      metadata:Metadata.t ->
      rng:Rng.t ->
      y:Target.regression Target.t ->
      (fitted, Error.t) result

    val transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result

    val inverse_transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result
  end

  type transformer

  val transformer :
    (module TRANSFORMER with type t = 'specification and type fitted = 'fitted) ->
    'specification ->
    transformer
  (** Captures the fit request once and clones the specification for every fit.
      Weights, groups and callbacks follow the same policies as other consumers.
  *)

  val functions :
    transform:(float -> float) ->
    inverse_transform:(float -> float) ->
    transformer
  (** Pure, deterministic scalar functions, for example [log1p] and [expm1].
      Non-finite results are typed errors; exceptions from user code propagate.
  *)

  val create :
    ?rtol:float ->
    ?atol:float ->
    name:string ->
    transformer:transformer ->
    regressor:
      ( Target.regression Target.t,
        Target.regression Target.t )
      Pipeline.estimator ->
    unit ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result
  (** Packages a terminal regressor for ordinary or supervised pipelines, CV and
      search. Every fit learns the target transformation on that training
      partition, checks [inverse_transform (transform y)] against every training
      target, then fits the regressor on transformed targets. Defaults are
      [rtol=1e-7] and [atol=1e-9]; both must be finite and nonnegative. The
      check uses [abs (restored - y) <= atol + rtol * abs y], evaluated without
      overflowing the tolerance calculation. There is no opt-out.

      Predictions are inverse-transformed before scoring, so scorers always
      receive original-space targets. Target and prediction lengths are checked;
      finite values are guaranteed by {!val:Target.regression}. Row order and
      invertibility away from training targets remain implementer obligations.
      Separate deterministic RNG streams fit the transformer and regressor. Fit
      metadata is independently routed to both consumers; inverse prediction
      requires no metadata. Child errors carry stage context. Target transforms
      have no artifact codec, so saving this wrapper returns a typed error. *)
end

(** Partition sizes shared by resampling and train/test splitting. *)
module Split_size : sig
  type t =
    | Count of int
    | Fraction of float
        (** Counts must be positive; fractions must be finite and strictly
            between zero and one. Training fractions round down, test fractions
            round up. A missing size is the complement of the other. Supplying
            both may leave unused rows; neither partition may be empty and their
            sum cannot exceed the source size. *)
end

(** Independent shuffled partitions; defaults to ten splits and a 10% test
    fraction. Rows are disjoint within each split but may recur across splits.
    Output rows retain permutation order, not sorted source order. Random
    streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Shuffle_split : sig
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t

  val create :
    ?splits:int ->
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Shuffled partitions with proportional class allocation; defaults to ten
    splits and a 10% test fraction. Requires aligned classification labels, at
    least two rows per class, and at least as many rows per partition as
    classes. Largest-remainder allocation fills training first, then test from
    remaining rows, with seeded tie breaking. Extreme imbalance can still omit a
    rare class from a partition. Classes follow first appearance order. Random
    streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Stratified_shuffle_split : sig
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t

  val create :
    ?splits:int ->
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** One train/test partition; defaults to shuffling and a 25% test fraction.
    With [shuffle=false], training takes the first rows, test takes the next
    rows, and any unused rows follow. Without shuffling the RNG is ignored.
    Random streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Holdout : sig
  type params = {
    train_size : Split_size.t option;
    test_size : Split_size.t option;
    shuffle : bool;
  }

  type t

  val create :
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    ?shuffle:bool ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Repeated shuffled K-fold, defaulting to five folds and ten repetitions. Each
    repetition tests every row exactly once. Splits are emitted in
    repetition-major, then fold-major order; row indices retain source order.
    Extending the repetition count preserves earlier repetitions. Random streams
    are deterministic for the same input and seed, independent of execution
    scheduling; they do not reproduce NumPy random streams. Groups do not
    constrain these splits; use group-aware splitting when group exclusion is
    required. *)
module Repeated_k_fold : sig
  type params = { folds : int; repeats : int }
  type t

  val create : ?folds:int -> ?repeats:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Repeated shuffled stratified K-fold, with five folds and ten repetitions by
    default. Uses the class allocation and feasibility rules of
    {!Stratified_k_fold}; rare classes can be absent in a fold. Splits are
    repetition-major, then fold-major, and indices retain source order.
    Extending the repetition count preserves earlier repetitions. Random streams
    are deterministic for the same input and seed, independent of execution
    scheduling; they do not reproduce NumPy random streams. Groups do not
    constrain these splits; use group-aware splitting when group exclusion is
    required. *)
module Repeated_stratified_k_fold : sig
  type params = { folds : int; repeats : int }
  type t

  val create : ?folds:int -> ?repeats:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** Materialized train/test datasets with all row-aligned fields selected
    together, preserving feature schema and finiteness policy. *)
module Train_test_split : sig
  val split :
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    ?shuffle:bool ->
    ?stratify:Target.classification Target.t ->
    rng:Rng.t ->
    'kind Dataset.t ->
    unit ->
    ('kind Dataset.t * 'kind Dataset.t, Error.t) result
  (** Defaults to shuffling and a 25% test fraction, like {!Holdout}. Optional
      classification labels must match the source row count and require
      [shuffle=true]; stratification uses {!Stratified_shuffle_split}'s
      allocation rules. Dataset groups are copied, not kept exclusive. All
      fields follow the exact selected order; source data is immutable.
      Materialization can fail if a selected sample-weight partition has zero
      total weight. For row views instead of copies, use the splitter modules
      and {!Split.of_views}. *)
end

(** User-defined test folds with explicit always-training rows. *)
module Predefined_split : sig
  type params = { test_folds : int array }
  type t

  val create : test_folds:int array -> unit -> (t, Error.t) result
  (** Copies one assignment per source row. [-1] means always in training;
      nonnegative IDs identify test folds and need not be contiguous. Other
      negative IDs, no test folds, and an empty training partition are rejected.
  *)

  val fold_ids : t -> int array
  (** A fresh array of distinct test IDs in ascending emission order. *)

  (** [params] returns a fresh assignment array. [split] checks source length;
      each assigned row is tested once, while [-1] rows are never tested. Each
      training partition contains every source row outside its test fold. Row
      views retain source order. RNG, targets, and groups are ignored;
      caller-defined assignments are not checked for group exclusion. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Exhaustive single-row testing in source order. *)
module Leave_one_out : sig
  type t
  type params = unit

  val create : unit -> t

  (** Requires at least two rows. Each row is tested once against all remaining
      training rows. RNG, targets, and groups are ignored. Views are emitted
      eagerly and require quadratic total index storage. Single-row scores such
      as R-squared can be undefined; select an appropriate scorer. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Exhaustive single-group testing with complete group exclusion. *)
module Leave_one_group_out : sig
  type t
  type params = unit

  val create : unit -> t

  (** Requires aligned groups and at least two distinct group IDs. Test groups
      are emitted in ascending integer-ID order, with rows in source order.
      Every row is tested once; each training partition contains all other
      groups. RNG and targets are ignored. Eager views require storage
      proportional to row count times distinct group count. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Greedy class balancing while keeping every group intact. *)
module Stratified_group_k_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  (** Defaults to five folds without shuffling. Requires at least two folds,
      aligned classification labels and groups, and at least as many distinct
      groups as folds. Every fold is nonempty, every row is tested exactly once,
      and a group never crosses training/test boundaries within a fold.

      Groups are considered in descending standard deviation of their class
      counts. Ties use ascending group IDs, or seeded shuffled order when
      [shuffle=true]; unequal dispersions retain their ordering. Classes are
      accumulated in ascending label order. Each group minimizes the mean across
      classes of the standard deviation of per-fold fractions of that class.
      Objectives within [1e-12] tie on fewer rows already allocated to the fold,
      then the lowest fold index. Remaining groups fill empty folds when
      necessary to guarantee nonempty partitions. Output row views retain source
      order.

      Class balance is a heuristic, not an optimum or a guarantee of class
      coverage. Classes confined to too few groups may be absent from training
      or test partitions. This remains a valid split; estimator/scorer
      requirements are checked downstream. RNG behavior is portable but does not
      reproduce NumPy streams. Sparse group histograms use space linear in the
      observed group/class pairs, plus folds times class count; returned row
      indices require space proportional to folds times row count. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** Leakage-safe validation curves over one typed immutable parameter axis.

    A specification retains a base configuration and an ordered, nonempty copy
    of the caller's typed values. Its setter returns a new configuration for
    each value, and its builder packages that configuration as a pipeline. The
    splitter runs exactly once per evaluation; every value is therefore scored
    on identical folds, with preprocessing and supervised selection fitted only
    on that value's training partition.

    Values are evaluated sequentially in declaration order. Folds for one value
    use the supplied bounded {!Execution.t}. Each point retains its original
    typed value, stable encoded parameter, aggregate scores, timings, and
    ordinary cross-validation report. Models are not retained or refitted.
    [Record] is the default failure policy and preserves setter, builder, fold,
    prediction, and scorer failures; [Abort] returns the first error in value
    and fold order. *)
module Validation_curve : sig
  type ('configuration, 'value, 'target, 'prediction) t

  val create :
    ?max_fits:int ->
    name:string ->
    base:'configuration ->
    values:'value array ->
    encode:('value -> Grid_search.parameter_value) ->
    set:('configuration -> 'value -> ('configuration, Error.t) result) ->
    build:
      ('configuration -> (('target, 'prediction) Pipeline.t, Error.t) result) ->
    unit ->
    (('configuration, 'value, 'target, 'prediction) t, Error.t) result
  (** Creates a one-parameter curve specification. [name] must not be blank and
      [values] must not be empty. Values and configurations must be immutable;
      [set] must not mutate the supplied base. The values array is copied.
      [max_fits], when supplied, must be positive and bounds [values * folds]
      before any setter, builder, or fit runs. *)

  val parameter_name :
    ('configuration, 'value, 'target, 'prediction) t -> string

  val parameter_values :
    ('configuration, 'value, 'target, 'prediction) t -> 'value array
  (** Returns a defensive copy in evaluation order. *)

  val max_fits : ('configuration, 'value, 'target, 'prediction) t -> int option

  type ('value, 'model) point = {
    point_index : int;
    parameter_value : 'value;
    parameter : Grid_search.parameter;
    mean_fit_time : float;
    mean_score_time : float;
    scores : Grid_search.score_summary array;
    evaluation : 'model Cross_validation.report option;
    build_error : Error.t option;
  }
  (** [evaluation] is absent only when the setter or builder failed. Fold and
      scorer failures remain inside a present evaluation and its score
      summaries. Training scores are always requested. Report timings are
      observational rather than reproducibility guarantees. *)

  type ('value, 'model) report

  val points : ('value, 'model) report -> ('value, 'model) point array
  (** Returns points and their score arrays as defensive copies. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:
        ( 'configuration,
          'value,
          Target.regression Target.t,
          Target.regression Target.t )
        t ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (('value, model) report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:
        ( 'configuration,
          'value,
          Target.classification Target.t,
          Target.classification Target.t )
        t ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (('value, model) report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:
        ( 'configuration,
          'value,
          Target.classification Target.t,
          Target.classification Target.t )
        t ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (('value, model) report, Error.t) result
  end
end

(** Cross-validated permutation significance tests.

    The observed target and every permuted target are evaluated on one shared,
    validated split plan. Preprocessing and supervised pipeline stages are
    fitted independently inside every training fold. A single scorer is used
    because the corrected upper-tail p-value is defined for one higher-is-better
    statistic.

    Targets shuffle globally when the dataset has no groups. When dataset groups
    are present, values move only among rows with the same group ID; groups
    still reach the splitter and metadata consumers normally. Permutation and
    fit seeds derive from logical permutation identities, so results do not
    depend on scheduling. ModelKit random streams intentionally do not reproduce
    NumPy streams.

    The observed evaluation runs first. Permutations use the supplied bounded
    {!Execution.t}; folds within each permutation run sequentially to prevent
    nested parallelism. Any split, fit, prediction, scoring, callback, or
    aggregation failure aborts with a typed error because omitting failed
    permutations would invalidate the p-value. Fitted models from permutations
    are never retained. *)
module Permutation_test : sig
  type t

  val create : ?permutations:int -> ?max_fits:int -> unit -> (t, Error.t) result
  (** Defaults to 100 permutations. Both arguments must be positive. [max_fits],
      when supplied, bounds [(permutations + 1) * folds] before any fit runs;
      the additional evaluation is the observed target. *)

  val permutation_count : t -> int
  val max_fits : t -> int option

  type 'model report

  val observed_score : 'model report -> float
  (** Mean validation-fold score for the unpermuted target. *)

  val permutation_scores : 'model report -> float array
  (** Mean validation-fold scores in logical permutation order. Returns a
      defensive copy. *)

  val p_value : 'model report -> float
  (** Corrected upper-tail estimate
      [(1 + count (permuted >= observed)) / (1 + permutations)]. *)

  val observed_evaluation : 'model report -> 'model Cross_validation.report
  (** The ordinary unpermuted fold report. Models are omitted; indices are
      included only when requested. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val evaluate :
      ?return_indices:bool ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:t ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorer:Regression_scorer.t ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:t ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorer:Binary_classification_scorer.t ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val evaluate :
      ?return_indices:bool ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      specification:t ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorer:Multiclass_classification_scorer.t ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Typed immutable parameter distributions with explicit random streams. *)
module Parameter_distribution : sig
  type 'a t

  val choice : 'a array -> ('a t, Error.t) result
  (** Copies a nonempty array. Values themselves must be immutable. *)

  val uniform : low:float -> high:float -> unit -> (float t, Error.t) result
  val log_uniform : low:float -> high:float -> unit -> (float t, Error.t) result

  val int_uniform : low:int -> high:int -> unit -> (int t, Error.t) result
  (** Bounds are lower-inclusive and upper-exclusive. Float bounds must be
      finite; log-uniform bounds must also be positive. Integer sampling avoids
      modulo bias and supports the complete OCaml integer range of bounds. *)

  val custom : (Rng.t -> ('a, Error.t) result) -> 'a t
  (** Samplers must be deterministic for their RNG and must not retain mutable
      random state. Returned errors are recorded as candidate build failures
      under [Record]; exceptions propagate. *)

  val sample : rng:Rng.t -> 'a t -> ('a, Error.t) result
end

(** Randomized search using the same evaluation, metadata, callback and refit
    contracts as {!Grid_search}. Finite choice-only spaces sample Cartesian
    positions without replacement, capping iterations at the product size.
    Duplicate choice values can still yield equal configurations. If any axis
    uses a distribution, all axes sample with replacement. Candidate and axis
    identities derive deterministic sampling streams, separately from fit RNGs.
    Increasing iterations preserves the sampled prefix. Random identities do not
    reproduce NumPy streams. *)
module Randomized_search : sig
  type 'configuration axis

  val axis :
    name:string ->
    distribution:'value Parameter_distribution.t ->
    encode:('value -> Grid_search.parameter_value) ->
    set:('configuration -> 'value -> ('configuration, Error.t) result) ->
    ('configuration axis, Error.t) result
  (** Setters return new immutable configurations. Axis names must be nonblank
      and unique. Encoders and setters must be deterministic and must not mutate
      inputs. *)

  type ('configuration, 'target, 'prediction) space

  val create :
    ?iterations:int ->
    base:'configuration ->
    build:
      ('configuration -> (('target, 'prediction) Pipeline.t, Error.t) result) ->
    'configuration axis array ->
    (('configuration, 'target, 'prediction) space, Error.t) result
  (** Defaults to ten iterations. An empty axis array samples the base once.
      Iterations must fit an array; finite Cartesian products must fit an OCaml
      integer. Sampling finite spaces uses storage proportional to iterations,
      without expanding the complete Cartesian product. *)

  val candidate_count : ('configuration, 'target, 'prediction) space -> int

  type 'configuration sampled_candidate = {
    sampled_parameters : Grid_search.parameter array;
    sampled_configuration : ('configuration, Error.t) result;
  }

  val sample :
    seed:Seed.t ->
    ('configuration, 'target, 'prediction) space ->
    'configuration sampled_candidate array
  (** Previews typed configurations and encoded parameters without building or
      fitting pipelines. Failed draws omit that axis's parameter; subsequent
      axes still draw after ordinary failures. The first ordinary failure is
      retained with candidate/axis context; control errors override it and stop
      remaining axis draws. Search draws candidates lazily inside candidate
      callbacks, so cancellation prevents sampling later candidates when no
      checkpoint is supplied. Checkpointed search prepares all configurations
      before evaluation to validate their identities; see {!Search_checkpoint}.
  *)

  type 'model report = 'model Grid_search.report

  val candidates : 'model report -> 'model Grid_search.candidate array
  val selection : 'model report -> ('model Grid_search.selected, Error.t) result

  val refit_result :
    'model report -> ('model Grid_search.selected option, Error.t) result

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.regression Target.t, Target.regression Target.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        space ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.regression Target.t, Target.regression Target.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        space ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Binary_prediction.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Binary_prediction.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Multiclass_prediction.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      ?custom_scorers:
        (Target.classification Target.t, Multiclass_prediction.t) Scorer.t array ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Successive halving with nested training-row budgets and fixed validation
    folds. Each round fits fresh pipelines; fitted-state continuation is not
    supported. All candidates in a round receive identical rows. Training
    subsets are seeded prefixes, with one row per class placed first for
    classification. Every base training and validation fold must contain every
    class; at least one fold is required. Group separation from the splitter is
    preserved by subsetting.

    Promotion uses descending mean test score, breaking ties by original
    candidate index, and retains [ceil(candidate_count / factor)] eligible
    candidates. Failed promotion aggregates are ineligible. Rounds continue to
    the maximum budget even with one survivor. Final selection sees only
    final-round candidates; refitting uses the entire input dataset. Custom
    selectors return an array position, while reported candidate indices retain
    their original identities.

    Configurations are sampled once and reused, with fresh builds and fits each
    round. Specifications and user functions must obey the same purity contracts
    as grid and randomized search. Candidate/fold seeds remain stable across
    rounds. Callbacks include a zero-based [Stage "halving round N"] context,
    with one outer search lifecycle and progress after each completed round.
    Cancellation and callback failures stop evaluation and return an error. With
    a checkpoint, completed candidates remain available through
    {!val:Search_checkpoint.snapshot}. Timings are observational; score and
    promotion reproducibility follows the pipeline and execution contracts. *)
module Successive_halving : sig
  type budget

  val budget :
    ?max_fits:int ->
    min_samples:int ->
    max_samples:int ->
    factor:int ->
    unit ->
    (budget, Error.t) result
  (** Training rows per fold grow geometrically, capped at [max_samples].
      Require [1 <= min_samples <= max_samples] and [factor >= 2]. The maximum
      must fit every base training fold; the minimum must cover every class.
      Optional [max_fits] bounds scheduled candidate/fold fits plus an optional
      full-data refit. This conservative bound assumes successful promotion and
      is checked before building or fitting candidates. Overflow is a validation
      error. Row alignment and positive total weights in all scheduled subsets
      are also checked before candidate fitting. *)

  val resources : budget -> int array
  (** A copied array of training-row counts in round order. *)

  type ('configuration, 'target, 'prediction) candidates

  val of_grid :
    ('configuration, 'target, 'prediction) Grid_search.grid ->
    ('configuration, 'target, 'prediction) candidates

  val of_randomized :
    ('configuration, 'target, 'prediction) Randomized_search.space ->
    ('configuration, 'target, 'prediction) candidates

  type 'model round = {
    round_index : int;
    training_samples : int;
    candidates : 'model Grid_search.candidate array;
    promoted_candidate_indices : int array;
  }
  (** Candidates are in original-index order, ranked on the promotion score.
      Promoted indices are in score order; the final round promotes none.
      Evaluation reports retain train/test row indices, without fitted fold
      models. Returned arrays are defensive copies. *)

  type 'model report

  val rounds : 'model report -> 'model round array
  val selection : 'model report -> ('model Grid_search.selected, Error.t) result

  val refit_result :
    'model report -> ('model Grid_search.selected option, Error.t) result
  (** [No_refit] yields [Ok None] after a completed final round. If promotion
      cannot proceed, [Record] retains completed rounds with an error selection;
      [Abort] returns the error directly. Ordinary final selection/refit
      failures follow the same policy. Inspect candidate reports for recorded
      evaluation failures. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        candidates ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        candidates ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      promotion_score:string ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        candidates ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        candidates ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      promotion_score:string ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        candidates ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      ?checkpoint:'configuration Search_checkpoint.t ->
      budget:budget ->
      candidates:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        candidates ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      promotion_score:string ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end
