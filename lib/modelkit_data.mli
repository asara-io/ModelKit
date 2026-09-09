(** Portable classical machine learning workflows for OCaml. *)

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

  val unsafe_init : int -> (int -> float) -> t
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

  val storage : t -> bigarray
  (** Zero-copy access to the immutable backing storage for internal kernels.
      Callers must never write through the result. Not part of the public
      [Modelkit] API. *)
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

  val storage : t -> bigarray
  (** Zero-copy access to the immutable backing storage for internal kernels.
      Callers must never write through the result. Not part of the public
      [Modelkit] API. *)
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

  val storage : t -> int array * int array * Vector.bigarray
  (** Zero-copy access to the row offsets, column indices, and value storage for
      internal kernels. Callers must never write through the result. Not part of
      the public [Modelkit] API. *)

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
  val cache_identity : _ t -> string
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

(** Shared convention for immutable configured components.

    Concrete modules expose [params] as a public typed value. [clone] returns an
    equivalent unfitted specification and may return the same value because
    specifications contain no mutable fitted state. *)
