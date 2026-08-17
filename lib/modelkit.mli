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
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Column-wise centering and population-standard-deviation scaling.

    Fitting and transformation require finite inputs. Constant features use a
    scale of one, so centering maps them to zero without division by zero.
    Sample weights are rejected. Fit and transform are [O(rows * columns)] and
    transform allocates one dense output matrix. *)
module Standard_scaler : sig
  type params = { with_mean : bool; with_std : bool }
  type t
  type fitted

  val create : ?with_mean:bool -> ?with_std:bool -> unit -> t
  val mean : fitted -> Vector.t
  val variance : fitted -> Vector.t
  val scale : fitted -> Vector.t

  include
    TRANSFORMER
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

(** Immutable sequential composition of fitted preprocessing and an estimator.

    Transformer stages are fitted only from the matrix supplied to [fit]. Their
    fitted values are then reused by [transform], [predict],
    [decision_function], and [predict_proba]. Stage names are non-empty and
    unique across the whole pipeline, and failures carry the responsible
    [Error.Stage] context.

    Current transformers are unsupervised and do not receive targets or sample
    weights. Sample weights route to the terminal estimator. Each stage receives
    a child RNG derived from its logical name and position. Fit and inference
    are sequential and allocate one dense matrix per transformer stage. *)
module Pipeline : sig
  type transformer
  type builder
  type ('target, 'prediction) estimator
  type ('target, 'prediction) t
  type ('target, 'prediction) fitted
  type capabilities = { decision_function : bool; predict_proba : bool }

  val transformer :
    name:string ->
    (module TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result
  (** Packages an unsupervised transformer specification as a named stage. *)

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

  val empty : builder
  val add_transformer : builder -> transformer -> (builder, Error.t) result

  val set_estimator :
    builder ->
    ('target, 'prediction) estimator ->
    (('target, 'prediction) t, Error.t) result

  val clone : ('target, 'prediction) t -> ('target, 'prediction) t
  val transformer_names : ('target, 'prediction) t -> string array
  val estimator_name : ('target, 'prediction) t -> string
  val capabilities : ('target, 'prediction) t -> capabilities

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
    F1, and balanced accuracy, and [0.5] for ROC AUC. Scalar label and loss
    metrics are [O(samples)] with [O(1)] scratch. Ranking curves are
    [O(samples * log samples)] time and [O(samples)] space. *)
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

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

(** Higher-is-better binary classification scorer specifications.

    Label metrics request {!Binary_prediction.labels}; log loss and ROC AUC
    request positive-class probabilities. Log loss is negated for selection. *)
module Binary_classification_scorer : sig
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision
    | Recall
    | F1
    | Log_loss
    | Roc_auc

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

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.classification Target.t
       and type prediction = Binary_prediction.t
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
    lowest-index failure in logical fold order. *)
module Cross_validation : sig
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
      splitter:Target.regression Target.t splitter ->
      scorers:Regression_scorer.t array ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
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
      splitter:Target.classification Target.t splitter ->
      scorers:Binary_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Typed exhaustive search over finite immutable configuration grids.

    Axes retain declaration order and their values retain caller order. The
    Cartesian product varies the last axis fastest. Each candidate is evaluated
    on identical split membership, while fitted fold RNGs derive from the
    logical candidate and fold identities. Ranking uses the named [refit]
    scorer's mean test score in descending order; equal scores receive equal
    competition ranks and the lowest candidate index wins a tie.

    [Record] keeps failed candidates and selects from candidates whose primary
    test score aggregates successfully. [Abort] returns the first failure in
    candidate order. The winning immutable specification is fitted once on the
    complete dataset. For [c] candidates, [f] folds, and [s] scorers, search
    performs at most [c * f + 1] fits and retains [O(c * (f + s))] report data.
    An empty axis array evaluates the base configuration once. [execution]
    controls each candidate's fold evaluation and defaults to sequential
    execution; candidates themselves are evaluated in stable sequential order.
*)
module Grid_search : sig
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

  type 'model selected = {
    selected_candidate_index : int;
    selected_model : 'model;
  }

  type 'model report

  val candidates : 'model report -> 'model candidate array
  val selection : 'model report -> ('model selected, Error.t) result

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
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
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
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
    name:string -> Standard_scaler.t -> (Pipeline.transformer, Error.t) result

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
