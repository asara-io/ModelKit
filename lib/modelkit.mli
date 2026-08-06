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
    | Index_out_of_bounds of {
        name : string;
        index : int;
        upper_bound : int;
      }
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

      @raise Invalid_argument if [index] is outside the vector. *)

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
    rows:int ->
    columns:int ->
    (int -> int -> float) ->
    (t, Data_error.t) result

  val of_arrays : float array array -> (t, Data_error.t) result
  val of_bigarray : bigarray -> t
  val rows : t -> int
  val columns : t -> int
  val shape : t -> int * int

  val get : t -> int -> int -> float
  (** [get matrix row column] returns one matrix element.

      @raise Invalid_argument if either index is outside the matrix. *)

  val row : t -> int -> Vector.t
  (** [row matrix index] returns a zero-copy immutable view of one row.

      @raise Invalid_argument if [index] is outside the matrix. *)

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

      @raise Invalid_argument if [position] is outside the view. *)

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

  val create :
    expected_count:int -> string array -> (t, Data_error.t) result

  val length : t -> int

  val get : t -> int -> Feature_name.t
  (** @raise Invalid_argument if the index is outside the collection. *)

  val to_array : t -> string array
end

(** Finite, non-negative sample weights aligned to samples.

    At least one weight must be strictly positive. *)
module Sample_weight : sig
  type t

  val create :
    expected_length:int -> Vector.t -> (t, Data_error.t) result

  val of_array :
    expected_length:int -> float array -> (t, Data_error.t) result

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
  (** @raise Invalid_argument if the index is outside the collection. *)

  val to_array : t -> int array
  val distinct_count : t -> int
  val select : t -> Row_view.t -> (t, Data_error.t) result
end
