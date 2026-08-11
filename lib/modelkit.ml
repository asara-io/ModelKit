module Data_error = struct
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

  let pp formatter = function
    | Negative_dimension { name; value } ->
        Format.fprintf formatter "%s must be non-negative, observed %d" name
          value
    | Ragged_matrix { row; expected_columns; observed_columns } ->
        Format.fprintf formatter
          "matrix row %d has %d columns; expected %d columns" row
          observed_columns expected_columns
    | Length_mismatch { name; expected; observed } ->
        Format.fprintf formatter "%s has length %d; expected %d" name observed
          expected
    | Index_out_of_bounds { name; index; upper_bound } ->
        Format.fprintf formatter
          "%s index %d is outside the half-open range [0, %d)" name index
          upper_bound
    | Non_finite { name; index; value } ->
        Format.fprintf formatter "%s contains non-finite value %g at index %d"
          name value index
    | Negative_weight { index; value } ->
        Format.fprintf formatter "sample weight %g at index %d is negative"
          value index
    | All_zero_weights ->
        Format.pp_print_string formatter
          "sample weights must contain at least one positive value"
    | Empty_feature_name { index } ->
        Format.fprintf formatter "feature name at index %d is empty" index
    | Duplicate_feature_name { name; first_index; duplicate_index } ->
        Format.fprintf formatter
          "feature name %S at index %d duplicates index %d" name duplicate_index
          first_index

  let to_string error = Format.asprintf "%a" pp error
end

let check_index ~name ~length index =
  if index < 0 || index >= length then
    invalid_arg
      (Format.asprintf "%a" Data_error.pp
         (Data_error.Index_out_of_bounds { name; index; upper_bound = length }))

module Vector = struct
  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  type t = bigarray

  let unsafe_init length f =
    let values =
      Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout length
    in
    for index = 0 to length - 1 do
      Bigarray.Array1.set values index (f index)
    done;
    values

  let create ~length value =
    if length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "vector length"; value = length })
    else Ok (unsafe_init length (fun _ -> value))

  let init ~length f =
    if length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "vector length"; value = length })
    else Ok (unsafe_init length f)

  let of_array values = unsafe_init (Array.length values) (Array.get values)

  let of_bigarray values =
    unsafe_init (Bigarray.Array1.dim values) (Bigarray.Array1.get values)

  let length = Bigarray.Array1.dim

  let get values index =
    check_index ~name:"vector" ~length:(length values) index;
    Bigarray.Array1.get values index

  let to_array values = Array.init (length values) (get values)
  let to_bigarray values = of_bigarray values
end

module Matrix = struct
  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array2.t

  type t = bigarray

  let dimensions_error ~rows ~columns =
    if rows < 0 then
      Some
        (Data_error.Negative_dimension { name = "matrix rows"; value = rows })
    else if columns < 0 then
      Some
        (Data_error.Negative_dimension
           { name = "matrix columns"; value = columns })
    else None

  let unsafe_init ~rows ~columns f =
    let values =
      Bigarray.Array2.create Bigarray.float64 Bigarray.c_layout rows columns
    in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        Bigarray.Array2.set values row column (f row column)
      done
    done;
    values

  let init ~rows ~columns f =
    match dimensions_error ~rows ~columns with
    | Some error -> Error error
    | None -> Ok (unsafe_init ~rows ~columns f)

  let create ~rows ~columns value = init ~rows ~columns (fun _ _ -> value)

  let of_arrays values =
    let rows = Array.length values in
    let columns = if rows = 0 then 0 else Array.length values.(0) in
    let rec validate row =
      if row = rows then Ok ()
      else
        let observed_columns = Array.length values.(row) in
        if observed_columns <> columns then
          Error
            (Data_error.Ragged_matrix
               { row; expected_columns = columns; observed_columns })
        else validate (row + 1)
    in
    match validate 0 with
    | Error _ as error -> error
    | Ok () ->
        Ok
          (unsafe_init ~rows ~columns (fun row column -> values.(row).(column)))

  let of_bigarray values =
    let rows = Bigarray.Array2.dim1 values in
    let columns = Bigarray.Array2.dim2 values in
    unsafe_init ~rows ~columns (Bigarray.Array2.get values)

  let rows = Bigarray.Array2.dim1
  let columns = Bigarray.Array2.dim2
  let shape values = (rows values, columns values)

  let get values row column =
    check_index ~name:"matrix row" ~length:(rows values) row;
    check_index ~name:"matrix column" ~length:(columns values) column;
    Bigarray.Array2.get values row column

  let row values index =
    check_index ~name:"matrix row" ~length:(rows values) index;
    Bigarray.Array2.slice_left values index

  let to_arrays values =
    Array.init (rows values) (fun row ->
        Array.init (columns values) (fun column -> get values row column))

  let to_bigarray values = of_bigarray values
end

module Row_view = struct
  type t = { source_size : int; indices : int array }

  let create ~source_size indices =
    if source_size < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "row-view source size"; value = source_size })
    else
      let rec validate position =
        if position = Array.length indices then Ok ()
        else
          let index = indices.(position) in
          if index < 0 || index >= source_size then
            Error
              (Data_error.Index_out_of_bounds
                 { name = "row view"; index; upper_bound = source_size })
          else validate (position + 1)
      in
      match validate 0 with
      | Error _ as error -> error
      | Ok () -> Ok { source_size; indices = Array.copy indices }

  let all ~source_size =
    if source_size < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "row-view source size"; value = source_size })
    else Ok { source_size; indices = Array.init source_size Fun.id }

  let source_size view = view.source_size
  let length view = Array.length view.indices

  let get view position =
    check_index ~name:"row-view position" ~length:(length view) position;
    view.indices.(position)

  let indices view = Array.copy view.indices
end

let check_view_alignment ~name ~length view =
  let observed = Row_view.source_size view in
  if observed = length then Ok ()
  else Error (Data_error.Length_mismatch { name; expected = length; observed })

module Target = struct
  type regression
  type classification

  type _ t =
    | Regression_values : Vector.t -> regression t
    | Classification_values : int array -> classification t

  let regression values =
    let rec validate index =
      if index = Vector.length values then Ok (Regression_values values)
      else
        let value = Vector.get values index in
        if Float.is_finite value then validate (index + 1)
        else
          Error
            (Data_error.Non_finite { name = "regression target"; index; value })
    in
    validate 0

  let classification values = Classification_values (Array.copy values)

  let length : type kind. kind t -> int = function
    | Regression_values values -> Vector.length values
    | Classification_values values -> Array.length values

  let regression_values (Regression_values values) = values
  let classification_values (Classification_values values) = Array.copy values

  let select : type kind. kind t -> Row_view.t -> (kind t, Data_error.t) result
      =
   fun target view ->
    match
      check_view_alignment ~name:"target row view" ~length:(length target) view
    with
    | Error _ as error -> error
    | Ok () -> (
        match target with
        | Regression_values values ->
            let selected =
              Vector.unsafe_init (Row_view.length view) (fun position ->
                  Vector.get values (Row_view.get view position))
            in
            Ok (Regression_values selected)
        | Classification_values values ->
            let selected =
              Array.init (Row_view.length view) (fun position ->
                  values.(Row_view.get view position))
            in
            Ok (Classification_values selected))
end

module Feature_name = struct
  type t = string

  let create name =
    if String.length name = 0 then
      Error (Data_error.Empty_feature_name { index = 0 })
    else Ok name

  let to_string name = name
end

module Feature_names = struct
  type t = Feature_name.t array

  let create ~expected_count names =
    if expected_count < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected feature count"; value = expected_count })
    else
      let observed = Array.length names in
      if observed <> expected_count then
        Error
          (Data_error.Length_mismatch
             { name = "feature names"; expected = expected_count; observed })
      else
        let seen = Hashtbl.create observed in
        let rec validate index =
          if index = observed then Ok (Array.copy names)
          else
            let name = names.(index) in
            if String.length name = 0 then
              Error (Data_error.Empty_feature_name { index })
            else
              match Hashtbl.find_opt seen name with
              | Some first_index ->
                  Error
                    (Data_error.Duplicate_feature_name
                       { name; first_index; duplicate_index = index })
              | None ->
                  Hashtbl.add seen name index;
                  validate (index + 1)
        in
        validate 0

  let length = Array.length

  let get names index =
    check_index ~name:"feature name" ~length:(length names) index;
    names.(index)

  let to_array names = Array.map Feature_name.to_string names
end

module Sample_weight = struct
  type t = Vector.t

  let create ~expected_length values =
    if expected_length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected sample-weight length"; value = expected_length })
    else
      let observed = Vector.length values in
      if observed <> expected_length then
        Error
          (Data_error.Length_mismatch
             { name = "sample weights"; expected = expected_length; observed })
      else
        let rec validate index has_positive =
          if index = observed then
            if has_positive then Ok values
            else Error Data_error.All_zero_weights
          else
            let value = Vector.get values index in
            if not (Float.is_finite value) then
              Error
                (Data_error.Non_finite { name = "sample weights"; index; value })
            else if value < 0.0 then
              Error (Data_error.Negative_weight { index; value })
            else validate (index + 1) (has_positive || value > 0.0)
        in
        validate 0 false

  let of_array ~expected_length values =
    create ~expected_length (Vector.of_array values)

  let length = Vector.length
  let get = Vector.get
  let to_vector weights = weights

  let select weights view =
    match
      check_view_alignment ~name:"sample-weight row view"
        ~length:(length weights) view
    with
    | Error _ as error -> error
    | Ok () ->
        let selected =
          Vector.unsafe_init (Row_view.length view) (fun position ->
              get weights (Row_view.get view position))
        in
        create ~expected_length:(Row_view.length view) selected
end

module Groups = struct
  type t = int array

  let create ~expected_length groups =
    if expected_length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected group length"; value = expected_length })
    else
      let observed = Array.length groups in
      if observed = expected_length then Ok (Array.copy groups)
      else
        Error
          (Data_error.Length_mismatch
             { name = "groups"; expected = expected_length; observed })

  let length = Array.length

  let get groups index =
    check_index ~name:"group" ~length:(length groups) index;
    groups.(index)

  let to_array = Array.copy

  let distinct_count groups =
    let distinct = Hashtbl.create (length groups) in
    Array.iter (fun group -> Hashtbl.replace distinct group ()) groups;
    Hashtbl.length distinct

  let select groups view =
    match
      check_view_alignment ~name:"group row view" ~length:(length groups) view
    with
    | Error _ as error -> error
    | Ok () ->
        Ok
          (Array.init (Row_view.length view) (fun position ->
               get groups (Row_view.get view position)))
end

module Feature_schema = struct
  type t = { feature_count : int; names : Feature_names.t option }

  let anonymous ~feature_count =
    if feature_count < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "feature-schema width"; value = feature_count })
    else Ok { feature_count; names = None }

  let named names =
    { feature_count = Feature_names.length names; names = Some names }

  let of_matrix ?names matrix =
    let feature_count = Matrix.columns matrix in
    match names with
    | None -> Ok { feature_count; names = None }
    | Some names ->
        let observed = Feature_names.length names in
        if observed = feature_count then
          Ok { feature_count; names = Some names }
        else
          Error
            (Data_error.Length_mismatch
               { name = "feature names"; expected = feature_count; observed })

  let feature_count schema = schema.feature_count
  let names schema = schema.names

  let equal_names left right =
    let length = Feature_names.length left in
    if length <> Feature_names.length right then false
    else
      let rec loop index =
        index = length
        || String.equal
             (Feature_name.to_string (Feature_names.get left index))
             (Feature_name.to_string (Feature_names.get right index))
           && loop (index + 1)
      in
      loop 0

  let equal left right =
    left.feature_count = right.feature_count
    &&
    match (left.names, right.names) with
    | None, None -> true
    | Some left, Some right -> equal_names left right
    | None, Some _ | Some _, None -> false

  let validate_matrix schema matrix =
    let observed = Matrix.columns matrix in
    if observed = schema.feature_count then Ok ()
    else
      Error
        (Data_error.Length_mismatch
           {
             name = "matrix feature count";
             expected = schema.feature_count;
             observed;
           })

  let pp formatter schema =
    match schema.names with
    | None ->
        Format.fprintf formatter "%d anonymous features" schema.feature_count
    | Some names ->
        Format.fprintf formatter "@[<hov 1>[";
        for index = 0 to Feature_names.length names - 1 do
          if index > 0 then Format.fprintf formatter ";@ ";
          Format.fprintf formatter "%S"
            (Feature_name.to_string (Feature_names.get names index))
        done;
        Format.fprintf formatter "]@]"

  let to_string schema = Format.asprintf "%a" pp schema
end

module Error = struct
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

  type t = { kind : kind; context : context list; remediation : string }

  let make ?(context = []) ~remediation kind = { kind; context; remediation }

  let of_data_error ?context ~remediation error =
    make ?context ~remediation (Data error)

  let kind error = error.kind
  let context error = error.context
  let remediation error = error.remediation
  let with_context outer error = { error with context = outer :: error.context }

  let pp_dimensions formatter dimensions =
    Format.fprintf formatter "(";
    List.iteri
      (fun index dimension ->
        if index > 0 then Format.fprintf formatter ", ";
        Format.pp_print_int formatter dimension)
      dimensions;
    Format.fprintf formatter ")"

  let pp_kind formatter = function
    | Data error -> Data_error.pp formatter error
    | Shape_mismatch { name; expected; observed } ->
        Format.fprintf formatter "%s has shape %a; expected %a" name
          pp_dimensions observed pp_dimensions expected
    | Feature_schema_mismatch { expected; observed } ->
        Format.fprintf formatter
          "feature schema %a does not match fitted schema %a" Feature_schema.pp
          observed Feature_schema.pp expected
    | Validation { name; reason } ->
        Format.fprintf formatter "%s is invalid: %s" name reason
    | Numerical { operation; reason } ->
        Format.fprintf formatter "%s failed numerically: %s" operation reason
    | Convergence { algorithm; reason } ->
        Format.fprintf formatter "%s did not converge: %s" algorithm reason
    | Compatibility { component; reason } ->
        Format.fprintf formatter "%s is incompatible: %s" component reason
    | Artifact { operation; reason } ->
        Format.fprintf formatter "artifact %s failed: %s" operation reason
    | Cancelled -> Format.pp_print_string formatter "operation was cancelled"

  let pp_context formatter = function
    | Stage stage -> Format.fprintf formatter "stage %S" stage
    | Fold fold -> Format.fprintf formatter "fold %d" fold
    | Candidate candidate -> Format.fprintf formatter "candidate %d" candidate
    | Feature feature ->
        Format.fprintf formatter "feature %S" (Feature_name.to_string feature)

  let pp formatter error =
    Format.fprintf formatter "%a" pp_kind error.kind;
    (match error.context with
    | [] -> ()
    | context ->
        Format.fprintf formatter " (@[<hov>";
        List.iteri
          (fun index item ->
            if index > 0 then Format.fprintf formatter ",@ ";
            pp_context formatter item)
          context;
        Format.fprintf formatter "@])");
    Format.fprintf formatter ". Remediation: %s" error.remediation

  let to_string error = Format.asprintf "%a" pp error
end

module type SPECIFICATION = sig
  type t
  type params

  val clone : t -> t
  val params : t -> params
end

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

module type CLASSIFIER = sig
  include
    ESTIMATOR
      with type target = Target.classification Target.t
       and type prediction = Target.classification Target.t
end

module type REGRESSOR = sig
  include
    ESTIMATOR
      with type target = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

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

module type EXECUTION = sig
  type t

  val concurrency : t -> int

  val map :
    t ->
    f:(index:int -> 'input -> ('output, 'error) result) ->
    'input array ->
    ('output array, 'error) result
end

module type RNG = sig
  type seed
  type t

  val create : seed -> t
  val derive : seed -> operation:string -> index:int -> seed
  val next_int64 : t -> int64 * t
  val next_float : t -> float * t
end

module type NUMERICAL_BACKEND = sig
  val name : string
  val sum : Vector.t -> float
  val dot : Vector.t -> Vector.t -> (float, Error.t) result
  val matrix_vector_product : Matrix.t -> Vector.t -> (Vector.t, Error.t) result

  val transposed_matrix_vector_product :
    Matrix.t -> Vector.t -> (Vector.t, Error.t) result
end

module Seed = struct
  type t = int64

  let of_int = Int64.of_int
  let of_int64 seed = seed
  let to_int64 seed = seed
  let equal = Int64.equal

  (* FNV-1a. *)
  let mix value =
    let value =
      Int64.mul
        (Int64.logxor value (Int64.shift_right_logical value 30))
        (-4658895280553007687L)
    in
    let value =
      Int64.mul
        (Int64.logxor value (Int64.shift_right_logical value 27))
        (-7723592293110705685L)
    in
    Int64.logxor value (Int64.shift_right_logical value 31)

  let hash_operation operation =
    let hash = ref (-3750763034362895579L) in
    for index = 0 to String.length operation - 1 do
      hash :=
        Int64.mul
          (Int64.logxor !hash (Int64.of_int (Char.code operation.[index])))
          1099511628211L
    done;
    mix !hash

  let derive seed ~operation ~index =
    let operation_hash = hash_operation operation in
    let index_hash = mix (Int64.of_int index) in
    mix (Int64.logxor seed (Int64.logxor operation_hash index_hash))

  let pp formatter seed = Format.fprintf formatter "0x%016Lx" seed
  let to_string seed = Format.asprintf "%a" pp seed
end

module Rng = struct
  type seed = Seed.t
  type t = int64

  let create = Seed.to_int64
  let derive = Seed.derive

  let next_int64 state =
    let successor = Int64.add state (-7046029254386353131L) in
    (Seed.mix successor, successor)

  let next_float state =
    let bits, successor = next_int64 state in
    let significand = Int64.shift_right_logical bits 11 in
    (Int64.to_float significand /. 9_007_199_254_740_992.0, successor)
end

module Sequential_execution = struct
  type t = unit

  let default = ()
  let concurrency () = 1

  let map () ~f inputs =
    let rec loop index outputs =
      if index = Array.length inputs then Ok (Array.of_list (List.rev outputs))
      else
        match f ~index inputs.(index) with
        | Error _ as error -> error
        | Ok output -> loop (index + 1) (output :: outputs)
    in
    loop 0 []
end

module Reference_backend = struct
  let name = "reference"

  module Accumulator = struct
    type t = {
      mutable total : float;
      mutable correction : float;
      mutable positive_infinity : bool;
      mutable negative_infinity : bool;
      mutable nan : bool;
    }

    let create () =
      {
        total = 0.0;
        correction = 0.0;
        positive_infinity = false;
        negative_infinity = false;
        nan = false;
      }

    let record_non_finite accumulator value =
      if Float.is_nan value then accumulator.nan <- true
      else if value > 0.0 then accumulator.positive_infinity <- true
      else accumulator.negative_infinity <- true

    let add accumulator value =
      if not (Float.is_finite value) then record_non_finite accumulator value
      else if
        not
          (accumulator.nan || accumulator.positive_infinity
         || accumulator.negative_infinity)
      then (
        let next = accumulator.total +. value in
        if not (Float.is_finite next) then record_non_finite accumulator next
        else
          let correction =
            if Float.abs accumulator.total >= Float.abs value then
              accumulator.total -. next +. value
            else value -. next +. accumulator.total
          in
          let corrected = accumulator.correction +. correction in
          if Float.is_finite corrected then accumulator.correction <- corrected
          else record_non_finite accumulator corrected;
          accumulator.total <- next)

    let value accumulator =
      if
        accumulator.nan
        || (accumulator.positive_infinity && accumulator.negative_infinity)
      then Float.nan
      else if accumulator.positive_infinity then Float.infinity
      else if accumulator.negative_infinity then Float.neg_infinity
      else accumulator.total +. accumulator.correction
  end

  let length_error ~name ~expected ~observed =
    Error.of_data_error ~remediation:"provide operands with aligned dimensions"
      (Data_error.Length_mismatch { name; expected; observed })

  let vector_result = function
    | Ok vector -> Ok vector
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"provide a representable output dimension" error)

  let sum vector =
    let accumulator = Accumulator.create () in
    for index = 0 to Vector.length vector - 1 do
      Accumulator.add accumulator (Vector.get vector index)
    done;
    Accumulator.value accumulator

  let dot left right =
    let expected = Vector.length left in
    let observed = Vector.length right in
    if expected <> observed then
      Error (length_error ~name:"dot-product right operand" ~expected ~observed)
    else
      let accumulator = Accumulator.create () in
      for index = 0 to expected - 1 do
        Accumulator.add accumulator
          (Vector.get left index *. Vector.get right index)
      done;
      Ok (Accumulator.value accumulator)

  let matrix_vector_product matrix vector =
    let expected = Matrix.columns matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error (length_error ~name:"matrix-vector operand" ~expected ~observed)
    else
      vector_result
        (Vector.init ~length:(Matrix.rows matrix) (fun row ->
             let accumulator = Accumulator.create () in
             for column = 0 to expected - 1 do
               Accumulator.add accumulator
                 (Matrix.get matrix row column *. Vector.get vector column)
             done;
             Accumulator.value accumulator))

  let transposed_matrix_vector_product matrix vector =
    let expected = Matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (length_error ~name:"transposed-matrix-vector operand" ~expected
           ~observed)
    else
      vector_result
        (Vector.init ~length:(Matrix.columns matrix) (fun column ->
             let accumulator = Accumulator.create () in
             for row = 0 to expected - 1 do
               Accumulator.add accumulator
                 (Matrix.get matrix row column *. Vector.get vector row)
             done;
             Accumulator.value accumulator))
end
