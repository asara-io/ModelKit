open Modelkit_data

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
  let to_seed = Seed.of_int64

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

module Execution = struct
  type t =
    | Backend :
        (module EXECUTION with type t = 'configuration) * 'configuration
        -> t

  let of_backend backend configuration = Backend (backend, configuration)

  let sequential =
    of_backend (module Sequential_execution) Sequential_execution.default

  let concurrency (Backend ((module Backend), configuration)) =
    Backend.concurrency configuration

  let map (Backend ((module Backend), configuration)) ~f inputs =
    Backend.map configuration ~f inputs
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
