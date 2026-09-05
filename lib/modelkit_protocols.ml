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

  val feature_matrix_vector_product :
    Feature_matrix.t -> Vector.t -> (Vector.t, Error.t) result

  val transposed_feature_matrix_vector_product :
    Feature_matrix.t -> Vector.t -> (Vector.t, Error.t) result
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

  (* Neumaier compensated summation with explicit non-finite tracking.

     The state flags are kept as bits of a float so that every field of the
     accumulator is a float and the record is stored flat with unboxed fields;
     [add] then allocates nothing. The kernels below inline the same update on
     local references so that their inner loops allocate nothing either and
     produce bit-identical results to folding with [Accumulator.add] in the
     same order. *)
  let positive_infinity_flag = 1
  let negative_infinity_flag = 2
  let nan_flag = 4

  let non_finite_flag value =
    if Float.is_nan value then nan_flag
    else if value > 0.0 then positive_infinity_flag
    else negative_infinity_flag

  let resolve ~total ~correction ~flags =
    if
      flags land nan_flag <> 0
      || flags land positive_infinity_flag <> 0
         && flags land negative_infinity_flag <> 0
    then Float.nan
    else if flags land positive_infinity_flag <> 0 then Float.infinity
    else if flags land negative_infinity_flag <> 0 then Float.neg_infinity
    else total +. correction

  module Accumulator = struct
    type t = {
      mutable total : float;
      mutable correction : float;
      mutable state : float;
    }

    let create () = { total = 0.0; correction = 0.0; state = 0.0 }
    let flags accumulator = Float.to_int accumulator.state

    let record_non_finite accumulator value =
      accumulator.state <-
        Float.of_int (flags accumulator lor non_finite_flag value)

    let add accumulator value =
      if not (Float.is_finite value) then record_non_finite accumulator value
      else if accumulator.state = 0.0 then (
        let total = accumulator.total in
        let next = total +. value in
        if not (Float.is_finite next) then record_non_finite accumulator next
        else
          let correction =
            if Float.abs total >= Float.abs value then total -. next +. value
            else value -. next +. total
          in
          let corrected = accumulator.correction +. correction in
          if Float.is_finite corrected then accumulator.correction <- corrected
          else record_non_finite accumulator corrected;
          accumulator.total <- next)

    let value accumulator =
      resolve ~total:accumulator.total ~correction:accumulator.correction
        ~flags:(flags accumulator)
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

  (* [ACCUMULATE value] is the inlined Neumaier step over the local references
     [total], [correction], and [flags] declared by the enclosing kernel. It is
     written out in each kernel because a shared closure would box every
     float that crosses it. *)

  let sum vector =
    let storage = Vector.storage vector in
    let total = ref 0.0 and correction = ref 0.0 and flags = ref 0 in
    for index = 0 to Bigarray.Array1.dim storage - 1 do
      let value = Bigarray.Array1.unsafe_get storage index in
      if not (Float.is_finite value) then
        flags := !flags lor non_finite_flag value
      else if !flags = 0 then begin
        let next = !total +. value in
        if not (Float.is_finite next) then
          flags := !flags lor non_finite_flag next
        else begin
          let step =
            if Float.abs !total >= Float.abs value then !total -. next +. value
            else value -. next +. !total
          in
          let corrected = !correction +. step in
          if Float.is_finite corrected then correction := corrected
          else flags := !flags lor non_finite_flag corrected;
          total := next
        end
      end
    done;
    resolve ~total:!total ~correction:!correction ~flags:!flags

  let dot left right =
    let expected = Vector.length left in
    let observed = Vector.length right in
    if expected <> observed then
      Error (length_error ~name:"dot-product right operand" ~expected ~observed)
    else
      let left = Vector.storage left and right = Vector.storage right in
      let total = ref 0.0 and correction = ref 0.0 and flags = ref 0 in
      for index = 0 to expected - 1 do
        let value =
          Bigarray.Array1.unsafe_get left index
          *. Bigarray.Array1.unsafe_get right index
        in
        if not (Float.is_finite value) then
          flags := !flags lor non_finite_flag value
        else if !flags = 0 then begin
          let next = !total +. value in
          if not (Float.is_finite next) then
            flags := !flags lor non_finite_flag next
          else begin
            let step =
              if Float.abs !total >= Float.abs value then
                !total -. next +. value
              else value -. next +. !total
            in
            let corrected = !correction +. step in
            if Float.is_finite corrected then correction := corrected
            else flags := !flags lor non_finite_flag corrected;
            total := next
          end
        end
      done;
      Ok (resolve ~total:!total ~correction:!correction ~flags:!flags)

  let matrix_vector_product matrix vector =
    let expected = Matrix.columns matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error (length_error ~name:"matrix-vector operand" ~expected ~observed)
    else
      let storage = Matrix.storage matrix and operand = Vector.storage vector in
      vector_result
        (Vector.init ~length:(Matrix.rows matrix) (fun row ->
             let total = ref 0.0 and correction = ref 0.0 and flags = ref 0 in
             for column = 0 to expected - 1 do
               let value =
                 Bigarray.Array2.unsafe_get storage row column
                 *. Bigarray.Array1.unsafe_get operand column
               in
               if not (Float.is_finite value) then
                 flags := !flags lor non_finite_flag value
               else if !flags = 0 then begin
                 let next = !total +. value in
                 if not (Float.is_finite next) then
                   flags := !flags lor non_finite_flag next
                 else begin
                   let step =
                     if Float.abs !total >= Float.abs value then
                       !total -. next +. value
                     else value -. next +. !total
                   in
                   let corrected = !correction +. step in
                   if Float.is_finite corrected then correction := corrected
                   else flags := !flags lor non_finite_flag corrected;
                   total := next
                 end
               end
             done;
             resolve ~total:!total ~correction:!correction ~flags:!flags))

  let transposed_matrix_vector_product matrix vector =
    let expected = Matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (length_error ~name:"transposed-matrix-vector operand" ~expected
           ~observed)
    else
      let storage = Matrix.storage matrix and operand = Vector.storage vector in
      vector_result
        (Vector.init ~length:(Matrix.columns matrix) (fun column ->
             let total = ref 0.0 and correction = ref 0.0 and flags = ref 0 in
             for row = 0 to expected - 1 do
               let value =
                 Bigarray.Array2.unsafe_get storage row column
                 *. Bigarray.Array1.unsafe_get operand row
               in
               if not (Float.is_finite value) then
                 flags := !flags lor non_finite_flag value
               else if !flags = 0 then begin
                 let next = !total +. value in
                 if not (Float.is_finite next) then
                   flags := !flags lor non_finite_flag next
                 else begin
                   let step =
                     if Float.abs !total >= Float.abs value then
                       !total -. next +. value
                     else value -. next +. !total
                   in
                   let corrected = !correction +. step in
                   if Float.is_finite corrected then correction := corrected
                   else flags := !flags lor non_finite_flag corrected;
                   total := next
                 end
               end
             done;
             resolve ~total:!total ~correction:!correction ~flags:!flags))

  let csr_matrix_vector_product matrix vector =
    let row_offsets, column_indices, values = Csr_matrix.storage matrix in
    let operand = Vector.storage vector in
    vector_result
      (Vector.init ~length:(Csr_matrix.rows matrix) (fun row ->
           let total = ref 0.0 and correction = ref 0.0 and flags = ref 0 in
           for
             entry = Array.unsafe_get row_offsets row
             to Array.unsafe_get row_offsets (row + 1) - 1
           do
             let value =
               Bigarray.Array1.unsafe_get values entry
               *. Bigarray.Array1.unsafe_get operand
                    (Array.unsafe_get column_indices entry)
             in
             if not (Float.is_finite value) then
               flags := !flags lor non_finite_flag value
             else if !flags = 0 then begin
               let next = !total +. value in
               if not (Float.is_finite next) then
                 flags := !flags lor non_finite_flag next
               else begin
                 let step =
                   if Float.abs !total >= Float.abs value then
                     !total -. next +. value
                   else value -. next +. !total
                 in
                 let corrected = !correction +. step in
                 if Float.is_finite corrected then correction := corrected
                 else flags := !flags lor non_finite_flag corrected;
                 total := next
               end
             end
           done;
           resolve ~total:!total ~correction:!correction ~flags:!flags))

  let transposed_csr_matrix_vector_product matrix vector =
    let row_offsets, column_indices, values = Csr_matrix.storage matrix in
    let operand = Vector.storage vector in
    let columns = Csr_matrix.columns matrix in
    let totals = Array.make columns 0.0 in
    let corrections = Array.make columns 0.0 in
    let flags = Array.make columns 0 in
    for row = 0 to Csr_matrix.rows matrix - 1 do
      let factor = Bigarray.Array1.unsafe_get operand row in
      for
        entry = Array.unsafe_get row_offsets row
        to Array.unsafe_get row_offsets (row + 1) - 1
      do
        let column = Array.unsafe_get column_indices entry in
        let value = Bigarray.Array1.unsafe_get values entry *. factor in
        let column_flags = Array.unsafe_get flags column in
        if not (Float.is_finite value) then
          Array.unsafe_set flags column (column_flags lor non_finite_flag value)
        else if column_flags = 0 then begin
          let total = Array.unsafe_get totals column in
          let next = total +. value in
          if not (Float.is_finite next) then
            Array.unsafe_set flags column (column_flags lor non_finite_flag next)
          else begin
            let step =
              if Float.abs total >= Float.abs value then total -. next +. value
              else value -. next +. total
            in
            let corrected = Array.unsafe_get corrections column +. step in
            if Float.is_finite corrected then
              Array.unsafe_set corrections column corrected
            else
              Array.unsafe_set flags column
                (column_flags lor non_finite_flag corrected);
            Array.unsafe_set totals column next
          end
        end
      done
    done;
    vector_result
      (Vector.init ~length:columns (fun column ->
           resolve ~total:totals.(column) ~correction:corrections.(column)
             ~flags:flags.(column)))

  let feature_matrix_vector_product matrix vector =
    let expected = Feature_matrix.columns matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (length_error ~name:"feature-matrix-vector operand" ~expected ~observed)
    else
      match matrix with
      | Feature_matrix.Dense_matrix matrix ->
          matrix_vector_product matrix vector
      | Feature_matrix.Csr_matrix matrix ->
          csr_matrix_vector_product matrix vector

  let transposed_feature_matrix_vector_product matrix vector =
    let expected = Feature_matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (length_error ~name:"transposed-feature-matrix-vector operand" ~expected
           ~observed)
    else
      match matrix with
      | Feature_matrix.Dense_matrix matrix ->
          transposed_matrix_vector_product matrix vector
      | Feature_matrix.Csr_matrix matrix ->
          transposed_csr_matrix_vector_product matrix vector
end
