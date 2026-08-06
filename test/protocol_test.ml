open Modelkit

module Test_rng : RNG with type seed = int64 = struct
  type seed = int64
  type t = int64

  let create seed = seed

  let derive seed ~operation ~index =
    Int64.logxor seed (Int64.of_int (Hashtbl.seeded_hash index operation))

  let next_int64 state = (state, Int64.add state 1L)

  let next_float state =
    let value = Int64.to_float (Int64.logand state 0x1f_ffff_ffff_ffffL) in
    (value /. 9_007_199_254_740_992.0, Int64.add state 1L)
end

module Test_regressor :
  REGRESSOR
    with type t = unit
     and type fitted = float
     and type error = Data_error.t
     and type rng = Test_rng.t = struct
  type t = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = float
  type error = Data_error.t
  type rng = Test_rng.t

  let fit () ?sample_weight:_ ~rng:_ ~x:_ ~y () =
    let values = Target.regression_values y in
    let fitted =
      if Vector.length values = 0 then 0.0 else Vector.get values 0
    in
    Ok fitted

  let predict fitted x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) fitted))
end

module Test_classifier :
  CLASSIFIER
    with type t = unit
     and type fitted = int
     and type error = Data_error.t
     and type rng = Test_rng.t = struct
  type t = unit
  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type fitted = int
  type error = Data_error.t
  type rng = Test_rng.t

  let fit () ?sample_weight:_ ~rng:_ ~x:_ ~y () =
    let labels = Target.classification_values y in
    Ok (if Array.length labels = 0 then 0 else labels.(0))

  let predict fitted x =
    Ok (Target.classification (Array.make (Matrix.rows x) fitted))
end

module Test_transformer :
  TRANSFORMER
    with type t = unit
     and type target = Target.regression Target.t
     and type fitted = unit
     and type error = Data_error.t
     and type rng = Test_rng.t = struct
  type t = unit
  type target = Target.regression Target.t
  type fitted = unit
  type error = Data_error.t
  type rng = Test_rng.t

  let fit () ?sample_weight:_ ~rng:_ ~x:_ ~y:_ () = Ok ()
  let transform () matrix = Ok matrix
end

module Test_scorer :
  SCORER
    with type t = unit
     and type truth = Target.regression Target.t
     and type prediction = Target.regression Target.t
     and type error = Data_error.t = struct
  type t = unit
  type truth = Target.regression Target.t
  type prediction = Target.regression Target.t
  type error = Data_error.t

  let name () = "test"
  let score () ?sample_weight:_ ~truth:_ ~prediction:_ () = Ok 0.0
end

module Test_splitter :
  SPLITTER
    with type t = unit
     and type target = Target.regression Target.t
     and type rng = Test_rng.t
     and type error = Data_error.t = struct
  type t = unit
  type target = Target.regression Target.t
  type rng = Test_rng.t
  type error = Data_error.t

  let split () ~rng:_ ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    match Row_view.all ~source_size:sample_count with
    | Error _ as error -> error
    | Ok rows -> Ok [| (rows, rows) |]
end

module Test_execution : EXECUTION with type t = unit = struct
  type t = unit

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

module Test_backend : NUMERICAL_BACKEND with type error = Data_error.t = struct
  type error = Data_error.t

  let name = "test"

  let sum vector =
    let total = ref 0.0 in
    for index = 0 to Vector.length vector - 1 do
      total := !total +. Vector.get vector index
    done;
    !total

  let dot left right =
    let expected = Vector.length left in
    let observed = Vector.length right in
    if expected <> observed then
      Error
        (Data_error.Length_mismatch
           { name = "dot-product right operand"; expected; observed })
    else
      let total = ref 0.0 in
      for index = 0 to expected - 1 do
        total := !total +. (Vector.get left index *. Vector.get right index)
      done;
      Ok !total

  let matrix_vector_product matrix vector =
    let expected = Matrix.columns matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (Data_error.Length_mismatch
           { name = "matrix-vector operand"; expected; observed })
    else
      Vector.init ~length:(Matrix.rows matrix) (fun row ->
          let total = ref 0.0 in
          for column = 0 to expected - 1 do
            total :=
              !total
              +. (Matrix.get matrix row column *. Vector.get vector column)
          done;
          !total)

  let transposed_matrix_vector_product matrix vector =
    let expected = Matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (Data_error.Length_mismatch
           { name = "transposed-matrix-vector operand"; expected; observed })
    else
      Vector.init ~length:(Matrix.columns matrix) (fun column ->
          let total = ref 0.0 in
          for row = 0 to expected - 1 do
            total :=
              !total +. (Matrix.get matrix row column *. Vector.get vector row)
          done;
          !total)
end

let () =
  let rng = Test_rng.create 42L in
  let x = Result.get_ok (Matrix.of_arrays [| [| 1.0 |]; [| 2.0 |] |]) in
  let y = Result.get_ok (Target.regression (Vector.of_array [| 1.0; 2.0 |])) in
  let classes = Target.classification [| 0; 1 |] in
  let fitted = Result.get_ok (Test_regressor.fit () ~rng ~x ~y ()) in
  let classifier =
    Result.get_ok (Test_classifier.fit () ~rng ~x ~y:classes ())
  in
  ignore (Result.get_ok (Test_regressor.predict fitted x));
  ignore (Result.get_ok (Test_classifier.predict classifier x));
  ignore (Result.get_ok (Test_transformer.fit () ~rng ~x ~y:(Some y) ()));
  ignore (Result.get_ok (Test_scorer.score () ~truth:y ~prediction:y ()));
  ignore (Result.get_ok (Test_splitter.split () ~rng ~x ~y:(Some y) ()));
  ignore
    (Result.get_ok
       (Test_execution.map ()
          ~f:(fun ~index value -> Ok (index, value))
          [| 1; 2 |]));
  ignore
    (Result.get_ok
       (Test_backend.dot
          (Vector.of_array [| 1.0 |])
          (Vector.of_array [| 2.0 |])))
