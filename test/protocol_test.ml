open Modelkit

let wrap_data_error error =
  Error.of_data_error ~remediation:"provide aligned input data" error

let wrap_data_result = function
  | Ok value -> Ok value
  | Error error -> Error (wrap_data_error error)

let validate_matrix schema matrix =
  match Feature_schema.validate_matrix schema matrix with
  | Ok () -> Ok ()
  | Error error -> Error (wrap_data_error error)

let validate_schema expected observed =
  if Feature_schema.equal expected observed then Ok ()
  else
    Error
      (Error.make ~remediation:"supply features with the fitted schema"
         (Error.Feature_schema_mismatch { expected; observed }))

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

type regressor_params = { constant : float }
type regressor_fitted = { params : regressor_params; schema : Feature_schema.t }

module Test_regressor :
  REGRESSOR
    with type t = regressor_params
     and type params = regressor_params
     and type fitted = regressor_fitted
     and type rng = Test_rng.t = struct
  type t = regressor_params
  type params = regressor_params
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = regressor_fitted
  type rng = Test_rng.t

  let clone specification = specification
  let params specification = specification

  let fit params ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    match validate_matrix feature_schema x with
    | Error _ as error -> error
    | Ok () -> Ok ({ params; schema = feature_schema } : fitted)

  let predict (fitted : fitted) ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () -> (
        match validate_matrix feature_schema x with
        | Error _ as error -> error
        | Ok () -> (
            match
              Target.regression
                (Vector.of_array
                   (Array.make (Matrix.rows x) fitted.params.constant))
            with
            | Ok prediction -> Ok prediction
            | Error error -> Error (wrap_data_error error)))

  let fitted_params (fitted : fitted) = fitted.params
  let feature_schema (fitted : fitted) = fitted.schema
end

type classifier_params = { label : int }

type classifier_fitted = {
  params : classifier_params;
  schema : Feature_schema.t;
}

module Test_classifier :
  CLASSIFIER
    with type t = classifier_params
     and type params = classifier_params
     and type fitted = classifier_fitted
     and type rng = Test_rng.t = struct
  type t = classifier_params
  type params = classifier_params
  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type fitted = classifier_fitted
  type rng = Test_rng.t

  let clone specification = specification
  let params specification = specification

  let fit params ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    match validate_matrix feature_schema x with
    | Error _ as error -> error
    | Ok () -> Ok ({ params; schema = feature_schema } : fitted)

  let predict (fitted : fitted) ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () ->
        Ok
          (Target.classification
             (Array.make (Matrix.rows x) fitted.params.label))

  let fitted_params (fitted : fitted) = fitted.params
  let feature_schema (fitted : fitted) = fitted.schema
end

type transformer_fitted = { schema : Feature_schema.t }

module Test_transformer :
  TRANSFORMER
    with type t = unit
     and type params = unit
     and type target = Target.regression Target.t
     and type fitted = transformer_fitted
     and type rng = Test_rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type fitted = transformer_fitted
  type rng = Test_rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    match validate_matrix feature_schema x with
    | Error _ as error -> error
    | Ok () -> Ok ({ schema = feature_schema } : fitted)

  let transform (fitted : fitted) ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () -> Ok x

  let fitted_params _ = ()
  let input_schema (fitted : fitted) = fitted.schema
  let output_schema (fitted : fitted) = fitted.schema
end

module Test_scorer :
  SCORER
    with type t = unit
     and type params = unit
     and type truth = Target.regression Target.t
     and type prediction = Target.regression Target.t = struct
  type t = unit
  type params = unit
  type truth = Target.regression Target.t
  type prediction = Target.regression Target.t

  let clone () = ()
  let params () = ()
  let name () = "test"
  let score () ?sample_weight:_ ~truth:_ ~prediction:_ () = Ok 0.0
end

module Test_splitter :
  SPLITTER
    with type t = unit
     and type params = unit
     and type target = Target.regression Target.t
     and type rng = Test_rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type rng = Test_rng.t

  let clone () = ()
  let params () = ()

  let split () ~rng:_ ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    match Row_view.all ~source_size:sample_count with
    | Error error -> Error (wrap_data_error error)
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

module Test_backend : NUMERICAL_BACKEND = struct
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
        (wrap_data_error
           (Data_error.Length_mismatch
              { name = "dot-product right operand"; expected; observed }))
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
        (wrap_data_error
           (Data_error.Length_mismatch
              { name = "matrix-vector operand"; expected; observed }))
    else
      wrap_data_result
        (Vector.init ~length:(Matrix.rows matrix) (fun row ->
             let total = ref 0.0 in
             for column = 0 to expected - 1 do
               total :=
                 !total
                 +. (Matrix.get matrix row column *. Vector.get vector column)
             done;
             !total))

  let transposed_matrix_vector_product matrix vector =
    let expected = Matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (wrap_data_error
           (Data_error.Length_mismatch
              { name = "transposed-matrix-vector operand"; expected; observed }))
    else
      wrap_data_result
        (Vector.init ~length:(Matrix.columns matrix) (fun column ->
             let total = ref 0.0 in
             for row = 0 to expected - 1 do
               total :=
                 !total
                 +. (Matrix.get matrix row column *. Vector.get vector row)
             done;
             !total))
end

let () =
  let rng = Test_rng.create 42L in
  let x = Result.get_ok (Matrix.of_arrays [| [| 1.0 |]; [| 2.0 |] |]) in
  let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
  let y = Result.get_ok (Target.regression (Vector.of_array [| 1.0; 2.0 |])) in
  let classes = Target.classification [| 0; 1 |] in
  let specification = { constant = 1.5 } in
  let fitted =
    Result.get_ok
      (Test_regressor.fit specification ~rng ~feature_schema ~x ~y ())
  in
  let classifier =
    Result.get_ok
      (Test_classifier.fit { label = 1 } ~rng ~feature_schema ~x ~y:classes ())
  in
  ignore (Test_regressor.params (Test_regressor.clone specification));
  ignore (Test_regressor.fitted_params fitted);
  ignore (Test_regressor.feature_schema fitted);
  ignore (Result.get_ok (Test_regressor.predict fitted ~feature_schema ~x));
  ignore (Result.get_ok (Test_classifier.predict classifier ~feature_schema ~x));
  let transformed =
    Result.get_ok
      (Test_transformer.fit () ~rng ~feature_schema ~x ~y:(Some y) ())
  in
  ignore
    (Result.get_ok (Test_transformer.transform transformed ~feature_schema ~x));
  let incompatible_schema =
    Result.get_ok (Feature_schema.anonymous ~feature_count:2)
  in
  (match
     Test_regressor.predict fitted ~feature_schema:incompatible_schema ~x
   with
  | Error error -> (
      match Error.kind error with
      | Error.Feature_schema_mismatch _ -> ()
      | Error.Data _ | Error.Shape_mismatch _ | Error.Validation _
      | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
      | Error.Artifact _ | Error.Cancelled ->
          raise (Failure "unexpected prediction error"))
  | Ok _ -> raise (Failure "incompatible prediction schema was accepted"));
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
