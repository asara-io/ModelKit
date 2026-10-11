open Modelkit

let ( let* ) = Result.bind

let validation ~name reason =
  Error
    (Error.make
       ~remediation:
         "Use values satisfying the documented XGBoost input contract."
       (Error.Validation { name; reason }))

let map_data = function
  | Ok value -> Ok value
  | Error error ->
      Error
        (Error.of_data_error
           ~remediation:"Align the feature schema with the input matrix." error)

let float32 value = Int32.float_of_bits (Int32.bits_of_float value)
let max_float32 = Int32.float_of_bits 0x7f7fffffl

module Specification = struct
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

  type t = params

  let integer ~name ~minimum value =
    if value < minimum || Int64.of_int value > 2147483647L then
      validation ~name
        (Format.sprintf "must be between %d and 2147483647" minimum)
    else Ok ()

  let bounded ~name ~minimum ~maximum ~strict value =
    if
      (not (Float.is_finite value))
      || value > maximum
      ||
      if strict then value <= minimum || float32 value <= minimum
      else value < minimum
    then
      validation ~name
        (Format.sprintf "must be finite and in %s%.17g,%.17g]%s"
           (if strict then "(" else "[")
           minimum maximum
           (if strict then ", remaining positive after float32 rounding" else ""))
    else Ok ()

  let create ?(rounds = 100) ?(max_depth = 6) ?(max_bin = 256)
      ?(learning_rate = 0.3) ?(min_child_weight = 1.) ?(min_split_loss = 0.)
      ?(reg_alpha = 0.) ?(reg_lambda = 1.) ?(subsample = 1.)
      ?(colsample_bytree = 1.) ?(threads = 1) () =
    let* () = integer ~name:"rounds" ~minimum:1 rounds in
    let* () = integer ~name:"max_depth" ~minimum:1 max_depth in
    let* () = integer ~name:"max_bin" ~minimum:2 max_bin in
    let* () = integer ~name:"threads" ~minimum:1 threads in
    let* () =
      bounded ~name:"learning_rate" ~minimum:0. ~maximum:1. ~strict:false
        learning_rate
    in
    let rec nonnegative = function
      | [] -> Ok ()
      | (name, value) :: rest ->
          let* () =
            bounded ~name ~minimum:0. ~maximum:max_float32 ~strict:false value
          in
          nonnegative rest
    in
    let* () =
      nonnegative
        [
          ("min_child_weight", min_child_weight);
          ("min_split_loss", min_split_loss);
          ("reg_alpha", reg_alpha);
          ("reg_lambda", reg_lambda);
        ]
    in
    let* () =
      bounded ~name:"subsample" ~minimum:0. ~maximum:1. ~strict:true subsample
    in
    let* () =
      bounded ~name:"colsample_bytree" ~minimum:0. ~maximum:1. ~strict:true
        colsample_bytree
    in
    Ok
      {
        rounds;
        max_depth;
        max_bin;
        learning_rate;
        min_child_weight;
        min_split_loss;
        reg_alpha;
        reg_lambda;
        subsample;
        colsample_bytree;
        threads;
      }

  let clone specification = specification
  let params specification = specification

  let native_parameters specification ~rng =
    let bits, _ = Rng.next_int64 rng in
    let seed = Int64.to_string (Int64.logand bits 0x7fffffffL) in
    let float value = Printf.sprintf "%.17g" value in
    [
      ("booster", "gbtree");
      ("device", "cpu");
      ("tree_method", "hist");
      ("validate_parameters", "true");
      ("verbosity", "0");
      ("seed", seed);
      ("nthread", string_of_int specification.threads);
      ("max_depth", string_of_int specification.max_depth);
      ("max_bin", string_of_int specification.max_bin);
      ("eta", float specification.learning_rate);
      ("min_child_weight", float specification.min_child_weight);
      ("gamma", float specification.min_split_loss);
      ("alpha", float specification.reg_alpha);
      ("lambda", float specification.reg_lambda);
      ("subsample", float specification.subsample);
      ("colsample_bytree", float specification.colsample_bytree);
    ]
end

module Input = struct
  type report = {
    shape : int array;
    bytes : int64;
    temporary_bytes : int64;
    rounded : int;
  }

  let source_shape report = Array.copy report.shape
  let payload_bytes report = report.bytes
  let temporary_payload_bytes report = report.temporary_bytes
  let rounded_values report = report.rounded

  type buffer =
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  type features = {
    feature_matrix :
      (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t;
    schema : Feature_schema.t;
    feature_conversion : report;
  }

  type target = { target_values : buffer; target_conversion : report }
  type weights = { weight_values : buffer; weight_conversion : report }

  let feature_schema (features : features) = features.schema
  let feature_report (features : features) = features.feature_conversion
  let target_report (target : target) = target.target_conversion
  let weight_report (weights : weights) = weights.weight_conversion

  let feature_value features ~row ~column =
    features.feature_matrix.{row, column}

  let target_value (target : target) index = target.target_values.{index}
  let weight_value (weights : weights) index = weights.weight_values.{index}

  let convert ~name ~index ~column ~allow_nan ~positive value =
    let invalid reason =
      let name =
        if column < 0 then Format.sprintf "%s[%d]" name index
        else Format.sprintf "%s[%d,%d]" name index column
      in
      validation ~name reason
    in
    let converted = float32 value in
    if allow_nan && Float.is_nan value then Ok converted
    else if not (Float.is_finite value && Float.is_finite converted) then
      invalid "must remain finite after float32 conversion"
    else if positive && value > 0. && converted = 0. then
      invalid "positive weight would round to zero in float32"
    else Ok converted

  let features ~feature_schema x =
    let* () = Feature_schema.validate_matrix feature_schema x |> map_data in
    let rows, columns = Matrix.shape x in
    if rows = 0 || columns = 0 then
      validation ~name:"features" "must have at least one row and one column"
    else
      let matrix =
        Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout rows columns
      in
      let rounded = ref 0 in
      let rec fill row column =
        if row = rows then Ok ()
        else if column = columns then fill (row + 1) 0
        else
          let value = Matrix.get x row column in
          let* converted =
            convert ~name:"features" ~index:row ~column ~allow_nan:true
              ~positive:false value
          in
          matrix.{row, column} <- converted;
          if Float.is_finite value && value <> converted then incr rounded;
          fill row (column + 1)
      in
      let* () = fill 0 0 in
      Ok
        {
          feature_matrix = matrix;
          schema = feature_schema;
          feature_conversion =
            {
              shape = [| rows; columns |];
              bytes =
                Int64.mul 4L
                  (Int64.mul (Int64.of_int rows) (Int64.of_int columns));
              temporary_bytes = 0L;
              rounded = !rounded;
            };
        }

  let vector ~name ~positive ~temporary_bytes length get =
    let values =
      Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout length
    in
    let rounded = ref 0 in
    let rec fill index =
      if index = length then Ok ()
      else
        let value = get index in
        let* converted =
          convert ~name ~index ~column:(-1) ~allow_nan:false ~positive value
        in
        values.{index} <- converted;
        if value <> converted then incr rounded;
        fill (index + 1)
    in
    let* () = fill 0 in
    Ok
      ( values,
        {
          shape = [| length |];
          bytes = Int64.mul 4L (Int64.of_int length);
          temporary_bytes;
          rounded = !rounded;
        } )

  let regression_target y =
    let source = Target.regression_values y in
    let* values, report =
      vector ~name:"target" ~positive:false ~temporary_bytes:0L
        (Target.length y) (Vector.get source)
    in
    Ok { target_values = values; target_conversion = report }

  let encoded_classification_target ~class_count y =
    if class_count < 2 || class_count > 16777217 then
      validation ~name:"class_count" "must be between 2 and 16777217"
    else
      let source = Target.classification_values y in
      let length = Array.length source in
      let rec check index =
        if index = length then Ok ()
        else if source.(index) < 0 || source.(index) >= class_count then
          validation
            ~name:(Format.sprintf "target[%d]" index)
            "encoded label is outside [0,class_count)"
        else check (index + 1)
      in
      let* () = check 0 in
      let temporary_bytes =
        Int64.mul (Int64.of_int (Sys.word_size / 8)) (Int64.of_int length)
      in
      let* values, report =
        vector ~name:"target" ~positive:false ~temporary_bytes length
          (fun index -> float_of_int source.(index))
      in
      Ok { target_values = values; target_conversion = report }

  let sample_weight source =
    let* values, report =
      vector ~name:"sample_weight" ~positive:true ~temporary_bytes:0L
        (Sample_weight.length source)
        (Sample_weight.get source)
    in
    Ok { weight_values = values; weight_conversion = report }
end

module Native = struct
  let protect operation f =
    try f ()
    with Xgboost.Xgboost_error native ->
      let kind =
        match native with
        | Xgboost.Error.Invalid_argument reason ->
            Error.Validation { name = operation; reason }
        | Xgboost.Error.Shape_mismatch
            { expected = rows, columns; got = got_rows, got_columns } ->
            Error.Shape_mismatch
              {
                name = operation;
                expected = [ rows; columns ];
                observed = [ got_rows; got_columns ];
              }
        | Xgboost.Error.Xgb_error reason ->
            Error.Numerical { operation; reason }
      in
      Error
        (Error.make ~context:[ Error.Stage operation ]
           ~remediation:
             "Check XGBoost input, native runtime, and parameter compatibility."
           kind)

  let version () =
    protect "XGBoost version" (fun () ->
        let major, minor, patch = Xgboost.version () in
        if major = 3 then Ok (major, minor, patch)
        else
          Error
            (Error.make ~remediation:"Use libxgboost 3.x with binding 0.2.1."
               (Error.Compatibility
                  {
                    component = "libxgboost";
                    reason = "supported native major version is 3";
                  })))

  let aligned ~name ~rows length =
    if length = rows then Ok ()
    else
      Error
        (Error.make
           ~remediation:"Align target and weight lengths with feature rows."
           (Error.Shape_mismatch
              { name; expected = [ rows ]; observed = [ length ] }))

  let with_dmatrix ?target ?sample_weight (features : Input.features) callback =
    let rows = Bigarray.Array2.dim1 features.Input.feature_matrix in
    let* () =
      match target with
      | None -> Ok ()
      | Some (target : Input.target) ->
          aligned ~name:"target" ~rows
            (Bigarray.Array1.dim target.Input.target_values)
    in
    let* () =
      match sample_weight with
      | None -> Ok ()
      | Some (weights : Input.weights) ->
          aligned ~name:"sample_weight" ~rows
            (Bigarray.Array1.dim weights.Input.weight_values)
    in
    let* _ = version () in
    protect "XGBoost DMatrix" (fun () ->
        Xgboost.DMatrix.with_
          (fun () -> Xgboost.DMatrix.of_bigarray2 features.Input.feature_matrix)
          (fun matrix ->
            Option.iter
              (fun (target : Input.target) ->
                Xgboost.DMatrix.set_label matrix target.Input.target_values)
              target;
            Option.iter
              (fun (weights : Input.weights) ->
                Xgboost.DMatrix.set_weight matrix weights.Input.weight_values)
              sample_weight;
            callback matrix))

  let with_booster ?cache ~specification ~rng callback =
    let* _ = version () in
    protect "XGBoost booster" (fun () ->
        Xgboost.Booster.with_ ?cache (fun booster ->
            Xgboost.Booster.set_params booster
              (Specification.native_parameters specification ~rng);
            callback booster))
end
