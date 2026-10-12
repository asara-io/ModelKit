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

  let native_seed rng =
    let bits, _ = Rng.next_int64 rng in
    Int64.logand bits 0x7fffffffL

  let native_parameters specification ~rng =
    let seed = Int64.to_string (native_seed rng) in
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

  let features_internal ~allow_empty ~feature_schema x =
    let* () = Feature_schema.validate_matrix feature_schema x |> map_data in
    let rows, columns = Matrix.shape x in
    if (rows = 0 && not allow_empty) || columns = 0 then
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

  let features ~feature_schema x =
    features_internal ~allow_empty:false ~feature_schema x

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

  let validate_native_model ~columns ~rounds booster =
    if Xgboost.Booster.num_features booster <> columns then
      validation ~name:"XGBoost fitted features"
        "native feature count differs from the training schema"
    else if Xgboost.Booster.boosted_rounds booster <> rounds then
      validation ~name:"XGBoost fitted rounds"
        "native boosting rounds differ from the specification"
    else Ok ()

  let detached_booster specification ~columns bytes =
    protect "XGBoost fitted model" (fun () ->
        let booster = Xgboost.Booster.create () in
        let transferred = ref false in
        Fun.protect
          ~finally:(fun () ->
            if not !transferred then Xgboost.Booster.free booster)
          (fun () ->
            Xgboost.Booster.load_model_buffer booster bytes;
            Xgboost.Booster.set_params booster
              [
                ("device", "cpu");
                ( "nthread",
                  string_of_int
                    (Specification.params specification).Specification.threads
                );
              ];
            let* () =
              validate_native_model ~columns
                ~rounds:
                  (Specification.params specification).Specification.rounds
                booster
            in
            transferred := true;
            Ok booster))

  let train_booster ~parameters ~specification ~rng features target weights =
    let columns = Bigarray.Array2.dim2 features.Input.feature_matrix in
    let rounds = (Specification.params specification).Specification.rounds in
    let* bytes =
      with_dmatrix ~target ?sample_weight:weights features (fun matrix ->
          with_booster ~cache:[ matrix ] ~specification ~rng (fun booster ->
              Xgboost.Booster.set_params booster parameters;
              for iter = 0 to rounds - 1 do
                Xgboost.Booster.update_one_iter booster ~iter ~dtrain:matrix
              done;
              let* () = validate_native_model ~columns ~rounds booster in
              Ok (Xgboost.Booster.save_model_buffer booster)))
    in
    (* A reloaded model has no training cache or references to training buffers. *)
    detached_booster specification ~columns bytes

  let prediction_values ~operation ~booster ~mutex features =
    protect operation (fun () ->
        Mutex.lock mutex;
        let values =
          Fun.protect
            ~finally:(fun () -> Mutex.unlock mutex)
            (fun () ->
              let values =
                Xgboost.Booster.predict_dense booster
                  features.Input.feature_matrix
              in
              (* The binding's output copy must finish before native finalization. *)
              ignore (Sys.opaque_identity booster);
              values)
        in
        Ok values)
end

module Regressor_implementation = struct
  type t = Specification.t
  type params = Specification.params
  type target = Target.regression Target.t
  type prediction = target
  type rng = Rng.t

  type report = {
    rounds : int;
    native_version : int * int * int;
    native_seed : int64;
    feature_conversion : Input.report;
    target_conversion : Input.report;
    weight_conversion : Input.report option;
  }

  type fitted = {
    specification : t;
    schema : Feature_schema.t;
    booster : Xgboost.Booster.t;
    prediction_mutex : Mutex.t;
    fit_report : report;
  }

  let clone = Specification.clone
  let params = Specification.params
  let fitted_params fitted = Specification.params fitted.specification
  let feature_schema fitted = fitted.schema
  let report fitted = fitted.fit_report
  let capabilities = Capability.estimator ~sample_weight:Capability.Supported ()

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = Feature_schema.validate_matrix feature_schema x |> map_data in
    let rows = Matrix.rows x in
    let* () = Native.aligned ~name:"target" ~rows (Target.length y) in
    let* () =
      match sample_weight with
      | None -> Ok ()
      | Some weights ->
          Native.aligned ~name:"sample_weight" ~rows
            (Sample_weight.length weights)
    in
    let* native_version = Native.version () in
    let* features = Input.features ~feature_schema x in
    let* target = Input.regression_target y in
    let* weights =
      match sample_weight with
      | None -> Ok None
      | Some weights -> Input.sample_weight weights |> Result.map Option.some
    in
    let rounds = (Specification.params specification).Specification.rounds in
    let* booster =
      Native.train_booster
        ~parameters:[ ("objective", "reg:squarederror") ]
        ~specification ~rng features target weights
    in
    Ok
      {
        specification = clone specification;
        schema = feature_schema;
        booster;
        prediction_mutex = Mutex.create ();
        fit_report =
          {
            rounds;
            native_version;
            native_seed = Specification.native_seed rng;
            feature_conversion = Input.feature_report features;
            target_conversion = Input.target_report target;
            weight_conversion = Option.map Input.weight_report weights;
          };
      }

  let predict_with_report fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make
           ~remediation:"Pass the fitted feature names in their original order."
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      let* features =
        Input.features_internal ~allow_empty:true ~feature_schema x
      in
      let rows = Matrix.rows x in
      let* prediction =
        if rows = 0 then Target.regression (Vector.of_array [||]) |> map_data
        else
          Native.protect "XGBoost regression prediction" (fun () ->
              let* values =
                Native.prediction_values
                  ~operation:"XGBoost regression prediction"
                  ~booster:fitted.booster ~mutex:fitted.prediction_mutex
                  features
              in
              let* () =
                Native.aligned ~name:"prediction" ~rows
                  (Bigarray.Array1.dim values)
              in
              let rec finite row =
                if row = rows then Ok ()
                else if Float.is_finite values.{row} then finite (row + 1)
                else
                  Error
                    (Error.make
                       ~remediation:
                         "Rescale extreme targets or regularize the fitted \
                          model."
                       (Error.Numerical
                          {
                            operation = "XGBoost regression prediction";
                            reason =
                              Format.sprintf "nonfinite prediction at row %d"
                                row;
                          }))
              in
              let* () = finite 0 in
              let* vector =
                Vector.init ~length:rows (fun row -> values.{row}) |> map_data
              in
              Target.regression vector |> map_data)
      in
      Ok (prediction, Input.feature_report features)

  let predict fitted ~feature_schema ~x =
    predict_with_report fitted ~feature_schema ~x |> Result.map fst
end

module Regressor = struct
  include Regressor_implementation

  let estimator ?(name = "xgboost") specification =
    let* provenance =
      Pipeline.provenance ~package:"modelkit-xgboost"
        ~version:Modelkit_xgboost_version.value ~implementation:"Regressor"
    in
    Pipeline.estimator ~provenance ~name
      (module Regressor_implementation)
      specification
end

module Classifier_implementation = struct
  type t = Specification.t
  type params = Specification.params
  type target = Target.classification Target.t
  type prediction = target
  type rng = Rng.t

  type report = {
    rounds : int;
    native_version : int * int * int;
    native_seed : int64;
    class_count : int;
    training_rows : int;
    effective_rows : int;
    feature_conversion : Input.report;
    target_conversion : Input.report;
    weight_conversion : Input.report option;
  }

  type fitted = {
    specification : t;
    schema : Feature_schema.t;
    class_labels : int array;
    booster : Xgboost.Booster.t;
    prediction_mutex : Mutex.t;
    fit_report : report;
  }

  let clone = Specification.clone
  let params = Specification.params
  let fitted_params fitted = Specification.params fitted.specification
  let feature_schema fitted = fitted.schema
  let classes fitted = Array.copy fitted.class_labels
  let report fitted = fitted.fit_report

  let capabilities =
    Capability.estimator ~sample_weight:Capability.Supported
      ~predict_proba:Capability.Supported ()

  let positive sample_weight row =
    match sample_weight with
    | None -> true
    | Some weights -> Sample_weight.get weights row > 0.

  let encoding sample_weight labels =
    let distinct = Hashtbl.create 16 in
    let count = ref 0 in
    Array.iteri
      (fun row label ->
        if positive sample_weight row then (
          incr count;
          Hashtbl.replace distinct label ()))
      labels;
    let class_labels = Hashtbl.to_seq_keys distinct |> Array.of_seq in
    Array.sort Int.compare class_labels;
    let class_count = Array.length class_labels in
    if class_count < 2 then
      validation ~name:"XGBoost classification classes"
        "at least two positively weighted classes are required"
    else if class_count > 16777217 then
      validation ~name:"XGBoost classification classes"
        "encoded labels must be exactly representable as float32"
    else
      let lookup = Hashtbl.create class_count in
      Array.iteri
        (fun index label -> Hashtbl.add lookup label index)
        class_labels;
      let selected =
        if !count = Array.length labels then None
        else
          let rows = Array.make !count 0 in
          let destination = ref 0 in
          Array.iteri
            (fun row _ ->
              if positive sample_weight row then (
                rows.(!destination) <- row;
                incr destination))
            labels;
          Some rows
      in
      Ok (class_labels, lookup, selected, !count)

  let selected_row selected row =
    match selected with None -> row | Some rows -> rows.(row)

  let select_features selected (features : Input.features) =
    match selected with
    | None -> features
    | Some rows ->
        let columns = Bigarray.Array2.dim2 features.Input.feature_matrix in
        let matrix =
          Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout
            (Array.length rows) columns
        in
        Array.iteri
          (fun destination source ->
            for column = 0 to columns - 1 do
              matrix.{destination, column} <-
                features.Input.feature_matrix.{source, column}
            done)
          rows;
        let previous = features.Input.feature_conversion in
        {
          Input.feature_matrix = matrix;
          schema = features.Input.schema;
          feature_conversion =
            {
              previous with
              Input.bytes =
                Int64.mul 4L
                  (Int64.mul
                     (Int64.of_int (Array.length rows))
                     (Int64.of_int columns));
              temporary_bytes =
                Int64.add previous.Input.temporary_bytes previous.Input.bytes;
            };
        }

  let select_weights selected (weights : Input.weights) =
    match selected with
    | None -> weights
    | Some rows ->
        let values =
          Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout
            (Array.length rows)
        in
        Array.iteri
          (fun destination source ->
            values.{destination} <- weights.Input.weight_values.{source})
          rows;
        let previous = weights.Input.weight_conversion in
        {
          Input.weight_values = values;
          weight_conversion =
            {
              previous with
              Input.bytes = Int64.mul 4L (Int64.of_int (Array.length rows));
              temporary_bytes =
                Int64.add previous.Input.temporary_bytes previous.Input.bytes;
            };
        }

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = Feature_schema.validate_matrix feature_schema x |> map_data in
    let training_rows = Matrix.rows x in
    let* () =
      Native.aligned ~name:"target" ~rows:training_rows (Target.length y)
    in
    let* () =
      match sample_weight with
      | None -> Ok ()
      | Some weights ->
          Native.aligned ~name:"sample_weight" ~rows:training_rows
            (Sample_weight.length weights)
    in
    let labels = Target.classification_values y in
    let* class_labels, lookup, selected, effective_rows =
      encoding sample_weight labels
    in
    let* native_version = Native.version () in
    let* features = Input.features ~feature_schema x in
    let features = select_features selected features in
    let row_bytes = Int64.of_int (Sys.word_size / 8) in
    let selection_bytes =
      match selected with
      | None -> 0L
      | Some rows -> Int64.mul row_bytes (Int64.of_int (Array.length rows))
    in
    let temporary_bytes =
      Int64.add selection_bytes
        (Int64.mul row_bytes (Int64.of_int training_rows))
    in
    let* values, conversion =
      Input.vector ~name:"classification_target" ~positive:false
        ~temporary_bytes effective_rows (fun row ->
          float_of_int (Hashtbl.find lookup labels.(selected_row selected row)))
    in
    let target =
      {
        Input.target_values = values;
        target_conversion =
          { conversion with Input.shape = [| training_rows |] };
      }
    in
    let* weights =
      match sample_weight with
      | None -> Ok None
      | Some weights ->
          let* converted = Input.sample_weight weights in
          Ok (Some (select_weights selected converted))
    in
    let class_count = Array.length class_labels in
    let parameters =
      if class_count = 2 then [ ("objective", "binary:logistic") ]
      else
        [
          ("objective", "multi:softprob");
          ("num_class", string_of_int class_count);
        ]
    in
    let* booster =
      Native.train_booster ~parameters ~specification ~rng features target
        weights
    in
    Ok
      {
        specification = clone specification;
        schema = feature_schema;
        class_labels;
        booster;
        prediction_mutex = Mutex.create ();
        fit_report =
          {
            rounds = (Specification.params specification).Specification.rounds;
            native_version;
            native_seed = Specification.native_seed rng;
            class_count;
            training_rows;
            effective_rows;
            feature_conversion = Input.feature_report features;
            target_conversion = Input.target_report target;
            weight_conversion = Option.map Input.weight_report weights;
          };
      }

  let probability_error reason =
    Error
      (Error.make
         ~remediation:
           "Use a tested native runtime and numerically representable weights."
         (Error.Numerical
            { operation = "XGBoost classification probabilities"; reason }))

  let prediction_values fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make
           ~remediation:"Pass the fitted feature names in their original order."
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      let rows = Matrix.rows x in
      let count = Array.length fitted.class_labels in
      if rows > max_int / count then
        validation ~name:"classification prediction dimensions"
          "rows times classes must fit OCaml int"
      else
        let* features =
          Input.features_internal ~allow_empty:true ~feature_schema x
        in
        let width = if count = 2 then 1 else count in
        let* values =
          if rows = 0 then
            Ok (Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 0)
          else
            Native.prediction_values
              ~operation:"XGBoost classification prediction"
              ~booster:fitted.booster ~mutex:fitted.prediction_mutex features
        in
        let* () =
          Native.aligned ~name:"classification probabilities"
            ~rows:(rows * width)
            (Bigarray.Array1.dim values)
        in
        let rec check row column total =
          if row = rows then Ok ()
          else if column = width then
            if width = 1 || Float.abs (total -. 1.) <= 1e-6 then
              check (row + 1) 0 0.
            else
              probability_error
                (Format.sprintf "row %d probability sum is %.17g" row total)
          else
            let value = values.{(row * width) + column} in
            if (not (Float.is_finite value)) || value < 0. || value > 1. then
              probability_error
                (Format.sprintf "invalid probability at row %d column %d" row
                   column)
            else check row (column + 1) (total +. value)
        in
        let* () = check 0 0 0. in
        Ok (values, Input.feature_report features)

  let predict_proba_with_report fitted ~feature_schema ~x =
    let* values, conversion = prediction_values fitted ~feature_schema ~x in
    let rows = Matrix.rows x in
    let count = Array.length fitted.class_labels in
    let totals =
      if count = 2 then [||]
      else
        Array.init rows (fun row ->
            let total = ref 0. in
            for column = 0 to count - 1 do
              total := !total +. values.{(row * count) + column}
            done;
            !total)
    in
    let* probabilities =
      Matrix.init ~rows ~columns:count (fun row column ->
          if count = 2 then
            let positive = values.{row} in
            if column = 0 then 1. -. positive else positive
          else values.{(row * count) + column} /. totals.(row))
      |> map_data
    in
    Ok (probabilities, conversion)

  let predict_proba fitted ~feature_schema ~x =
    predict_proba_with_report fitted ~feature_schema ~x |> Result.map fst

  let predict fitted ~feature_schema ~x =
    let* values, _ = prediction_values fitted ~feature_schema ~x in
    let count = Array.length fitted.class_labels in
    let labels =
      Array.init (Matrix.rows x) (fun row ->
          let index =
            if count = 2 then if values.{row} > 0.5 then 1 else 0
            else
              let best = ref 0 in
              for column = 1 to count - 1 do
                if
                  values.{(row * count) + column}
                  > values.{(row * count) + !best}
                then best := column
              done;
              !best
          in
          fitted.class_labels.(index))
    in
    Ok (Target.classification labels)
end

module Classifier = struct
  include Classifier_implementation

  let estimator ?(name = "xgboost") specification =
    let* provenance =
      Pipeline.provenance ~package:"modelkit-xgboost"
        ~version:Modelkit_xgboost_version.value ~implementation:"Classifier"
    in
    Pipeline.estimator ~provenance ~name
      (module Classifier_implementation)
      ~predict_proba ~classes specification
end
