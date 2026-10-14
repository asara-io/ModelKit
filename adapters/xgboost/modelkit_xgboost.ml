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

module Origin = struct
  type t = {
    fitted_adapter_version : string;
    writer_adapter_version : string;
    binding_version : string;
    source_word_size : int;
    model_native_version : int * int * int;
    runtime_native_version : int * int * int;
    inference_threads : int;
  }
end

type model_origin = Origin.t

let fitted_origin specification native_version =
  {
    Origin.fitted_adapter_version = Modelkit_xgboost_version.value;
    writer_adapter_version = Modelkit_xgboost_version.value;
    binding_version = "0.2.1";
    source_word_size = Sys.word_size;
    model_native_version = native_version;
    runtime_native_version = native_version;
    inference_threads =
      (Specification.params specification).Specification.threads;
  }

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

  let with_loaded_model ~threads bytes callback =
    let* () =
      Specification.integer ~name:"inference_threads" ~minimum:1 threads
    in
    let* _ = version () in
    protect "XGBoost loaded model" (fun () ->
        let booster = Xgboost.Booster.create () in
        let retained = ref false in
        Fun.protect
          ~finally:(fun () ->
            if not !retained then Xgboost.Booster.free booster)
          (fun () ->
            let configure () =
              Xgboost.Booster.set_params booster
                [ ("device", "cpu"); ("nthread", string_of_int threads) ]
            in
            configure ();
            Xgboost.Booster.load_model_buffer booster bytes;
            configure ();
            let* value, keep = callback booster in
            retained := keep;
            Ok value))

  let with_model_buffer ?(threads = 1) bytes callback =
    with_loaded_model ~threads bytes (fun booster ->
        Result.map (fun value -> (value, false)) (callback booster))

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

module Batches = struct
  type batch = {
    feature_schema : Feature_schema.t;
    x : Matrix.t;
    y : Target.regression Target.t;
    sample_weight : Vector.t option;
  }

  type report = {
    batches : int;
    rows : int;
    max_batch_rows : int;
    resets : int;
    passes : int;
    peak_feature_buffer_bytes : int64;
    peak_staging_buffer_bytes : int64;
    replay_fingerprint_bytes : int64;
  }

  let callback operation f =
    try f () |> Result.map_error (Error.with_context (Error.Stage operation))
    with exn ->
      Error
        (Error.make ~context:[ Error.Stage operation ]
           ~remediation:
             "Return typed callback errors and replay the same batches on \
              reset."
           (Error.Callback_failure { reason = Printexc.to_string exn }))

  let io f =
    try f () with
    | End_of_file ->
        validation ~name:"streaming disk cache"
          "the replay fingerprint manifest ended unexpectedly"
    | Sys_error reason ->
        Error
          (Error.make
             ~remediation:
               "Provide a writable cache directory with sufficient disk space."
             (Error.Validation { name = "streaming disk cache"; reason }))
    | Unix.Unix_error (error, operation, _) ->
        Error
          (Error.make ~context:[ Error.Stage operation ]
             ~remediation:
               "Provide a writable cache directory with sufficient disk space."
             (Error.Validation
                {
                  name = "streaming disk cache";
                  reason = Unix.error_message error;
                }))

  let with_cache directory callback =
    io (fun () ->
        let random = Random.State.make_self_init () in
        let rec create attempts =
          if attempts = 0 then
            raise
              (Unix.Unix_error
                 (Unix.EEXIST, "streaming cache reservation", directory))
          else
            let path =
              Filename.concat directory
                (Printf.sprintf "modelkit-xgboost-%d-%08x" (Unix.getpid ())
                   (Random.State.bits random))
            in
            try
              Unix.mkdir path 0o700;
              path
            with Unix.Unix_error (Unix.EEXIST, _, _) -> create (attempts - 1)
        in
        let path = create 32 in
        let cleanup = ref (Ok ()) in
        let outcome =
          Fun.protect
            ~finally:(fun () ->
              cleanup :=
                io (fun () ->
                    Array.iter
                      (fun name -> Unix.unlink (Filename.concat path name))
                      (Sys.readdir path);
                    Unix.rmdir path;
                    Ok ()))
            (fun () -> io (fun () -> callback path))
        in
        match (outcome, !cleanup) with
        | Error error, Error cleanup_error ->
            Error
              (Error.with_context
                 (Error.Stage
                    ("streaming cache cleanup: " ^ Error.to_string cleanup_error))
                 error)
        | Error error, Ok () -> Error error
        | Ok _, Error error -> Error error
        | Ok value, Ok () -> Ok value)

  let fingerprint batch =
    let rows, columns = Matrix.shape batch.x in
    let weighted = Option.is_some batch.sample_weight in
    let width = columns + 1 + if weighted then 1 else 0 in
    let length = (rows * width * 8) + 17 in
    let bytes = Bytes.create length in
    Bytes.set_int64_le bytes 0 (Int64.of_int rows);
    Bytes.set_int64_le bytes 8 (Int64.of_int columns);
    Bytes.set bytes 16 (if weighted then '\001' else '\000');
    let target = Target.regression_values batch.y in
    let index = ref 17 in
    let write value =
      let bits =
        if Float.is_nan value then Int64.bits_of_float Float.nan
        else Int64.bits_of_float value
      in
      Bytes.set_int64_le bytes !index bits;
      index := !index + 8
    in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        write (Matrix.get batch.x row column)
      done;
      write (Vector.get target row);
      Option.iter
        (fun weights -> write (Vector.get weights row))
        batch.sample_weight
    done;
    (Digest.BLAKE256.bytes bytes, Int64.of_int length)

  let prepare schema batch =
    if not (Feature_schema.equal schema batch.feature_schema) then
      Error
        (Error.make
           ~remediation:"Keep the same ordered named schema in every batch."
           (Error.Feature_schema_mismatch
              { expected = schema; observed = batch.feature_schema }))
    else
      let rows, columns = Matrix.shape batch.x in
      let* () =
        Native.aligned ~name:"batch target" ~rows (Target.length batch.y)
      in
      let* () =
        match batch.sample_weight with
        | None -> Ok ()
        | Some weights ->
            Native.aligned ~name:"batch sample_weight" ~rows
              (Vector.length weights)
      in
      if
        columns > ((max_int - 17) / 8) - 2
        || rows > (max_int - 17) / ((columns + 2) * 8)
      then
        validation ~name:"batch size"
          "conversion and fingerprint payload exceed the addressable size"
      else
        let* features = Input.features ~feature_schema:schema batch.x in
        let* target = Input.regression_target batch.y in
        let* weights =
          match batch.sample_weight with
          | None -> Ok None
          | Some source ->
              let rec nonnegative index =
                if index = rows then Ok ()
                else if Vector.get source index < 0. then
                  validation ~name:"batch sample_weight"
                    "weights must be nonnegative"
                else nonnegative (index + 1)
              in
              let* () = nonnegative 0 in
              let* values, report =
                Input.vector ~name:"sample_weight" ~positive:true
                  ~temporary_bytes:0L rows (Vector.get source)
              in
              Ok
                (Some
                   { Input.weight_values = values; weight_conversion = report })
        in
        let digest, fingerprint_bytes = fingerprint batch in
        Ok
          ( {
              Modelkit_xgboost_streaming_native.features =
                features.Input.feature_matrix;
              labels = target.Input.target_values;
              weights =
                Option.map (fun weight -> weight.Input.weight_values) weights;
            },
            digest,
            fingerprint_bytes,
            features.Input.feature_conversion,
            target.Input.target_conversion,
            Option.map Input.weight_report weights )

  let train specification ~rng ~feature_schema ~expected_rows ~cache_directory
      ~next ~reset =
    let columns = Feature_schema.feature_count feature_schema in
    let* () =
      if expected_rows < 1 || columns < 1 then
        validation ~name:"expected_rows"
          "training requires positive row and feature counts"
      else if Option.is_none (Feature_schema.names feature_schema) then
        validation ~name:"feature_schema"
          "batch training requires named features"
      else if
        columns > ((max_int - 17) / 8) - 2
        || expected_rows > max_int / columns
        || Int64.of_int expected_rows > Int64.div Int64.max_int 32L
        || Int64.of_int expected_rows
           > Int64.div Int64.max_int (Int64.mul 8L (Int64.of_int (columns + 2)))
      then
        validation ~name:"expected_rows" "conversion accounting would overflow"
      else if cache_directory = "" || String.contains cache_directory '\000'
      then
        validation ~name:"cache_directory"
          "provide an existing cache directory without NUL characters"
      else Ok ()
    in
    let batches = ref 0
    and max_batch_rows = ref 0
    and resets = ref 0
    and passes = ref 0 in
    let peak_features = ref 0L and peak_staging = ref 0L in
    let feature_rounded = ref 0
    and target_rounded = ref 0
    and weight_rounded = ref 0 in
    let weighted = ref None and has_positive_weight = ref false in
    let invoke_reset () =
      incr resets;
      callback "batch reset" reset
    in
    let accept batch consumed =
      let rows = Matrix.rows batch.x in
      if rows > expected_rows - consumed then
        validation ~name:"batch row count"
          "batches exceed the expected total row count"
      else
        let present = Option.is_some batch.sample_weight in
        let* () =
          match !weighted with
          | None ->
              weighted := Some present;
              Ok ()
          | Some expected when expected = present -> Ok ()
          | Some _ ->
              validation ~name:"batch sample_weight"
                "weight presence must be consistent across batches and resets"
        in
        prepare feature_schema batch
        |> Result.map_error
             (Error.with_context
                (Error.Stage
                   (Printf.sprintf "batch beginning at row %d" consumed)))
    in
    let* bytes =
      with_cache cache_directory (fun directory ->
          let manifest = Filename.concat directory "replay.fingerprints" in
          let output = open_out_bin manifest in
          let* () =
            Fun.protect
              ~finally:(fun () -> close_out_noerr output)
              (fun () ->
                let* () = invoke_reset () in
                let rec collect consumed =
                  let* batch = callback "batch next" next in
                  match batch with
                  | None ->
                      if consumed <> expected_rows then
                        validation ~name:"batch row count"
                          "end of input precedes the expected total row count"
                      else if !weighted = Some true && not !has_positive_weight
                      then
                        validation ~name:"batch sample_weight"
                          "the dataset must contain a positive weight"
                      else (
                        incr passes;
                        flush output;
                        close_out output;
                        Ok ())
                  | Some batch ->
                      let* ( _,
                             digest,
                             fingerprint_bytes,
                             feature,
                             target,
                             weights ) =
                        accept batch consumed
                      in
                      let rows = Matrix.rows batch.x in
                      incr batches;
                      max_batch_rows := max !max_batch_rows rows;
                      feature_rounded :=
                        !feature_rounded + feature.Input.rounded;
                      target_rounded := !target_rounded + target.Input.rounded;
                      let weight_bytes =
                        match weights with
                        | None -> 0L
                        | Some report ->
                            weight_rounded :=
                              !weight_rounded + report.Input.rounded;
                            Option.iter
                              (fun weights ->
                                for row = 0 to rows - 1 do
                                  if Vector.get weights row > 0. then
                                    has_positive_weight := true
                                done)
                              batch.sample_weight;
                            report.Input.bytes
                      in
                      peak_features :=
                        Int64.max !peak_features feature.Input.bytes;
                      peak_staging :=
                        Int64.max !peak_staging
                          (Int64.add fingerprint_bytes
                             (Int64.add feature.Input.bytes
                                (Int64.add target.Input.bytes weight_bytes)));
                      output_string output digest;
                      collect (consumed + rows)
                in
                collect 0)
          in
          let input = open_in_bin manifest in
          Fun.protect
            ~finally:(fun () -> close_in_noerr input)
            (fun () ->
              let consumed = ref 0 and ended = ref true and delivered = ref 0 in
              let replay_reset () =
                io (fun () ->
                    if !consumed <> 0 && not !ended then
                      validation ~name:"batch reset"
                        "reset interrupted an incomplete dataset pass"
                    else
                      let* () = invoke_reset () in
                      consumed := 0;
                      delivered := 0;
                      ended := false;
                      seek_in input 0;
                      Ok ())
              in
              let replay_next () =
                io (fun () ->
                    let* batch = callback "batch next" next in
                    match batch with
                    | None ->
                        if !consumed <> expected_rows || !delivered <> !batches
                        then
                          validation ~name:"batch replay"
                            "reset did not reproduce the complete ordered \
                             dataset"
                        else (
                          if not !ended then incr passes;
                          ended := true;
                          Ok None)
                    | Some batch ->
                        let* native, digest, _, _, _, _ =
                          accept batch !consumed
                        in
                        if !ended || !delivered >= !batches then
                          validation ~name:"batch replay"
                            "replay supplied unexpected batches"
                        else
                          let expected = really_input_string input 32 in
                          if digest <> expected then
                            validation ~name:"batch replay"
                              "reset changed batch boundaries, values, \
                               targets, weights, or row order"
                          else (
                            consumed := !consumed + Matrix.rows batch.x;
                            incr delivered;
                            Ok (Some native)))
              in
              let p = Specification.params specification in
              Modelkit_xgboost_streaming_native.train
                ~parameters:
                  (("objective", "reg:squarederror")
                  :: Specification.native_parameters specification ~rng)
                ~rounds:p.Specification.rounds ~threads:p.Specification.threads
                ~rows:expected_rows ~columns
                ~cache_prefix:(Filename.concat directory "pages")
                ~next:replay_next ~reset:replay_reset ()))
    in
    let conversion shape rounded =
      let count =
        Array.fold_left
          (fun total value -> Int64.mul total (Int64.of_int value))
          1L shape
      in
      { Input.shape; bytes = Int64.mul 4L count; temporary_bytes = 0L; rounded }
    in
    Ok
      ( bytes,
        {
          batches = !batches;
          rows = expected_rows;
          max_batch_rows = !max_batch_rows;
          resets = !resets;
          passes = !passes;
          peak_feature_buffer_bytes = !peak_features;
          peak_staging_buffer_bytes = !peak_staging;
          replay_fingerprint_bytes = Int64.mul 32L (Int64.of_int !batches);
        },
        conversion [| expected_rows; columns |] !feature_rounded,
        conversion [| expected_rows |] !target_rounded,
        if !weighted = Some true then
          Some (conversion [| expected_rows |] !weight_rounded)
        else None )
end

module Regressor_implementation = struct
  type t = Specification.t
  type params = Specification.params
  type target = Target.regression Target.t
  type prediction = target
  type rng = Rng.t

  type batch = Batches.batch = {
    feature_schema : Feature_schema.t;
    x : Matrix.t;
    y : Target.regression Target.t;
    sample_weight : Vector.t option;
  }

  type batches_report = Batches.report = {
    batches : int;
    rows : int;
    max_batch_rows : int;
    resets : int;
    passes : int;
    peak_feature_buffer_bytes : int64;
    peak_staging_buffer_bytes : int64;
    replay_fingerprint_bytes : int64;
  }

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
    batches_report : batches_report option;
    origin : model_origin;
  }

  let clone = Specification.clone
  let params = Specification.params
  let fitted_params fitted = Specification.params fitted.specification
  let feature_schema fitted = fitted.schema
  let report fitted = fitted.fit_report
  let batches_report fitted = fitted.batches_report
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
        origin = fitted_origin specification native_version;
        batches_report = None;
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

  let fit_batches specification ~rng ~feature_schema ~expected_rows
      ~cache_directory ~next ~reset () =
    let* native_version = Native.version () in
    let* ( bytes,
           batches_report,
           feature_conversion,
           target_conversion,
           weight_conversion ) =
      Batches.train specification ~rng ~feature_schema ~expected_rows
        ~cache_directory ~next ~reset
    in
    let* booster =
      Native.detached_booster specification
        ~columns:(Feature_schema.feature_count feature_schema)
        bytes
    in
    Ok
      {
        specification = clone specification;
        schema = feature_schema;
        booster;
        prediction_mutex = Mutex.create ();
        origin = fitted_origin specification native_version;
        batches_report = Some batches_report;
        fit_report =
          {
            rounds = (Specification.params specification).Specification.rounds;
            native_version;
            native_seed = Specification.native_seed rng;
            feature_conversion;
            target_conversion;
            weight_conversion;
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
    origin : model_origin;
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
        origin = fitted_origin specification native_version;
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

module Artifact = struct
  module Io = Modelkit_xgboost_io
  module W = Io.Writer
  module R = Io.Reader

  type limits = Io.limits

  let default_limits = Io.default_limits
  let limits = Io.limits
  let failure = Io.failure

  let compatibility component reason =
    Error
      (Error.make
         ~remediation:
           "Use binding 0.2.1 and a tested libxgboost 3.x runtime at least as \
            new as the model writer."
         (Error.Compatibility { component; reason }))

  type provenance = {
    fitted_adapter_version : string;
    writer_adapter_version : string;
    binding_version : string;
    training_native_version : int * int * int;
    model_native_version : int * int * int;
    runtime_native_version : int * int * int;
    source_word_size : int;
    inference_threads : int;
    prediction_begin : int;
    prediction_end : int;
  }

  let provenance (origin : model_origin) training_native_version rounds =
    {
      fitted_adapter_version = origin.Origin.fitted_adapter_version;
      writer_adapter_version = origin.Origin.writer_adapter_version;
      binding_version = origin.Origin.binding_version;
      training_native_version;
      model_native_version = origin.Origin.model_native_version;
      runtime_native_version = origin.Origin.runtime_native_version;
      source_word_size = origin.Origin.source_word_size;
      inference_threads = origin.Origin.inference_threads;
      prediction_begin = 0;
      prediction_end = rounds;
    }

  let regression_provenance fitted =
    provenance fitted.Regressor_implementation.origin
      fitted.Regressor_implementation.fit_report
        .Regressor_implementation.native_version
      fitted.Regressor_implementation.fit_report.Regressor_implementation.rounds

  let classification_provenance fitted =
    provenance fitted.Classifier_implementation.origin
      fitted.Classifier_implementation.fit_report
        .Classifier_implementation.native_version
      fitted.Classifier_implementation.fit_report
        .Classifier_implementation.rounds

  type stored = {
    task : int;
    specification : Specification.t;
    schema : Feature_schema.t;
    classes : int array;
    origin : model_origin;
    training_native_version : int * int * int;
    native_seed : int64;
    training_rows : int;
    effective_rows : int;
    feature_conversion : Input.report;
    target_conversion : Input.report;
    weight_conversion : Input.report option;
    batches_report : Batches.report option;
    model : bytes;
  }

  let objective task =
    match task with
    | 1 -> "reg:squarederror"
    | 2 -> "binary:logistic"
    | 3 -> "multi:softprob"
    | _ -> assert false

  let text limits value =
    if
      String.length value > limits.Io.max_string_bytes
      || not (String.is_valid_utf_8 value)
    then failure "model text exceeds its limit or is not UTF-8"
    else Ok ()

  let write_version writer (major, minor, patch) =
    W.int writer major;
    W.int writer minor;
    W.int writer patch

  let read_version reader =
    let* major = R.int reader in
    let* minor = R.int reader in
    let* patch = R.int reader in
    if major <> 3 then
      compatibility "XGBoost model"
        "supported native model major version is three"
    else if minor < 0 || patch < 0 then failure "invalid native version"
    else Ok (major, minor, patch)

  let write_ints writer values =
    W.int writer (Array.length values);
    Array.iter (W.int writer) values

  let write_schema ~limits writer schema =
    if
      Feature_schema.feature_count schema < 1
      || Feature_schema.feature_count schema > limits.Io.max_features
    then failure "feature count exceeds the model reader contract"
    else
      let () = W.int writer (Feature_schema.feature_count schema) in
      let* () =
        match Feature_schema.names schema with
        | None ->
            W.byte writer 0;
            Ok ()
        | Some names ->
            let values = Feature_names.to_array names in
            let rec check index =
              if index = Array.length values then Ok ()
              else
                let* () = text limits values.(index) in
                check (index + 1)
            in
            let* () = check 0 in
            W.byte writer 1;
            Array.iter (W.string writer) values;
            Ok ()
      in
      let fingerprint =
        Schema_fingerprint.to_string (Feature_schema.fingerprint schema)
      in
      let* () = text limits fingerprint in
      W.string writer fingerprint;
      Ok ()

  let read_schema ~limits reader =
    let* columns = R.length reader ~maximum:limits.Io.max_features in
    let* named = R.byte reader in
    let* schema =
      match named with
      | 0 -> Feature_schema.anonymous ~feature_count:columns |> map_data
      | 1 ->
          let* () =
            R.require reader
              (if columns > limits.Io.max_bytes / 8 then limits.Io.max_bytes
               else columns * 8)
          in
          let names = Array.make columns "" in
          let rec read index =
            if index = columns then Ok ()
            else
              let* name = R.string reader in
              names.(index) <- name;
              read (index + 1)
          in
          let* () = read 0 in
          let* names =
            Feature_names.create ~expected_count:columns names |> map_data
          in
          Ok (Feature_schema.named names)
      | _ -> failure "invalid model feature-schema tag"
    in
    let* fingerprint = R.string reader in
    if
      columns = 0
      || fingerprint
         <> Schema_fingerprint.to_string (Feature_schema.fingerprint schema)
    then failure "model feature-schema fingerprint is inconsistent"
    else Ok schema

  let write_specification writer specification =
    let p = Specification.params specification in
    List.iter (W.int writer)
      [
        p.Specification.rounds;
        p.Specification.max_depth;
        p.Specification.max_bin;
        p.Specification.threads;
      ];
    List.iter (W.float writer)
      [
        p.Specification.learning_rate;
        p.Specification.min_child_weight;
        p.Specification.min_split_loss;
        p.Specification.reg_alpha;
        p.Specification.reg_lambda;
        p.Specification.subsample;
        p.Specification.colsample_bytree;
      ]

  let read_specification reader =
    let* rounds = R.int reader in
    let* max_depth = R.int reader in
    let* max_bin = R.int reader in
    let* threads = R.int reader in
    let* learning_rate = R.float reader in
    let* min_child_weight = R.float reader in
    let* min_split_loss = R.float reader in
    let* reg_alpha = R.float reader in
    let* reg_lambda = R.float reader in
    let* subsample = R.float reader in
    let* colsample_bytree = R.float reader in
    Specification.create ~rounds ~max_depth ~max_bin ~threads ~learning_rate
      ~min_child_weight ~min_split_loss ~reg_alpha ~reg_lambda ~subsample
      ~colsample_bytree ()
    |> Result.map_error
         (Error.with_context (Error.Stage "stored XGBoost specification"))

  let write_conversion writer report =
    write_ints writer report.Input.shape;
    W.i64 writer report.Input.bytes;
    W.i64 writer report.Input.temporary_bytes;
    W.int writer report.Input.rounded

  let read_conversion reader =
    let* shape = R.ints reader ~maximum:2 in
    let* bytes = R.i64 reader in
    let* temporary_bytes = R.i64 reader in
    let* rounded = R.int reader in
    if bytes < 0L || temporary_bytes < 0L || rounded < 0 then
      failure "invalid stored conversion counts"
    else Ok { Input.shape; bytes; temporary_bytes; rounded }

  let validate_reports stored =
    let rows = stored.training_rows and effective = stored.effective_rows in
    let columns = Feature_schema.feature_count stored.schema in
    let values = Int64.of_int rows in
    if
      rows < 1 || effective < 1 || effective > rows
      || values > Int64.div Int64.max_int 16L
      || values > Int64.div Int64.max_int (Int64.mul 4L (Int64.of_int columns))
    then failure "invalid training/effective row counts"
    else
      let* () =
        match stored.batches_report with
        | None -> Ok ()
        | Some report ->
            let weighted = Option.is_some stored.weight_conversion in
            let width =
              Int64.add (Int64.of_int columns) (if weighted then 2L else 1L)
            in
            let batch_rows = Int64.of_int report.Batches.max_batch_rows in
            let feature_peak =
              Int64.mul 4L (Int64.mul batch_rows (Int64.of_int columns))
            in
            let staging_peak =
              Int64.add 17L (Int64.mul 12L (Int64.mul batch_rows width))
            in
            if
              stored.task <> 1
              || report.Batches.rows <> rows
              || report.Batches.batches < 1
              || report.Batches.batches > rows
              || Int64.of_int report.Batches.batches
                 > Int64.div Int64.max_int 32L
              || report.Batches.max_batch_rows < 1
              || report.Batches.max_batch_rows > rows
              || columns > ((max_int - 17) / 8) - 2
              || report.Batches.max_batch_rows
                 > (max_int - 17) / ((columns + 2) * 8)
              || report.Batches.max_batch_rows
                 < ((rows - 1) / report.Batches.batches) + 1
              || report.Batches.resets < 2 || report.Batches.passes < 2
              || report.Batches.passes > report.Batches.resets
              || report.Batches.peak_feature_buffer_bytes <> feature_peak
              || report.Batches.peak_staging_buffer_bytes <> staging_peak
              || report.Batches.replay_fingerprint_bytes
                 <> Int64.mul 32L (Int64.of_int report.Batches.batches)
            then failure "stored batch-training diagnostics are inconsistent"
            else Ok ()
      in
      let source_values = Int64.mul values (Int64.of_int columns) in
      let size count = Int64.mul 4L count in
      let filtered = effective <> rows in
      let feature_bytes =
        size (Int64.mul (Int64.of_int effective) (Int64.of_int columns))
      in
      let target_bytes = size (Int64.of_int effective) in
      let word = Int64.of_int (stored.origin.Origin.source_word_size / 8) in
      let target_temporary =
        if stored.task = 1 then 0L
        else
          Int64.mul word
            (Int64.add values (if filtered then Int64.of_int effective else 0L))
      in
      let feature = stored.feature_conversion
      and target = stored.target_conversion in
      if
        feature.Input.shape <> [| rows; columns |]
        || target.Input.shape <> [| rows |]
        || feature.Input.bytes <> feature_bytes
        || target.Input.bytes <> target_bytes
        || (feature.Input.temporary_bytes
           <> if filtered then size source_values else 0L)
        || target.Input.temporary_bytes <> target_temporary
        || Int64.of_int feature.Input.rounded > source_values
        || target.Input.rounded > rows
        || (stored.task <> 1 && target.Input.rounded <> 0)
        || (stored.task = 1 && filtered)
      then failure "stored training conversion reports are inconsistent"
      else
        match stored.weight_conversion with
        | None ->
            if filtered then
              failure "filtered training rows require a weight report"
            else Ok ()
        | Some report ->
            if
              report.Input.shape <> [| rows |]
              || report.Input.bytes <> target_bytes
              || (report.Input.temporary_bytes
                 <> if filtered then size values else 0L)
              || report.Input.rounded > rows
            then failure "stored weight conversion report is inconsistent"
            else Ok ()

  let validate_classes ~limits task classes effective =
    let count = Array.length classes in
    if
      (task = 1 && count <> 0)
      || (task = 2 && count <> 2)
      || (task = 3 && count < 3)
      || count > limits.Io.max_classes
      || count > 16777217 || count > effective
    then
      failure "class count is incompatible with the model task or reader limit"
    else
      let rec ordered index =
        if index >= count then Ok ()
        else if classes.(index - 1) >= classes.(index) then
          failure "model class mapping must be strictly increasing"
        else ordered (index + 1)
      in
      ordered 1

  let snapshot ~mutex booster =
    Native.protect "XGBoost serialization" (fun () ->
        Mutex.lock mutex;
        Fun.protect
          ~finally:(fun () -> Mutex.unlock mutex)
          (fun () ->
            let bytes =
              Xgboost.Booster.save_model_buffer ~format:{|{"format":"json"}|}
                booster
            in
            ignore (Sys.opaque_identity booster);
            Ok bytes))

  let encode_stored ~limits stored =
    let* () = validate_reports stored in
    let* () =
      validate_classes ~limits stored.task stored.classes stored.effective_rows
    in
    let* json =
      Io.Json.parse ~max_bytes:limits.Io.max_bytes
        (Bytes.to_string stored.model)
    in
    let* model_native_version = Io.Json.version json in
    let* () =
      Io.Json.check_model
        ~columns:(Feature_schema.feature_count stored.schema)
        ~classes:(Array.length stored.classes)
        ~rounds:(Specification.params stored.specification).Specification.rounds
        ~objective:(objective stored.task) ~version:model_native_version json
    in
    let writer = W.create () in
    let values =
      [
        "modelkit-xgboost";
        (if stored.task = 1 then "Regressor" else "Classifier");
        stored.origin.Origin.fitted_adapter_version;
        Modelkit_xgboost_version.value;
        stored.origin.Origin.binding_version;
        objective stored.task;
      ]
    in
    let rec validate_text = function
      | [] -> Ok ()
      | value :: rest ->
          let* () = text limits value in
          validate_text rest
    in
    let* () = validate_text values in
    W.string writer "modelkit-xgboost";
    W.string writer (if stored.task = 1 then "Regressor" else "Classifier");
    W.int writer (if Option.is_none stored.batches_report then 1 else 2);
    W.string writer stored.origin.Origin.fitted_adapter_version;
    W.string writer Modelkit_xgboost_version.value;
    W.string writer stored.origin.Origin.binding_version;
    write_version writer stored.training_native_version;
    write_version writer model_native_version;
    W.int writer stored.origin.Origin.source_word_size;
    W.byte writer 1;
    W.byte writer 1;
    W.byte writer 1;
    W.string writer (objective stored.task);
    W.int writer 0;
    W.int writer
      (Specification.params stored.specification).Specification.rounds;
    write_specification writer stored.specification;
    let* () = write_schema ~limits writer stored.schema in
    write_ints writer stored.classes;
    W.i64 writer stored.native_seed;
    W.int writer stored.training_rows;
    W.int writer stored.effective_rows;
    write_conversion writer stored.feature_conversion;
    write_conversion writer stored.target_conversion;
    (match stored.weight_conversion with
    | None -> W.byte writer 0
    | Some report ->
        W.byte writer 1;
        write_conversion writer report);
    Option.iter
      (fun report ->
        List.iter (W.int writer)
          [
            report.Batches.batches;
            report.Batches.rows;
            report.Batches.max_batch_rows;
            report.Batches.resets;
            report.Batches.passes;
          ];
        List.iter (W.i64 writer)
          [
            report.Batches.peak_feature_buffer_bytes;
            report.Batches.peak_staging_buffer_bytes;
            report.Batches.replay_fingerprint_bytes;
          ])
      stored.batches_report;
    W.bytes writer stored.model;
    Io.frame ~limits ~task:stored.task (W.contents writer)

  let decode_stored ~limits ~tasks ~threads bytes =
    let* () =
      Specification.integer ~name:"inference_threads" ~minimum:1 threads
    in
    let* task, payload = Io.unframe ~limits ~tasks bytes in
    let reader = R.create ~limits payload in
    let* package = R.string reader in
    let* implementation = R.string reader in
    let* codec_version = R.int reader in
    if
      package <> "modelkit-xgboost"
      || (implementation <> if task = 1 then "Regressor" else "Classifier")
      || (codec_version <> 1 && (codec_version <> 2 || task <> 1))
    then failure "unsupported model codec identity/version"
    else
      let* fitted_adapter_version = R.string reader in
      let* writer_adapter_version = R.string reader in
      let* binding_version = R.string reader in
      if
        String.trim fitted_adapter_version = ""
        || String.trim writer_adapter_version = ""
      then failure "invalid adapter provenance"
      else if binding_version <> "0.2.1" then
        compatibility "xgboost OCaml binding"
          "model provenance requires a different binding version"
      else
        let* training_native_version = read_version reader in
        let* model_native_version = read_version reader in
        let* source_word_size = R.int reader in
        let* precision = R.byte reader in
        let* missing = R.byte reader in
        let* execution = R.byte reader in
        let* declared_objective = R.string reader in
        let* prediction_begin = R.int reader in
        let* prediction_end = R.int reader in
        if
          (source_word_size <> 32 && source_word_size <> 64)
          || precision <> 1 || missing <> 1 || execution <> 1
          || declared_objective <> objective task
          || prediction_begin <> 0
        then
          failure
            "unsupported precision, missing-value, execution, or objective \
             policy"
        else
          let* specification = read_specification reader in
          if
            prediction_end
            <> (Specification.params specification).Specification.rounds
          then failure "prediction range differs from the specification"
          else
            let* schema = read_schema ~limits reader in
            let* classes =
              R.ints reader ~maximum:(min limits.Io.max_classes 16777217)
            in
            let* native_seed = R.i64 reader in
            let* training_rows = R.int reader in
            let* effective_rows = R.int reader in
            let* feature_conversion = read_conversion reader in
            let* target_conversion = read_conversion reader in
            let* has_weights = R.byte reader in
            let* weight_conversion =
              match has_weights with
              | 0 -> Ok None
              | 1 -> Result.map Option.some (read_conversion reader)
              | _ -> failure "invalid weight-report tag"
            in
            let* batches_report =
              if codec_version = 1 then Ok None
              else
                let* batches = R.int reader in
                let* rows = R.int reader in
                let* max_batch_rows = R.int reader in
                let* resets = R.int reader in
                let* passes = R.int reader in
                let* peak_feature_buffer_bytes = R.i64 reader in
                let* peak_staging_buffer_bytes = R.i64 reader in
                let* replay_fingerprint_bytes = R.i64 reader in
                Ok
                  (Some
                     {
                       Batches.batches;
                       rows;
                       max_batch_rows;
                       resets;
                       passes;
                       peak_feature_buffer_bytes;
                       peak_staging_buffer_bytes;
                       replay_fingerprint_bytes;
                     })
            in
            let* model = R.bytes reader in
            let* () = R.finish reader in
            let origin =
              {
                Origin.fitted_adapter_version;
                writer_adapter_version;
                binding_version;
                source_word_size;
                model_native_version;
                runtime_native_version = model_native_version;
                inference_threads = threads;
              }
            in
            let stored =
              {
                task;
                specification;
                schema;
                classes;
                origin;
                training_native_version;
                native_seed;
                training_rows;
                effective_rows;
                feature_conversion;
                target_conversion;
                weight_conversion;
                batches_report;
                model;
              }
            in
            if
              native_seed < 0L || native_seed > 2147483647L
              || training_native_version > model_native_version
            then
              failure "invalid native seed or training/model version ordering"
            else
              let* () = validate_reports stored in
              let* () = validate_classes ~limits task classes effective_rows in
              let* json =
                Io.Json.parse ~max_bytes:limits.Io.max_bytes
                  (Bytes.to_string model)
              in
              let* () =
                Io.Json.check_model
                  ~columns:(Feature_schema.feature_count schema)
                  ~classes:(Array.length classes) ~rounds:prediction_end
                  ~objective:declared_objective ~version:model_native_version
                  json
              in
              let* runtime_native_version = Native.version () in
              if runtime_native_version < model_native_version then
                compatibility "libxgboost"
                  "model JSON requires a native runtime at least as new as its \
                   writer"
              else
                Ok
                  {
                    stored with
                    origin = { origin with Origin.runtime_native_version };
                  }

  let restore ~limits stored construct =
    Native.with_loaded_model ~threads:stored.origin.Origin.inference_threads
      stored.model (fun booster ->
        let* () =
          Native.validate_native_model
            ~columns:(Feature_schema.feature_count stored.schema)
            ~rounds:
              (Specification.params stored.specification).Specification.rounds
            booster
        in
        let config = Xgboost.Booster.save_json_config booster in
        let* config = Io.Json.parse ~max_bytes:limits.Io.max_bytes config in
        let* () =
          Io.Json.check_config
            ~columns:(Feature_schema.feature_count stored.schema)
            ~classes:(Array.length stored.classes)
            ~objective:(objective stored.task)
            ~threads:stored.origin.Origin.inference_threads config
        in
        Ok (construct booster, true))

  let restore_regression ~limits stored =
    restore ~limits stored (fun booster ->
        {
          Regressor_implementation.specification = stored.specification;
          batches_report = stored.batches_report;
          schema = stored.schema;
          booster;
          prediction_mutex = Mutex.create ();
          origin = stored.origin;
          fit_report =
            {
              Regressor_implementation.rounds =
                (Specification.params stored.specification).Specification.rounds;
              native_version = stored.training_native_version;
              native_seed = stored.native_seed;
              feature_conversion = stored.feature_conversion;
              target_conversion = stored.target_conversion;
              weight_conversion = stored.weight_conversion;
            };
        })

  let restore_classification ~limits stored =
    restore ~limits stored (fun booster ->
        {
          Classifier_implementation.specification = stored.specification;
          schema = stored.schema;
          booster;
          class_labels = Array.copy stored.classes;
          prediction_mutex = Mutex.create ();
          origin = stored.origin;
          fit_report =
            {
              Classifier_implementation.rounds =
                (Specification.params stored.specification).Specification.rounds;
              native_version = stored.training_native_version;
              native_seed = stored.native_seed;
              class_count = Array.length stored.classes;
              training_rows = stored.training_rows;
              effective_rows = stored.effective_rows;
              feature_conversion = stored.feature_conversion;
              target_conversion = stored.target_conversion;
              weight_conversion = stored.weight_conversion;
            };
        })

  let encode_regression ?(limits = default_limits) fitted =
    let report = fitted.Regressor_implementation.fit_report in
    let* model =
      snapshot ~mutex:fitted.Regressor_implementation.prediction_mutex
        fitted.Regressor_implementation.booster
    in
    encode_stored ~limits
      {
        task = 1;
        specification = fitted.Regressor_implementation.specification;
        schema = fitted.Regressor_implementation.schema;
        classes = [||];
        origin = fitted.Regressor_implementation.origin;
        training_native_version = report.Regressor_implementation.native_version;
        native_seed = report.Regressor_implementation.native_seed;
        training_rows =
          report.Regressor_implementation.feature_conversion.Input.shape.(0);
        effective_rows =
          report.Regressor_implementation.feature_conversion.Input.shape.(0);
        feature_conversion = report.Regressor_implementation.feature_conversion;
        target_conversion = report.Regressor_implementation.target_conversion;
        weight_conversion = report.Regressor_implementation.weight_conversion;
        batches_report = fitted.Regressor_implementation.batches_report;
        model;
      }

  let encode_classification ?(limits = default_limits) fitted =
    let report = fitted.Classifier_implementation.fit_report in
    let* model =
      snapshot ~mutex:fitted.Classifier_implementation.prediction_mutex
        fitted.Classifier_implementation.booster
    in
    encode_stored ~limits
      {
        task =
          (if Array.length fitted.Classifier_implementation.class_labels = 2
           then 2
           else 3);
        specification = fitted.Classifier_implementation.specification;
        schema = fitted.Classifier_implementation.schema;
        classes = fitted.Classifier_implementation.class_labels;
        origin = fitted.Classifier_implementation.origin;
        training_native_version =
          report.Classifier_implementation.native_version;
        native_seed = report.Classifier_implementation.native_seed;
        training_rows = report.Classifier_implementation.training_rows;
        effective_rows = report.Classifier_implementation.effective_rows;
        feature_conversion = report.Classifier_implementation.feature_conversion;
        target_conversion = report.Classifier_implementation.target_conversion;
        weight_conversion = report.Classifier_implementation.weight_conversion;
        batches_report = None;
        model;
      }

  let decode_regression ?(limits = default_limits) ?(threads = 1) bytes =
    let* stored = decode_stored ~limits ~tasks:[ 1 ] ~threads bytes in
    restore_regression ~limits stored

  let decode_classification ?(limits = default_limits) ?(threads = 1) bytes =
    let* stored = decode_stored ~limits ~tasks:[ 2; 3 ] ~threads bytes in
    restore_classification ~limits stored

  let save_regression ?limits ~path fitted =
    let* bytes = encode_regression ?limits fitted in
    Io.save ~path bytes

  let save_classification ?limits ~path fitted =
    let* bytes = encode_classification ?limits fitted in
    Io.save ~path bytes

  let load_regression ?(limits = default_limits) ?threads ~path () =
    let* bytes = Io.read ~limits ~path in
    decode_regression ~limits ?threads bytes

  let load_classification ?(limits = default_limits) ?threads ~path () =
    let* bytes = Io.read ~limits ~path in
    decode_classification ~limits ?threads bytes

  let check_component ~provenance ~feature_schema ~classes stored =
    if
      Pipeline.provenance_package provenance <> "modelkit-xgboost"
      || Pipeline.provenance_version provenance
         <> stored.origin.Origin.fitted_adapter_version
      || Pipeline.provenance_implementation provenance
         <> if stored.task = 1 then "Regressor" else "Classifier"
    then
      failure "pipeline component provenance differs from the native envelope"
    else if
      (not (Feature_schema.equal feature_schema stored.schema))
      || classes <> stored.classes
    then
      failure
        "pipeline component schema/classes differ from the native envelope"
    else Ok ()

  let regression_codec ?(limits = default_limits) ?(threads = 1) () =
    let* () =
      Specification.integer ~name:"inference_threads" ~minimum:1 threads
    in
    Modelkit.Artifact.Codec.regression ~package:"modelkit-xgboost"
      ~name:"Regressor" ~version:1
      (module Regressor_implementation)
      ~encode:(encode_regression ~limits)
      ~decode:(fun ~provenance ~feature_schema bytes ->
        let* stored = decode_stored ~limits ~tasks:[ 1 ] ~threads bytes in
        let* () =
          check_component ~provenance ~feature_schema ~classes:[||] stored
        in
        restore_regression ~limits stored)

  let classification_codec ?(limits = default_limits) ?(threads = 1) () =
    let* () =
      Specification.integer ~name:"inference_threads" ~minimum:1 threads
    in
    Modelkit.Artifact.Codec.classification ~package:"modelkit-xgboost"
      ~name:"Classifier" ~version:1
      (module Classifier_implementation)
      ~classes:Classifier_implementation.classes
      ~predict_proba:Classifier_implementation.predict_proba
      ~encode:(encode_classification ~limits)
      ~decode:(fun ~provenance ~feature_schema ~classes bytes ->
        let* stored = decode_stored ~limits ~tasks:[ 2; 3 ] ~threads bytes in
        let* () = check_component ~provenance ~feature_schema ~classes stored in
        restore_classification ~limits stored)
      ()
end

module Regressor = struct
  include Regressor_implementation

  let estimator ?(name = "xgboost") specification =
    let* provenance =
      Pipeline.provenance ~package:"modelkit-xgboost"
        ~version:Modelkit_xgboost_version.value ~implementation:"Regressor"
    in
    let* codec = Artifact.regression_codec () in
    Modelkit.Artifact.Codec.estimator ~provenance ~name codec specification
end

module Classifier = struct
  include Classifier_implementation

  let estimator ?(name = "xgboost") specification =
    let* provenance =
      Pipeline.provenance ~package:"modelkit-xgboost"
        ~version:Modelkit_xgboost_version.value ~implementation:"Classifier"
    in
    let* codec = Artifact.classification_codec () in
    Modelkit.Artifact.Codec.estimator ~provenance ~name codec specification
end
