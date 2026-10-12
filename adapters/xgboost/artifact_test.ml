open Modelkit
module Xgb = Modelkit_xgboost
module A = Xgb.Artifact

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let rng () = Rng.create (Seed.of_int 137)

let spec () =
  Xgb.Specification.create ~rounds:4 ~max_depth:3 ~max_bin:32 ~learning_rate:0.2
    ~min_child_weight:0.5 ~min_split_loss:0.1 ~reg_alpha:0.2 ~reg_lambda:1.5
    ~subsample:0.8 ~colsample_bytree:0.9 ~threads:2 ()
  |> get

let inputs () =
  let x =
    Matrix.init ~rows:18 ~columns:3 (fun row column ->
        if column = 0 then float_of_int row /. 3.
        else if column = 1 && row mod 3 = 0 then Float.nan
        else float_of_int ((row + (column * 2)) mod 7))
    |> data
  in
  let names =
    Feature_names.create ~expected_count:3 [| "row"; "café"; "signal" |] |> data
  in
  let schema = Feature_schema.named names in
  let weights =
    Sample_weight.of_array ~expected_length:18
      (Array.init 18 (fun row ->
           if row mod 5 = 0 then 0. else 0.3 +. float_of_int (row mod 4)))
    |> data
  in
  (x, schema, weights)

let regression () =
  let x, schema, sample_weight = inputs () in
  let y =
    Array.init 18 (fun row ->
        (0.7 *. float_of_int row) +. float_of_int (row mod 3))
    |> Vector.of_array |> Target.regression |> data
  in
  Xgb.Regressor.fit (spec ()) ~sample_weight ~rng:(rng ())
    ~feature_schema:schema ~x ~y ()
  |> get

let classification count =
  let x, schema, sample_weight = inputs () in
  let labels = [| -7; 16777219; 61 |] in
  let y =
    Target.classification (Array.init 18 (fun row -> labels.(row mod count)))
  in
  Xgb.Classifier.fit (spec ()) ~sample_weight ~rng:(rng ())
    ~feature_schema:schema ~x ~y ()
  |> get

let check_conversion expected observed =
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "conversion shape"
    (Xgb.Input.source_shape expected)
    (Xgb.Input.source_shape observed);
  Alcotest.check Alcotest.int64 "retained allocation"
    (Xgb.Input.payload_bytes expected)
    (Xgb.Input.payload_bytes observed);
  Alcotest.check Alcotest.int64 "temporary allocation"
    (Xgb.Input.temporary_payload_bytes expected)
    (Xgb.Input.temporary_payload_bytes observed);
  Alcotest.check Alcotest.int "rounded values"
    (Xgb.Input.rounded_values expected)
    (Xgb.Input.rounded_values observed)

let check_weights expected observed =
  match (expected, observed) with
  | None, None -> ()
  | Some expected, Some observed -> check_conversion expected observed
  | None, Some _ | Some _, None -> Alcotest.fail "weight report differs"

let check_regression expected observed =
  let x, schema, _ = inputs () in
  let predict model =
    Xgb.Regressor.predict model ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "native predictions" (predict expected) (predict observed);
  Alcotest.check Alcotest.bool "immutable parameters" true
    (Xgb.Regressor.fitted_params expected = Xgb.Regressor.fitted_params observed);
  Alcotest.check Alcotest.bool "feature schema" true
    (Feature_schema.equal
       (Xgb.Regressor.feature_schema expected)
       (Xgb.Regressor.feature_schema observed));
  let before = Xgb.Regressor.report expected
  and after = Xgb.Regressor.report observed in
  Alcotest.check Alcotest.int "rounds" before.Xgb.Regressor.rounds
    after.Xgb.Regressor.rounds;
  Alcotest.check Alcotest.int64 "training seed" before.Xgb.Regressor.native_seed
    after.Xgb.Regressor.native_seed;
  check_conversion before.Xgb.Regressor.feature_conversion
    after.Xgb.Regressor.feature_conversion;
  check_conversion before.Xgb.Regressor.target_conversion
    after.Xgb.Regressor.target_conversion;
  check_weights before.Xgb.Regressor.weight_conversion
    after.Xgb.Regressor.weight_conversion

let check_classification expected observed =
  let x, schema, _ = inputs () in
  let predict model =
    Xgb.Classifier.predict model ~feature_schema:schema ~x
    |> get |> Target.classification_values
  in
  let probabilities model =
    Xgb.Classifier.predict_proba model ~feature_schema:schema ~x
    |> get |> Matrix.to_arrays
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "original labels" (predict expected) (predict observed);
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "class order"
    (Xgb.Classifier.classes expected)
    (Xgb.Classifier.classes observed);
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.)))
    "probabilities" (probabilities expected) (probabilities observed);
  Alcotest.check Alcotest.bool "immutable classification parameters" true
    (Xgb.Classifier.fitted_params expected
    = Xgb.Classifier.fitted_params observed);
  let before = Xgb.Classifier.report expected
  and after = Xgb.Classifier.report observed in
  Alcotest.check Alcotest.int64 "classifier training seed"
    before.Xgb.Classifier.native_seed after.Xgb.Classifier.native_seed;
  Alcotest.check Alcotest.int "training rows"
    before.Xgb.Classifier.training_rows after.Xgb.Classifier.training_rows;
  Alcotest.check Alcotest.int "effective rows"
    before.Xgb.Classifier.effective_rows after.Xgb.Classifier.effective_rows;
  check_conversion before.Xgb.Classifier.feature_conversion
    after.Xgb.Classifier.feature_conversion;
  check_conversion before.Xgb.Classifier.target_conversion
    after.Xgb.Classifier.target_conversion;
  check_weights before.Xgb.Classifier.weight_conversion
    after.Xgb.Classifier.weight_conversion

let test_regression () =
  let fitted = regression () in
  let bytes = A.encode_regression fitted |> get in
  let restored = A.decode_regression bytes |> get in
  check_regression fitted restored;
  let info = A.regression_provenance restored in
  Alcotest.check Alcotest.int "default inference thread cap" 1
    info.A.inference_threads;
  Alcotest.check Alcotest.int "training thread parameter retained" 2
    (Xgb.Regressor.fitted_params restored).Xgb.Specification.threads;
  Alcotest.check Alcotest.int "complete prediction range" 4
    info.A.prediction_end;
  Alcotest.check Alcotest.string "binding provenance" "0.2.1"
    info.A.binding_version;
  Alcotest.check Alcotest.string "canonical model round trip"
    (Bytes.to_string bytes)
    (A.encode_regression restored |> get |> Bytes.to_string);
  let override = A.decode_regression ~threads:3 bytes |> get in
  check_regression fitted override;
  Alcotest.check Alcotest.int "explicit inference thread cap" 3
    (A.regression_provenance override).A.inference_threads;
  let path = Filename.temp_file "modelkit-xgboost-regression-" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      A.save_regression ~path fitted |> get;
      check_regression fitted (A.load_regression ~threads:2 ~path () |> get))

let test_classification count =
  let fitted = classification count in
  let bytes = A.encode_classification fitted |> get in
  check_classification fitted (A.decode_classification bytes |> get);
  let path = Filename.temp_file "modelkit-xgboost-classification-" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      A.save_classification ~path fitted |> get;
      check_classification fitted
        (A.load_classification ~threads:2 ~path () |> get))

let expect_error name = function
  | Error _ -> ()
  | Ok _ -> Alcotest.fail (name ^ " should fail")

let raw_model () =
  let x, schema, _ = inputs () in
  let converted = Xgb.Input.features ~feature_schema:schema x |> get in
  let y =
    Target.regression (Vector.of_array (Array.init 18 float_of_int)) |> data
  in
  let target = Xgb.Input.regression_target y |> get in
  Xgb.Native.with_dmatrix ~target converted (fun dtrain ->
      Xgb.Native.with_booster ~cache:[ dtrain ] ~specification:(spec ())
        ~rng:(rng ()) (fun booster ->
          Xgboost.Booster.set_param booster "objective" "reg:squarederror";
          for iter = 0 to 3 do
            Xgboost.Booster.update_one_iter booster ~iter ~dtrain
          done;
          Ok (Xgboost.Booster.save_model_buffer booster)))
  |> get

let test_scoped_cleanup () =
  let model = raw_model () in
  let run callback =
    let borrowed = ref None in
    let result () =
      Xgb.Native.with_model_buffer model (fun booster ->
          borrowed := Some booster;
          callback booster)
    in
    let check () =
      match !borrowed with
      | None -> Alcotest.fail "loaded booster not observed"
      | Some booster ->
          (try
             ignore (Xgboost.Booster.num_features booster);
             Alcotest.fail "borrowed booster remained live"
           with Xgboost.Xgboost_error _ -> ());
          Xgboost.Booster.free booster
    in
    (result, check)
  in
  let result, check = run (fun _ -> Ok ()) in
  result () |> get;
  check ();
  let result, check =
    run (fun _ ->
        Error
          (Error.make ~remediation:"test"
             (Error.Artifact { operation = "test"; reason = "test" })))
  in
  expect_error "typed callback rejection" (result ());
  check ();
  let result, check = run (fun _ -> raise Exit) in
  Alcotest.check_raises "callback exception" Exit (fun () -> ignore (result ()));
  check ();
  let result, check =
    run (fun booster ->
        Xgboost.Booster.set_param booster "objective" "no-such-objective";
        ignore (Xgboost.Booster.save_json_config booster);
        Ok ())
  in
  expect_error "native callback rejection" (result ());
  check ()

type offsets = {
  codec_version : int;
  fitted_version : int;
  binding : int;
  model_version : int;
  policy : int;
  objective : int;
  prediction_end : int;
  rounds : int;
  columns : int;
  fingerprint : int;
  classes : int;
  seed : int;
  effective_rows : int;
  feature_bytes : int;
  model_length : int;
}

let offsets bytes =
  let position = ref 36 in
  let integer offset = Bytes.get_int64_be bytes offset |> Int64.to_int in
  let string () =
    let offset = !position in
    position := offset + 8 + integer offset;
    offset
  in
  let take count =
    let offset = !position in
    position := !position + count;
    offset
  in
  ignore (string ());
  ignore (string ());
  let codec_version = take 8 in
  let fitted_version = string () in
  ignore (string ());
  let binding = string () in
  ignore (take 24);
  let model_version = take 24 in
  ignore (take 8);
  let policy = take 3 in
  let objective = string () in
  ignore (take 8);
  let prediction_end = take 8 in
  let rounds = take (11 * 8) in
  let columns = take 8 in
  let named = Char.code (Bytes.get bytes (take 1)) in
  if named = 1 then
    for _ = 1 to integer columns do
      ignore (string ())
    done;
  let fingerprint = string () in
  let classes = !position in
  ignore (take (8 + (8 * integer classes)));
  let seed = take 8 in
  ignore (take 8);
  let effective_rows = take 8 in
  let conversion () =
    let shape = !position in
    ignore (take (8 + (8 * integer shape)));
    take 24
  in
  let feature_bytes = conversion () in
  ignore (conversion ());
  let weighted = Char.code (Bytes.get bytes (take 1)) in
  if weighted = 1 then ignore (conversion ());
  let model_length = !position in
  {
    codec_version;
    fitted_version;
    binding;
    model_version;
    policy;
    objective;
    prediction_end;
    rounds;
    columns;
    fingerprint;
    classes;
    seed;
    effective_rows;
    feature_bytes;
    model_length;
  }

let checksum bytes =
  Bytes.set_int64_be bytes 11 (Int64.of_int (Bytes.length bytes - 36));
  Bytes.blit_string
    (Digest.string (Bytes.sub_string bytes 36 (Bytes.length bytes - 36)))
    0 bytes 20 16

let mutate bytes change =
  let copy = Bytes.copy bytes in
  change copy;
  checksum copy;
  copy

let native_json bytes =
  let point = (offsets bytes).model_length in
  Bytes.sub_string bytes (point + 8) (Bytes.length bytes - point - 8)

let replace_json bytes json =
  let point = (offsets bytes).model_length in
  let prefix = Bytes.sub bytes 0 (point + 8) in
  Bytes.set_int64_be prefix point (Int64.of_int (String.length json));
  let changed = Bytes.cat prefix (Bytes.of_string json) in
  checksum changed;
  changed

let[@warning "-4"] rec set_json path value json =
  match path with
  | [] -> value
  | name :: rest -> (
      match json with
      | `Assoc fields ->
          `Assoc
            (List.map
               (fun (key, current) ->
                 ( key,
                   if key = name then set_json rest value current else current
                 ))
               fields)
      | _ -> Alcotest.fail "native JSON test path missing")

let test_corruption () =
  let encoded = regression () |> A.encode_regression |> get in
  let fields = offsets encoded in
  let reject name change =
    expect_error name (A.decode_regression (mutate encoded change))
  in
  reject "unknown container major" (fun bytes -> Bytes.set bytes 8 '\002');
  reject "unknown container minor" (fun bytes -> Bytes.set bytes 9 '\001');
  reject "unknown checksum" (fun bytes -> Bytes.set bytes 19 '\002');
  reject "unknown codec version" (fun bytes ->
      Bytes.set_int64_be bytes fields.codec_version 2L);
  reject "unknown binding version" (fun bytes ->
      Bytes.set bytes (fields.binding + 12) '2');
  reject "blank fitted producer" (fun bytes ->
      Bytes.fill bytes
        (fields.fitted_version + 8)
        (Bytes.get_int64_be bytes fields.fitted_version |> Int64.to_int)
        ' ');
  reject "unknown precision" (fun bytes -> Bytes.set bytes fields.policy '\002');
  reject "unknown missing-value policy" (fun bytes ->
      Bytes.set bytes (fields.policy + 1) '\002');
  reject "unknown execution policy" (fun bytes ->
      Bytes.set bytes (fields.policy + 2) '\002');
  reject "wrong objective" (fun bytes ->
      Bytes.set bytes (fields.objective + 8) 'X');
  reject "wrong prediction range" (fun bytes ->
      Bytes.set_int64_be bytes fields.prediction_end 1L);
  reject "invalid immutable parameters" (fun bytes ->
      Bytes.set_int64_be bytes fields.rounds 0L);
  reject "huge feature count" (fun bytes ->
      Bytes.set_int64_be bytes fields.columns Int64.max_int);
  reject "schema fingerprint" (fun bytes ->
      Bytes.set bytes (fields.fingerprint + 8) 'X');
  reject "negative native seed" (fun bytes ->
      Bytes.set_int64_be bytes fields.seed (-1L));
  reject "inconsistent effective rows" (fun bytes ->
      Bytes.set_int64_be bytes fields.effective_rows 1L);
  reject "inconsistent allocation report" (fun bytes ->
      Bytes.set_int64_be bytes fields.feature_bytes 1L);
  reject "native payload length bound" (fun bytes ->
      Bytes.set_int64_be bytes fields.model_length Int64.max_int);
  expect_error "wrong classification task" (A.decode_classification encoded);
  expect_error "corrupt payload checksum"
    (A.decode_regression
       (let bytes = Bytes.copy encoded in
        Bytes.set bytes (Bytes.length bytes - 1) 'X';
        bytes));
  expect_error "trailing payload"
    (A.decode_regression
       (let bytes = Bytes.cat encoded (Bytes.of_string "X") in
        checksum bytes;
        bytes));
  for count = 0 to Bytes.length encoded - 1 do
    expect_error "all truncated prefixes"
      (A.decode_regression (Bytes.sub encoded 0 count))
  done;
  List.iter
    (fun limits ->
      expect_error "configured reader bound"
        (A.decode_regression ~limits encoded))
    [
      A.limits ~max_bytes:(Bytes.length encoded - 1) () |> get;
      A.limits ~max_features:2 () |> get;
      A.limits ~max_string_bytes:8 () |> get;
    ];
  expect_error "invalid inference cap" (A.decode_regression ~threads:0 encoded);
  expect_error "invalid codec inference cap" (A.regression_codec ~threads:0 ());
  let bytes = classification 3 |> A.encode_classification |> get in
  let fields = offsets bytes in
  expect_error "duplicate class label"
    (A.decode_classification
       (mutate bytes (fun bytes ->
            Bytes.set_int64_be bytes (fields.classes + 16)
              (Bytes.get_int64_be bytes (fields.classes + 8)))));
  expect_error "class-count bound"
    (A.decode_classification ~limits:(A.limits ~max_classes:2 () |> get) bytes)

let test_native_consistency () =
  let encoded = regression () |> A.encode_regression |> get in
  let model = Yojson.Basic.from_string (native_json encoded) in
  List.iter
    (fun (name, path, value) ->
      let json = set_json path value model |> Yojson.Basic.to_string in
      expect_error name (A.decode_regression (replace_json encoded json)))
    [
      ( "native feature mismatch",
        [ "learner"; "learner_model_param"; "num_feature" ],
        `String "4" );
      ( "native output mismatch",
        [ "learner"; "learner_model_param"; "num_target" ],
        `String "2" );
      ( "native class mismatch",
        [ "learner"; "learner_model_param"; "num_class" ],
        `String "3" );
      ( "native objective mismatch",
        [ "learner"; "objective"; "name" ],
        `String "binary:logistic" );
      ( "native booster mismatch",
        [ "learner"; "gradient_booster"; "name" ],
        `String "dart" );
      ( "native round mismatch",
        [
          "learner";
          "gradient_booster";
          "model";
          "gbtree_model_param";
          "num_trees";
        ],
        `String "1" );
      ( "native iteration mismatch",
        [ "learner"; "gradient_booster"; "model"; "iteration_indptr" ],
        `List [ `Int 0; `Int 2; `Int 2; `Int 3; `Int 4 ] );
      ( "unsupported categorical metadata",
        [ "learner"; "feature_types" ],
        `List [ `String "c" ] );
    ];
  expect_error "invalid native JSON"
    (A.decode_regression (replace_json encoded "{"));
  expect_error "native JSON nesting bound"
    (A.decode_regression
       (replace_json encoded (String.make 65 '[' ^ String.make 65 ']')));
  let future =
    set_json [ "version" ] (`List [ `Int 3; `Int 99; `Int 0 ]) model
    |> Yojson.Basic.to_string
  in
  let future = replace_json encoded future in
  let point = (offsets future).model_version in
  let future =
    mutate future (fun bytes ->
        Bytes.set_int64_be bytes point 3L;
        Bytes.set_int64_be bytes (point + 8) 99L;
        Bytes.set_int64_be bytes (point + 16) 0L)
  in
  expect_error "forward native compatibility rejected before loading"
    (A.decode_regression future)

let test_codec_attachment () =
  let codec = A.regression_codec ~threads:3 () |> get in
  let provenance =
    Pipeline.provenance ~package:"modelkit-xgboost" ~version:"0.6.0-dev"
      ~implementation:"Regressor"
    |> get
  in
  let terminal =
    Modelkit.Artifact.Codec.estimator ~provenance codec (spec ()) |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let x, schema, sample_weight = inputs () in
  let y =
    Target.regression (Vector.of_array (Array.init 18 float_of_int)) |> data
  in
  let fitted =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng ()) ~feature_schema:schema ~x
      ~y ()
    |> get
  in
  let bytes = Modelkit.Artifact.encode_regression fitted |> get in
  let restored =
    Modelkit.Artifact.decode_regression
      ~decoders:[| Modelkit.Artifact.Codec.decoder codec |]
      bytes
    |> get |> Modelkit.Artifact.model
  in
  let predict model =
    Pipeline.predict model ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "attached native codec predictions" (predict fitted) (predict restored)

let test_concurrent_save () =
  let fitted = regression () in
  let x, schema, _ = inputs () in
  let expected =
    Xgb.Regressor.predict fitted ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  let writer =
    Domain.spawn (fun () ->
        for _ = 1 to 12 do
          let bytes = A.encode_regression fitted |> get in
          check_regression fitted (A.decode_regression bytes |> get)
        done)
  in
  for _ = 1 to 24 do
    let observed =
      Xgb.Regressor.predict fitted ~feature_schema:schema ~x
      |> get |> Target.regression_values |> Vector.to_array
    in
    Alcotest.check
      (Alcotest.array (Alcotest.float 0.))
      "save/predict independence" expected observed
  done;
  Domain.join writer

let test_classification_codec count =
  let codec = A.classification_codec () |> get in
  let provenance =
    Pipeline.provenance ~package:"modelkit-xgboost" ~version:"0.6.0-dev"
      ~implementation:"Classifier"
    |> get
  in
  let terminal =
    Modelkit.Artifact.Codec.estimator ~provenance codec (spec ()) |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let x, schema, sample_weight = inputs () in
  let labels = [| -7; 16777219; 61 |] in
  let y =
    Target.classification (Array.init 18 (fun row -> labels.(row mod count)))
  in
  let fitted =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng ()) ~feature_schema:schema ~x
      ~y ()
    |> get
  in
  let bytes = Modelkit.Artifact.encode_classification fitted |> get in
  let restored =
    Modelkit.Artifact.decode_classification
      ~decoders:[| Modelkit.Artifact.Codec.decoder codec |]
      bytes
    |> get |> Modelkit.Artifact.model
  in
  let probabilities model =
    Pipeline.predict_proba model ~feature_schema:schema ~x
    |> get |> Matrix.to_arrays
  in
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.)))
    "native classification codec dispatch" (probabilities fitted)
    (probabilities restored)

let test_unweighted_and_empty () =
  let x, schema, _ = inputs () in
  let y =
    Target.regression (Vector.of_array (Array.init 18 float_of_int)) |> data
  in
  let fitted =
    Xgb.Regressor.fit (spec ()) ~rng:(rng ()) ~feature_schema:schema ~x ~y ()
    |> get
  in
  let restored =
    A.encode_regression fitted |> get |> A.decode_regression |> get
  in
  check_regression fitted restored;
  let empty = Matrix.create ~rows:0 ~columns:3 0. |> data in
  Alcotest.check Alcotest.int "restored empty regression" 0
    (Xgb.Regressor.predict restored ~feature_schema:schema ~x:empty
    |> get |> Target.length);
  let classes =
    Xgb.Classifier.fit (spec ()) ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:
        (Target.classification
           (Array.init 18 (fun row -> if row mod 2 = 0 then -7 else 16777219)))
      ()
    |> get
  in
  let restored =
    A.encode_classification classes |> get |> A.decode_classification |> get
  in
  check_classification classes restored;
  Alcotest.check Alcotest.bool "restored empty probabilities" true
    (Matrix.shape
       (Xgb.Classifier.predict_proba restored ~feature_schema:schema ~x:empty
       |> get)
    = (0, 2))

let test_writer_bounds () =
  let fitted = regression () in
  let limits = A.limits ~max_bytes:1 () |> get in
  expect_error "writer byte limit" (A.encode_regression ~limits fitted);
  expect_error "writer feature limit"
    (A.encode_regression ~limits:(A.limits ~max_features:2 () |> get) fitted);
  expect_error "writer schema-string limit"
    (A.encode_regression
       ~limits:(A.limits ~max_string_bytes:40 () |> get)
       fitted);
  let path = Filename.temp_file "modelkit-xgboost-untouched-" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      let channel = open_out_bin path in
      output_string channel "original";
      close_out channel;
      expect_error "encoding precedes destination open"
        (A.save_regression ~limits ~path fitted);
      let channel = open_in_bin path in
      let text = really_input_string channel (in_channel_length channel) in
      close_in channel;
      Alcotest.check Alcotest.string "destination preserved on encoding failure"
        "original" text;
      expect_error "file-size reader bound" (A.load_regression ~limits ~path ()))

let () =
  Alcotest.run "XGBoost model persistence"
    [
      ( "models",
        [
          ("regression byte/file round trip", `Quick, test_regression);
          ( "binary byte/file round trip",
            `Quick,
            fun () -> test_classification 2 );
          ( "multiclass byte/file round trip",
            `Quick,
            fun () -> test_classification 3 );
          ("loaded-model native cleanup", `Quick, test_scoped_cleanup);
          ("bounded envelope/corruption failures", `Quick, test_corruption);
          ("native model consistency", `Quick, test_native_consistency);
          ("public codec attachment", `Quick, test_codec_attachment);
          ( "concurrent serialization and prediction",
            `Quick,
            test_concurrent_save );
          ( "binary codec dispatch",
            `Quick,
            fun () -> test_classification_codec 2 );
          ( "multiclass codec dispatch",
            `Quick,
            fun () -> test_classification_codec 3 );
          ("unweighted and empty inference", `Quick, test_unweighted_and_empty);
          ("writer/file bounds", `Quick, test_writer_bounds);
        ] );
    ]
