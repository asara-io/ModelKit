open Modelkit
module Fixture = Codec_fixture

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let regression values = Vector.of_array values |> Target.regression |> data
let seed = Seed.of_int 87

let input () =
  let x =
    Matrix.init ~rows:6 ~columns:3 (fun row column ->
        if column = 0 then float_of_int row
        else if column = 2 then 1.
        else if row mod 2 = 0 then Float.nan
        else float_of_int (row * row))
    |> data
  in
  let names =
    Feature_names.create ~expected_count:3 [| "length"; "café"; "constant" |]
    |> data
  in
  let schema = Feature_schema.named names in
  let weights =
    Sample_weight.of_array ~expected_length:6 [| 1.; 2.; 1.; 3.; 1.; 2. |]
    |> data
  in
  (x, schema, weights)

let pipeline terminal =
  let stages =
    [|
      Artifact.simple_imputer_stage ~name:"impute" (Simple_imputer.mean ())
      |> get;
      Artifact.standard_scaler_stage ~name:"scale" (Standard_scaler.create ())
      |> get;
      Artifact.variance_threshold_stage ~name:"select"
        (Variance_threshold.create () |> get)
      |> get;
    |]
  in
  let builder =
    Array.fold_left
      (fun builder stage -> Pipeline.add_transformer builder stage |> get)
      Pipeline.empty stages
  in
  Pipeline.set_estimator builder terminal |> get

let fit terminal y =
  let x, schema, sample_weight = input () in
  Pipeline.fit (pipeline terminal) ~rng:(Rng.create seed) ~feature_schema:schema
    ~sample_weight ~x ~y ()
  |> get

let fitted_mean codec =
  let provenance = Fixture.provenance "Mean" |> get in
  Artifact.Codec.estimator ~name:"mean" ~provenance codec 0.5 |> get
  |> fun terminal -> fit terminal (regression [| 1.; 3.; 5.; 7.; 9.; 11. |])

let fitted_classifier codec count =
  let provenance = Fixture.provenance "Classifier" |> get in
  let labels = Fixture.Classifier.labels count in
  Artifact.Codec.estimator ~name:"classifier" ~provenance codec count |> get
  |> fun terminal ->
  fit terminal
    (Target.classification (Array.init 6 (fun row -> labels.(row mod count))))

let check_schemas expected observed =
  Alcotest.check Alcotest.bool "input schema" true
    (Feature_schema.equal
       (Pipeline.input_schema expected)
       (Pipeline.input_schema observed));
  Alcotest.check Alcotest.bool "output schema" true
    (Feature_schema.equal
       (Pipeline.output_schema expected)
       (Pipeline.output_schema observed))

let check_mean expected observed =
  let x, schema, _ = input () in
  let predict model =
    Pipeline.predict model ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "regression predictions" (predict expected) (predict observed);
  check_schemas expected observed

let check_classifier expected observed =
  let x, schema, _ = input () in
  let predict model =
    Pipeline.predict model ~feature_schema:schema ~x
    |> get |> Target.classification_values
  in
  let probabilities model =
    Pipeline.predict_proba model ~feature_schema:schema ~x
    |> get |> Matrix.to_arrays
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "classification predictions" (predict expected) (predict observed);
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "class order"
    (Pipeline.classes expected |> get)
    (Pipeline.classes observed |> get);
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.)))
    "aligned probabilities" (probabilities expected) (probabilities observed);
  check_schemas expected observed

let[@warning "-4"] expect_error message = function
  | Error error -> (
      match Error.kind error with
      | Error.Artifact _ -> error
      | _ -> Alcotest.fail (message ^ ": " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail (message ^ ": expected an artifact error")

let rejects message result = ignore (expect_error message result)

let check_provenance model implementation =
  let report = Pipeline.artifact_report model in
  Alcotest.check Alcotest.bool "codec support" true
    (Pipeline.artifact_supported report);
  Alcotest.check Alcotest.bool
    "external runtime excluded from portable guarantee" false
    (Pipeline.portable_artifact_supported report);
  let terminal = Pipeline.artifact_estimator report in
  Alcotest.check Alcotest.bool "external codec reported" true
    (terminal.Pipeline.serialization_support = Pipeline.External_artifact);
  let provenance = Option.get terminal.Pipeline.provenance in
  Alcotest.check Alcotest.string "external package" Fixture.package
    (Pipeline.provenance_package provenance);
  Alcotest.check Alcotest.string "external producer version" "1.2.3"
    (Pipeline.provenance_version provenance);
  Alcotest.check Alcotest.string "external implementation" implementation
    (Pipeline.provenance_implementation provenance)

let test_regression () =
  let calls = ref 0 in
  let codec = Fixture.mean_codec ~calls () |> get in
  let fitted = fitted_mean codec in
  let metadata =
    Artifact.metadata ~training_rows:6 ~root_seed:seed ~sample_weighted:true
      ~labels:[| ("owner", "codec fixture") |]
      ()
    |> get
  in
  let encoded = Artifact.encode_regression ~metadata fitted |> get in
  Alcotest.check Alcotest.int "external container version" 2
    (Char.code (Bytes.get encoded 8));
  rejects "decoder must be selected" (Artifact.decode_regression encoded);
  Alcotest.check Alcotest.int "no callback without selection" 0 !calls;
  let decoders = [| Artifact.Codec.decoder codec |] in
  let loaded = Artifact.decode_regression ~decoders encoded |> get in
  Alcotest.check Alcotest.int "one explicit decoder call" 1 !calls;
  let restored = Artifact.model loaded in
  check_mean fitted restored;
  check_provenance fitted "Mean";
  check_provenance restored "Mean";
  Alcotest.check
    (Alcotest.option Alcotest.int)
    "metadata rows" (Some 6)
    (Artifact.training_rows (Artifact.metadata_of_loaded loaded));
  Alcotest.check
    (Alcotest.array (Alcotest.pair Alcotest.string Alcotest.string))
    "metadata labels" (Artifact.labels metadata)
    (Artifact.labels (Artifact.metadata_of_loaded loaded));
  Alcotest.check Alcotest.string "canonical re-encoding"
    (Bytes.to_string encoded)
    (Artifact.encode_regression ~metadata restored |> get |> Bytes.to_string);
  rejects "decoder selection does not become ambient"
    (Artifact.decode_regression encoded);
  Alcotest.check Alcotest.int "no ambient callback" 1 !calls;
  let path = Filename.temp_file "modelkit-external-regression-" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Artifact.save_regression ~metadata ~path fitted |> get;
      check_mean fitted
        (Artifact.load_regression ~decoders ~path () |> get |> Artifact.model))

let test_classification count =
  let calls = ref 0 in
  let codec = Fixture.classifier_codec ~calls () |> get in
  let fitted = fitted_classifier codec count in
  let decoders = [| Artifact.Codec.decoder codec |] in
  let encoded = Artifact.encode_classification fitted |> get in
  let restored =
    Artifact.decode_classification ~decoders encoded |> get |> Artifact.model
  in
  check_classifier fitted restored;
  check_provenance restored "Classifier";
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "unsorted class order preserved"
    (Fixture.Classifier.labels count)
    (Pipeline.classes restored |> get);
  rejects "binary reader rejects generalized task"
    (Artifact.decode_binary_classification ~decoders encoded);
  let before = !calls in
  rejects "wrong regression task" (Artifact.decode_regression encoded);
  rejects "unselected classification decoder"
    (Artifact.decode_classification encoded);
  Alcotest.check Alcotest.int "task rejected before callback" before !calls;
  if count = 2 then (
    let binary = Artifact.encode_binary_classification fitted |> get in
    check_classifier fitted
      (Artifact.decode_binary_classification ~decoders binary
      |> get |> Artifact.model);
    check_classifier fitted
      (Artifact.decode_classification ~decoders binary |> get |> Artifact.model))
  else
    rejects "binary writer rejects three classes"
      (Artifact.encode_binary_classification fitted);
  let path = Filename.temp_file "modelkit-external-classification-" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Artifact.save_classification ~path fitted |> get;
      check_classifier fitted
        (Artifact.load_classification ~decoders ~path ()
        |> get |> Artifact.model))

let checksum bytes =
  let digest =
    Digest.string (Bytes.sub_string bytes 36 (Bytes.length bytes - 36))
  in
  Bytes.blit_string digest 0 bytes 20 16

let framed payload =
  let encoded = Bytes.create (36 + Bytes.length payload) in
  Bytes.blit_string "MDLKIT01" 0 encoded 0 8;
  Bytes.set encoded 8 (Char.chr 2);
  Bytes.set encoded 9 (Char.chr 0);
  Bytes.set encoded 10 (Char.chr 1);
  Bytes.set_int64_be encoded 11 (Int64.of_int (Bytes.length payload));
  Bytes.set encoded 19 (Char.chr 1);
  Bytes.blit payload 0 encoded 36 (Bytes.length payload);
  checksum encoded;
  encoded

let string_field text =
  let bytes = Bytes.create (8 + String.length text) in
  Bytes.set_int64_be bytes 0 (Int64.of_int (String.length text));
  Bytes.blit_string text 0 bytes 8 (String.length text);
  Bytes.to_string bytes

let find bytes needle =
  let rec scan offset =
    if offset > Bytes.length bytes - String.length needle then
      Alcotest.fail "fixture marker missing"
    else if Bytes.sub_string bytes offset (String.length needle) = needle then
      offset
    else scan (offset + 1)
  in
  scan 0

let codec_offsets bytes name =
  let marker =
    String.make 1 (Char.chr 255)
    ^ string_field Fixture.package
    ^ string_field name
  in
  let tag = find bytes marker in
  (tag, tag + String.length marker)

let test_preflight () =
  let calls = ref 0 in
  let codec = Fixture.mean_codec ~calls () |> get in
  let fitted = fitted_mean codec in
  let encoded = Artifact.encode_regression fitted |> get in
  let decoders = [| Artifact.Codec.decoder codec |] in
  let reject message bytes =
    rejects message (Artifact.decode_regression ~decoders bytes);
    Alcotest.check Alcotest.int (message ^ " precedes callback") 0 !calls
  in
  let mutate change =
    let bytes = Bytes.copy encoded in
    change bytes;
    checksum bytes;
    bytes
  in
  let _, version = codec_offsets encoded "Mean" in
  reject "unsupported codec version"
    (mutate (fun bytes -> Bytes.set_int64_be bytes version 2L));
  reject "oversized component payload"
    (mutate (fun bytes -> Bytes.set_int64_be bytes (version + 8) Int64.max_int));
  reject "unknown package-qualified name"
    (mutate (fun bytes -> Bytes.set bytes (version - 1) 'X'));
  reject "unsupported container version"
    (mutate (fun bytes -> Bytes.set bytes 8 (Char.chr 3)));
  reject "bad checksum"
    (let bytes = Bytes.copy encoded in
     Bytes.set bytes (Bytes.length bytes - 1) 'X';
     bytes);
  reject "truncated framing" (Bytes.sub encoded 0 (Bytes.length encoded - 1));
  let payload = Bytes.sub encoded 36 (Bytes.length encoded - 36) in
  let producer_length = Bytes.get_int64_be payload 0 |> Int64.to_int in
  let without_producer =
    Bytes.sub payload (8 + producer_length)
      (Bytes.length payload - 8 - producer_length)
  in
  reject "blank producer version"
    (framed (Bytes.cat (Bytes.make 8 '\000') without_producer));
  reject "trailing envelope payload"
    (framed (Bytes.cat payload (Bytes.of_string "X")));
  let fingerprint =
    Pipeline.input_schema fitted
    |> Feature_schema.fingerprint |> Schema_fingerprint.to_string
  in
  reject "bad schema fingerprint"
    (mutate (fun bytes -> Bytes.set bytes (find bytes fingerprint) 'X'));
  reject "provenance package mismatch"
    (mutate (fun bytes -> Bytes.set bytes (find bytes Fixture.package) 'X'));
  for length = 0 to Bytes.length encoded - 1 do
    reject "all truncated prefixes" (Bytes.sub encoded 0 length)
  done;
  rejects "duplicate decoder selection"
    (Artifact.decode_regression
       ~decoders:[| decoders.(0); decoders.(0) |]
       encoded);
  Alcotest.check Alcotest.int "duplicate selection precedes callback" 0 !calls;
  List.iter
    (fun (name, limits) ->
      rejects name (Artifact.decode_regression ~limits ~decoders encoded);
      Alcotest.check Alcotest.int (name ^ " precedes callback") 0 !calls)
    [
      ( "byte limit",
        Artifact.limits ~max_bytes:(Bytes.length encoded - 1) () |> get );
      ("component limit", Artifact.limits ~max_components:3 () |> get);
      ("feature limit", Artifact.limits ~max_features:2 () |> get);
      ("string limit", Artifact.limits ~max_string_bytes:8 () |> get);
    ];
  let bad_codec = Fixture.mean_codec ~bad_schema:true () |> get in
  rejects "decoded fitted schema mismatch"
    (Artifact.decode_regression
       ~decoders:[| Artifact.Codec.decoder bad_codec |]
       encoded);
  let wrong_codec = Fixture.mean_codec ~name:"Other" () |> get in
  rejects "wrong decoder identity"
    (Artifact.decode_regression
       ~decoders:[| Artifact.Codec.decoder wrong_codec |]
       encoded)

let test_callback_contract () =
  let codec = Fixture.mean_codec () |> get in
  let fitted = fitted_mean codec in
  let encoded = Artifact.encode_regression fitted |> get in
  let failing =
    Artifact.Codec.regression ~package:Fixture.package ~name:"Mean" ~version:1
      (module Fixture.Mean)
      ~encode:(fun _ -> Fixture.failure "encoder refused")
      ~decode:(fun ~provenance:_ ~feature_schema:_ _ ->
        Fixture.failure "decoder refused")
    |> get
  in
  let error =
    Artifact.decode_regression
      ~decoders:[| Artifact.Codec.decoder failing |]
      encoded
    |> expect_error "decoder failure"
  in
  Alcotest.check Alcotest.bool "decoder failure retains stage context" true
    (List.mem (Error.Stage "mean") (Error.context error));
  rejects "encoder failure remains typed"
    (Artifact.encode_regression (fitted_mean failing));
  let raising =
    Artifact.Codec.regression ~package:Fixture.package ~name:"Mean" ~version:1
      (module Fixture.Mean)
      ~encode:(fun _ -> Ok Bytes.empty)
      ~decode:(fun ~provenance:_ ~feature_schema:_ _ ->
        failwith "codec programmer error")
    |> get
  in
  Alcotest.check_raises "unrelated callback exceptions propagate"
    (Failure "codec programmer error") (fun () ->
      ignore
        (Artifact.decode_regression
           ~decoders:[| Artifact.Codec.decoder raising |]
           encoded));
  let second = Fixture.mean_codec ~version:2 () |> get in
  let bytes = Artifact.encode_regression (fitted_mean second) |> get in
  rejects "version one does not select version two"
    (Artifact.decode_regression
       ~decoders:[| Artifact.Codec.decoder codec |]
       bytes);
  check_mean (fitted_mean second)
    (Artifact.decode_regression
       ~decoders:
         [| Artifact.Codec.decoder codec; Artifact.Codec.decoder second |]
       bytes
    |> get |> Artifact.model)

let test_class_mapping_preflight () =
  let calls = ref 0 in
  let codec = Fixture.classifier_codec ~calls () |> get in
  let encoded =
    fitted_classifier codec 3 |> Artifact.encode_classification |> get
  in
  let decoders = [| Artifact.Codec.decoder codec |] in
  let tag, _ = codec_offsets encoded "Classifier" in
  let mutate change =
    let bytes = Bytes.copy encoded in
    change bytes;
    checksum bytes;
    bytes
  in
  let reject name bytes =
    rejects name (Artifact.decode_classification ~decoders bytes);
    Alcotest.check Alcotest.int "class validation precedes callback" 0 !calls
  in
  reject "duplicate class labels"
    (mutate (fun bytes ->
         Bytes.set_int64_be bytes (tag - 16)
           (Bytes.get_int64_be bytes (tag - 24))));
  reject "huge class count"
    (mutate (fun bytes -> Bytes.set_int64_be bytes (tag - 32) Int64.max_int));
  let limits = Artifact.limits ~max_classes:2 () |> get in
  rejects "class limit"
    (Artifact.decode_classification ~decoders ~limits encoded);
  Alcotest.check Alcotest.int "class limit precedes callback" 0 !calls;
  let bad_codec = Fixture.classifier_codec ~bad_classes:true () |> get in
  rejects "decoded class order mismatch"
    (Artifact.decode_classification
       ~decoders:[| Artifact.Codec.decoder bad_codec |]
       encoded)

let test_independent_class_limit () =
  let codec = Fixture.classifier_codec () |> get in
  let provenance = Fixture.provenance "Classifier" |> get in
  let terminal = Artifact.Codec.estimator ~provenance codec 3 |> get in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let x =
    Matrix.init ~rows:6 ~columns:1 (fun row _ -> float_of_int row) |> data
  in
  let schema = Feature_schema.of_matrix x |> data in
  let y = Target.classification [| 19; -7; 61; 19; -7; 61 |] in
  let fitted =
    Pipeline.fit pipeline ~rng:(Rng.create seed) ~feature_schema:schema ~x ~y ()
    |> get
  in
  let limits = Artifact.limits ~max_features:1 ~max_classes:3 () |> get in
  let bytes = Artifact.encode_classification ~limits fitted |> get in
  let restored =
    Artifact.decode_classification ~limits
      ~decoders:[| Artifact.Codec.decoder codec |]
      bytes
    |> get |> Artifact.model
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "class count is independent of feature count" [| 19; -7; 61 |]
    (Pipeline.classes restored |> get)

let test_codec_validation () =
  let make package name version =
    Artifact.Codec.regression ~package ~name ~version
      (module Fixture.Mean)
      ~encode:(fun _ -> Ok Bytes.empty)
      ~decode:(fun ~provenance:_ ~feature_schema:_ _ ->
        Fixture.failure "unused")
  in
  List.iter
    (fun (package, name, version) ->
      rejects "invalid codec identity" (make package name version))
    [
      ("modelkit", "Mean", 1);
      ("", "Mean", 1);
      (Fixture.package, "", 1);
      (Fixture.package, "a:b", 1);
      (Fixture.package, "Mean", 0);
      (Fixture.package, "Mean", 65536);
    ];
  let codec = Fixture.mean_codec () |> get in
  let provenance =
    Pipeline.provenance ~package:"other-package" ~version:"1"
      ~implementation:"Mean"
    |> get
  in
  rejects "codec/provenance package mismatch"
    (Artifact.Codec.estimator ~provenance codec 0.);
  let long_codec = Fixture.mean_codec ~name:(String.make 129 'x') () |> get in
  let limits = Artifact.limits ~max_string_bytes:128 () |> get in
  rejects "writer bounds codec identity"
    (Artifact.encode_regression ~limits (fitted_mean long_codec))

let test_builtin_classification () =
  let terminal =
    Artifact.logistic_regression_estimator ~name:"logistic"
      (Logistic_regression.create () |> get)
    |> get
  in
  let fitted =
    fit terminal (Target.classification [| -7; -7; -7; 19; 19; 19 |])
  in
  let legacy = Artifact.encode_binary_classification fitted |> get in
  Alcotest.check Alcotest.int "binary built-in writer stays v1" 1
    (Char.code (Bytes.get legacy 8));
  check_classifier fitted
    (Artifact.decode_classification legacy |> get |> Artifact.model);
  let generalized = Artifact.encode_classification fitted |> get in
  check_classifier fitted
    (Artifact.decode_classification generalized |> get |> Artifact.model)

let test_unsupported_stage () =
  let codec = Fixture.mean_codec () |> get in
  let provenance = Fixture.provenance "Mean" |> get in
  let terminal = Artifact.Codec.estimator ~provenance codec 0.5 |> get in
  let stage =
    Pipeline.transformer ~name:"unpersistable"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty stage |> get in
  let pipeline = Pipeline.set_estimator builder terminal |> get in
  let x, schema, _ = input () in
  let fitted =
    Pipeline.fit pipeline ~rng:(Rng.create seed) ~feature_schema:schema ~x
      ~y:(regression [| 1.; 2.; 3.; 4.; 5.; 6. |])
      ()
    |> get
  in
  Alcotest.check Alcotest.bool
    "unsupported preprocessing prevents complete encoding" false
    (Pipeline.artifact_report fitted |> Pipeline.artifact_supported);
  let error =
    Artifact.encode_regression fitted |> expect_error "unsupported stage"
  in
  ignore (find (Bytes.of_string (Error.to_string error)) "unpersistable")

let fixtures () =
  let mean = Fixture.mean_codec () |> get in
  let classifier = Fixture.classifier_codec () |> get in
  [
    ( "external_regression_v2.hex",
      Artifact.encode_regression (fitted_mean mean) |> get );
    ( "external_binary_v2.hex",
      Artifact.encode_binary_classification (fitted_classifier classifier 2)
      |> get );
    ( "external_multiclass_v2.hex",
      Artifact.encode_classification (fitted_classifier classifier 3) |> get );
  ]

let hex bytes =
  String.init
    (Bytes.length bytes * 2)
    (fun index ->
      let value = Char.code (Bytes.get bytes (index / 2)) in
      "0123456789abcdef".[if index mod 2 = 0 then value lsr 4 else value land 15])

let unhex text =
  let text = String.trim text in
  let digit char = String.index "0123456789abcdef" char in
  Bytes.init
    (String.length text / 2)
    (fun index ->
      Char.chr ((digit text.[2 * index] * 16) + digit text.[(2 * index) + 1]))

let test_goldens () =
  let root = Sys.getenv "MODELKIT_EXTERNAL_ARTIFACT_FIXTURES" in
  let read name =
    let channel = open_in_bin (Filename.concat root name) in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        really_input_string channel (in_channel_length channel) |> unhex)
  in
  let mean = Fixture.mean_codec () |> get in
  let classifier = Fixture.classifier_codec () |> get in
  let loaded =
    Artifact.decode_regression
      ~decoders:[| Artifact.Codec.decoder mean |]
      (read "external_regression_v2.hex")
    |> get
  in
  Alcotest.check Alcotest.string "golden producer" "0.6.0-dev"
    (Artifact.producer_version loaded);
  check_mean (fitted_mean mean) (Artifact.model loaded);
  check_classifier
    (fitted_classifier classifier 2)
    (Artifact.decode_binary_classification
       ~decoders:[| Artifact.Codec.decoder classifier |]
       (read "external_binary_v2.hex")
    |> get |> Artifact.model);
  check_classifier
    (fitted_classifier classifier 3)
    (Artifact.decode_classification
       ~decoders:[| Artifact.Codec.decoder classifier |]
       (read "external_multiclass_v2.hex")
    |> get |> Artifact.model)

let () =
  if Array.length Sys.argv = 2 && Sys.argv.(1) = "--write-fixtures" then
    List.iter
      (fun (name, bytes) ->
        let channel =
          open_out (Filename.concat "test/fixtures/artifacts" name)
        in
        Fun.protect
          ~finally:(fun () -> close_out_noerr channel)
          (fun () -> output_string channel (hex bytes ^ "\n")))
      (fixtures ())
  else
    Alcotest.run "public external codecs"
      [
        ( "boundary",
          [
            ("regression", `Quick, test_regression);
            ("binary", `Quick, fun () -> test_classification 2);
            ("multiclass", `Quick, fun () -> test_classification 3);
            ( "single-class general classification",
              `Quick,
              fun () -> test_classification 1 );
            ("envelope preflight", `Quick, test_preflight);
            ("class mapping preflight", `Quick, test_class_mapping_preflight);
            ("independent class limit", `Quick, test_independent_class_limit);
            ("codec identity", `Quick, test_codec_validation);
            ("unsupported component reporting", `Quick, test_unsupported_stage);
            ( "callback and explicit version contracts",
              `Quick,
              test_callback_contract );
            ( "legacy/generalized built-in classification",
              `Quick,
              test_builtin_classification );
            ("schema-v2 golden readers", `Quick, test_goldens);
          ] );
      ]
