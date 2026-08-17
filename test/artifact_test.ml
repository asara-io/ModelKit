open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let classification values = Target.classification values
let rng () = Rng.create (Seed.of_int 2048)

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let add_stage stage builder = Pipeline.add_transformer builder stage |> get

let regression_pipeline terminal =
  let imputer =
    Artifact.simple_imputer_stage ~name:"impute" (Simple_imputer.mean ()) |> get
  in
  let scaler =
    Artifact.standard_scaler_stage ~name:"scale" (Standard_scaler.create ())
    |> get
  in
  let selector =
    Artifact.variance_threshold_stage ~name:"select"
      (Variance_threshold.create () |> get)
    |> get
  in
  Pipeline.empty |> add_stage imputer |> add_stage scaler |> add_stage selector
  |> fun builder -> Pipeline.set_estimator builder terminal |> get

let check_regression_predictions message expected observed =
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    message
    (Target.regression_values expected |> Vector.to_array)
    (Target.regression_values observed |> Vector.to_array)

let fitted_ridge_pipeline () =
  let x =
    matrix
      [|
        [| 0.0; 1.0; 3.0 |];
        [| 1.0; 2.0; 1.0 |];
        [| 2.0; 3.0; 4.0 |];
        [| 3.0; 5.0; 2.0 |];
        [| 4.0; 8.0; 5.0 |];
      |]
  in
  let schema = named_schema [| "length"; "width"; "café" |] in
  let specification = Ridge_regression.create ~alpha:0.5 () |> get in
  let terminal =
    Artifact.ridge_regression_estimator ~name:"ridge" specification |> get
  in
  let pipeline = regression_pipeline terminal in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| 1.0; 2.5; 3.0; 5.0; 8.0 |])
      ()
    |> get
  in
  (x, schema, fitted)

let test_regression_round_trip () =
  let x, schema, fitted = fitted_ridge_pipeline () in
  let metadata =
    Artifact.metadata ~training_rows:5 ~root_seed:(Seed.of_int 2048)
      ~sample_weighted:false
      ~labels:[| ("dataset", "fixture") |]
      ()
    |> get
  in
  let encoded = Artifact.encode_regression ~metadata fitted |> get in
  let loaded = Artifact.decode_regression encoded |> get in
  let restored = Artifact.model loaded in
  let before = Pipeline.predict fitted ~feature_schema:schema ~x |> get in
  let after = Pipeline.predict restored ~feature_schema:schema ~x |> get in
  check_regression_predictions "regression predictions" before after;
  Alcotest.(check (option int))
    "training rows" (Some 5)
    (Artifact.metadata_of_loaded loaded |> Artifact.training_rows);
  Alcotest.(check string)
    "producer version" "0.3.0"
    (Artifact.producer_version loaded);
  Alcotest.(check string)
    "canonical bytes survive another write" (Bytes.to_string encoded)
    (Artifact.encode_regression
       ~metadata:(Artifact.metadata_of_loaded loaded)
       restored
    |> get |> Bytes.to_string)

let test_linear_and_classification_codecs () =
  let x = matrix [| [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let schema = Feature_schema.of_matrix x |> get_data in
  let linear_terminal =
    Artifact.linear_regression_estimator ~name:"ols"
      (Linear_regression.create ())
    |> get
  in
  let linear_pipeline =
    Pipeline.set_estimator Pipeline.empty linear_terminal |> get
  in
  let linear_fitted =
    Pipeline.fit linear_pipeline ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| -3.0; -1.0; 3.0; 5.0 |])
      ()
    |> get
  in
  let linear_loaded =
    Artifact.encode_regression linear_fitted
    |> get |> Artifact.decode_regression |> get |> Artifact.model
  in
  check_regression_predictions "OLS predictions"
    (Pipeline.predict linear_fitted ~feature_schema:schema ~x |> get)
    (Pipeline.predict linear_loaded ~feature_schema:schema ~x |> get);
  let logistic = Logistic_regression.create ~c:2.0 () |> get in
  let logistic_terminal =
    Artifact.logistic_regression_estimator ~name:"logistic" logistic |> get
  in
  let scaler =
    Artifact.standard_scaler_stage ~name:"scale" (Standard_scaler.create ())
    |> get
  in
  let logistic_pipeline =
    Pipeline.empty |> add_stage scaler |> fun builder ->
    Pipeline.set_estimator builder logistic_terminal |> get
  in
  let logistic_fitted =
    Pipeline.fit logistic_pipeline ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(classification [| -4; -4; 9; 9 |])
      ()
    |> get
  in
  let logistic_loaded =
    Artifact.encode_binary_classification logistic_fitted
    |> get |> Artifact.decode_binary_classification |> get |> Artifact.model
  in
  Alcotest.(check (array int))
    "classification predictions"
    (Pipeline.predict logistic_fitted ~feature_schema:schema ~x
    |> get |> Target.classification_values)
    (Pipeline.predict logistic_loaded ~feature_schema:schema ~x
    |> get |> Target.classification_values);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "classification probabilities"
    (Pipeline.predict_proba logistic_fitted ~feature_schema:schema ~x
    |> get |> Matrix.to_arrays |> Array.to_list |> Array.concat)
    (Pipeline.predict_proba logistic_loaded ~feature_schema:schema ~x
    |> get |> Matrix.to_arrays |> Array.to_list |> Array.concat)

let[@warning "-4"] expect_artifact_error message = function
  | Error error -> (
      match Error.kind error with
      | Error.Artifact _ -> ()
      | _ -> Alcotest.fail (message ^ ": " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail (message ^ ": expected an artifact error")

let recompute_checksum bytes =
  let payload = Bytes.sub bytes 36 (Bytes.length bytes - 36) in
  let digest = Digest.string (Bytes.to_string payload) in
  Bytes.blit_string digest 0 bytes 20 16

let int64_at bytes offset =
  let value = ref 0L in
  for index = offset to offset + 7 do
    value :=
      Int64.logor
        (Int64.shift_left !value 8)
        (Int64.of_int (Char.code (Bytes.get bytes index)))
  done;
  Int64.to_int !value

let skip_string bytes offset = offset + 8 + int64_at bytes offset

let terminal_component_offsets bytes =
  let offset = ref 36 in
  offset := skip_string bytes !offset;
  offset := skip_string bytes !offset;
  offset := !offset + 1 + 1 + 1 + 8;
  let skip_schema () =
    offset := !offset + 8;
    let named = Char.code (Bytes.get bytes !offset) in
    incr offset;
    if named = 1 then (
      let count = int64_at bytes !offset in
      offset := !offset + 8;
      for _ = 1 to count do
        offset := skip_string bytes !offset
      done);
    offset := skip_string bytes !offset
  in
  skip_schema ();
  skip_schema ();
  let stage_count = int64_at bytes !offset in
  offset := !offset + 8;
  Alcotest.(check int) "golden model has no transformers" 0 stage_count;
  offset := skip_string bytes !offset;
  let tag_offset = !offset in
  let version_offset = tag_offset + 1 in
  (tag_offset, version_offset)

let test_rejections () =
  let _, _, fitted = fitted_ridge_pipeline () in
  let encoded = Artifact.encode_regression fitted |> get in
  let corrupted = Bytes.copy encoded in
  Bytes.set corrupted
    (Bytes.length corrupted - 1)
    (Char.chr
       (Char.code (Bytes.get corrupted (Bytes.length corrupted - 1)) lxor 1));
  expect_artifact_error "checksum mismatch"
    (Artifact.decode_regression corrupted);
  let future = Bytes.copy encoded in
  Bytes.set future 8 (Char.chr 2);
  expect_artifact_error "unknown container version"
    (Artifact.decode_regression future);
  let checksum = Bytes.copy encoded in
  Bytes.set checksum 19 (Char.chr 2);
  expect_artifact_error "unknown checksum algorithm"
    (Artifact.decode_regression checksum);
  expect_artifact_error "task mismatch"
    (Artifact.decode_binary_classification encoded);
  expect_artifact_error "truncation"
    (Artifact.decode_regression
       (Bytes.sub encoded 0 (Bytes.length encoded - 1)));
  let limits =
    Artifact.limits ~max_bytes:(Bytes.length encoded - 1) () |> get
  in
  expect_artifact_error "reader byte limit"
    (Artifact.decode_regression ~limits encoded);
  let limits = Artifact.limits ~max_features:2 () |> get in
  expect_artifact_error "reader feature limit"
    (Artifact.decode_regression ~limits encoded);
  let limits = Artifact.limits ~max_components:3 () |> get in
  expect_artifact_error "reader component limit"
    (Artifact.decode_regression ~limits encoded)

let test_unsupported_component () =
  let x = matrix [| [| 0.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let schema = Feature_schema.of_matrix x |> get_data in
  let terminal =
    Pipeline.estimator ~name:"ols"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| 1.0; 3.0; 5.0 |])
      ()
    |> get
  in
  expect_artifact_error "generic pipeline component"
    (Artifact.encode_regression fitted)

let hex_value = function
  | '0' .. '9' as value -> Char.code value - Char.code '0'
  | 'a' .. 'f' as value -> Char.code value - Char.code 'a' + 10
  | 'A' .. 'F' as value -> Char.code value - Char.code 'A' + 10
  | _ -> Alcotest.fail "golden artifact contains non-hexadecimal data"

let bytes_of_hex value =
  let value = String.trim value in
  if String.length value mod 2 <> 0 then Alcotest.fail "odd golden hex length";
  Bytes.init
    (String.length value / 2)
    (fun index ->
      Char.chr
        ((hex_value value.[index * 2] lsl 4)
        lor hex_value value.[(index * 2) + 1]))

let hex_of_bytes bytes =
  let digits = "0123456789abcdef" in
  String.init
    (Bytes.length bytes * 2)
    (fun index ->
      let byte = Char.code (Bytes.get bytes (index / 2)) in
      if index mod 2 = 0 then digits.[byte lsr 4] else digits.[byte land 0xf])

let golden_artifact () =
  let x = matrix [| [| 0.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let schema = named_schema [| "input" |] in
  let terminal =
    Artifact.linear_regression_estimator ~name:"ols"
      (Linear_regression.create ())
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema:schema ~x
    ~y:(regression [| 1.0; 3.0; 5.0 |])
    ()
  |> get |> Artifact.encode_regression |> get

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let test_golden_reader_and_component_versions () =
  let path = Sys.getenv "MODELKIT_ARTIFACT_FIXTURE" in
  let golden = read_file path |> bytes_of_hex in
  let generated = golden_artifact () in
  if Bytes.length golden = 1 then Alcotest.fail (hex_of_bytes generated);
  Alcotest.(check string)
    "writer remains canonical" (Bytes.to_string golden)
    (Bytes.to_string generated);
  let loaded = Artifact.decode_regression golden |> get in
  let rewritten = Artifact.encode_regression (Artifact.model loaded) |> get in
  Alcotest.(check string)
    "golden artifact rewrites canonically" (Bytes.to_string golden)
    (Bytes.to_string rewritten);
  let tag_offset, version_offset = terminal_component_offsets golden in
  let unknown_tag = Bytes.copy golden in
  Bytes.set unknown_tag tag_offset (Char.chr 127);
  recompute_checksum unknown_tag;
  expect_artifact_error "unknown component tag"
    (Artifact.decode_regression unknown_tag);
  let unknown_version = Bytes.copy golden in
  Bytes.set unknown_version (version_offset + 7) (Char.chr 2);
  recompute_checksum unknown_version;
  expect_artifact_error "unknown component version"
    (Artifact.decode_regression unknown_version);
  let non_finite = Bytes.copy golden in
  let component_payload = version_offset + 8 + 8 in
  let coefficient = component_payload + 1 + 8 in
  Bytes.blit_string "\127\248\000\000\000\000\000\000" 0 non_finite coefficient
    8;
  recompute_checksum non_finite;
  expect_artifact_error "non-finite coefficient"
    (Artifact.decode_regression non_finite)

let test_file_round_trip () =
  let x, schema, fitted = fitted_ridge_pipeline () in
  let path = Filename.temp_file "modelkit-artifact-" ".bin" in
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists path then Sys.remove path)
    (fun () ->
      Artifact.save_regression ~path fitted |> get;
      let restored =
        Artifact.load_regression ~path () |> get |> Artifact.model
      in
      check_regression_predictions "file predictions"
        (Pipeline.predict fitted ~feature_schema:schema ~x |> get)
        (Pipeline.predict restored ~feature_schema:schema ~x |> get))

let () =
  Alcotest.run "artifact"
    [
      ( "round trip",
        [
          Alcotest.test_case "regression pipeline" `Quick
            test_regression_round_trip;
          Alcotest.test_case "OLS and classification" `Quick
            test_linear_and_classification_codecs;
          Alcotest.test_case "file convenience" `Quick test_file_round_trip;
        ] );
      ( "validation",
        [
          Alcotest.test_case "malformed and bounded inputs" `Quick
            test_rejections;
          Alcotest.test_case "unsupported components" `Quick
            test_unsupported_component;
          Alcotest.test_case "golden and component versions" `Quick
            test_golden_reader_and_component_versions;
        ] );
    ]
