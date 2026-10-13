open Modelkit
module Xgb = Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

let check name condition = if not condition then failwith name

let read path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let length = in_channel_length channel in
      check "deployment input exceeds byte limit" (length <= 128 * 1024 * 1024);
      really_input_string channel length)

let bytes path =
  let text = read path |> String.trim in
  check "hex artifact has odd length" (String.length text mod 2 = 0);
  let digit char = String.index "0123456789abcdef" char in
  Bytes.init
    (String.length text / 2)
    (fun index ->
      Char.chr ((16 * digit text.[2 * index]) + digit text.[(2 * index) + 1]))

let field name json = Yojson.Basic.Util.member name json
let string name json = field name json |> Yojson.Basic.Util.to_string

let strings json =
  Yojson.Basic.Util.to_list json
  |> List.map Yojson.Basic.Util.to_string
  |> Array.of_list

let number = function
  | `Null -> Float.nan
  | value -> Yojson.Basic.Util.to_number value

let matrix json =
  Yojson.Basic.Util.to_list json
  |> List.map (fun row ->
      Yojson.Basic.Util.to_list row |> List.map number |> Array.of_list)
  |> Array.of_list |> Matrix.of_arrays |> data

let schema json =
  let names = strings json in
  Feature_names.create ~expected_count:(Array.length names) names
  |> data |> Feature_schema.named

let names schema =
  Feature_schema.names schema |> Option.get |> Feature_names.to_array

let json_ints values =
  `List (Array.to_list (Array.map (fun value -> `Int value) values))

let json_floats values =
  `List (Array.to_list (Array.map (fun value -> `Float value) values))

let json_matrix matrix =
  `List (Array.to_list (Array.map json_floats (Matrix.to_arrays matrix)))

let json_strings values =
  `List (Array.to_list (Array.map (fun value -> `String value) values))

let[@warning "-4"] incompatible = function
  | Error error -> (
      match Error.kind error with Error.Compatibility _ -> true | _ -> false)
  | Ok _ -> false

let[@warning "-4"] rec verify expected observed =
  match (expected, observed) with
  | `List left, `List right ->
      check "deployment response shape differs"
        (List.length left = List.length right);
      List.iter2 verify left right
  | `Int left, `Int right ->
      check "deployment integer response differs" (left = right)
  | (`Float _ | `Int _), (`Float _ | `Int _) ->
      let value = Yojson.Basic.Util.to_number expected in
      check "deployment numeric response differs"
        (abs_float (value -. Yojson.Basic.Util.to_number observed)
        <= 2e-6 +. (2e-6 *. abs_float value))
  | _ -> check "deployment response differs" (expected = observed)

let infer directory metadata expect_incompatible =
  let native_bytes =
    bytes (Filename.concat directory (string "native_artifact" metadata))
  in
  let pipeline_bytes =
    bytes (Filename.concat directory (string "pipeline_artifact" metadata))
  in
  let task = string "task" metadata in
  check "unknown deployment task"
    (task = "regression" || task = "binary" || task = "multiclass");
  if task = "regression" then (
    let reader =
      Xgb.Artifact.regression_codec ~threads:1 ()
      |> get |> Modelkit.Artifact.Codec.decoder
    in
    let native = Xgb.Artifact.decode_regression ~threads:1 native_bytes in
    let pipeline =
      Modelkit.Artifact.decode_regression ~decoders:[| reader |] pipeline_bytes
    in
    if expect_incompatible then (
      check "forward model load must fail typed" (incompatible native);
      check "forward pipeline load must fail typed" (incompatible pipeline);
      `Assoc [ ("status", `String "incompatible") ])
    else
      let native = get native
      and pipeline = get pipeline |> Modelkit.Artifact.model in
      let x = field "probe" metadata |> matrix in
      let input_schema = field "input_names" metadata |> schema in
      let native_x = field "transformed_probe" metadata |> matrix in
      let native_schema = field "output_names" metadata |> schema in
      check "pipeline input schema"
        (Feature_schema.equal input_schema (Pipeline.input_schema pipeline));
      check "pipeline output schema"
        (Feature_schema.equal native_schema (Pipeline.output_schema pipeline));
      check "native feature schema"
        (Feature_schema.equal native_schema
           (Xgb.Regressor.feature_schema native));
      let predictions =
        Pipeline.predict pipeline ~feature_schema:input_schema ~x
        |> get |> Target.regression_values |> Vector.to_array
      in
      let native_predictions =
        Xgb.Regressor.predict native ~feature_schema:native_schema ~x:native_x
        |> get |> Target.regression_values |> Vector.to_array
      in
      `Assoc
        [
          ("status", `String "loaded");
          ("task", `String task);
          ("input_names", json_strings (names (Pipeline.input_schema pipeline)));
          ( "output_names",
            json_strings (names (Pipeline.output_schema pipeline)) );
          ("predictions", json_floats predictions);
          ("native_predictions", json_floats native_predictions);
          ( "transformed_probe",
            json_matrix
              (Pipeline.transform pipeline ~feature_schema:input_schema ~x
              |> get) );
        ])
  else
    let reader =
      Xgb.Artifact.classification_codec ~threads:1 ()
      |> get |> Modelkit.Artifact.Codec.decoder
    in
    let native = Xgb.Artifact.decode_classification ~threads:1 native_bytes in
    let pipeline =
      Modelkit.Artifact.decode_classification ~decoders:[| reader |]
        pipeline_bytes
    in
    if expect_incompatible then (
      check "forward class model load must fail typed" (incompatible native);
      check "forward class pipeline load must fail typed"
        (incompatible pipeline);
      `Assoc [ ("status", `String "incompatible") ])
    else
      let native = get native
      and pipeline = get pipeline |> Modelkit.Artifact.model in
      let x = field "probe" metadata |> matrix in
      let input_schema = field "input_names" metadata |> schema in
      let native_x = field "transformed_probe" metadata |> matrix in
      let native_schema = field "output_names" metadata |> schema in
      check "pipeline input schema"
        (Feature_schema.equal input_schema (Pipeline.input_schema pipeline));
      check "pipeline output schema"
        (Feature_schema.equal native_schema (Pipeline.output_schema pipeline));
      check "native feature schema"
        (Feature_schema.equal native_schema
           (Xgb.Classifier.feature_schema native));
      `Assoc
        [
          ("status", `String "loaded");
          ("task", `String task);
          ("input_names", json_strings (names (Pipeline.input_schema pipeline)));
          ( "output_names",
            json_strings (names (Pipeline.output_schema pipeline)) );
          ("classes", json_ints (Pipeline.classes pipeline |> get));
          ("native_classes", json_ints (Xgb.Classifier.classes native));
          ( "predictions",
            json_ints
              (Pipeline.predict pipeline ~feature_schema:input_schema ~x
              |> get |> Target.classification_values) );
          ( "native_predictions",
            json_ints
              (Xgb.Classifier.predict native ~feature_schema:native_schema
                 ~x:native_x
              |> get |> Target.classification_values) );
          ( "probabilities",
            json_matrix
              (Pipeline.predict_proba pipeline ~feature_schema:input_schema ~x
              |> get) );
          ( "native_probabilities",
            json_matrix
              (Xgb.Classifier.predict_proba native ~feature_schema:native_schema
                 ~x:native_x
              |> get) );
          ( "transformed_probe",
            json_matrix
              (Pipeline.transform pipeline ~feature_schema:input_schema ~x
              |> get) );
        ]

let () =
  try
    check
      "usage: deployment_worker DIRECTORY METADATA \
       [--expect-incompatible|--auto-compat]"
      (Array.length Sys.argv = 3 || Array.length Sys.argv = 4);
    let metadata =
      read (Filename.concat Sys.argv.(1) Sys.argv.(2))
      |> Yojson.Basic.from_string
    in
    let expect_incompatible =
      Array.length Sys.argv = 4
      &&
      if Sys.argv.(3) = "--expect-incompatible" then true
      else if Sys.argv.(3) = "--auto-compat" then
        let major, minor, patch = Xgb.Native.version () |> get in
        [ major; minor; patch ]
        < (field "writer_native_version" metadata
          |> Yojson.Basic.Util.to_list
          |> List.map Yojson.Basic.Util.to_int)
      else failwith "unknown deployment option"
    in
    let response = infer Sys.argv.(1) metadata expect_incompatible in
    if not expect_incompatible then
      List.iter
        (fun name -> verify (field name metadata) (field name response))
        ([
           "task";
           "input_names";
           "output_names";
           "transformed_probe";
           "predictions";
           "native_predictions";
         ]
        @
        if string "task" metadata = "regression" then []
        else
          [
            "classes"; "native_classes"; "probabilities"; "native_probabilities";
          ]);
    print_endline (Yojson.Basic.to_string response)
  with Failure reason ->
    prerr_endline reason;
    exit 1
