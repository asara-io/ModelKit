open Modelkit
module Xgb = Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let seed = Seed.of_int 73

let specification depth =
  Xgb.Specification.create ~rounds:4 ~max_depth:depth ~threads:1
    ~learning_rate:0.2 ~min_child_weight:0.5 ~reg_alpha:0.25 ~reg_lambda:2. ()
  |> get

let input_names = [| "length"; "café"; "constant" |]

let schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> data |> Feature_schema.named

let names schema =
  Feature_schema.names schema |> Option.get |> Feature_names.to_array

let input_schema = schema input_names

let training () =
  let x =
    Matrix.init ~rows:18 ~columns:3 (fun row column ->
        if column = 0 then float_of_int row /. 3.
        else if column = 2 then 1.
        else if row mod 4 = 0 then Float.nan
        else float_of_int ((row * 7 mod 11) - 5))
    |> data
  in
  let weights =
    Sample_weight.of_array ~expected_length:18
      (Array.init 18 (fun row ->
           if row mod 5 = 0 then 0. else 1. +. (0.5 *. float_of_int (row mod 3))))
    |> data
  in
  (x, weights)

let probe () =
  Matrix.of_arrays
    [|
      [| 0.2; Float.nan; 1. |];
      [| 4.5; 1.2; 1. |];
      [| -2.; -5.; 1. |];
      [| 10.5; Float.nan; 1. |];
      [| 1.1; 3.6; 1. |];
    |]
  |> data

let regression_target () =
  Vector.of_array
    (Array.init 18 (fun row ->
         (0.6 *. float_of_int row) +. float_of_int (row mod 3)))
  |> Target.regression |> data

let classification_target count =
  let labels =
    if count = 2 then [| -7; 16777219 |] else [| -7; 61; 16777219 |]
  in
  Target.classification (Array.init 18 (fun row -> labels.(row mod count)))

let preprocessing () =
  let stages =
    [|
      Artifact.simple_imputer_stage ~name:"impute" (Simple_imputer.mean ())
      |> get;
      Artifact.standard_scaler_stage ~route_sample_weight:true ~name:"scale"
        (Standard_scaler.create ())
      |> get;
      Artifact.variance_threshold_stage ~name:"select"
        (Variance_threshold.create () |> get)
      |> get;
    |]
  in
  Array.fold_left
    (fun builder stage -> Pipeline.add_transformer builder stage |> get)
    Pipeline.empty stages

let regression_pipeline depth =
  Pipeline.set_estimator (preprocessing ())
    (Xgb.Regressor.estimator (specification depth) |> get)

let classification_pipeline depth =
  Pipeline.set_estimator (preprocessing ())
    (Xgb.Classifier.estimator (specification depth) |> get)

let fit pipeline y =
  let x, sample_weight = training () in
  Pipeline.fit pipeline ~sample_weight ~rng:(Rng.create seed)
    ~feature_schema:input_schema ~x ~y ()
  |> get

let json_floats values =
  `List
    (Array.to_list
       (Array.map
          (fun value -> if Float.is_nan value then `Null else `Float value)
          values))

let json_matrix matrix =
  `List (Array.to_list (Array.map json_floats (Matrix.to_arrays matrix)))

let json_ints values =
  `List (Array.to_list (Array.map (fun value -> `Int value) values))

let json_strings values =
  `List (Array.to_list (Array.map (fun value -> `String value) values))

let hex bytes =
  String.init
    (Bytes.length bytes * 2)
    (fun index ->
      let value = Char.code (Bytes.get bytes (index / 2)) in
      "0123456789abcdef".[if index mod 2 = 0 then value lsr 4 else value land 15])

let write path contents =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_string channel contents)

let read path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let native_version () = Xgb.Native.version () |> get

let version_name (major, minor, patch) =
  Printf.sprintf "%d.%d.%d" major minor patch

let common_metadata task pipeline =
  [
    ("format_version", `Int 1);
    ("task", `String task);
    ( "writer_native_version",
      let major, minor, patch = native_version () in
      json_ints [| major; minor; patch |] );
    ("input_names", json_strings input_names);
    ("output_names", json_strings (names (Pipeline.output_schema pipeline)));
    ("probe", json_matrix (probe ()));
    ( "transformed_probe",
      json_matrix
        (Pipeline.transform pipeline ~feature_schema:input_schema ~x:(probe ())
        |> get) );
    ("native_artifact", `String (task ^ "_model_v1.hex"));
    ("pipeline_artifact", `String (task ^ "_pipeline_v2.hex"));
  ]

let write_regression directory pipeline y =
  let x, sample_weight = training () in
  let output_schema = Pipeline.output_schema pipeline in
  let transformed_x =
    Pipeline.transform pipeline ~feature_schema:input_schema ~x |> get
  in
  let native =
    Xgb.Regressor.fit (specification 2) ~sample_weight ~rng:(Rng.create seed)
      ~feature_schema:output_schema ~x:transformed_x ~y ()
    |> get
  in
  let native_probe =
    Pipeline.transform pipeline ~feature_schema:input_schema ~x:(probe ())
    |> get
  in
  let prediction =
    Pipeline.predict pipeline ~feature_schema:input_schema ~x:(probe ())
    |> get |> Target.regression_values |> Vector.to_array
  in
  let native_prediction =
    Xgb.Regressor.predict native ~feature_schema:output_schema ~x:native_probe
    |> get |> Target.regression_values |> Vector.to_array
  in
  write
    (Filename.concat directory "regression_model_v1.hex")
    (hex (Xgb.Artifact.encode_regression native |> get) ^ "\n");
  write
    (Filename.concat directory "regression_pipeline_v2.hex")
    (hex (Artifact.encode_regression pipeline |> get) ^ "\n");
  let metadata =
    `Assoc
      (common_metadata "regression" pipeline
      @ [
          ("predictions", json_floats prediction);
          ("native_predictions", json_floats native_prediction);
        ])
  in
  write
    (Filename.concat directory "regression.metadata.json")
    (Yojson.Basic.pretty_to_string metadata ^ "\n")

let write_classification directory count =
  let task = if count = 2 then "binary" else "multiclass" in
  let y = classification_target count in
  let pipeline = fit (classification_pipeline 2 |> get) y in
  let x, sample_weight = training () in
  let output_schema = Pipeline.output_schema pipeline in
  let transformed_x =
    Pipeline.transform pipeline ~feature_schema:input_schema ~x |> get
  in
  let native =
    Xgb.Classifier.fit (specification 2) ~sample_weight ~rng:(Rng.create seed)
      ~feature_schema:output_schema ~x:transformed_x ~y ()
    |> get
  in
  let native_probe =
    Pipeline.transform pipeline ~feature_schema:input_schema ~x:(probe ())
    |> get
  in
  let prediction =
    Pipeline.predict pipeline ~feature_schema:input_schema ~x:(probe ())
    |> get |> Target.classification_values
  in
  let native_prediction =
    Xgb.Classifier.predict native ~feature_schema:output_schema ~x:native_probe
    |> get |> Target.classification_values
  in
  write
    (Filename.concat directory (task ^ "_model_v1.hex"))
    (hex (Xgb.Artifact.encode_classification native |> get) ^ "\n");
  let bytes =
    if count = 2 then Artifact.encode_binary_classification pipeline |> get
    else Artifact.encode_classification pipeline |> get
  in
  write
    (Filename.concat directory (task ^ "_pipeline_v2.hex"))
    (hex bytes ^ "\n");
  let metadata =
    `Assoc
      (common_metadata task pipeline
      @ [
          ("predictions", json_ints prediction);
          ("native_predictions", json_ints native_prediction);
          ("classes", json_ints (Pipeline.classes pipeline |> get));
          ("native_classes", json_ints (Xgb.Classifier.classes native));
          ( "probabilities",
            json_matrix
              (Pipeline.predict_proba pipeline ~feature_schema:input_schema
                 ~x:(probe ())
              |> get) );
          ( "native_probabilities",
            json_matrix
              (Xgb.Classifier.predict_proba native ~feature_schema:output_schema
                 ~x:native_probe
              |> get) );
        ])
  in
  write
    (Filename.concat directory (task ^ ".metadata.json"))
    (Yojson.Basic.pretty_to_string metadata ^ "\n")

let generate root =
  let directory = Filename.concat root (version_name (native_version ())) in
  if not (Sys.file_exists directory) then Unix.mkdir directory 0o755;
  let y = regression_target () in
  write_regression directory (fit (regression_pipeline 2 |> get) y) y;
  write_classification directory 2;
  write_classification directory 3

let field name json = Yojson.Basic.Util.member name json

let[@warning "-4"] rec check_json name expected observed =
  match (expected, observed) with
  | `List left, `List right ->
      Alcotest.check Alcotest.int name (List.length left) (List.length right);
      List.iter2 (check_json name) left right
  | `Int left, `Int right -> Alcotest.check Alcotest.int name left right
  | (`Float _ | `Int _), (`Float _ | `Int _) ->
      let value = Yojson.Basic.Util.to_number expected in
      Alcotest.check
        (Alcotest.float (2e-6 +. (2e-6 *. abs_float value)))
        name value
        (Yojson.Basic.Util.to_number observed)
  | _ -> Alcotest.check Alcotest.bool name true (expected = observed)

let deploy directory metadata_filename expect_incompatible =
  let worker = Sys.getenv "MODELKIT_XGBOOST_DEPLOYMENT_WORKER" in
  let output = Filename.temp_file "modelkit-deployment-result-" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove output)
    (fun () ->
      let channel =
        Unix.openfile output [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600
      in
      let arguments =
        Array.of_list
          ([ worker; directory; metadata_filename ]
          @ if expect_incompatible then [ "--expect-incompatible" ] else [])
      in
      let status =
        Fun.protect
          ~finally:(fun () -> Unix.close channel)
          (fun () ->
            let pid =
              Unix.create_process worker arguments Unix.stdin channel
                Unix.stderr
            in
            snd (Unix.waitpid [] pid))
      in
      Alcotest.check Alcotest.bool "fresh inference process succeeds" true
        (status = Unix.WEXITED 0);
      read output |> Yojson.Basic.from_string)

let test_golden version task () =
  let root = Sys.getenv "MODELKIT_XGBOOST_ARTIFACT_FIXTURES" in
  let directory = Filename.concat root version in
  let metadata_filename = task ^ ".metadata.json" in
  let expected =
    read (Filename.concat directory metadata_filename)
    |> Yojson.Basic.from_string
  in
  let writer =
    field "writer_native_version" expected
    |> Yojson.Basic.Util.to_list
    |> List.map Yojson.Basic.Util.to_int
  in
  let current =
    let a, b, c = native_version () in
    [ a; b; c ]
  in
  let incompatible = current < writer in
  let observed = deploy directory metadata_filename incompatible in
  if incompatible then
    check_json "typed forward-version rejection" (`String "incompatible")
      (field "status" observed)
  else (
    check_json "fresh process loaded" (`String "loaded")
      (field "status" observed);
    List.iter
      (fun name -> check_json name (field name expected) (field name observed))
      ([
         "task";
         "input_names";
         "output_names";
         "transformed_probe";
         "predictions";
         "native_predictions";
       ]
      @
      if task = "regression" then []
      else
        [ "classes"; "native_classes"; "probabilities"; "native_probabilities" ]
      ))

let test_search_refit () =
  let x, sample_weight = training () in
  let feature_names =
    Feature_names.create ~expected_count:3 input_names |> data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Allow_nan ~feature_names ~sample_weight
      ~x ~y:(regression_target ()) ()
    |> data
  in
  let axis =
    Grid_search.axis ~name:"max_depth" ~values:[| 1; 2 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let grid =
    Grid_search.create ~base:1 ~build:regression_pipeline [| axis |] |> get
  in
  let splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let search =
    Grid_search.Regression.search ~grid ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed dataset
    |> get
  in
  let selected = Grid_search.selection search |> get in
  let pipeline = selected.Grid_search.selected_model in
  let reader =
    Xgb.Artifact.regression_codec () |> get |> Artifact.Codec.decoder
  in
  let bytes = Artifact.encode_regression pipeline |> get in
  let restored =
    Artifact.decode_regression ~decoders:[| reader |] bytes
    |> get |> Artifact.model
  in
  let prediction model =
    Pipeline.predict model ~feature_schema:input_schema ~x:(probe ())
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.))
    "search-refitted pipeline survives serialization" (prediction pipeline)
    (prediction restored);
  let directory = Filename.temp_file "modelkit-search-deployment-" "" in
  Sys.remove directory;
  Unix.mkdir directory 0o755;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun file -> Sys.remove (Filename.concat directory file))
        (Sys.readdir directory);
      Unix.rmdir directory)
    (fun () ->
      write_regression directory pipeline (regression_target ());
      let observed = deploy directory "regression.metadata.json" false in
      let expected =
        read (Filename.concat directory "regression.metadata.json")
        |> Yojson.Basic.from_string
      in
      check_json "search-refit deployed predictions"
        (field "predictions" expected)
        (field "predictions" observed))

let[@warning "-4"] test_unsupported_stage () =
  let stage =
    Pipeline.transformer ~name:"unsupported-imputer"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty stage |> get in
  let pipeline =
    Pipeline.set_estimator builder
      (Xgb.Regressor.estimator (specification 2) |> get)
    |> get
  in
  let fitted = fit pipeline (regression_target ()) in
  match Artifact.encode_regression fitted with
  | Ok _ -> Alcotest.fail "unsupported component was silently dropped"
  | Error error ->
      (match Error.kind error with
      | Error.Artifact { reason; _ } ->
          Alcotest.check Alcotest.string "unsupported stage reported"
            "pipeline stage \"unsupported-imputer\" has no artifact codec"
            reason
      | _ -> Alcotest.fail (Error.to_string error));
      Alcotest.check Alcotest.bool "pipeline cannot serialize" false
        (Pipeline.artifact_report fitted |> Pipeline.artifact_supported)

let () =
  if Array.length Sys.argv = 3 && Sys.argv.(1) = "--write-fixtures" then
    generate Sys.argv.(2)
  else
    Alcotest.run "XGBoost fitted-pipeline deployment"
      [
        ( "deployment",
          List.concat_map
            (fun version ->
              List.map
                (fun task ->
                  (version ^ " " ^ task, `Quick, test_golden version task))
                [ "regression"; "binary"; "multiclass" ])
            [ "3.0.4"; "3.4.2" ]
          @ [
              ("search refit/save/load/deploy", `Quick, test_search_refit);
              ("unsupported preprocessing", `Quick, test_unsupported_stage);
            ] );
      ]
