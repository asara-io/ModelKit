open Modelkit
module X = Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

let schema () =
  let names =
    Feature_names.create ~expected_count:2 [| "signal"; "context" |] |> data
  in
  Feature_schema.named names

let predict model =
  let x = Matrix.of_arrays [| [| 1.; Float.nan |]; [| 8.; 2. |] |] |> data in
  X.Regressor.predict model ~feature_schema:(schema ()) ~x
  |> get |> Target.regression_values |> Vector.to_array

let load_pipeline path =
  let reader =
    X.Artifact.regression_codec ~threads:1 () |> get |> Artifact.Codec.decoder
  in
  Artifact.load_regression ~decoders:[| reader |] ~path ()
  |> get |> Artifact.model

let check_pipeline fitted pipeline =
  let x = Matrix.of_arrays [| [| 1.; Float.nan |]; [| 8.; 2. |] |] |> data in
  let observed =
    Pipeline.predict pipeline ~feature_schema:(schema ()) ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  assert (predict fitted = observed)

let train cache_directory =
  let feature_schema = schema () in
  let cursor = ref 0 in
  let reset () =
    cursor := 0;
    Ok ()
  in
  let next () =
    if !cursor = 32 then Ok None
    else
      let start = !cursor in
      cursor := start + 8;
      let x =
        Matrix.init ~rows:8 ~columns:2 (fun row column ->
            if column = 1 && row mod 3 = 0 then Float.nan
            else float_of_int (start + row + column))
        |> data
      in
      let y =
        Vector.init ~length:8 (fun row -> float_of_int (start + row))
        |> data |> Target.regression |> data
      in
      Ok (Some { X.Regressor.feature_schema; x; y; sample_weight = None })
  in
  let specification = X.Specification.create ~rounds:3 ~max_depth:2 () |> get in
  let fitted =
    X.Regressor.fit_batches specification
      ~rng:(Rng.create (Seed.of_int 42))
      ~feature_schema ~expected_rows:32 ~cache_directory ~next ~reset ()
    |> get
  in
  assert (Sys.readdir cache_directory = [||]);
  let report = X.Regressor.batches_report fitted |> Option.get in
  assert (
    report.batches = 4 && report.rows = 32
    && report.peak_feature_buffer_bytes = 64L);
  let path = Filename.concat cache_directory "streamed.mkxgb" in
  X.Artifact.save_regression ~path fitted |> get;
  let restored = X.Artifact.load_regression ~path () |> get in
  assert (predict fitted = predict restored);
  assert (X.Regressor.batches_report restored = Some report);
  let pipeline = X.Regressor.to_pipeline fitted |> get in
  let path = Filename.concat cache_directory "streamed.modelkit" in
  Artifact.save_regression ~path pipeline |> get;
  check_pipeline fitted (load_pipeline path);
  print_endline
    "Installed streaming fit, cache cleanup, and native/pipeline persistence \
     passed"

let () =
  let directory = Sys.argv.(1) in
  if Array.length Sys.argv = 3 && Sys.argv.(2) = "--load-pipeline" then (
    let fitted =
      X.Artifact.load_regression
        ~path:(Filename.concat directory "streamed.mkxgb")
        ()
      |> get
    in
    check_pipeline fitted
      (load_pipeline (Filename.concat directory "streamed.modelkit"));
    print_endline "Installed streamed pipeline loaded in a fresh process")
  else train directory
