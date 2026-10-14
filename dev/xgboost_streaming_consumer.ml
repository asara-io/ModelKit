open Modelkit
module X = Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

let () =
  let cache_directory = Sys.argv.(1) in
  let names =
    Feature_names.create ~expected_count:2 [| "signal"; "context" |] |> data
  in
  let feature_schema = Feature_schema.named names in
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
  let x = Matrix.of_arrays [| [| 1.; Float.nan |]; [| 8.; 2. |] |] |> data in
  let predict model =
    X.Regressor.predict model ~feature_schema ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  assert (predict fitted = predict restored);
  assert (X.Regressor.batches_report restored = Some report);
  print_endline
    "Installed streaming fit, cache cleanup, and persisted inference passed"
