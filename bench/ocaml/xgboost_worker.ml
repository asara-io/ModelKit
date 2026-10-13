open Modelkit
module Xgb = Modelkit_xgboost

let require message condition = if not condition then failwith message

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

let json_int64 value = `Intlit (Int64.to_string value)

let json_floats values =
  `List (Array.to_list (Array.map (fun value -> `Float value) values))

let runs = 3
let seed = Seed.of_int 73

let measure operation =
  Gc.full_major ();
  let allocated_before = Gc.allocated_bytes () in
  let started = Unix.gettimeofday () in
  let result = operation () in
  let elapsed = Unix.gettimeofday () -. started in
  let allocated = Gc.allocated_bytes () -. allocated_before in
  require "invalid measurement" (elapsed >= 0. && allocated >= 0.);
  ( result,
    `Assoc
      [
        ("elapsed_ns", json_int64 (Int64.of_float (elapsed *. 1e9)));
        ("ocaml_allocated_bytes", `Float allocated);
      ] )

let payload reports =
  let sum field =
    List.fold_left
      (fun total report -> Int64.add total (field report))
      0L reports
  in
  `Assoc
    [
      ("float32_payload_bytes", json_int64 (sum Xgb.Input.payload_bytes));
      ( "temporary_payload_bytes",
        json_int64 (sum Xgb.Input.temporary_payload_bytes) );
      ( "rounded_values",
        `Int
          (List.fold_left
             (fun total report -> total + Xgb.Input.rounded_values report)
             0 reports) );
    ]

let conversion name expected_bytes convert report =
  let check value =
    require "conversion payload differs"
      (Xgb.Input.payload_bytes (report value) = expected_bytes)
  in
  let warm = convert () in
  check warm;
  let measurements =
    List.init runs (fun _ ->
        let value, measurement = measure convert in
        check value;
        measurement)
  in
  ( name,
    `Assoc
      [
        ("measurements", `List measurements);
        ("payload", payload [ report warm ]);
      ] )

let task name ~fit ~predict ~signature ~reports ~check =
  let warm = fit () in
  let response, _ = predict warm in
  check warm response;
  let expected = signature response in
  let measurements =
    List.init runs (fun _ ->
        let fitted, fit_measurement = measure fit in
        let (response, conversion), predict_measurement =
          measure (fun () -> predict fitted)
        in
        check fitted response;
        require "repeated fits changed responses" (signature response = expected);
        `Assoc
          [
            ("fit", fit_measurement);
            ("predict", predict_measurement);
            ("fit_conversion_payload", payload (reports fitted));
            ("predict_conversion_payload", payload [ conversion ]);
          ])
  in
  ( name,
    `Assoc
      [
        ("measurements", `List measurements); ("signature", json_floats expected);
      ] )

let finite values =
  Array.iter
    (fun value -> require "nonfinite response" (Float.is_finite value))
    values

let () =
  require "start with single-thread OpenMP bounds"
    (Sys.getenv_opt "OMP_NUM_THREADS" = Some "1"
    && Sys.getenv_opt "OMP_THREAD_LIMIT" = Some "1");
  let rows, columns, rounds =
    if Array.length Sys.argv = 1 then (4096, 16, 12)
    else if Array.length Sys.argv = 4 then
      ( int_of_string Sys.argv.(1),
        int_of_string Sys.argv.(2),
        int_of_string Sys.argv.(3) )
    else failwith "usage: xgboost_worker [ROWS COLUMNS ROUNDS]"
  in
  require "invalid smoke dimensions" (rows >= 6 && columns >= 2 && rounds > 0);
  let features row column =
    if (row + column) mod 17 = 0 then Float.nan
    else float_of_int ((((row * 17) + (column * 31)) mod 101) - 50) /. 13.
  in
  let x = Matrix.init ~rows ~columns features |> data in
  let probe_rows = 64 in
  let probe =
    Matrix.init ~rows:probe_rows ~columns (fun row -> features (row + rows))
    |> data
  in
  let schema = Feature_schema.of_matrix x |> data in
  let y =
    Vector.of_array
      (Array.init rows (fun row -> float_of_int (row mod 19) /. 3.))
    |> Target.regression |> data
  in
  let weights =
    Sample_weight.of_array ~expected_length:rows
      (Array.init rows (fun row ->
           if row mod 11 = 0 then 0. else 1. +. (float_of_int (row mod 3) /. 2.)))
    |> data
  in
  let specification =
    Xgb.Specification.create ~rounds ~max_depth:3 ~max_bin:32 ~threads:1 ()
    |> get
  in
  let bytes count = Int64.mul 4L (Int64.of_int count) in
  let conversions =
    [
      conversion "features"
        (bytes (rows * columns))
        (fun () -> Xgb.Input.features ~feature_schema:schema x |> get)
        Xgb.Input.feature_report;
      conversion "regression_target" (bytes rows)
        (fun () -> Xgb.Input.regression_target y |> get)
        Xgb.Input.target_report;
      conversion "sample_weight" (bytes rows)
        (fun () -> Xgb.Input.sample_weight weights |> get)
        Xgb.Input.weight_report;
    ]
  in
  let regression =
    task "regression"
      ~fit:(fun () ->
        Xgb.Regressor.fit specification ~sample_weight:weights
          ~rng:(Rng.create seed) ~feature_schema:schema ~x ~y ()
        |> get)
      ~predict:(fun fitted ->
        Xgb.Regressor.predict_with_report fitted ~feature_schema:schema ~x:probe
        |> get)
      ~signature:(fun response ->
        Target.regression_values response |> Vector.to_array)
      ~reports:(fun fitted ->
        let report = Xgb.Regressor.report fitted in
        [
          report.Xgb.Regressor.feature_conversion;
          report.Xgb.Regressor.target_conversion;
          Option.get report.Xgb.Regressor.weight_conversion;
        ])
      ~check:(fun _ response ->
        require "regression prediction length"
          (Target.length response = probe_rows);
        Target.regression_values response |> Vector.to_array |> finite)
  in
  let classification count =
    let classes =
      if count = 2 then [| -7; 16777219 |] else [| -7; 61; 16777219 |]
    in
    let target =
      Target.classification
        (Array.init rows (fun row -> classes.(row mod count)))
    in
    task
      (if count = 2 then "binary" else "multiclass")
      ~fit:(fun () ->
        Xgb.Classifier.fit specification ~sample_weight:weights
          ~rng:(Rng.create seed) ~feature_schema:schema ~x ~y:target ()
        |> get)
      ~predict:(fun fitted ->
        Xgb.Classifier.predict_proba_with_report fitted ~feature_schema:schema
          ~x:probe
        |> get)
      ~signature:(fun response ->
        Matrix.to_arrays response |> Array.to_list |> Array.concat)
      ~reports:(fun fitted ->
        let report = Xgb.Classifier.report fitted in
        [
          report.Xgb.Classifier.feature_conversion;
          report.Xgb.Classifier.target_conversion;
          Option.get report.Xgb.Classifier.weight_conversion;
        ])
      ~check:(fun fitted response ->
        require "class mapping differs" (Xgb.Classifier.classes fitted = classes);
        require "probability shape" (Matrix.shape response = (probe_rows, count));
        let labels =
          Xgb.Classifier.predict fitted ~feature_schema:schema ~x:probe
          |> get |> Target.classification_values
        in
        for row = 0 to probe_rows - 1 do
          let sum = ref 0. and best = ref 0 in
          for column = 0 to count - 1 do
            let value = Matrix.get response row column in
            require "invalid probability"
              (Float.is_finite value && value >= 0. && value <= 1.);
            sum := !sum +. value;
            if value > Matrix.get response row !best then best := column
          done;
          require "probability normalization" (abs_float (!sum -. 1.) <= 1e-12);
          require "label/probability alignment" (labels.(row) = classes.(!best))
        done)
  in
  let fitted =
    Xgb.Regressor.fit specification ~rng:(Rng.create seed)
      ~feature_schema:schema ~x ~y ()
    |> get
  in
  let provenance = Xgb.Artifact.regression_provenance fitted in
  let major, minor, patch = Xgb.Native.version () |> get in
  `Assoc
    [
      ("schema_version", `Int 1);
      ("scenario", `String "xgboost_smoke_v1");
      ("claim_eligible", `Bool false);
      ("warmup_runs", `Int 1);
      ("measured_runs", `Int runs);
      ("rows", `Int rows);
      ("columns", `Int columns);
      ("prediction_rows", `Int probe_rows);
      ("rounds", `Int rounds);
      ("max_depth", `Int 3);
      ("max_bin", `Int 32);
      ("learning_rate", `Float 0.3);
      ("min_child_weight", `Float 1.);
      ("min_split_loss", `Float 0.);
      ("reg_alpha", `Float 0.);
      ("reg_lambda", `Float 1.);
      ("subsample", `Float 1.);
      ("colsample_bytree", `Float 1.);
      ("seed", `Int 73);
      ("threads", `Int 1);
      ("omp_num_threads", `String "1");
      ("omp_thread_limit", `String "1");
      ("ocaml", `String Sys.ocaml_version);
      ("word_size", `Int Sys.word_size);
      ("modelkit", `String provenance.Xgb.Artifact.writer_adapter_version);
      ( "modelkit_xgboost",
        `String provenance.Xgb.Artifact.writer_adapter_version );
      ("binding", `String provenance.Xgb.Artifact.binding_version);
      ("native", `String (Printf.sprintf "%d.%d.%d" major minor patch));
      ("conversions", `Assoc conversions);
      ("tasks", `Assoc [ regression; classification 2; classification 3 ]);
    ]
  |> Yojson.Safe.to_string |> print_endline
