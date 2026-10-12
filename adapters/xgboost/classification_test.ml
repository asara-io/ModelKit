open Modelkit
open Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let matrix values = Matrix.of_arrays values |> data
let schema x = Feature_schema.of_matrix x |> data
let rng seed = Rng.create (Seed.of_int seed)
let labels target = Target.classification_values target

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values |> data

let specification () = Specification.create ~rounds:5 () |> get

let fit ?sample_weight ?(seed = 17) specification x values feature_schema =
  Classifier.fit specification ?sample_weight ~rng:(rng seed) ~feature_schema ~x
    ~y:(Target.classification values)
    ()
  |> get

let source class_count =
  let x =
    Matrix.init ~rows:18 ~columns:2 (fun row column ->
        if row = 7 && column = 1 then Float.nan
        else float_of_int (((row * 7) + (column * 3)) mod 11))
    |> data
  in
  let vocabulary = [| 19; -7; 61 |] in
  (x, Array.init 18 (fun row -> vocabulary.(row mod class_count)), schema x)

let check_probabilities classes rows probabilities =
  Alcotest.check
    (Alcotest.pair Alcotest.int Alcotest.int)
    "probability shape"
    (rows, Array.length classes)
    (Matrix.shape probabilities);
  for row = 0 to rows - 1 do
    let total = ref 0. in
    for column = 0 to Array.length classes - 1 do
      let value = Matrix.get probabilities row column in
      Alcotest.check Alcotest.bool "finite probability" true
        (Float.is_finite value && value >= 0. && value <= 1.);
      total := !total +. value
    done;
    Alcotest.check (Alcotest.float 1e-14) "normalized row" 1. !total
  done

type category = Validation | Shape | Schema | Numerical | Artifact | Other

let category = function
  | Error.Validation _ -> Validation
  | Error.Shape_mismatch _ | Error.Data _ -> Shape
  | Error.Feature_schema_mismatch _ -> Schema
  | Error.Numerical _ -> Numerical
  | Error.Artifact _ -> Artifact
  | Error.Convergence _ | Error.Compatibility _ | Error.Callback_failure _
  | Error.Cancelled ->
      Other

let expect expected = function
  | Error error when category (Error.kind error) = expected -> ()
  | Error error -> Alcotest.fail (Error.to_string error)
  | Ok _ -> Alcotest.fail "expected typed failure"

let test_conformance class_count () =
  let x, y, feature_schema = source class_count in
  let fixture =
    {
      Conformance.Estimator.specification = specification ();
      rng = (fun () -> rng 31);
      feature_schema;
      x;
      y = Target.classification y;
      sample_weight =
        Some (weights (Array.init 18 (fun row -> if row = 0 then 0. else 1.)));
      equal_params = ( = );
      prediction_length = Target.length;
      equal_prediction = (fun left right -> labels left = labels right);
    }
  in
  let checks = Conformance.Estimator.check (module Classifier) fixture in
  Array.iter
    (fun check ->
      match check.Conformance.outcome with
      | Conformance.Passed -> ()
      | Conformance.Failed issue ->
          Alcotest.fail (Conformance.issue_to_string issue))
    (Conformance.failures checks);
  Alcotest.check Alcotest.bool "published classifier conformance" true
    (Conformance.passed checks)

let test_label_mapping () =
  let x, _, feature_schema = source 3 in
  let vocabulary = [| max_int; min_int; 17 |] in
  let y = Array.init 18 (fun row -> vocabulary.(row mod 3)) in
  let target = Target.classification y in
  y.(0) <- 99;
  let fitted =
    Classifier.fit (specification ()) ~rng:(rng 7) ~feature_schema ~x ~y:target
      ()
    |> get
  in
  let classes = Classifier.classes fitted in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "original labels survive encoding" [| min_int; 17; max_int |] classes;
  classes.(0) <- 0;
  let classes = Classifier.classes fitted in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "class inspection cannot mutate model" [| min_int; 17; max_int |] classes;
  let probabilities =
    Classifier.predict_proba fitted ~feature_schema ~x |> get
  in
  check_probabilities classes 18 probabilities;
  let predicted =
    Classifier.predict fitted ~feature_schema ~x |> get |> labels
  in
  Array.iteri
    (fun row label ->
      let best = ref 0 in
      for column = 1 to 2 do
        if
          Matrix.get probabilities row column
          > Matrix.get probabilities row !best
        then best := column
      done;
      Alcotest.check Alcotest.int "probability column decodes to original label"
        classes.(!best) label)
    predicted

let test_zero_weight_rows () =
  let x =
    matrix [| [| -2.; 1. |]; [| 2.; -1. |]; [| 0.; 0. |]; [| 1e6; -1e6 |] |]
  in
  let feature_schema = schema x in
  let spec = specification () in
  let fitted =
    fit
      ~sample_weight:(weights [| 1.; 2.; 1.; 0. |])
      spec x [| -9; 7; -9; max_int |] feature_schema
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "zero-only class excluded" [| -9; 7 |]
    (Classifier.classes fitted);
  let short_x = matrix [| [| -2.; 1. |]; [| 2.; -1. |]; [| 0.; 0. |] |] in
  let baseline =
    fit
      ~sample_weight:(weights [| 1.; 2.; 1. |])
      spec short_x [| -9; 7; -9 |] feature_schema
  in
  let actual =
    Classifier.predict_proba fitted ~feature_schema ~x:short_x
    |> get |> Matrix.to_arrays
  in
  let expected =
    Classifier.predict_proba baseline ~feature_schema ~x:short_x
    |> get |> Matrix.to_arrays
  in
  Alcotest.check Alcotest.bool "zero-weight outlier does not change training"
    true (actual = expected);
  let report = Classifier.report fitted in
  Alcotest.check Alcotest.int "original rows" 4 report.Classifier.training_rows;
  Alcotest.check Alcotest.int "effective rows" 3
    report.Classifier.effective_rows;
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "source shape retained" [| 4; 2 |]
    (Input.source_shape report.Classifier.feature_conversion);
  Alcotest.check Alcotest.int64 "retained filtered features" 24L
    (Input.payload_bytes report.Classifier.feature_conversion);
  Alcotest.check Alcotest.int64 "full feature staging" 32L
    (Input.temporary_payload_bytes report.Classifier.feature_conversion);
  Alcotest.check Alcotest.int64 "label and selection staging"
    (Int64.of_int (7 * (Sys.word_size / 8)))
    (Input.temporary_payload_bytes report.Classifier.target_conversion);
  let weight_report = Option.get report.Classifier.weight_conversion in
  Alcotest.check Alcotest.int64 "retained filtered weights" 12L
    (Input.payload_bytes weight_report);
  Alcotest.check Alcotest.int64 "full weight staging" 16L
    (Input.temporary_payload_bytes weight_report)

let test_degenerate_targets () =
  let spec = specification () in
  let run ?sample_weight x y =
    Classifier.fit spec ?sample_weight ~rng:(rng 1) ~feature_schema:(schema x)
      ~x ~y:(Target.classification y) ()
  in
  expect Validation (run (Matrix.create ~rows:0 ~columns:2 0. |> data) [||]);
  expect Validation (run (matrix [| [| 1.; 2. |] |]) [| 7 |]);
  expect Validation (run (matrix [| [| 1. |]; [| 2. |] |]) [| 7; 7 |]);
  expect Validation
    (run
       ~sample_weight:(weights [| 1.; 0.; 0. |])
       (matrix [| [| 1. |]; [| 2. |]; [| 3. |] |])
       [| 7; 9; 11 |]);
  let x = matrix [| [| 1. |]; [| 2. |]; [| 3. |] |] in
  let fitted =
    fit
      ~sample_weight:(weights [| 1.; 1.; 0. |])
      spec x [| 7; 9; 11 |] (schema x)
  in
  Alcotest.check Alcotest.int
    "multiclass target with two effective classes uses binary" 2
    (Array.length (Classifier.classes fitted))

let test_schema_and_numeric_errors () =
  let x, y, feature_schema = source 3 in
  let spec = specification () in
  expect Shape
    (Classifier.fit spec ~rng:(rng 1) ~feature_schema ~x
       ~y:(Target.classification [| 1 |])
       ());
  expect Shape
    (Classifier.fit spec ~sample_weight:(weights [| 1. |]) ~rng:(rng 1)
       ~feature_schema ~x ~y:(Target.classification y) ());
  let named =
    Feature_names.create ~expected_count:2 [| "a"; "b" |]
    |> data |> Feature_schema.named
  in
  let fitted = fit spec x y named in
  let reversed =
    Feature_names.create ~expected_count:2 [| "b"; "a" |]
    |> data |> Feature_schema.named
  in
  expect Schema (Classifier.predict fitted ~feature_schema:reversed ~x);
  expect Shape
    (Classifier.predict_proba fitted ~feature_schema:named
       ~x:(matrix [| [| 1. |] |]));
  List.iter
    (fun value ->
      expect Validation
        (Classifier.predict_proba fitted ~feature_schema:named
           ~x:(matrix [| [| value; 1. |] |])))
    [ infinity; neg_infinity; Float.max_float ];
  let bad_weights =
    weights (Array.init 18 (fun row -> if row = 0 then 1e-300 else 1.))
  in
  expect Validation
    (Classifier.fit spec ~sample_weight:bad_weights ~rng:(rng 1) ~feature_schema
       ~x ~y:(Target.classification y) ());
  let bad_x = matrix [| [| 1.; 2. |]; [| 2.; 1. |]; [| infinity; 0. |] |] in
  expect Validation
    (Classifier.fit spec
       ~sample_weight:(weights [| 1.; 1.; 0. |])
       ~rng:(rng 1) ~feature_schema:named ~x:bad_x
       ~y:(Target.classification [| 1; 2; 3 |])
       ());
  let empty = Matrix.create ~rows:0 ~columns:2 0. |> data in
  let probabilities, report =
    Classifier.predict_proba_with_report fitted ~feature_schema:named ~x:empty
    |> get
  in
  Alcotest.check
    (Alcotest.pair Alcotest.int Alcotest.int)
    "empty class columns preserved" (0, 3)
    (Matrix.shape probabilities);
  Alcotest.check Alcotest.int "empty labels" 0
    (Classifier.predict fitted ~feature_schema:named ~x:empty
    |> get |> Target.length);
  Alcotest.check Alcotest.int64 "empty input payload" 0L
    (Input.payload_bytes report)

let test_ties () =
  List.iter
    (fun vocabulary ->
      let rows = 12 in
      let x = Matrix.create ~rows ~columns:1 1. |> data in
      let y =
        Array.init rows (fun row ->
            vocabulary.(row mod Array.length vocabulary))
      in
      let feature_schema = schema x in
      let fitted =
        fit
          (Specification.create ~rounds:3 ~learning_rate:0. () |> get)
          x y feature_schema
      in
      let predicted =
        Classifier.predict fitted ~feature_schema ~x |> get |> labels
      in
      let lowest = Array.fold_left min max_int vocabulary in
      Alcotest.check
        (Alcotest.array Alcotest.int)
        "exact ties choose lowest original label" (Array.make rows lowest)
        predicted;
      check_probabilities
        (Classifier.classes fitted)
        rows
        (Classifier.predict_proba fitted ~feature_schema ~x |> get))
    [ [| 9; -7 |]; [| 9; -7; 61 |] ]

let test_fresh_fits_and_concurrency () =
  let x, y, feature_schema = source 3 in
  let spec = Specification.create ~rounds:5 ~subsample:0.75 () |> get in
  let fitted = fit spec x y feature_schema in
  let expected =
    Classifier.predict_proba fitted ~feature_schema ~x
    |> get |> Matrix.to_arrays
  in
  let clone = fit (Classifier.clone spec) x y feature_schema in
  Alcotest.check Alcotest.bool "repeat fit is deterministic" true
    (expected
    = (Classifier.predict_proba clone ~feature_schema ~x
      |> get |> Matrix.to_arrays));
  ignore (fit spec x (Array.map (( + ) 100) y) feature_schema);
  Alcotest.check Alcotest.bool "fitted parameters preserved" true
    (Classifier.fitted_params fitted = Specification.params spec);
  let work () =
    for index = 0 to 29 do
      if index mod 7 = 0 then Gc.full_major ();
      Alcotest.check Alcotest.bool "shared probability output" true
        (expected
        = (Classifier.predict_proba fitted ~feature_schema ~x
          |> get |> Matrix.to_arrays));
      Alcotest.check Alcotest.int "concurrent label rows" 18
        (Classifier.predict fitted ~feature_schema ~x |> get |> Target.length)
    done
  in
  let workers = Array.init 3 (fun _ -> Domain.spawn work) in
  work ();
  Array.iter Domain.join workers

let test_pipeline class_count () =
  let x, y, feature_schema = source class_count in
  let sample_weight =
    weights (Array.init 18 (fun row -> 1. +. float_of_int (row mod 3)))
  in
  let spec = specification () in
  let estimator = Classifier.estimator ~name:"trees" spec |> get in
  let imputer =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty imputer |> get in
  let pipeline = Pipeline.set_estimator builder estimator |> get in
  let target = Target.classification y in
  let fitted =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng 17) ~feature_schema ~x
      ~y:target ()
    |> get
  in
  let classes = Pipeline.classes fitted |> get in
  let probabilities = Pipeline.predict_proba fitted ~feature_schema ~x |> get in
  check_probabilities classes 18 probabilities;
  let loss =
    Multiclass_classification_metrics.log_loss ~sample_weight ~truth:target
      ~classes ~probabilities ()
    |> get
  in
  Alcotest.check Alcotest.bool "probabilities enter public log-loss scoring"
    true
    (Float.is_finite loss && loss >= 0.);
  let prediction = Pipeline.predict fitted ~feature_schema ~x |> get in
  Alcotest.check Alcotest.int "pipeline label rows" 18
    (Target.length prediction);
  let stage =
    Simple_imputer.fit (Simple_imputer.mean ()) ~rng:(rng 0) ~feature_schema ~x
      ~y:None ()
    |> get
  in
  let transformed = Simple_imputer.transform stage ~feature_schema ~x |> get in
  let direct = fit ~sample_weight spec transformed y feature_schema in
  Alcotest.check Alcotest.bool "pipeline routes weights and probabilities" true
    (Matrix.to_arrays probabilities
    = (Classifier.predict_proba direct ~feature_schema ~x:transformed
      |> get |> Matrix.to_arrays));
  let component =
    Pipeline.artifact_report fitted |> Pipeline.artifact_estimator
  in
  let provenance = Option.get component.Pipeline.provenance in
  Alcotest.check Alcotest.string "classifier provenance" "Classifier"
    (Pipeline.provenance_implementation provenance);
  Alcotest.check Alcotest.string "current package version" "0.6.0-dev"
    (Pipeline.provenance_version provenance);
  Alcotest.check Alcotest.bool "codec remains unsupported" true
    (component.Pipeline.serialization_support = Pipeline.Unsupported);
  let bare = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let bare_fitted =
    Pipeline.fit bare ~sample_weight ~rng:(rng 17) ~feature_schema ~x ~y:target
      ()
    |> get
  in
  expect Artifact (Modelkit.Artifact.encode_binary_classification bare_fitted)

type fixture = {
  vectors : (string, string array) Hashtbl.t;
  matrices : (string, (int, float array) Hashtbl.t) Hashtbl.t;
}

let split values = String.split_on_char ',' values |> Array.of_list
let vector fixture name = Hashtbl.find fixture.vectors name
let scalar fixture name = float_of_string (vector fixture name).(0)
let integer fixture name = int_of_string (vector fixture name).(0)
let integers fixture name = Array.map int_of_string (vector fixture name)

let fixture_matrix fixture name =
  let rows = Hashtbl.find fixture.matrices name in
  Array.init (Hashtbl.length rows) (fun row -> Hashtbl.find rows row) |> matrix

let read_fixture () =
  let _, minor, _ = Native.version () |> get in
  let path =
    Sys.getenv
      (if minor = 0 then "MODELKIT_XGBOOST_CLASSIFICATION_FIXTURE_3_0"
       else "MODELKIT_XGBOOST_CLASSIFICATION_FIXTURE")
  in
  let fixtures = Hashtbl.create 20 in
  In_channel.with_open_text path (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if line <> "" && line.[0] <> '#' then
            let fields = String.split_on_char '\t' line in
            let name = List.hd fields in
            let fixture =
              match Hashtbl.find_opt fixtures name with
              | Some fixture -> fixture
              | None ->
                  let fixture =
                    { vectors = Hashtbl.create 20; matrices = Hashtbl.create 3 }
                  in
                  Hashtbl.add fixtures name fixture;
                  fixture
            in
            match fields with
            | [ _; field; values ] ->
                Hashtbl.add fixture.vectors field (split values)
            | [ _; field; index; values ] ->
                let rows =
                  match Hashtbl.find_opt fixture.matrices field with
                  | Some rows -> rows
                  | None ->
                      let rows = Hashtbl.create 80 in
                      Hashtbl.add fixture.matrices field rows;
                      rows
                in
                Hashtbl.add rows (int_of_string index)
                  (Array.map float_of_string (split values))
            | _ -> Alcotest.fail "invalid fixture row"));
  Hashtbl.to_seq fixtures |> List.of_seq
  |> List.sort (fun (left, _) (right, _) -> String.compare left right)

let test_parity fixture () =
  let x = fixture_matrix fixture "x_train" in
  let feature_schema = schema x in
  let y = integers fixture "target" in
  let spec =
    Specification.create ~rounds:(integer fixture "rounds")
      ~max_depth:(integer fixture "max_depth")
      ~max_bin:(integer fixture "max_bin")
      ~learning_rate:(scalar fixture "learning_rate")
      ~min_child_weight:(scalar fixture "min_child_weight")
      ~min_split_loss:(scalar fixture "min_split_loss")
      ~reg_alpha:(scalar fixture "reg_alpha")
      ~reg_lambda:(scalar fixture "reg_lambda")
      ~subsample:(scalar fixture "subsample")
      ~colsample_bytree:(scalar fixture "colsample_bytree")
      ()
    |> get
  in
  let sample_weight =
    Option.map
      (fun values -> weights (Array.map float_of_string values))
      (Hashtbl.find_opt fixture.vectors "weight")
  in
  let fitted =
    fit ?sample_weight ~seed:(integer fixture "seed") spec x y feature_schema
  in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "class mapping parity"
    (integers fixture "classes")
    (Classifier.classes fitted);
  let report = Classifier.report fitted in
  Alcotest.check Alcotest.int "effective row parity"
    (integer fixture "effective_rows")
    report.Classifier.effective_rows;
  Alcotest.check Alcotest.int64 "RNG parity"
    (Int64.of_string (vector fixture "native_seed").(0))
    report.Classifier.native_seed;
  let test = fixture_matrix fixture "x_predict" in
  let actual = Classifier.predict_proba fitted ~feature_schema ~x:test |> get in
  let expected = fixture_matrix fixture "probabilities" in
  check_probabilities (Classifier.classes fitted) (Matrix.rows test) actual;
  for row = 0 to Matrix.rows actual - 1 do
    for column = 0 to Matrix.columns actual - 1 do
      let value = Matrix.get expected row column in
      Alcotest.check
        (Alcotest.float (2e-6 +. (2e-6 *. Float.abs value)))
        "probability parity" value
        (Matrix.get actual row column)
    done
  done;
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "original-label prediction parity"
    (integers fixture "prediction")
    (Classifier.predict fitted ~feature_schema ~x:test |> get |> labels)

let () =
  let major, minor, patch = Native.version () |> get in
  Alcotest.run
    (Format.sprintf "XGBoost classification (native %d.%d.%d)" major minor patch)
    [
      ( "protocol",
        [
          Alcotest.test_case "binary conformance" `Quick (test_conformance 2);
          Alcotest.test_case "multiclass conformance" `Quick
            (test_conformance 3);
          Alcotest.test_case "lossless original labels" `Quick
            test_label_mapping;
          Alcotest.test_case "zero-weight filtering and allocation" `Quick
            test_zero_weight_rows;
          Alcotest.test_case "degenerate targets" `Quick test_degenerate_targets;
          Alcotest.test_case "schema and numeric errors" `Quick
            test_schema_and_numeric_errors;
          Alcotest.test_case "deterministic ties" `Quick test_ties;
          Alcotest.test_case "independent fits, GC, and concurrency" `Quick
            test_fresh_fits_and_concurrency;
          Alcotest.test_case "binary pipeline dispatch" `Quick (test_pipeline 2);
          Alcotest.test_case "multiclass pipeline dispatch" `Quick
            (test_pipeline 3);
        ] );
      ( "Python parity",
        List.map
          (fun (name, fixture) ->
            Alcotest.test_case name `Quick (test_parity fixture))
          (read_fixture ()) );
    ]
