open Modelkit
open Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let target values = Vector.of_array values |> Target.regression |> data
let array values = Target.regression_values values |> Vector.to_array
let matrix values = Matrix.of_arrays values |> data
let schema x = Feature_schema.of_matrix x |> data
let rng seed = Rng.create (Seed.of_int seed)
let equal_prediction left right = array left = array right

let source () =
  let x =
    matrix [| [| 0.; 1. |]; [| 1.; 2. |]; [| 2.; Float.nan |]; [| 3.; 4. |] |]
  in
  let y = target [| 0.; 1.; 4.; 9. |] in
  (x, y, schema x)

let fit ?sample_weight ?(seed = 1) specification x y feature_schema =
  Regressor.fit specification ?sample_weight ~rng:(rng seed) ~feature_schema ~x
    ~y ()
  |> get

let test_conformance () =
  let x, y, feature_schema = source () in
  let sample_weight =
    Sample_weight.of_array ~expected_length:4 [| 0.; 1.; 2.; 3. |] |> data
  in
  let fixture =
    {
      Conformance.Estimator.specification =
        Specification.create ~rounds:5 () |> get;
      rng = (fun () -> rng 19);
      feature_schema;
      x;
      y;
      sample_weight = Some sample_weight;
      equal_params = ( = );
      prediction_length = Target.length;
      equal_prediction;
    }
  in
  let checks = Conformance.Estimator.check (module Regressor) fixture in
  Array.iter
    (fun check ->
      match check.Conformance.outcome with
      | Conformance.Passed -> ()
      | Conformance.Failed issue ->
          Alcotest.fail (Conformance.issue_to_string issue))
    (Conformance.failures checks);
  Alcotest.check Alcotest.bool "public conformance" true
    (Conformance.passed checks)

let expect_failure result =
  match result with
  | Error error -> error
  | Ok _ -> Alcotest.fail "expected typed failure"

let test_alignment_and_schema () =
  let x, y, feature_schema = source () in
  let specification = Specification.create ~rounds:2 () |> get in
  ignore
    (expect_failure
       (Regressor.fit specification ~rng:(rng 1) ~feature_schema ~x
          ~y:(target [| 1. |]) ()));
  let short = Sample_weight.of_array ~expected_length:1 [| 1. |] |> data in
  ignore
    (expect_failure
       (Regressor.fit specification ~sample_weight:short ~rng:(rng 1)
          ~feature_schema ~x ~y ()));
  let names = Feature_names.create ~expected_count:2 [| "a"; "b" |] |> data in
  let named = Feature_schema.named names in
  let fitted = fit specification x y named in
  let reversed =
    Feature_names.create ~expected_count:2 [| "b"; "a" |]
    |> data |> Feature_schema.named
  in
  let mismatch =
    expect_failure (Regressor.predict fitted ~feature_schema:reversed ~x)
  in
  (match Error.kind mismatch with
  | Error.Feature_schema_mismatch _ -> ()
  | Error.Data _ | Error.Shape_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Callback_failure _ | Error.Cancelled ->
      Alcotest.fail "wrong schema error");
  ignore
    (expect_failure
       (Regressor.predict fitted ~feature_schema:named
          ~x:(matrix [| [| 1. |] |])));
  let prediction, report =
    Regressor.predict_with_report fitted ~feature_schema:named
      ~x:(Matrix.create ~rows:0 ~columns:2 0. |> data)
    |> get
  in
  Alcotest.check Alcotest.int "empty prediction" 0 (Target.length prediction);
  Alcotest.check Alcotest.int64 "empty conversion payload" 0L
    (Input.payload_bytes report)

let test_invalid_numeric_input () =
  let x, y, feature_schema = source () in
  let specification = Specification.create ~rounds:2 () |> get in
  let fitted = fit specification x y feature_schema in
  List.iter
    (fun value ->
      ignore
        (expect_failure
           (Regressor.predict fitted ~feature_schema
              ~x:(matrix [| [| value; 1. |] |]))))
    [ infinity; neg_infinity; Float.max_float ];
  ignore
    (expect_failure
       (Regressor.fit specification ~rng:(rng 1) ~feature_schema ~x
          ~y:(target [| Float.max_float; 1.; 2.; 3. |])
          ()));
  let sample_weight =
    Sample_weight.of_array ~expected_length:4 [| 1.; 1e-300; 1.; 1. |] |> data
  in
  ignore
    (expect_failure
       (Regressor.fit specification ~sample_weight ~rng:(rng 1) ~feature_schema
          ~x ~y ()))

let test_reports_and_fresh_fits () =
  let x, y, feature_schema = source () in
  let specification =
    Specification.create ~rounds:7 ~subsample:0.75 () |> get
  in
  let first = fit specification x y feature_schema in
  let observed = Regressor.predict first ~feature_schema ~x |> get in
  let second = fit (Regressor.clone specification) x y feature_schema in
  Alcotest.check Alcotest.bool "repeat fit prediction" true
    (equal_prediction observed
       (Regressor.predict second ~feature_schema ~x |> get));
  ignore (fit specification x (target [| 99.; 98.; 97.; 96. |]) feature_schema);
  Gc.full_major ();
  Alcotest.check Alcotest.bool "later fit and GC do not change earlier model"
    true
    (equal_prediction observed
       (Regressor.predict first ~feature_schema ~x |> get));
  Alcotest.check Alcotest.bool "parameters preserved" true
    (Regressor.fitted_params first = Specification.params specification);
  let report = Regressor.report first in
  Alcotest.check Alcotest.int "actual rounds" 7 report.Regressor.rounds;
  Alcotest.check Alcotest.bool "no weight conversion" true
    (report.Regressor.weight_conversion = None);
  Alcotest.check Alcotest.int64 "training float32 feature payload" 32L
    (Input.payload_bytes report.Regressor.feature_conversion);
  Alcotest.check Alcotest.int64 "training target payload" 16L
    (Input.payload_bytes report.Regressor.target_conversion)

let test_concurrent_prediction () =
  let x, y, feature_schema = source () in
  let fitted =
    fit (Specification.create ~rounds:3 () |> get) x y feature_schema
  in
  let expected = Regressor.predict fitted ~feature_schema ~x |> get in
  let work () =
    for _ = 1 to 40 do
      Alcotest.check Alcotest.bool "shared fitted predictions" true
        (equal_prediction expected
           (Regressor.predict fitted ~feature_schema ~x |> get))
    done
  in
  let workers = Array.init 3 (fun _ -> Domain.spawn work) in
  work ();
  Array.iter Domain.join workers

let test_pipeline () =
  let x, y, feature_schema = source () in
  let specification = Specification.create ~rounds:4 () |> get in
  let estimator = Regressor.estimator ~name:"trees" specification |> get in
  let stage =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty stage |> get in
  let pipeline = Pipeline.set_estimator builder estimator |> get in
  let sample_weight =
    Sample_weight.of_array ~expected_length:4 [| 0.; 1.; 2.; 3. |] |> data
  in
  let fitted =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng 11) ~feature_schema ~x ~y ()
    |> get
  in
  let prediction = Pipeline.predict fitted ~feature_schema ~x |> get in
  Alcotest.check Alcotest.int "pipeline prediction rows" 4
    (Target.length prediction);
  let imputed =
    Simple_imputer.fit (Simple_imputer.mean ()) ~rng:(rng 0) ~feature_schema ~x
      ~y:None ()
    |> get
  in
  let transformed =
    Simple_imputer.transform imputed ~feature_schema ~x |> get
  in
  let direct = fit ~sample_weight specification transformed y feature_schema in
  (* No sampling is enabled, so independently seeded fits verify weight routing. *)
  Alcotest.check Alcotest.bool "pipeline routes sample weights" true
    (equal_prediction prediction
       (Regressor.predict direct ~feature_schema ~x:transformed |> get));
  let component =
    Pipeline.artifact_report fitted |> Pipeline.artifact_estimator
  in
  let provenance = component.Pipeline.provenance |> Option.get in
  Alcotest.check Alcotest.string "package provenance" "modelkit-xgboost"
    (Pipeline.provenance_package provenance);
  Alcotest.check Alcotest.bool "codec explicitly unsupported" true
    (component.Pipeline.serialization_support = Pipeline.Unsupported);
  ignore (expect_failure (Artifact.encode_regression fitted))

type fixture = {
  vectors : (string, float array) Hashtbl.t;
  matrices : (string, (int, float array) Hashtbl.t) Hashtbl.t;
}

let floats text =
  String.split_on_char ',' text |> List.map float_of_string |> Array.of_list

let read_fixture () =
  let fixtures = Hashtbl.create 12 in
  let _, minor, _ = Native.version () |> get in
  let path =
    Sys.getenv
      (if minor = 0 then "MODELKIT_XGBOOST_REGRESSION_FIXTURE_3_0"
       else "MODELKIT_XGBOOST_REGRESSION_FIXTURE")
  in
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
                    { vectors = Hashtbl.create 20; matrices = Hashtbl.create 2 }
                  in
                  Hashtbl.add fixtures name fixture;
                  fixture
            in
            match fields with
            | [ _; field; values ] ->
                Hashtbl.add fixture.vectors field (floats values)
            | [ _; field; index; values ] ->
                let rows =
                  match Hashtbl.find_opt fixture.matrices field with
                  | Some rows -> rows
                  | None ->
                      let rows = Hashtbl.create 64 in
                      Hashtbl.add fixture.matrices field rows;
                      rows
                in
                Hashtbl.add rows (int_of_string index) (floats values)
            | _ -> Alcotest.fail "invalid fixture row"));
  Hashtbl.to_seq fixtures |> List.of_seq
  |> List.sort (fun (left, _) (right, _) -> String.compare left right)

let fixture_vector fixture name = Hashtbl.find fixture.vectors name
let scalar fixture name = (fixture_vector fixture name).(0)
let integer fixture name = int_of_float (scalar fixture name)

let fixture_matrix fixture name =
  let rows = Hashtbl.find fixture.matrices name in
  Array.init (Hashtbl.length rows) (fun row -> Hashtbl.find rows row) |> matrix

let test_parity fixture () =
  let x = fixture_matrix fixture "x_train" in
  let y = fixture_vector fixture "target" |> target in
  let x_predict = fixture_matrix fixture "x_predict" in
  let feature_schema = schema x in
  let specification =
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
      ~threads:1 ()
    |> get
  in
  let sample_weight =
    Option.map
      (fun values ->
        Sample_weight.of_array ~expected_length:(Matrix.rows x) values |> data)
      (Hashtbl.find_opt fixture.vectors "weight")
  in
  let fitted =
    fit ?sample_weight ~seed:(integer fixture "seed") specification x y
      feature_schema
  in
  Alcotest.check Alcotest.int64 "RNG seed parity"
    (Int64.of_float (scalar fixture "native_seed"))
    (Regressor.report fitted).Regressor.native_seed;
  let observed =
    Regressor.predict fitted ~feature_schema ~x:x_predict |> get |> array
  in
  let expected = fixture_vector fixture "expected" in
  Alcotest.check Alcotest.int "prediction shape" (Array.length expected)
    (Array.length observed);
  Array.iteri
    (fun row expected ->
      let tolerance = 2e-6 +. (2e-6 *. Float.abs expected) in
      Alcotest.check (Alcotest.float tolerance)
        (Format.sprintf "prediction[%d]" row)
        expected observed.(row))
    expected

let () =
  let major, minor, patch = Native.version () |> get in
  Alcotest.run
    (Format.sprintf "XGBoost regression (native %d.%d.%d)" major minor patch)
    [
      ( "protocol",
        [
          Alcotest.test_case "published conformance" `Quick test_conformance;
          Alcotest.test_case "row alignment and schemas" `Quick
            test_alignment_and_schema;
          Alcotest.test_case "numeric admission" `Quick
            test_invalid_numeric_input;
          Alcotest.test_case "reports and independent fits" `Quick
            test_reports_and_fresh_fits;
          Alcotest.test_case "concurrent inference" `Quick
            test_concurrent_prediction;
          Alcotest.test_case "pipeline, weights, and provenance" `Quick
            test_pipeline;
        ] );
      ( "Python parity",
        List.map
          (fun (name, fixture) ->
            Alcotest.test_case name `Quick (test_parity fixture))
          (read_fixture ()) );
    ]
