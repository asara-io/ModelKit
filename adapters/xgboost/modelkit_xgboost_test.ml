open Modelkit
open Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

type category = Validation | Shape | Data | Numerical | Other

let category = function
  | Error.Validation _ -> Validation
  | Error.Shape_mismatch _ -> Shape
  | Error.Data _ -> Data
  | Error.Numerical _ -> Numerical
  | Error.Feature_schema_mismatch _ | Error.Convergence _
  | Error.Compatibility _ | Error.Artifact _ | Error.Callback_failure _
  | Error.Cancelled ->
      Other

let expect expected = function
  | Error error when category (Error.kind error) = expected -> error
  | Error error -> Alcotest.fail (Error.to_string error)
  | Ok _ -> Alcotest.fail "expected a typed failure"

let features values =
  let x = Matrix.of_arrays values |> data in
  let feature_schema = Feature_schema.of_matrix x |> data in
  Input.features ~feature_schema x |> get

let target values =
  Vector.of_array values |> Target.regression |> data |> Input.regression_target
  |> get

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> data |> Input.sample_weight |> get

let test_specification () =
  let specification = Specification.create () |> get in
  let parameters = Specification.params specification in
  Alcotest.check Alcotest.int "rounds" 100 parameters.Specification.rounds;
  Alcotest.check Alcotest.int "one booster thread" 1
    parameters.Specification.threads;
  Alcotest.check Alcotest.bool "clone preserves all parameters" true
    (Specification.params (Specification.clone specification) = parameters);
  let changed = { parameters with Specification.rounds = 3; threads = 2 } in
  Alcotest.check Alcotest.int "independent parameter value" 3
    changed.Specification.rounds;
  Alcotest.check Alcotest.int "specification is unchanged" 100
    (Specification.params specification).Specification.rounds;
  ignore (Specification.create ~learning_rate:0. ~reg_lambda:0. () |> get);
  ignore (Specification.create ~learning_rate:1. ~max_bin:2 () |> get)

let test_invalid_specifications () =
  let cases =
    [
      Specification.create ~rounds:0 ();
      Specification.create ~max_depth:0 ();
      Specification.create ~max_bin:1 ();
      Specification.create ~threads:0 ();
      Specification.create ~learning_rate:(-0.1) ();
      Specification.create ~learning_rate:1.1 ();
      Specification.create ~learning_rate:Float.nan ();
      Specification.create ~min_child_weight:(-1.) ();
      Specification.create ~min_split_loss:infinity ();
      Specification.create ~reg_alpha:Float.nan ();
      Specification.create ~reg_lambda:Float.max_float ();
      Specification.create ~subsample:0. ();
      Specification.create ~subsample:1e-300 ();
      Specification.create ~subsample:infinity ();
      Specification.create ~colsample_bytree:1.01 ();
      Specification.create ~colsample_bytree:1e-300 ();
    ]
  in
  List.iter (fun result -> ignore (expect Validation result)) cases;
  if Sys.word_size = 64 then
    ignore (expect Validation (Specification.create ~rounds:max_int ()))

let test_conversion () =
  let x = features [| [| 1.; 0.1 |]; [| Float.nan; -0. |] |] in
  Alcotest.check (Alcotest.float 0.) "float32 rounding" 0.10000000149011612
    (Input.feature_value x ~row:0 ~column:1);
  Alcotest.check Alcotest.bool "missing NaN" true
    (Float.is_nan (Input.feature_value x ~row:1 ~column:0));
  Alcotest.check Alcotest.bool "signed zero preserved" true
    (Float.sign_bit (Input.feature_value x ~row:1 ~column:1));
  let report = Input.feature_report x in
  Alcotest.check Alcotest.int64 "float32 payload" 16L
    (Input.payload_bytes report);
  Alcotest.check Alcotest.int64 "no staging payload" 0L
    (Input.temporary_payload_bytes report);
  Alcotest.check Alcotest.int "one rounded finite value" 1
    (Input.rounded_values report);
  let shape = Input.source_shape report in
  shape.(0) <- 99;
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "shape inspection does not mutate report" [| 2; 2 |]
    (Input.source_shape report)

let test_schema_and_ownership () =
  let source =
    Bigarray.Array2.of_array Bigarray.float64 Bigarray.c_layout
      [| [| 2.; 3. |] |]
  in
  let matrix = Matrix.of_bigarray source in
  let names = Feature_names.create ~expected_count:2 [| "a"; "b" |] |> data in
  let schema = Feature_schema.named names in
  let converted = Input.features ~feature_schema:schema matrix |> get in
  source.{0, 0} <- 100.;
  Alcotest.check (Alcotest.float 0.) "conversion owns its values" 2.
    (Input.feature_value converted ~row:0 ~column:0);
  Alcotest.check Alcotest.bool "named schema retained" true
    (Feature_schema.equal schema (Input.feature_schema converted));
  let wrong = Feature_schema.anonymous ~feature_count:1 |> data in
  ignore (expect Data (Input.features ~feature_schema:wrong matrix));
  let empty = Matrix.create ~rows:0 ~columns:2 0. |> data in
  ignore (expect Validation (Input.features ~feature_schema:schema empty));
  let no_columns = Matrix.create ~rows:1 ~columns:0 0. |> data in
  let schema = Feature_schema.of_matrix no_columns |> data in
  ignore (expect Validation (Input.features ~feature_schema:schema no_columns))

let test_numeric_boundaries () =
  let maximum = Int32.float_of_bits 0x7f7fffffl in
  let tiny = Int64.float_of_bits 1L in
  let converted = features [| [| maximum; -.maximum; tiny |] |] in
  Alcotest.check (Alcotest.float 0.) "largest finite float32" maximum
    (Input.feature_value converted ~row:0 ~column:0);
  Alcotest.check (Alcotest.float 0.) "feature underflow is explicit" 0.
    (Input.feature_value converted ~row:0 ~column:2);
  Alcotest.check Alcotest.int "underflow counted" 1
    (Input.rounded_values (Input.feature_report converted));
  List.iter
    (fun value ->
      let matrix = Matrix.of_arrays [| [| 1.; value |] |] |> data in
      let feature_schema = Feature_schema.of_matrix matrix |> data in
      let error = expect Validation (Input.features ~feature_schema matrix) in
      match Error.kind error with
      | Error.Validation { name; _ } ->
          Alcotest.check Alcotest.string "coordinates" "features[0,1]" name
      | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
      | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
      | Error.Artifact _ | Error.Callback_failure _ | Error.Cancelled ->
          Alcotest.fail "unexpected category")
    [ infinity; neg_infinity; Float.max_float ];
  let source =
    Vector.of_array [| Float.max_float |] |> Target.regression |> data
  in
  ignore (expect Validation (Input.regression_target source));
  List.iter
    (fun value ->
      let source =
        Sample_weight.of_array ~expected_length:2 [| 1.; value |] |> data
      in
      ignore (expect Validation (Input.sample_weight source)))
    [ Float.max_float; tiny ];
  let converted = weights [| 0.; 0.1 |] in
  Alcotest.check (Alcotest.float 0.) "zero weights preserved" 0.
    (Input.weight_value converted 0);
  Alcotest.check Alcotest.int64 "weight payload" 8L
    (Input.payload_bytes (Input.weight_report converted))

let test_targets () =
  let source = [| 0.1; 2. |] in
  let converted = target source in
  source.(1) <- 99.;
  Alcotest.check (Alcotest.float 0.) "target owns conversion" 2.
    (Input.target_value converted 1);
  Alcotest.check Alcotest.int "target rounding counted" 1
    (Input.rounded_values (Input.target_report converted));
  let encoded =
    Input.encoded_classification_target ~class_count:16777217
      (Target.classification [| 0; 16777216 |])
    |> get
  in
  Alcotest.check (Alcotest.float 0.) "largest exact class label" 16777216.
    (Input.target_value encoded 1);
  Alcotest.check Alcotest.int "class labels do not round" 0
    (Input.rounded_values (Input.target_report encoded));
  Alcotest.check Alcotest.int64 "classification staging reported"
    (Int64.of_int (2 * (Sys.word_size / 8)))
    (Input.temporary_payload_bytes (Input.target_report encoded));
  List.iter
    (fun (class_count, labels) ->
      ignore
        (expect Validation
           (Input.encoded_classification_target ~class_count
              (Target.classification labels))))
    [ (1, [| 0 |]); (16777218, [| 0 |]); (2, [| -1 |]); (2, [| 2 |]) ]

let test_native_alignment () =
  let x = features [| [| 1. |]; [| 2. |] |] in
  let called = ref false in
  let callback _ =
    called := true;
    Ok ()
  in
  ignore
    (expect Shape (Native.with_dmatrix ~target:(target [| 1. |]) x callback));
  ignore
    (expect Shape
       (Native.with_dmatrix ~sample_weight:(weights [| 1. |]) x callback));
  Alcotest.check Alcotest.bool "invalid input never reaches callback" false
    !called

let expect_freed_matrix matrix =
  let labels = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 2 in
  match Xgboost.DMatrix.set_label matrix labels with
  | () -> Alcotest.fail "DMatrix escaped scope alive"
  | exception Xgboost.Xgboost_error (Xgboost.Error.Invalid_argument _) -> ()

let expect_freed_booster booster =
  match Xgboost.Booster.set_param booster "nthread" "1" with
  | () -> Alcotest.fail "booster escaped scope alive"
  | exception Xgboost.Xgboost_error (Xgboost.Error.Invalid_argument _) -> ()

exception Callback_failure

let routine_error = Error.make ~remediation:"test" Error.Cancelled

let test_native_lifetimes () =
  let x = features [| [| 0. |]; [| 1. |] |] in
  let matrix = ref None in
  let specification = Specification.create () |> get in
  let rng = Rng.create (Seed.of_int 12) in
  let booster = ref None in
  let run finish =
    Native.with_dmatrix
      ~target:(target [| 0.; 1. |])
      x
      (fun handle ->
        matrix := Some handle;
        Native.with_booster ~cache:[ handle ] ~specification ~rng (fun handle ->
            booster := Some handle;
            finish ()))
  in
  let check () =
    expect_freed_matrix (Option.get !matrix);
    expect_freed_booster (Option.get !booster)
  in
  run (fun () -> Ok ()) |> get;
  check ();
  (match run (fun () -> Error routine_error) with
  | Error error ->
      Alcotest.check Alcotest.bool "callback typed error preserved" true
        (error == routine_error)
  | Ok () -> Alcotest.fail "lost callback error");
  check ();
  Alcotest.check_raises "unexpected exceptions propagate" Callback_failure
    (fun () -> ignore (run (fun () -> raise Callback_failure)));
  check ();
  ignore
    (expect Numerical
       (run (fun () ->
            raise
              (Xgboost.Xgboost_error
                 (Xgboost.Error.Xgb_error "test native failure")))));
  check ();
  ignore
    (expect Shape
       (run (fun () ->
            raise
              (Xgboost.Xgboost_error
                 (Xgboost.Error.Shape_mismatch
                    { expected = (2, 1); got = (1, 1) })))));
  check ();
  ignore
    (expect Validation
       (run (fun () ->
            raise
              (Xgboost.Xgboost_error
                 (Xgboost.Error.Invalid_argument "invalid native input")))));
  check ()

let test_native_smoke () =
  let major, _, _ = Native.version () |> get in
  Alcotest.check Alcotest.int "native major" 3 major;
  let x = features [| [| 0.; Float.nan |]; [| 1.; 2. |] |] in
  let specification = Specification.create ~rounds:3 () |> get in
  Native.with_dmatrix
    ~target:(target [| 0.; 1. |])
    ~sample_weight:(weights [| 1.; 2. |])
    x
    (fun matrix ->
      Alcotest.check Alcotest.int "native rows" 2 (Xgboost.DMatrix.rows matrix);
      Alcotest.check Alcotest.int "native columns" 2
        (Xgboost.DMatrix.cols matrix);
      Alcotest.check Alcotest.int "native missing-value semantics" 3
        (Xgboost.DMatrix.num_non_missing matrix);
      Native.with_booster ~cache:[ matrix ] ~specification
        ~rng:(Rng.create (Seed.of_int 7))
        (fun booster ->
          Gc.full_major ();
          Xgboost.Booster.set_param booster "objective" "reg:squarederror";
          for
            iter = 0
            to (Specification.params specification).Specification.rounds - 1
          do
            Xgboost.Booster.update_one_iter booster ~iter ~dtrain:matrix
          done;
          let prediction = Xgboost.Booster.predict booster matrix in
          Alcotest.check Alcotest.int "prediction rows" 2
            (Bigarray.Array1.dim prediction);
          for row = 0 to 1 do
            Alcotest.check Alcotest.bool "finite predictions" true
              (Float.is_finite prediction.{row})
          done;
          Ok ()))
  |> get

let test_native_failure_cleanup () =
  let matrix_handle = ref None in
  let booster_handle = ref None in
  let result =
    Native.with_dmatrix
      ~target:(target [| 0.; 1. |])
      (features [| [| 0. |]; [| 1. |] |])
      (fun matrix ->
        matrix_handle := Some matrix;
        Native.with_booster ~cache:[ matrix ]
          ~specification:(Specification.create () |> get)
          ~rng:(Rng.create (Seed.of_int 1))
          (fun booster ->
            booster_handle := Some booster;
            Xgboost.Booster.set_param booster "objective"
              "modelkit-invalid-objective";
            Xgboost.Booster.update_one_iter booster ~iter:0 ~dtrain:matrix;
            Ok ()))
  in
  ignore (expect Numerical result);
  expect_freed_matrix (Option.get !matrix_handle);
  expect_freed_booster (Option.get !booster_handle);
  ignore
    (expect Validation
       (Native.with_booster
          ~cache:[ Option.get !matrix_handle ]
          ~specification:(Specification.create () |> get)
          ~rng:(Rng.create (Seed.of_int 1))
          (fun _ -> Alcotest.fail "freed cache reached callback")))

let test_native_configuration () =
  let specification =
    Specification.create ~max_depth:2 ~subsample:0.75 ~colsample_bytree:0.5 ()
    |> get
  in
  let config specification seed =
    Native.with_dmatrix (features [| [| 1. |] |]) (fun matrix ->
        Native.with_booster ~cache:[ matrix ] ~specification
          ~rng:(Rng.create (Seed.of_int seed))
          (fun booster -> Ok (Xgboost.Booster.save_json_config booster)))
    |> get
  in
  let original = config specification 17 in
  Alcotest.check Alcotest.string "cloned specifications preserve configuration"
    original
    (config (Specification.clone specification) 17);
  Alcotest.check Alcotest.bool "RNG stream controls native seed" true
    (original <> config specification 18)

let () =
  Alcotest.run "XGBoost admission"
    [
      ( "specification",
        [
          Alcotest.test_case "immutable defaults" `Quick test_specification;
          Alcotest.test_case "invalid parameters" `Quick
            test_invalid_specifications;
        ] );
      ( "conversion",
        [
          Alcotest.test_case "rounding, NaN, and accounting" `Quick
            test_conversion;
          Alcotest.test_case "schema and ownership" `Quick
            test_schema_and_ownership;
          Alcotest.test_case "numeric boundaries" `Quick test_numeric_boundaries;
          Alcotest.test_case "regression and encoded targets" `Quick
            test_targets;
        ] );
      ( "native",
        [
          Alcotest.test_case "preflight alignment" `Quick test_native_alignment;
          Alcotest.test_case "success and failure cleanup" `Quick
            test_native_lifetimes;
          Alcotest.test_case "weighted missing-value smoke" `Quick
            test_native_smoke;
          Alcotest.test_case "real native failure cleanup" `Quick
            test_native_failure_cleanup;
          Alcotest.test_case "native configuration and seeds" `Quick
            test_native_configuration;
        ] );
    ]
