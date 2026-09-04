open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail (Error.to_string error)
  | Ok _ -> Alcotest.fail "expected a typed error"

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_convergence = function
  | Error.Convergence _ -> true
  | _ -> false

let[@warning "-4"] is_numerical = function
  | Error.Numerical _ -> true
  | _ -> false

let[@warning "-4"] is_compatibility = function
  | Error.Compatibility _ -> true
  | _ -> false

let[@warning "-4"] is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | _ -> false

let check_close label expected observed =
  let tolerance = 1e-12 *. Float.max 1.0 (Float.abs expected) in
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed) <= tolerance)

let make x_values labels =
  let x = Matrix.of_arrays x_values |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  (x, Target.classification labels, feature_schema)

let rng seed = Rng.create (Seed.of_int seed)

let plain ?(loss = Sgd_classifier.Hinge) ?(fit_intercept = true)
    ?(max_epochs = 1) ?tolerance () =
  Sgd_classifier.create ~loss ~penalty:Sgd_classifier.No_penalty ~fit_intercept
    ~learning_rate:Sgd_classifier.Constant ~eta0:0.1 ~max_epochs ?tolerance
    ~shuffle:false ()
  |> get

let coefficient fitted model column =
  Matrix.get (Sgd_classifier.coefficients fitted) model column

let intercept fitted model = Vector.get (Sgd_classifier.intercepts fitted) model

let predictions fitted ~feature_schema ~x =
  Sgd_classifier.predict fitted ~feature_schema ~x
  |> get |> Target.classification_values

let test_binary_hinge_update () =
  let x, y, feature_schema = make [| [| 1.0 |]; [| 2.0 |] |] [| 0; 1 |] in
  let checkpoint =
    Sgd_classifier.start (plain ()) ~rng:(rng 1) ~feature_schema
      ~classes:[| 1; 0 |]
    |> get
  in
  Alcotest.(check (array int))
    "registered classes are sorted" [| 0; 1 |]
    (Sgd_classifier.checkpoint_classes checkpoint);
  let trained =
    Sgd_classifier.partial_fit checkpoint ~feature_schema ~x ~y () |> get
  in
  let fitted = Sgd_classifier.to_fitted trained |> get in
  Alcotest.(check (pair int int))
    "binary coefficient shape" (1, 1)
    (Matrix.shape (Sgd_classifier.coefficients fitted));
  check_close "hinge coefficient" 0.1 (coefficient fitted 0 0);
  check_close "hinge intercept" 0.0 (intercept fitted 0);
  Alcotest.(check int) "updates" 2 (Sgd_classifier.checkpoint_updates trained);
  let decisions =
    Sgd_classifier.binary_decision_function fitted ~feature_schema ~x |> get
  in
  check_close "first decision" 0.1 (Vector.get decisions 0);
  check_close "second decision" 0.2 (Vector.get decisions 1);
  Alcotest.(check (array int))
    "positive scores select the higher class" [| 1; 1 |]
    (predictions fitted ~feature_schema ~x);
  let zero = Matrix.of_arrays [| [| 0.0 |] |] |> get_data in
  Alcotest.(check (array int))
    "a zero score selects the lower class" [| 0 |]
    (predictions fitted ~feature_schema ~x:zero);
  Sgd_classifier.predict_proba fitted ~feature_schema ~x
  |> expect_error is_compatibility;
  let report = Sgd_classifier.report fitted in
  Alcotest.(check bool)
    "checkpoint snapshots report partial fit" true
    (report.Sgd_classifier.stopping_reason = Sgd_classifier.Partial_fit);
  Alcotest.(check bool)
    "objective is finite" true
    (Float.is_finite report.Sgd_classifier.objective)

let test_binary_log_loss_update () =
  let x, y, feature_schema = make [| [| 1.0 |]; [| 2.0 |] |] [| 0; 1 |] in
  let fitted =
    Sgd_classifier.fit
      (plain ~loss:Sgd_classifier.Log_loss ())
      ~rng:(rng 1) ~feature_schema ~x ~y ()
    |> get
  in
  let sigma = 1.0 /. (1.0 +. Float.exp 0.15) in
  check_close "log-loss coefficient"
    (-0.05 -. (0.2 *. (sigma -. 1.0)))
    (coefficient fitted 0 0);
  check_close "log-loss intercept"
    (-0.05 -. (0.1 *. (sigma -. 1.0)))
    (intercept fitted 0);
  let probabilities =
    Sgd_classifier.predict_proba fitted ~feature_schema ~x |> get
  in
  Alcotest.(check (pair int int))
    "binary probability shape" (2, 2)
    (Matrix.shape probabilities);
  for row = 0 to 1 do
    check_close "complementary probabilities" 1.0
      (Matrix.get probabilities row 0 +. Matrix.get probabilities row 1)
  done;
  let report = Sgd_classifier.report fitted in
  Alcotest.(check bool)
    "fixed epochs are not convergence" false report.Sgd_classifier.converged;
  Alcotest.(check bool)
    "epoch limit reason" true
    (report.Sgd_classifier.stopping_reason = Sgd_classifier.Epoch_limit)

let test_multiclass_update_and_ties () =
  let x, y, feature_schema = make [| [| 1.0 |] |] [| 3 |] in
  let checkpoint =
    Sgd_classifier.start (plain ()) ~rng:(rng 1) ~feature_schema
      ~classes:[| 7; -1; 3 |]
    |> get
  in
  let fitted =
    Sgd_classifier.partial_fit checkpoint ~feature_schema ~x ~y ()
    |> get |> Sgd_classifier.to_fitted |> get
  in
  Alcotest.(check (array int))
    "ascending classes" [| -1; 3; 7 |]
    (Sgd_classifier.classes fitted);
  Alcotest.(check (pair int int))
    "one-versus-rest coefficient shape" (3, 1)
    (Matrix.shape (Sgd_classifier.coefficients fitted));
  Array.iteri
    (fun model expected ->
      check_close
        (Format.sprintf "model %d coefficient" model)
        expected
        (coefficient fitted model 0);
      check_close
        (Format.sprintf "model %d intercept" model)
        expected (intercept fitted model))
    [| -0.1; 0.1; -0.1 |];
  Alcotest.(check int)
    "one shared update per row" 1
    (Sgd_classifier.checkpoint_updates (Sgd_classifier.checkpoint fitted));
  let decisions =
    Sgd_classifier.decision_function fitted ~feature_schema ~x |> get
  in
  Alcotest.(check (pair int int))
    "decision shape" (1, 3) (Matrix.shape decisions);
  Alcotest.(check (array int))
    "argmax prediction" [| 3 |]
    (predictions fitted ~feature_schema ~x);
  Sgd_classifier.binary_decision_function fitted ~feature_schema ~x
  |> expect_error is_compatibility;
  let zero_x, zero_y, zero_schema = make [| [| 0.0 |] |] [| 3 |] in
  let tied =
    Sgd_classifier.fit
      (plain ~fit_intercept:false ())
      ~rng:(rng 1) ~feature_schema:zero_schema ~x:zero_x ~y:zero_y
      ~sample_weight:
        (Sample_weight.of_array ~expected_length:1 [| 1.0 |] |> get_data)
      ()
  in
  (match tied with
  | Ok _ -> Alcotest.fail "one positively weighted class must be rejected"
  | Error _ -> ());
  let tie_x, tie_y, tie_schema =
    make [| [| 0.0 |]; [| 0.0 |]; [| 0.0 |] |] [| 8; -2; 4 |]
  in
  let tied =
    Sgd_classifier.fit
      (plain ~fit_intercept:false ())
      ~rng:(rng 1) ~feature_schema:tie_schema ~x:tie_x ~y:tie_y ()
    |> get
  in
  Alcotest.(check (array int))
    "ties choose the lowest class" [| -2; -2; -2 |]
    (predictions tied ~feature_schema:tie_schema ~x:tie_x);
  let uniform =
    Sgd_classifier.fit
      (plain ~loss:Sgd_classifier.Log_loss ~fit_intercept:false ())
      ~rng:(rng 1) ~feature_schema:tie_schema ~x:tie_x ~y:tie_y ()
    |> get
  in
  let probabilities =
    Sgd_classifier.predict_proba uniform ~feature_schema:tie_schema ~x:tie_x
    |> get
  in
  for row = 0 to 2 do
    for model = 0 to 2 do
      check_close "uniform tie probability" (1.0 /. 3.0)
        (Matrix.get probabilities row model)
    done
  done

let test_fit_matches_checkpoint_chain () =
  let x, y, feature_schema =
    make
      [|
        [| -2.0; 1.0 |];
        [| -1.0; 0.5 |];
        [| 0.0; -1.0 |];
        [| 1.0; 2.0 |];
        [| 2.0; -0.5 |];
        [| 3.0; 3.0 |];
      |]
      [| 5; 5; -2; 11; -2; 11 |]
  in
  let specification =
    Sgd_classifier.create ~loss:Sgd_classifier.Log_loss
      ~penalty:Sgd_classifier.Elastic_net ~alpha:0.01 ~l1_ratio:0.3
      ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = 0.2 })
      ~eta0:0.05 ~max_epochs:4 ~shuffle:true ()
    |> get
  in
  let fitted =
    Sgd_classifier.fit specification ~rng:(rng 1729) ~feature_schema ~x ~y ()
    |> get
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_classifier.partial_fit checkpoint ~feature_schema ~x ~y ()
      |> get
      |> train (remaining - 1)
  in
  let resumed =
    Sgd_classifier.start specification ~rng:(rng 1729) ~feature_schema
      ~classes:[| -2; 5; 11 |]
    |> get |> train 4 |> Sgd_classifier.to_fitted |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.0)))
    "continuation coefficients"
    (Sgd_classifier.coefficients fitted |> Matrix.to_arrays)
    (Sgd_classifier.coefficients resumed |> Matrix.to_arrays);
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "continuation intercepts"
    (Sgd_classifier.intercepts fitted |> Vector.to_array)
    (Sgd_classifier.intercepts resumed |> Vector.to_array);
  let report = Sgd_classifier.report fitted in
  Alcotest.(check int) "fit batches" 4 report.Sgd_classifier.batches_processed;
  Alcotest.(check int) "fit updates" 24 report.Sgd_classifier.updates;
  let before = Sgd_classifier.checkpoint fitted in
  let after =
    Sgd_classifier.partial_fit before ~feature_schema ~x ~y () |> get
  in
  Alcotest.(check int)
    "input checkpoint is unchanged" 24
    (Sgd_classifier.checkpoint_updates before);
  Alcotest.(check int)
    "successor advances" 30
    (Sgd_classifier.checkpoint_updates after);
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.0)))
    "fitted value is unchanged by continuation"
    (Sgd_classifier.coefficients fitted |> Matrix.to_arrays)
    (Sgd_classifier.coefficients resumed |> Matrix.to_arrays)

let test_class_registration_and_weights () =
  let x, y, feature_schema = make [| [| 1.0 |]; [| 4.0 |] |] [| 0; 1 |] in
  let specification = plain () in
  Sgd_classifier.start specification ~rng:(rng 3) ~feature_schema
    ~classes:[| 1 |]
  |> expect_error is_validation;
  Sgd_classifier.start specification ~rng:(rng 3) ~feature_schema
    ~classes:[| 2; 2 |]
  |> expect_error is_validation;
  let checkpoint =
    Sgd_classifier.start specification ~rng:(rng 3) ~feature_schema
      ~classes:[| 0; 1 |]
    |> get
  in
  let unregistered = Target.classification [| 0; 5 |] in
  Sgd_classifier.partial_fit checkpoint ~feature_schema ~x ~y:unregistered ()
  |> expect_error is_validation;
  let weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 0.0 |] |> get_data
  in
  let trained =
    Sgd_classifier.partial_fit checkpoint ~sample_weight:weights ~feature_schema
      ~x ~y:unregistered ()
    |> get
  in
  let fitted = Sgd_classifier.to_fitted trained |> get in
  check_close "zero-weight row leaves the first update" (-0.1)
    (coefficient fitted 0 0);
  Alcotest.(check int)
    "zero-weight rows still advance the counter" 2
    (Sgd_classifier.checkpoint_updates trained);
  Sgd_classifier.fit specification ~sample_weight:weights ~rng:(rng 3)
    ~feature_schema ~x ~y ()
  |> expect_error is_validation

let test_pipeline () =
  let x, y, feature_schema =
    make [| [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |] |] [| 3; 3; 8; 8 |]
  in
  let specification =
    Sgd_classifier.create ~loss:Sgd_classifier.Log_loss
      ~penalty:Sgd_classifier.No_penalty ~learning_rate:Sgd_classifier.Constant
      ~eta0:0.1 ~max_epochs:20 ~shuffle:false ()
    |> get
  in
  let terminal =
    Pipeline.estimator ~name:"SGD classifier"
      ~decision_function:Sgd_classifier.binary_decision_function
      ~predict_proba:Sgd_classifier.predict_proba
      ~classes:Sgd_classifier.classes
      (module Sgd_classifier)
      specification
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng 11) ~feature_schema ~x ~y () |> get
  in
  Alcotest.(check (array int))
    "pipeline prediction" [| 3; 3; 8; 8 |]
    (Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values);
  Alcotest.(check int)
    "pipeline decision length" 4
    (Pipeline.decision_function fitted ~feature_schema ~x
    |> get |> Vector.length);
  Alcotest.(check (pair int int))
    "pipeline probability shape" (4, 2)
    (Pipeline.predict_proba fitted ~feature_schema ~x |> get |> Matrix.shape);
  Alcotest.(check (array int))
    "pipeline classes" [| 3; 8 |]
    (Pipeline.classes fitted |> get)

let test_typed_failures () =
  let invalid results = List.iter (expect_error is_validation) results in
  invalid
    [
      Sgd_classifier.create ~alpha:(-1.0) ();
      Sgd_classifier.create ~l1_ratio:1.1 ();
      Sgd_classifier.create ~eta0:0.0 ();
      Sgd_classifier.create ~max_epochs:0 ();
      Sgd_classifier.create
        ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = -0.1 })
        ();
      Sgd_classifier.create ~tolerance:0.0 ();
    ];
  let x, y, feature_schema = make [| [| 1.0 |]; [| 2.0 |] |] [| 0; 1 |] in
  let initial =
    Sgd_classifier.start (plain ()) ~rng:(rng 1) ~feature_schema
      ~classes:[| 0; 1 |]
    |> get
  in
  Sgd_classifier.to_fitted initial |> expect_error is_validation;
  let empty_x = Matrix.create ~rows:0 ~columns:1 0.0 |> get_data in
  Sgd_classifier.partial_fit initial ~feature_schema ~x:empty_x
    ~y:(Target.classification [||])
    ()
  |> expect_error is_validation;
  let fitted =
    Sgd_classifier.fit (plain ()) ~rng:(rng 1) ~feature_schema ~x ~y () |> get
  in
  let wrong_schema = Feature_schema.anonymous ~feature_count:2 |> get_data in
  Sgd_classifier.predict fitted ~feature_schema:wrong_schema ~x
  |> expect_error is_schema_mismatch;
  let overflow_x =
    Matrix.of_arrays [| [| Float.max_float |]; [| -.Float.max_float |] |]
    |> get_data
  in
  Sgd_classifier.fit (plain ~max_epochs:2 ()) ~rng:(rng 1) ~feature_schema
    ~x:overflow_x ~y ()
  |> expect_error is_numerical;
  Sgd_classifier.fit
    (plain ~tolerance:1e-30 ())
    ~rng:(rng 1) ~feature_schema ~x ~y ()
  |> expect_error is_convergence;
  let separable_x, separable_y, separable_schema =
    make [| [| -1.0 |]; [| 1.0 |] |] [| 0; 1 |]
  in
  let converged =
    Sgd_classifier.fit
      (plain ~fit_intercept:false ~max_epochs:50 ~tolerance:1e-9 ())
      ~rng:(rng 1) ~feature_schema:separable_schema ~x:separable_x
      ~y:separable_y ()
    |> get
  in
  Alcotest.(check bool)
    "separable hinge data converges by step tolerance" true
    ((Sgd_classifier.report converged).Sgd_classifier.stopping_reason
   = Sgd_classifier.Step_tolerance)

let () =
  Alcotest.run "SGD classifier"
    [
      ( "incremental training",
        [
          Alcotest.test_case "binary hinge update" `Quick
            test_binary_hinge_update;
          Alcotest.test_case "binary log-loss update" `Quick
            test_binary_log_loss_update;
          Alcotest.test_case "multiclass update and ties" `Quick
            test_multiclass_update_and_ties;
          Alcotest.test_case "fit and checkpoint continuation" `Quick
            test_fit_matches_checkpoint_chain;
          Alcotest.test_case "class registration and weights" `Quick
            test_class_registration_and_weights;
        ] );
      ("integration", [ Alcotest.test_case "pipeline" `Quick test_pipeline ]);
      ( "errors",
        [ Alcotest.test_case "typed failures" `Quick test_typed_failures ] );
    ]
