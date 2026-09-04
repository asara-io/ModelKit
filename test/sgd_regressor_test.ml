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

let check_close label expected observed =
  let tolerance = 1e-12 *. Float.max 1.0 (Float.abs expected) in
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed) <= tolerance)

let make_regression x_values y_values =
  let x = Matrix.of_arrays x_values |> get_data in
  let y = Vector.of_array y_values |> Target.regression |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  (x, y, feature_schema)

let coefficient fitted index =
  Vector.get (Sgd_regressor.coefficients fitted) index

let test_one_batch_update () =
  let x, y, feature_schema =
    make_regression [| [| 1.0 |]; [| 2.0 |] |] [| 1.0; 2.0 |]
  in
  let specification =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty ~fit_intercept:false
      ~learning_rate:Sgd_regressor.Constant ~eta0:0.1 ~max_epochs:1
      ~shuffle:false ()
    |> get
  in
  let initial =
    Sgd_regressor.start specification
      ~rng:(Rng.create (Seed.of_int 7))
      ~feature_schema
  in
  let first =
    Sgd_regressor.partial_fit initial ~feature_schema ~x ~y () |> get
  in
  let first_model = Sgd_regressor.to_fitted first |> get in
  check_close "first coefficient" 0.46 (coefficient first_model 0);
  Alcotest.(check int)
    "first updates" 2
    (Sgd_regressor.checkpoint_updates first);
  Alcotest.(check int)
    "first batch" 1
    (Sgd_regressor.checkpoint_batches_processed first);
  let resumed = Sgd_regressor.checkpoint first_model in
  let second =
    Sgd_regressor.partial_fit resumed ~feature_schema ~x ~y () |> get
  in
  let second_model = Sgd_regressor.to_fitted second |> get in
  check_close "second coefficient" 0.7084 (coefficient second_model 0);
  check_close "first checkpoint remains immutable" 0.46
    (coefficient first_model 0)

let test_fit_matches_checkpoint_chain () =
  let x, y, feature_schema =
    make_regression
      [|
        [| -2.0; 1.0 |];
        [| -1.0; 0.5 |];
        [| 0.0; -1.0 |];
        [| 1.0; 2.0 |];
        [| 2.0; -0.5 |];
      |]
      [| -2.5; -1.25; 1.0; 2.0; 4.5 |]
  in
  let specification =
    Sgd_regressor.create ~penalty:Sgd_regressor.Elastic_net ~alpha:0.01
      ~l1_ratio:0.3
      ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.2 })
      ~eta0:0.02 ~max_epochs:4 ~shuffle:true ()
    |> get
  in
  let rng = Rng.create (Seed.of_int 1729) in
  let fitted =
    Sgd_regressor.fit specification ~rng ~feature_schema ~x ~y () |> get
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_regressor.partial_fit checkpoint ~feature_schema ~x ~y ()
      |> get
      |> train (remaining - 1)
  in
  let checkpoint =
    Sgd_regressor.start specification ~rng ~feature_schema |> train 4
  in
  let resumed = Sgd_regressor.to_fitted checkpoint |> get in
  Array.iteri
    (fun index expected ->
      check_close
        (Format.sprintf "continuation coefficient %d" index)
        expected
        (coefficient resumed index))
    (Sgd_regressor.coefficients fitted |> Vector.to_array);
  check_close "continuation intercept"
    (Sgd_regressor.intercept fitted)
    (Sgd_regressor.intercept resumed);
  let report = Sgd_regressor.report fitted in
  Alcotest.(check int) "fit batches" 4 report.Sgd_regressor.batches_processed;
  Alcotest.(check int) "fit updates" 20 report.Sgd_regressor.updates;
  Alcotest.(check bool)
    "fixed epochs are not convergence" false report.Sgd_regressor.converged;
  Alcotest.(check bool)
    "fixed epoch reason" true
    (report.Sgd_regressor.stopping_reason = Sgd_regressor.Epoch_limit)

let test_weights_and_resume () =
  let x, y, feature_schema =
    make_regression [| [| 1.0 |]; [| 4.0 |] |] [| 1.0; 100.0 |]
  in
  let weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 0.0 |] |> get_data
  in
  let specification =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty ~fit_intercept:false
      ~learning_rate:Sgd_regressor.Constant ~eta0:0.1 ~max_epochs:1
      ~shuffle:false ()
    |> get
  in
  let initial =
    Sgd_regressor.start specification
      ~rng:(Rng.create (Seed.of_int 3))
      ~feature_schema
  in
  let checkpoint =
    Sgd_regressor.partial_fit initial ~sample_weight:weights ~feature_schema ~x
      ~y ()
    |> get
  in
  let fitted = Sgd_regressor.to_fitted checkpoint |> get in
  check_close "zero-weight row" 0.1 (coefficient fitted 0);
  let resumed = Sgd_regressor.checkpoint fitted in
  Alcotest.(check int)
    "resumed update count" 2
    (Sgd_regressor.checkpoint_updates resumed)

let test_pipeline () =
  let x, y, feature_schema =
    make_regression
      [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |]; [| 2.0 |] |]
      [| -1.0; 1.0; 3.0; 5.0 |]
  in
  let specification =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
      ~learning_rate:Sgd_regressor.Constant ~eta0:0.05 ~max_epochs:20
      ~shuffle:false ()
    |> get
  in
  let terminal =
    Pipeline.estimator ~name:"SGD regressor"
      (module Sgd_regressor)
      specification
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let fitted =
    Pipeline.fit pipeline
      ~rng:(Rng.create (Seed.of_int 11))
      ~feature_schema ~x ~y ()
    |> get
  in
  Alcotest.(check int)
    "pipeline prediction length" 4
    (Pipeline.predict fitted ~feature_schema ~x |> get |> Target.length)

let test_typed_failures () =
  let invalid results = List.iter (expect_error is_validation) results in
  invalid
    [
      Sgd_regressor.create ~alpha:(-1.0) ();
      Sgd_regressor.create ~l1_ratio:1.1 ();
      Sgd_regressor.create ~eta0:0.0 ();
      Sgd_regressor.create ~max_epochs:0 ();
      Sgd_regressor.create
        ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = -0.1 })
        ();
      Sgd_regressor.create ~tolerance:0.0 ();
    ];
  let x, y, feature_schema = make_regression [| [| 1.0 |] |] [| 1.0 |] in
  let specification = Sgd_regressor.create () |> get in
  let initial =
    Sgd_regressor.start specification
      ~rng:(Rng.create (Seed.of_int 1))
      ~feature_schema
  in
  Sgd_regressor.to_fitted initial |> expect_error is_validation;
  let empty_x = Matrix.create ~rows:0 ~columns:1 0.0 |> get_data in
  let empty_y = Target.regression (Vector.of_array [||]) |> get_data in
  Sgd_regressor.partial_fit initial ~feature_schema ~x:empty_x ~y:empty_y ()
  |> expect_error is_validation;
  let overflow_x = Matrix.of_arrays [| [| Float.max_float |] |] |> get_data in
  let overflow_y =
    Target.regression (Vector.of_array [| Float.max_float |]) |> get_data
  in
  let unstable =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
      ~learning_rate:Sgd_regressor.Constant ~eta0:1.0 ~max_epochs:1
      ~shuffle:false ()
    |> get
  in
  Sgd_regressor.fit unstable
    ~rng:(Rng.create (Seed.of_int 1))
    ~feature_schema ~x:overflow_x ~y:overflow_y ()
  |> expect_error is_numerical;
  let strict =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
      ~learning_rate:Sgd_regressor.Constant ~eta0:0.1 ~max_epochs:1
      ~tolerance:1e-30 ~shuffle:false ()
    |> get
  in
  Sgd_regressor.fit strict
    ~rng:(Rng.create (Seed.of_int 1))
    ~feature_schema ~x ~y ()
  |> expect_error is_convergence

let () =
  Alcotest.run "SGD regressor"
    [
      ( "incremental training",
        [
          Alcotest.test_case "one batch update" `Quick test_one_batch_update;
          Alcotest.test_case "fit and checkpoint continuation" `Quick
            test_fit_matches_checkpoint_chain;
          Alcotest.test_case "weights and resume" `Quick test_weights_and_resume;
        ] );
      ("integration", [ Alcotest.test_case "pipeline" `Quick test_pipeline ]);
      ( "errors",
        [ Alcotest.test_case "typed failures" `Quick test_typed_failures ] );
    ]
