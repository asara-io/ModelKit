open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let rng () = Rng.create (Seed.of_int 2026)

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let check_float ?(epsilon = 1e-7) message expected observed =
  Alcotest.check (Alcotest.float epsilon) message expected observed

let poisson_fit ?sample_weight ?(alpha = 0.0) ?(fit_intercept = true)
    ?(tolerance = 1e-10) ?(max_iterations = 200) x y =
  Poisson_regression.fit
    (Poisson_regression.create ~alpha ~fit_intercept ~tolerance ~max_iterations
       ()
    |> get)
    ?sample_weight ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y:(regression y)
    ()
  |> get

let test_poisson_fit_predict_and_weights () =
  let x =
    matrix [| [| -2.0 |]; [| -1.0 |]; [| 0.0 |]; [| 1.0 |]; [| 2.0 |] |]
  in
  let y = Array.map Float.exp [| -0.8; -0.15; 0.5; 1.15; 1.8 |] in
  let fitted = poisson_fit x y in
  check_float "coefficient" 0.65
    (Poisson_regression.coefficients fitted |> fun values -> Vector.get values 0);
  check_float "intercept" 0.5 (Poisson_regression.intercept fitted);
  let predictions =
    Poisson_regression.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  Array.iteri
    (fun row expected -> check_float "prediction" expected predictions.(row))
    y;
  let weighted =
    poisson_fit ~sample_weight:(weights [| 1.0; 2.0; 3.0; 4.0; 5.0 |]) x y
  in
  let rescaled =
    poisson_fit ~sample_weight:(weights [| 10.0; 20.0; 30.0; 40.0; 50.0 |]) x y
  in
  check_float ~epsilon:1e-10 "weight-scale coefficient"
    ( Poisson_regression.coefficients weighted |> fun values ->
      Vector.get values 0 )
    ( Poisson_regression.coefficients rescaled |> fun values ->
      Vector.get values 0 );
  check_float ~epsilon:1e-10 "weight-scale intercept"
    (Poisson_regression.intercept weighted)
    (Poisson_regression.intercept rescaled);
  let report = Poisson_regression.report fitted in
  Alcotest.(check bool) "converged" true (Solver_report.converged report);
  Alcotest.(check bool)
    "iterative report" true
    (Solver_report.stopping_reason report <> Solver_report.Direct_solution)

let test_tweedie_links_and_domains () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let y = [| -1.0; 1.0; 3.0; 5.0 |] in
  let normal =
    Tweedie_regression.fit
      (Tweedie_regression.create ~power:0.0 ~alpha:0.0 ~tolerance:1e-10 ()
      |> get)
      ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y:(regression y) ()
    |> get
  in
  Alcotest.(check bool)
    "auto identity" true
    (Tweedie_regression.resolved_link normal = Tweedie_regression.Identity);
  check_float "normal coefficient" 2.0
    (Tweedie_regression.coefficients normal |> fun values -> Vector.get values 0);
  check_float "normal intercept" 1.0 (Tweedie_regression.intercept normal);
  let positive = [| 0.5; 1.0; 2.0; 4.0 |] in
  let compound =
    Tweedie_regression.fit
      (Tweedie_regression.create ~power:1.5 ~alpha:0.1 ~tolerance:1e-10 ()
      |> get)
      ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y:(regression positive) ()
    |> get
  in
  Alcotest.(check bool)
    "auto log" true
    (Tweedie_regression.resolved_link compound = Tweedie_regression.Log);
  let predictions =
    Tweedie_regression.predict compound ~feature_schema:(schema x) ~x
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.(check bool)
    "positive finite predictions" true
    (Array.for_all
       (fun value -> Float.is_finite value && value > 0.0)
       predictions)

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_numerical = function
  | Error.Numerical _ -> true
  | _ -> false

let[@warning "-4"] is_convergence = function
  | Error.Convergence _ -> true
  | _ -> false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_typed_failures_and_prediction_safeguards () =
  Poisson_regression.create ~alpha:(-1.0) () |> expect_error is_validation;
  Tweedie_regression.create ~power:Float.nan () |> expect_error is_validation;
  Tweedie_regression.create ~tolerance:0.0 () |> expect_error is_validation;
  Tweedie_regression.create ~max_iterations:0 () |> expect_error is_validation;
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  Poisson_regression.fit
    (Poisson_regression.create () |> get)
    ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(regression [| 1.0; -1.0; 2.0 |])
    ()
  |> expect_error is_validation;
  Tweedie_regression.fit
    (Tweedie_regression.create ~power:2.0 () |> get)
    ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(regression [| 1.0; 0.0; 2.0 |])
    ()
  |> expect_error is_validation;
  Poisson_regression.fit
    (Poisson_regression.create ~tolerance:1e-30 ~max_iterations:1 () |> get)
    ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(regression [| 0.5; 1.0; 4.0 |])
    ()
  |> expect_error is_convergence;
  let fitted = poisson_fit x [| 0.5; 1.0; 2.0 |] in
  let extreme = matrix [| [| 1e308 |] |] in
  Poisson_regression.predict fitted ~feature_schema:(schema x) ~x:extreme
  |> expect_error is_numerical

let test_pipeline_and_empty_prediction () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let y = regression [| 0.5; 1.0; 2.0 |] in
  let estimator =
    Pipeline.estimator ~name:"Poisson regression"
      (module Poisson_regression)
      (Poisson_regression.create ~alpha:0.0 () |> get)
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y ()
    |> get
  in
  Alcotest.(check int)
    "pipeline prediction length" 3
    (Pipeline.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.length);
  let empty = Matrix.init ~rows:0 ~columns:1 (fun _ _ -> 0.0) |> get_data in
  Alcotest.(check int)
    "empty prediction" 0
    (Pipeline.predict fitted ~feature_schema:(schema x) ~x:empty
    |> get |> Target.length)

let () =
  Alcotest.run "generalized linear models"
    [
      ( "estimators",
        [
          Alcotest.test_case "Poisson fit, predict, and weights" `Quick
            test_poisson_fit_predict_and_weights;
          Alcotest.test_case "Tweedie links and domains" `Quick
            test_tweedie_links_and_domains;
          Alcotest.test_case "pipeline and empty prediction" `Quick
            test_pipeline_and_empty_prediction;
        ] );
      ( "errors",
        [
          Alcotest.test_case "typed failures and safeguards" `Quick
            test_typed_failures_and_prediction_safeguards;
        ] );
    ]
