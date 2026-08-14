open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let rng () = Rng.create (Seed.of_int 2026)
let matrix values = Matrix.of_arrays values |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let classification values = Target.classification values

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let check_float ?(epsilon = 1e-10) message expected observed =
  Alcotest.check (Alcotest.float epsilon) message expected observed

let check_vector ?(epsilon = 1e-10) message expected observed =
  Alcotest.check
    (Alcotest.array (Alcotest.float epsilon))
    message expected (Vector.to_array observed)

let linear_fit ?sample_weight ?(fit_intercept = true) x values =
  Linear_regression.fit
    (Linear_regression.create ~fit_intercept ())
    ?sample_weight ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(regression values) ()
  |> get

let ridge_fit ?sample_weight ?(alpha = 1.0) ?(fit_intercept = true) x values =
  let specification = Ridge_regression.create ~alpha ~fit_intercept () |> get in
  Ridge_regression.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(regression values) ()
  |> get

let logistic_fit ?sample_weight ?(c = 1.0) ?(fit_intercept = true)
    ?(tolerance = 1e-10) ?(max_iterations = 100) x values =
  let specification =
    Logistic_regression.create ~c ~fit_intercept ~tolerance ~max_iterations ()
    |> get
  in
  Logistic_regression.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(classification values) ()
  |> get

let test_ordinary_least_squares () =
  let x =
    matrix [| [| 0.0; 0.0 |]; [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 2.0; -1.0 |] |]
  in
  let fitted = linear_fit x [| 4.0; 6.0; 1.0; 11.0 |] in
  check_vector "OLS coefficients" [| 2.0; -3.0 |]
    (Linear_regression.coefficients fitted);
  check_float "OLS intercept" 4.0 (Linear_regression.intercept fitted);
  let prediction =
    Linear_regression.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.regression_values
  in
  check_vector "OLS training prediction" [| 4.0; 6.0; 1.0; 11.0 |] prediction;
  let report = Linear_regression.report fitted in
  Alcotest.(check bool)
    "direct fit converged" true
    (Solver_report.converged report);
  Alcotest.(check int) "direct fit count" 1 (Solver_report.iterations report);
  Alcotest.(check (option int))
    "design rank" (Some 2)
    (Solver_report.rank report);
  Alcotest.(check bool)
    "direct stopping reason" true
    (Solver_report.stopping_reason report = Solver_report.Direct_solution)

let test_weighted_ordinary_least_squares () =
  let x = matrix [| [| 0.0 |]; [| 1.0 |]; [| 2.0 |]; [| 100.0 |] |] in
  let sample_weight = weights [| 1.0; 2.0; 3.0; 0.0 |] in
  let fitted = linear_fit ~sample_weight x [| 1.0; 3.0; 5.0; -1000.0 |] in
  check_vector "zero-weight outlier is ignored" [| 2.0 |]
    (Linear_regression.coefficients fitted);
  check_float "weighted intercept" 1.0 (Linear_regression.intercept fitted);
  let scaled_weight = weights [| 10.0; 20.0; 30.0; 0.0 |] in
  let scaled =
    linear_fit ~sample_weight:scaled_weight x [| 1.0; 3.0; 5.0; -1000.0 |]
  in
  check_vector "weight scaling preserves coefficients"
    (Vector.to_array (Linear_regression.coefficients fitted))
    (Linear_regression.coefficients scaled)

let test_rank_deficiency () =
  let x = matrix [| [| 1.0; 2.0 |]; [| 2.0; 4.0 |]; [| 3.0; 6.0 |] |] in
  let fitted = linear_fit ~fit_intercept:false x [| 3.0; 6.0; 9.0 |] in
  let prediction =
    Linear_regression.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.regression_values
  in
  check_vector "rank-deficient prediction" [| 3.0; 6.0; 9.0 |] prediction;
  Alcotest.(check (option int))
    "rank deficiency is reported" (Some 1)
    (Solver_report.rank (Linear_regression.report fitted))

let test_ridge () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let fitted = ridge_fit ~alpha:2.0 x [| -2.0; 0.0; 2.0 |] in
  check_vector "ridge coefficient" [| 1.0 |]
    (Ridge_regression.coefficients fitted);
  check_float "ridge intercept" 0.0 (Ridge_regression.intercept fitted);
  Alcotest.(check (option int))
    "augmented system rank" (Some 1)
    (Solver_report.rank (Ridge_regression.report fitted));
  let unregularized = ridge_fit ~alpha:0.0 x [| -2.0; 0.0; 2.0 |] in
  check_vector "zero alpha agrees with OLS" [| 2.0 |]
    (Ridge_regression.coefficients unregularized)

let test_logistic_regression () =
  let x =
    matrix
      [| [| -3.0 |]; [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |]; [| 3.0 |] |]
  in
  let fitted = logistic_fit ~c:10.0 x [| -7; -7; -7; 11; 11; 11 |] in
  Alcotest.(check (array int))
    "classes are ascending" [| -7; 11 |]
    (Logistic_regression.classes fitted);
  let decisions =
    Logistic_regression.decision_function fitted ~feature_schema:(schema x) ~x
    |> get
  in
  for row = 1 to Vector.length decisions - 1 do
    Alcotest.(check bool)
      "decision function is monotonic" true
      (Vector.get decisions (row - 1) < Vector.get decisions row)
  done;
  let probabilities =
    Logistic_regression.predict_proba fitted ~feature_schema:(schema x) ~x
    |> get
  in
  for row = 0 to Matrix.rows probabilities - 1 do
    let negative = Matrix.get probabilities row 0 in
    let positive = Matrix.get probabilities row 1 in
    Alcotest.(check bool)
      "negative probability is bounded" true
      (negative >= 0.0 && negative <= 1.0);
    Alcotest.(check bool)
      "positive probability is bounded" true
      (positive >= 0.0 && positive <= 1.0);
    check_float "probabilities sum to one" 1.0 (negative +. positive)
  done;
  let predicted =
    Logistic_regression.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int))
    "binary predictions"
    [| -7; -7; -7; 11; 11; 11 |]
    predicted;
  let report = Logistic_regression.report fitted in
  Alcotest.(check bool)
    "iterative fit converged" true
    (Solver_report.converged report);
  Alcotest.(check bool)
    "iterations are reported" true
    (Solver_report.iterations report > 0);
  Alcotest.(check bool)
    "objective is finite" true
    (Float.is_finite (Solver_report.objective report));
  Alcotest.(check (option int))
    "iterative report has no direct-solver rank" None
    (Solver_report.rank report)

let test_logistic_weights_and_stability () =
  let x = matrix [| [| -1.0 |]; [| 1.0 |]; [| 100.0 |] |] in
  let sample_weight = weights [| 1.0; 1.0; 0.0 |] in
  let fitted = logistic_fit ~sample_weight x [| 0; 1; 2 |] in
  Alcotest.(check (array int))
    "zero-weight class is ignored" [| 0; 1 |]
    (Logistic_regression.classes fitted);
  let extreme = matrix [| [| -1e300 |]; [| 1e300 |] |] in
  let probabilities =
    Logistic_regression.predict_proba fitted ~feature_schema:(schema x)
      ~x:extreme
    |> get
  in
  check_float "large negative decision remains stable" 0.0
    (Matrix.get probabilities 0 1);
  check_float "large positive decision remains stable" 1.0
    (Matrix.get probabilities 1 1)

let is_validation = function
  | Error.Validation _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_convergence = function
  | Error.Convergence _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Validation _ | Error.Numerical _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_typed_errors () =
  Ridge_regression.create ~alpha:(-1.0) () |> expect_error is_validation;
  Logistic_regression.create ~c:0.0 () |> expect_error is_validation;
  Logistic_regression.create ~tolerance:Float.nan ()
  |> expect_error is_validation;
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let specification =
    Logistic_regression.create ~tolerance:1e-30 ~max_iterations:1 () |> get
  in
  Logistic_regression.fit specification ~rng:(rng ()) ~feature_schema:(schema x)
    ~x
    ~y:(classification [| 0; 0; 1 |])
    ()
  |> expect_error is_convergence;
  let three_classes = classification [| 0; 1; 2 |] in
  let specification = Logistic_regression.create () |> get in
  Logistic_regression.fit specification ~rng:(rng ()) ~feature_schema:(schema x)
    ~x ~y:three_classes ()
  |> expect_error is_validation;
  let fitted = linear_fit x [| 1.0; 2.0; 3.0 |] in
  let wrong_schema = Feature_schema.anonymous ~feature_count:2 |> get_data in
  Linear_regression.predict fitted ~feature_schema:wrong_schema ~x
  |> expect_error is_schema_mismatch

let test_pipeline_dispatch () =
  let x = matrix [| [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let feature_schema = schema x in
  let specification = Logistic_regression.create ~c:10.0 () |> get in
  let estimator =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~decision_function:Logistic_regression.decision_function
      ~predict_proba:Logistic_regression.predict_proba specification
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema ~x
      ~y:(classification [| 0; 0; 1; 1 |])
      ()
    |> get
  in
  let decisions = Pipeline.decision_function fitted ~feature_schema ~x |> get in
  let probabilities = Pipeline.predict_proba fitted ~feature_schema ~x |> get in
  let predictions =
    Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values
  in
  Alcotest.(check int) "decision rows" 4 (Vector.length decisions);
  Alcotest.(check int) "probability rows" 4 (Matrix.rows probabilities);
  Alcotest.(check (array int))
    "pipeline predictions" [| 0; 0; 1; 1 |] predictions

let test_determinism () =
  let x =
    matrix
      [| [| -2.0; 1.0 |]; [| -1.0; 0.0 |]; [| 1.0; 0.0 |]; [| 2.0; 1.0 |] |]
  in
  let first = logistic_fit x [| 0; 0; 1; 1 |] in
  let second = logistic_fit x [| 0; 0; 1; 1 |] in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "fixed inputs reproduce coefficients"
    (Vector.to_array (Logistic_regression.coefficients first))
    (Vector.to_array (Logistic_regression.coefficients second));
  check_float ~epsilon:0.0 "fixed inputs reproduce intercept"
    (Logistic_regression.intercept first)
    (Logistic_regression.intercept second)

let test_empty_prediction_batches () =
  let x = matrix [| [| -1.0 |]; [| 1.0 |] |] in
  let feature_schema = schema x in
  let empty = Matrix.init ~rows:0 ~columns:1 (fun _ _ -> 0.0) |> get_data in
  let linear = linear_fit x [| -1.0; 1.0 |] in
  let linear_prediction =
    Linear_regression.predict linear ~feature_schema ~x:empty
    |> get |> Target.regression_values
  in
  Alcotest.(check int)
    "empty regression length" 0
    (Vector.length linear_prediction);
  let logistic = logistic_fit x [| 0; 1 |] in
  let logistic_prediction =
    Logistic_regression.predict logistic ~feature_schema ~x:empty
    |> get |> Target.classification_values
  in
  Alcotest.(check int)
    "empty classification length" 0
    (Array.length logistic_prediction);
  let probabilities =
    Logistic_regression.predict_proba logistic ~feature_schema ~x:empty |> get
  in
  Alcotest.(check (pair int int))
    "empty probability shape" (0, 2)
    (Matrix.shape probabilities)

let () =
  Alcotest.run "linear models"
    [
      ( "least squares",
        [
          Alcotest.test_case "ordinary least squares" `Quick
            test_ordinary_least_squares;
          Alcotest.test_case "sample weights" `Quick
            test_weighted_ordinary_least_squares;
          Alcotest.test_case "rank deficiency" `Quick test_rank_deficiency;
          Alcotest.test_case "ridge" `Quick test_ridge;
        ] );
      ( "logistic regression",
        [
          Alcotest.test_case "fit and predict" `Quick test_logistic_regression;
          Alcotest.test_case "weights and stability" `Quick
            test_logistic_weights_and_stability;
          Alcotest.test_case "pipeline dispatch" `Quick test_pipeline_dispatch;
          Alcotest.test_case "determinism" `Quick test_determinism;
          Alcotest.test_case "empty prediction batches" `Quick
            test_empty_prediction_batches;
        ] );
      ( "errors",
        [ Alcotest.test_case "typed failures" `Quick test_typed_errors ] );
    ]
