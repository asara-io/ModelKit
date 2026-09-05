open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let classification values = Target.classification values
let rng () = Rng.create (Seed.of_int 2026)

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let specification ?(c = 10.0) ?(fit_intercept = true) ?(tolerance = 1e-10)
    ?(max_iterations = 100) () =
  Multinomial_logistic_regression.create ~c ~fit_intercept ~tolerance
    ~max_iterations ()
  |> get

let fit ?sample_weight ?c ?fit_intercept ?tolerance ?max_iterations x labels =
  Multinomial_logistic_regression.fit
    (specification ?c ?fit_intercept ?tolerance ?max_iterations ())
    ?sample_weight ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(classification labels) ()
  |> get

let check_float ?(epsilon = 1e-8) message expected observed =
  Alcotest.check (Alcotest.float epsilon) message expected observed

let training_data () =
  let x =
    matrix
      [|
        [| 2.0; 0.0 |];
        [| 3.0; 0.0 |];
        [| 0.0; 2.0 |];
        [| 0.0; 3.0 |];
        [| -2.0; -2.0 |];
        [| -3.0; -3.0 |];
      |]
  in
  (x, [| -3; -3; 7; 7; 11; 11 |])

let test_fit_and_inference () =
  let x, labels = training_data () in
  let feature_schema = schema x in
  let fitted = fit x labels in
  Alcotest.(check (array int))
    "ascending classes" [| -3; 7; 11 |]
    (Multinomial_logistic_regression.classes fitted);
  Alcotest.(check (pair int int))
    "coefficient shape" (3, 2)
    (Multinomial_logistic_regression.coefficients fitted |> Matrix.shape);
  Alcotest.(check int)
    "intercept count" 3
    (Multinomial_logistic_regression.intercepts fitted |> Vector.length);
  let decisions =
    Multinomial_logistic_regression.decision_function fitted ~feature_schema ~x
    |> get
  in
  let probabilities =
    Multinomial_logistic_regression.predict_proba fitted ~feature_schema ~x
    |> get
  in
  Alcotest.(check (pair int int))
    "decision shape" (6, 3) (Matrix.shape decisions);
  Alcotest.(check (pair int int))
    "probability shape" (6, 3)
    (Matrix.shape probabilities);
  for row = 0 to Matrix.rows probabilities - 1 do
    let total = ref 0.0 in
    for class_index = 0 to Matrix.columns probabilities - 1 do
      let probability = Matrix.get probabilities row class_index in
      Alcotest.(check bool)
        "finite bounded probability" true
        (Float.is_finite probability && probability >= 0.0 && probability <= 1.0);
      total := !total +. probability
    done;
    check_float "probability total" 1.0 !total
  done;
  let prediction =
    Multinomial_logistic_regression.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int)) "training prediction" labels prediction;
  let report = Multinomial_logistic_regression.report fitted in
  Alcotest.(check bool) "solver converged" true (Solver_report.converged report);
  Alcotest.(check bool)
    "iterative stopping reason" true
    (Solver_report.stopping_reason report <> Solver_report.Direct_solution);
  Alcotest.(check bool)
    "finite objective" true
    (Float.is_finite (Solver_report.objective report));
  Alcotest.(check bool)
    "iterations recorded" true
    (Solver_report.iterations report > 0)

let test_weights_ties_and_extreme_scores () =
  let x =
    matrix
      [| [| 2.0; 0.0 |]; [| 0.0; 2.0 |]; [| -2.0; -2.0 |]; [| 100.0; 100.0 |] |]
  in
  let fitted =
    fit ~sample_weight:(weights [| 1.0; 1.0; 1.0; 0.0 |]) x [| -4; 2; 9; 20 |]
  in
  Alcotest.(check (array int))
    "zero-weight class excluded" [| -4; 2; 9 |]
    (Multinomial_logistic_regression.classes fitted);
  let zero_x = matrix [| [| 0.0 |]; [| 0.0 |]; [| 0.0 |] |] in
  let tied = fit ~fit_intercept:false zero_x [| 8; -2; 4 |] in
  let tied_probabilities =
    Multinomial_logistic_regression.predict_proba tied
      ~feature_schema:(schema zero_x) ~x:zero_x
    |> get
  in
  for row = 0 to 2 do
    for class_index = 0 to 2 do
      check_float "uniform tie probability" (1.0 /. 3.0)
        (Matrix.get tied_probabilities row class_index)
    done
  done;
  Alcotest.(check (array int))
    "ties choose lowest class" [| -2; -2; -2 |]
    (Multinomial_logistic_regression.predict tied
       ~feature_schema:(schema zero_x) ~x:zero_x
    |> get |> Target.classification_values);
  let training_x, labels = training_data () in
  let extreme = fit training_x labels in
  let production = matrix [| [| 1e150; -1e150 |] |] in
  let probabilities =
    Multinomial_logistic_regression.predict_proba extreme
      ~feature_schema:(schema training_x) ~x:production
    |> get
  in
  let total = ref 0.0 in
  for class_index = 0 to 2 do
    let probability = Matrix.get probabilities 0 class_index in
    Alcotest.(check bool)
      "extreme probability is finite" true
      (Float.is_finite probability);
    total := !total +. probability
  done;
  check_float "extreme probability total" 1.0 !total

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_convergence = function
  | Error.Convergence _ -> true
  | _ -> false

let[@warning "-4"] is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | _ -> false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_errors_and_empty_prediction () =
  Multinomial_logistic_regression.create ~c:0.0 () |> expect_error is_validation;
  Multinomial_logistic_regression.create ~tolerance:Float.nan ()
  |> expect_error is_validation;
  Multinomial_logistic_regression.create ~max_iterations:0 ()
  |> expect_error is_validation;
  let two_class_x = matrix [| [| -1.0 |]; [| 1.0 |] |] in
  Multinomial_logistic_regression.fit (specification ()) ~rng:(rng ())
    ~feature_schema:(schema two_class_x) ~x:two_class_x
    ~y:(classification [| 1; 2 |])
    ()
  |> expect_error is_validation;
  let x, labels = training_data () in
  let fitted = fit x labels in
  let wrong_schema = Feature_schema.anonymous ~feature_count:3 |> get_data in
  Multinomial_logistic_regression.predict fitted ~feature_schema:wrong_schema ~x
  |> expect_error is_schema_mismatch;
  let nonconverging = specification ~tolerance:1e-30 ~max_iterations:1 () in
  Multinomial_logistic_regression.fit nonconverging ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(classification labels) ()
  |> expect_error is_convergence;
  let empty = Matrix.init ~rows:0 ~columns:2 (fun _ _ -> 0.0) |> get_data in
  Alcotest.(check (pair int int))
    "empty decision shape" (0, 3)
    (Multinomial_logistic_regression.decision_function fitted
       ~feature_schema:(schema x) ~x:empty
    |> get |> Matrix.shape);
  Alcotest.(check (pair int int))
    "empty probability shape" (0, 3)
    (Multinomial_logistic_regression.predict_proba fitted
       ~feature_schema:(schema x) ~x:empty
    |> get |> Matrix.shape)

let test_pipeline_and_determinism () =
  let x, labels = training_data () in
  let feature_schema = schema x in
  let specification = specification () in
  let estimator =
    Pipeline.estimator ~name:"multinomial logistic"
      (module Multinomial_logistic_regression)
      ~predict_proba:Multinomial_logistic_regression.predict_proba
      ~classes:Multinomial_logistic_regression.classes specification
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema ~x
      ~y:(classification labels) ()
    |> get
  in
  Alcotest.(check (array int))
    "pipeline prediction" labels
    (Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values);
  Alcotest.(check (pair int int))
    "pipeline probabilities" (6, 3)
    (Pipeline.predict_proba fitted ~feature_schema ~x |> get |> Matrix.shape);
  Alcotest.(check (array int))
    "pipeline classes" [| -3; 7; 11 |]
    (Pipeline.classes fitted |> get);
  let first = fit x labels in
  let second = fit x labels in
  Alcotest.check
    (Alcotest.array (Alcotest.array (Alcotest.float 0.0)))
    "deterministic coefficients"
    (Multinomial_logistic_regression.coefficients first |> Matrix.to_arrays)
    (Multinomial_logistic_regression.coefficients second |> Matrix.to_arrays)

let () =
  Alcotest.run "multinomial logistic regression"
    [
      ( "estimator",
        [
          Alcotest.test_case "fit and inference" `Quick test_fit_and_inference;
          Alcotest.test_case "weights, ties, and stability" `Quick
            test_weights_ties_and_extreme_scores;
          Alcotest.test_case "pipeline and determinism" `Quick
            test_pipeline_and_determinism;
        ] );
      ( "errors",
        [
          Alcotest.test_case "typed failures and empty batches" `Quick
            test_errors_and_empty_prediction;
        ] );
    ]
