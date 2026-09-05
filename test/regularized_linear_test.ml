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

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let check_float ?(epsilon = 1e-9) message expected observed =
  Alcotest.check (Alcotest.float epsilon) message expected observed

let check_vector ?(epsilon = 1e-9) message expected observed =
  Alcotest.check
    (Alcotest.array (Alcotest.float epsilon))
    message expected (Vector.to_array observed)

let lasso_fit ?sample_weight ?(alpha = 1.0) ?(fit_intercept = true)
    ?(tolerance = 1e-10) ?(max_iterations = 1000) x values =
  let specification =
    Lasso_regression.create ~alpha ~fit_intercept ~tolerance ~max_iterations ()
    |> get
  in
  Lasso_regression.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(regression values) ()
  |> get

let elastic_fit ?sample_weight ?(alpha = 1.0) ?(l1_ratio = 0.5)
    ?(fit_intercept = true) ?(tolerance = 1e-10) ?(max_iterations = 1000) x
    values =
  let specification =
    Elastic_net_regression.create ~alpha ~l1_ratio ~fit_intercept ~tolerance
      ~max_iterations ()
    |> get
  in
  Elastic_net_regression.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(regression values) ()
  |> get

let test_lasso () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let fitted = lasso_fit ~alpha:(2.0 /. 3.0) x [| -2.0; 0.0; 2.0 |] in
  check_vector "lasso coefficient" [| 1.0 |]
    (Lasso_regression.coefficients fitted);
  check_float "lasso intercept" 0.0 (Lasso_regression.intercept fitted);
  let prediction =
    Lasso_regression.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.regression_values
  in
  check_vector "lasso prediction" [| -1.0; 0.0; 1.0 |] prediction;
  let report = Lasso_regression.report fitted in
  Alcotest.(check bool) "lasso converged" true (Solver_report.converged report);
  Alcotest.(check bool)
    "lasso iterated" true
    (Solver_report.iterations report > 0);
  Alcotest.(check (option int))
    "coordinate descent has no rank" None
    (Solver_report.rank report)

let test_elastic_net () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let fitted = elastic_fit ~alpha:1.0 ~l1_ratio:0.5 x [| -2.0; 0.0; 2.0 |] in
  check_vector "elastic-net coefficient"
    [| 5.0 /. 7.0 |]
    (Elastic_net_regression.coefficients fitted);
  check_float "elastic-net intercept" 0.0
    (Elastic_net_regression.intercept fitted);
  let lasso =
    elastic_fit ~alpha:(2.0 /. 3.0) ~l1_ratio:1.0 x [| -2.; 0.; 2. |]
  in
  check_vector "l1_ratio one is lasso" [| 1.0 |]
    (Elastic_net_regression.coefficients lasso)

let test_weights_and_constant_feature () =
  let x =
    matrix
      [| [| -1.0; 7.0 |]; [| 0.0; 7.0 |]; [| 1.0; 7.0 |]; [| 100.0; 7.0 |] |]
  in
  let sample_weight = weights [| 1.0; 1.0; 1.0; 0.0 |] in
  let fitted =
    lasso_fit ~sample_weight ~alpha:(2.0 /. 3.0) x [| -2.0; 0.0; 2.0; -1000.0 |]
  in
  check_vector "zero-weight row and constant feature" [| 1.0; 0.0 |]
    (Lasso_regression.coefficients fitted);
  check_float "constant feature intercept" 0.0
    (Lasso_regression.intercept fitted);
  let scaled_weight = weights [| 10.0; 10.0; 10.0; 0.0 |] in
  let scaled =
    lasso_fit ~sample_weight:scaled_weight ~alpha:(2.0 /. 3.0) x
      [| -2.0; 0.0; 2.0; -1000.0 |]
  in
  check_vector "uniform weight scaling"
    (Vector.to_array (Lasso_regression.coefficients fitted))
    (Lasso_regression.coefficients scaled)

let test_paths () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let y = regression [| -2.0; 0.0; 2.0 |] in
  let explicit = Vector.of_array [| 0.1; 1.0; 0.5 |] in
  let specification = Lasso_path.create ~count:4 ~tolerance:1e-10 () |> get in
  let fitted =
    Lasso_path.fit specification ~alphas:explicit ~rng:(rng ())
      ~feature_schema:(schema x) ~x ~y ()
    |> get
  in
  check_vector "descending explicit alphas" [| 1.0; 0.5; 0.1 |]
    (Lasso_path.alphas fitted);
  let coefficients = Lasso_path.coefficients fitted in
  Alcotest.(check (pair int int))
    "path shape" (3, 1)
    (Matrix.shape coefficients);
  Alcotest.(check bool)
    "weaker regularization grows the coefficient" true
    (Matrix.get coefficients 0 0 < Matrix.get coefficients 1 0
    && Matrix.get coefficients 1 0 < Matrix.get coefficients 2 0);
  let selected = Lasso_path.model fitted ~index:1 |> get in
  check_float "selected path model"
    (Matrix.get coefficients 1 0)
    (Vector.get (Lasso_regression.coefficients selected) 0);
  Alcotest.(check int)
    "one report per alpha" 3
    (Array.length (Lasso_path.reports fitted));
  let automatic =
    Lasso_path.fit specification ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y
      ()
    |> get
  in
  Alcotest.(check int)
    "automatic count" 4
    (Vector.length (Lasso_path.alphas automatic));
  check_float "alpha-max solution is zero" 0.0
    (Matrix.get (Lasso_path.coefficients automatic) 0 0);
  let elastic_spec =
    Elastic_net_path.create ~l1_ratio:0.5 ~count:3 ~tolerance:1e-10 () |> get
  in
  let elastic =
    Elastic_net_path.fit elastic_spec ~rng:(rng ()) ~feature_schema:(schema x)
      ~x ~y ()
    |> get
  in
  Alcotest.(check (pair int int))
    "elastic path shape" (3, 1)
    (Matrix.shape (Elastic_net_path.coefficients elastic))

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_convergence = function
  | Error.Convergence _ -> true
  | _ -> false

let[@warning "-4"] is_index_error = function
  | Error.Data (Data_error.Index_out_of_bounds _) -> true
  | _ -> false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_errors () =
  Lasso_regression.create ~alpha:(-1.0) () |> expect_error is_validation;
  Elastic_net_regression.create ~l1_ratio:1.1 () |> expect_error is_validation;
  Lasso_path.create ~epsilon:0.0 () |> expect_error is_validation;
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let y = regression [| -2.0; 0.0; 2.0 |] in
  let specification =
    Lasso_regression.create ~alpha:0.1 ~tolerance:1e-30 ~max_iterations:1 ()
    |> get
  in
  Lasso_regression.fit specification ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y ()
  |> expect_error is_convergence;
  let path =
    Lasso_path.fit
      (Lasso_path.create ~count:2 () |> get)
      ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y ()
    |> get
  in
  Lasso_path.model path ~index:2 |> expect_error is_index_error;
  let pure_l2 = Elastic_net_path.create ~l1_ratio:0.0 () |> get in
  Elastic_net_path.fit pure_l2 ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y ()
  |> expect_error is_validation

let test_pipeline () =
  let x = matrix [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |] in
  let feature_schema = schema x in
  let specification = Lasso_regression.create ~alpha:0.1 () |> get in
  let estimator =
    Pipeline.estimator ~name:"lasso" (module Lasso_regression) specification
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema ~x
      ~y:(regression [| -2.0; 0.0; 2.0 |])
      ()
    |> get
  in
  let prediction =
    Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.regression_values
  in
  Alcotest.(check int) "pipeline prediction rows" 3 (Vector.length prediction)

let () =
  Alcotest.run "regularized linear models"
    [
      ( "estimators",
        [
          Alcotest.test_case "lasso" `Quick test_lasso;
          Alcotest.test_case "elastic net" `Quick test_elastic_net;
          Alcotest.test_case "weights and constant feature" `Quick
            test_weights_and_constant_feature;
        ] );
      ("paths", [ Alcotest.test_case "regularization paths" `Quick test_paths ]);
      ("integration", [ Alcotest.test_case "pipeline" `Quick test_pipeline ]);
      ("errors", [ Alcotest.test_case "typed failures" `Quick test_errors ]);
    ]
