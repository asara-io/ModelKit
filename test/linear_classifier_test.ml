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

let fit ?sample_weight ?(alpha = 1.0) ?(fit_intercept = true) x labels =
  let specification = Ridge_classifier.create ~alpha ~fit_intercept () |> get in
  Ridge_classifier.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:(classification labels) ()
  |> get

let check_float ?(epsilon = 1e-9) message expected observed =
  Alcotest.check (Alcotest.float epsilon) message expected observed

let test_binary () =
  let x = matrix [| [| -2.0 |]; [| -1.0 |]; [| 1.0 |]; [| 2.0 |] |] in
  let fitted = fit ~alpha:1.0 x [| 10; 10; 20; 20 |] in
  Alcotest.(check (array int))
    "ascending classes" [| 10; 20 |]
    (Ridge_classifier.classes fitted);
  let coefficients = Ridge_classifier.coefficients fitted in
  Alcotest.(check (pair int int))
    "binary coefficient shape" (2, 1)
    (Matrix.shape coefficients);
  check_float "negative coefficient" (-6.0 /. 11.0)
    (Matrix.get coefficients 0 0);
  check_float "positive coefficient" (6.0 /. 11.0) (Matrix.get coefficients 1 0);
  check_float "negative intercept" 0.0
    (Vector.get (Ridge_classifier.intercepts fitted) 0);
  check_float "positive intercept" 0.0
    (Vector.get (Ridge_classifier.intercepts fitted) 1);
  let decisions =
    Ridge_classifier.decision_function fitted ~feature_schema:(schema x) ~x
    |> get
  in
  Alcotest.(check (pair int int))
    "binary decision shape" (4, 2) (Matrix.shape decisions);
  for row = 0 to Matrix.rows decisions - 1 do
    check_float "binary scores are opposites" 0.0
      (Matrix.get decisions row 0 +. Matrix.get decisions row 1)
  done;
  let prediction =
    Ridge_classifier.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int))
    "binary prediction" [| 10; 10; 20; 20 |] prediction;
  let reports = Ridge_classifier.reports fitted in
  Alcotest.(check int) "one report per class" 2 (Array.length reports);
  Array.iter
    (fun report ->
      Alcotest.(check bool)
        "direct solver converged" true
        (Solver_report.converged report);
      Alcotest.(check bool)
        "direct stopping reason" true
        (Solver_report.stopping_reason report = Solver_report.Direct_solution))
    reports

let test_multiclass () =
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
  let labels = [| -3; -3; 7; 7; 11; 11 |] in
  let fitted = fit ~alpha:0.5 x labels in
  Alcotest.(check (array int))
    "multiclass ordering" [| -3; 7; 11 |]
    (Ridge_classifier.classes fitted);
  Alcotest.(check (pair int int))
    "multiclass coefficient shape" (3, 2)
    (Matrix.shape (Ridge_classifier.coefficients fitted));
  Alcotest.(check (pair int int))
    "multiclass decision shape" (6, 3)
    (Ridge_classifier.decision_function fitted ~feature_schema:(schema x) ~x
    |> get |> Matrix.shape);
  let prediction =
    Ridge_classifier.predict fitted ~feature_schema:(schema x) ~x
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int)) "multiclass prediction" labels prediction

let test_weights_and_ties () =
  let x = matrix [| [| -1.0 |]; [| 1.0 |]; [| 100.0 |]; [| 200.0 |] |] in
  let sample_weight = weights [| 1.0; 1.0; 0.0; 0.0 |] in
  let fitted = fit ~sample_weight x [| 4; 9; 20; 20 |] in
  Alcotest.(check (array int))
    "zero-weight class is excluded" [| 4; 9 |]
    (Ridge_classifier.classes fitted);
  let zeros = matrix [| [| 0.0 |]; [| 0.0 |] |] in
  let tied = fit ~fit_intercept:false zeros [| 8; -2 |] in
  let prediction =
    Ridge_classifier.predict tied ~feature_schema:(schema zeros) ~x:zeros
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int))
    "ties choose the lowest class" [| -2; -2 |] prediction

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | _ -> false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_errors_and_empty_prediction () =
  Ridge_classifier.create ~alpha:(-1.0) () |> expect_error is_validation;
  let x = matrix [| [| 0.0 |]; [| 1.0 |] |] in
  let specification = Ridge_classifier.create () |> get in
  Ridge_classifier.fit specification ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:(classification [| 1; 1 |])
    ()
  |> expect_error is_validation;
  let fitted = fit x [| 1; 2 |] in
  let wrong_schema = Feature_schema.anonymous ~feature_count:2 |> get_data in
  Ridge_classifier.predict fitted ~feature_schema:wrong_schema ~x
  |> expect_error is_schema_mismatch;
  let empty = Matrix.init ~rows:0 ~columns:1 (fun _ _ -> 0.0) |> get_data in
  Alcotest.(check (pair int int))
    "empty decision shape" (0, 2)
    (Ridge_classifier.decision_function fitted ~feature_schema:(schema x)
       ~x:empty
    |> get |> Matrix.shape);
  Alcotest.(check int)
    "empty prediction length" 0
    (Ridge_classifier.predict fitted ~feature_schema:(schema x) ~x:empty
    |> get |> Target.classification_values |> Array.length)

let test_pipeline () =
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
  let feature_schema = schema x in
  let labels = classification [| -3; -3; 7; 7; 11; 11 |] in
  let estimator =
    Pipeline.estimator ~name:"ridge classifier"
      ~classes:Ridge_classifier.classes
      (module Ridge_classifier)
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let fitted =
    Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema ~x ~y:labels () |> get
  in
  Alcotest.(check (array int))
    "pipeline multiclass prediction"
    (Target.classification_values labels)
    (Pipeline.predict fitted ~feature_schema ~x
    |> get |> Target.classification_values)

let () =
  Alcotest.run "linear classifiers"
    [
      ( "ridge classifier",
        [
          Alcotest.test_case "binary" `Quick test_binary;
          Alcotest.test_case "multiclass" `Quick test_multiclass;
          Alcotest.test_case "weights and ties" `Quick test_weights_and_ties;
          Alcotest.test_case "pipeline" `Quick test_pipeline;
        ] );
      ( "errors",
        [
          Alcotest.test_case "typed failures and empty prediction" `Quick
            test_errors_and_empty_prediction;
        ] );
    ]
