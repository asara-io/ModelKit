open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let[@warning "-4"] expect_validation = function
  | Error error -> (
      match Error.kind error with
      | Error.Validation _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a validation error"

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let rng () = Rng.create (Seed.of_int 1729)

let feature_names schema =
  match Feature_schema.names schema with
  | Some names -> Feature_names.to_array names
  | None -> Alcotest.fail "expected a named feature schema"

let test_regression_scores_selection_and_schema () =
  let x =
    matrix
      (Array.init 8 (fun row ->
           let value = Float.of_int row in
           [| value; value; 7.0; Float.of_int (row * 5 mod 7) |]))
  in
  let y = regression [| 0.1; 1.2; 1.8; 3.4; 3.9; 5.1; 5.8; 7.3 |] in
  let schema = named_schema [| "first"; "tied"; "constant"; "noise" |] in
  let specification =
    Univariate_selection.Regression.create (Univariate_selection.Count 1) |> get
  in
  let fitted =
    Univariate_selection.Regression.fit specification ~rng:(rng ())
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  Alcotest.(check (array int))
    "lower input index wins an exact score tie" [| 0 |]
    (Univariate_selection.Regression.selected_indices fitted);
  let scores = Univariate_selection.Regression.scores fitted in
  Alcotest.check (Alcotest.float 0.0) "identical columns have identical scores"
    (Vector.get scores 0) (Vector.get scores 1);
  Alcotest.check (Alcotest.float 0.0) "constant column scores zero" 0.0
    (Vector.get scores 2);
  Alcotest.(check (array string))
    "selected feature name" [| "first" |]
    (feature_names (Univariate_selection.Regression.output_schema fitted));
  let transformed =
    Univariate_selection.Regression.transform fitted ~feature_schema:schema ~x
    |> get
  in
  Alcotest.(check (pair int int))
    "selected shape" (8, 1) (Matrix.shape transformed);
  Array.iteri
    (fun row _ ->
      Alcotest.check (Alcotest.float 0.0) "selected value" (Matrix.get x row 0)
        (Matrix.get transformed row 0))
    (Array.make 8 ());
  let percentile =
    Univariate_selection.Regression.create
      (Univariate_selection.Percentile 50.0)
    |> get
  in
  let percentile_fitted =
    Univariate_selection.Regression.fit percentile ~rng:(rng ())
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  Alcotest.(check (array int))
    "percentile floors selected width" [| 0; 1 |]
    (Univariate_selection.Regression.selected_indices percentile_fitted);
  let extreme_x =
    matrix [| [| -.Float.max_float |]; [| 0.0 |]; [| Float.max_float |] |]
  in
  let extreme_y = regression [| -.Float.max_float; 0.0; Float.max_float |] in
  let extreme_schema = named_schema [| "extreme" |] in
  let extreme_fitted =
    Univariate_selection.Regression.fit specification ~rng:(rng ())
      ~feature_schema:extreme_schema ~x:extreme_x ~y:(Some extreme_y) ()
    |> get
  in
  Alcotest.check (Alcotest.float 0.0) "perfect extreme correlation"
    Float.max_float
    (Vector.get (Univariate_selection.Regression.scores extreme_fitted) 0)

let test_classification_scores_and_schema () =
  let labels = Target.classification [| -5; -5; -5; 2; 2; 2; 9; 9; 9 |] in
  let x =
    matrix
      [|
        [| -0.2; 0.0; 3.0 |];
        [| 0.1; 1.0; 3.0 |];
        [| 0.2; 2.0; 3.0 |];
        [| 2.8; 2.0; 3.0 |];
        [| 3.1; 0.0; 3.0 |];
        [| 3.2; 1.0; 3.0 |];
        [| 5.8; 1.0; 3.0 |];
        [| 6.1; 2.0; 3.0 |];
        [| 6.2; 0.0; 3.0 |];
      |]
  in
  let schema = named_schema [| "class_signal"; "cycle"; "constant" |] in
  let specification =
    Univariate_selection.Classification.create (Univariate_selection.Count 1)
    |> get
  in
  let fitted =
    Univariate_selection.Classification.fit specification ~rng:(rng ())
      ~feature_schema:schema ~x ~y:(Some labels) ()
    |> get
  in
  let scores = Univariate_selection.Classification.scores fitted in
  Alcotest.(check bool)
    "class signal ranks first" true
    (Vector.get scores 0 > Vector.get scores 1);
  Alcotest.check (Alcotest.float 0.0) "constant ANOVA score" 0.0
    (Vector.get scores 2);
  Alcotest.(check (array int))
    "classification selection" [| 0 |]
    (Univariate_selection.Classification.selected_indices fitted);
  Alcotest.(check (array string))
    "classification schema" [| "class_signal" |]
    (feature_names (Univariate_selection.Classification.output_schema fitted));
  let perfect_x = matrix [| [| 0.0 |]; [| 0.0 |]; [| 9.0 |]; [| 9.0 |] |] in
  let perfect_schema = named_schema [| "perfect" |] in
  let perfect_fitted =
    Univariate_selection.Classification.fit specification ~rng:(rng ())
      ~feature_schema:perfect_schema ~x:perfect_x
      ~y:(Some (Target.classification [| 0; 0; 1; 1 |]))
      ()
    |> get
  in
  Alcotest.check (Alcotest.float 0.0) "zero within-class variance"
    Float.max_float
    (Vector.get (Univariate_selection.Classification.scores perfect_fitted) 0)

let[@warning "-4"] test_validation_boundaries () =
  expect_validation
    (Univariate_selection.Regression.create (Univariate_selection.Count 0));
  expect_validation
    (Univariate_selection.Regression.create
       (Univariate_selection.Percentile 0.0));
  expect_validation
    (Univariate_selection.Regression.create
       (Univariate_selection.Percentile Float.nan));
  expect_validation
    (Univariate_selection.Classification.create
       (Univariate_selection.Percentile 101.0));
  let x = matrix [| [| 0.0; 1.0 |]; [| 1.0; 0.0 |]; [| 2.0; 1.0 |] |] in
  let schema = named_schema [| "a"; "b" |] in
  let y = regression [| 0.0; 1.0; 2.0 |] in
  let count_three =
    Univariate_selection.Regression.create (Univariate_selection.Count 3) |> get
  in
  expect_validation
    (Univariate_selection.Regression.fit count_three ~rng:(rng ())
       ~feature_schema:schema ~x ~y:(Some y) ());
  let tiny_percentile =
    Univariate_selection.Regression.create
      (Univariate_selection.Percentile 10.0)
    |> get
  in
  expect_validation
    (Univariate_selection.Regression.fit tiny_percentile ~rng:(rng ())
       ~feature_schema:schema ~x ~y:(Some y) ());
  let one =
    Univariate_selection.Regression.create (Univariate_selection.Count 1) |> get
  in
  expect_validation
    (Univariate_selection.Regression.fit one ~rng:(rng ())
       ~feature_schema:schema ~x ~y:None ());
  let weights =
    Sample_weight.of_array ~expected_length:3 [| 1.0; 1.0; 1.0 |] |> get_data
  in
  expect_validation
    (Univariate_selection.Regression.fit one ~sample_weight:weights
       ~rng:(rng ()) ~feature_schema:schema ~x ~y:(Some y) ());
  let two_rows = matrix [| [| 0.0 |]; [| 1.0 |] |] in
  let one_schema = named_schema [| "value" |] in
  let two_targets = regression [| 0.0; 1.0 |] in
  expect_validation
    (Univariate_selection.Regression.fit one ~rng:(rng ())
       ~feature_schema:one_schema ~x:two_rows ~y:(Some two_targets) ());
  let class_one =
    Univariate_selection.Classification.create (Univariate_selection.Count 1)
    |> get
  in
  expect_validation
    (Univariate_selection.Classification.fit class_one ~rng:(rng ())
       ~feature_schema:schema ~x
       ~y:(Some (Target.classification [| 4; 4; 4 |]))
       ());
  let no_residual = matrix [| [| 0.0 |]; [| 1.0 |] |] in
  expect_validation
    (Univariate_selection.Classification.fit class_one ~rng:(rng ())
       ~feature_schema:one_schema ~x:no_residual
       ~y:(Some (Target.classification [| 0; 1 |]))
       ());
  let short_y = regression [| 0.0; 1.0 |] in
  (match
     Univariate_selection.Regression.fit one ~rng:(rng ())
       ~feature_schema:schema ~x ~y:(Some short_y) ()
   with
  | Error error -> (
      match Error.kind error with
      | Error.Data (Data_error.Length_mismatch _) -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a target length error");
  let nonfinite =
    matrix [| [| 0.0; Float.infinity |]; [| 1.0; 0.0 |]; [| 2.0; 1.0 |] |]
  in
  (match
     Univariate_selection.Regression.fit one ~rng:(rng ())
       ~feature_schema:schema ~x:nonfinite ~y:(Some y) ()
   with
  | Error error -> (
      match Error.kind error with
      | Error.Data (Data_error.Non_finite _) -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a non-finite input error");
  let fitted =
    Univariate_selection.Regression.fit one ~rng:(rng ()) ~feature_schema:schema
      ~x ~y:(Some y) ()
    |> get
  in
  match
    Univariate_selection.Regression.transform fitted
      ~feature_schema:(named_schema [| "changed"; "b" |])
      ~x
  with
  | Error error -> (
      match Error.kind error with
      | Error.Feature_schema_mismatch _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a schema mismatch"

let regression_pipeline () =
  let selector =
    Univariate_selection.Regression.create (Univariate_selection.Count 1)
    |> get
    |> Pipeline.Supervised.transformer ~name:"select"
         (module Univariate_selection.Regression)
    |> get
  in
  let builder =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty selector
    |> get
  in
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.Supervised.set_estimator builder estimator |> get

let test_fold_local_regression_pipeline () =
  let rows = 15 in
  let x =
    Matrix.init ~rows ~columns:3 (fun row column ->
        let value = Float.of_int row in
        match column with
        | 0 -> value
        | 1 -> Float.sin value
        | _ -> Float.of_int (row * 7 mod 5))
    |> get_data
  in
  let names = [| "signal"; "wave"; "cycle" |] in
  let admitted_names =
    Feature_names.create ~expected_count:3 names |> get_data
  in
  let run target =
    let dataset =
      Dataset.create ~finiteness:Dataset.Require_finite
        ~feature_names:admitted_names ~x ~y:target ()
      |> get_data
    in
    let splitter =
      K_fold.create ~folds:3 () |> get
      |> Cross_validation.target_independent_splitter (module K_fold)
    in
    Cross_validation.Regression.cross_validate ~return_models:true
      ~return_indices:true ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 23) (regression_pipeline ()) dataset
    |> get
  in
  let original_values =
    Array.init rows (fun row ->
        (2.0 *. Float.of_int row) +. Float.sin (Float.of_int row))
  in
  let original = run (regression original_values) in
  Alcotest.(check int)
    "all selector folds fit" 3
    (Cross_validation.successful_fold_count original);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let model = Option.get fold.Cross_validation.model in
      Alcotest.(check (array string))
        "fold output schema" [| "signal" |]
        (feature_names (Pipeline.output_schema model)))
    (Cross_validation.folds original);
  let first = (Cross_validation.folds original).(0) in
  let changed_values = Array.copy original_values in
  Array.iter
    (fun row -> changed_values.(row) <- 1e6 *. Matrix.get x row 2)
    (Option.get first.Cross_validation.test_indices);
  let changed = run (regression changed_values) in
  let changed_first = (Cross_validation.folds changed).(0) in
  Alcotest.(check (array string))
    "held-out targets cannot alter fitted selection"
    (feature_names
       (Pipeline.output_schema (Option.get first.Cross_validation.model)))
    (feature_names
       (Pipeline.output_schema
          (Option.get changed_first.Cross_validation.model)))

let classification_pipeline () =
  let selector =
    Univariate_selection.Classification.create (Univariate_selection.Count 1)
    |> get
    |> Pipeline.Supervised.transformer ~name:"select"
         (module Univariate_selection.Classification)
    |> get
  in
  let builder =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty selector
    |> get
  in
  let estimator =
    Pipeline.classifier ~name:"ridge" ~classes:Ridge_classifier.classes
      (module Ridge_classifier)
      (Ridge_classifier.create ~alpha:0.1 () |> get)
    |> get
  in
  Pipeline.Supervised.set_estimator builder estimator |> get

let test_classification_cross_validation () =
  let rows = 18 in
  let labels = Array.init rows (fun row -> [| -3; 4; 11 |].(row / 6)) in
  let x =
    Matrix.init ~rows ~columns:3 (fun row column ->
        match column with
        | 0 -> Float.of_int labels.(row) +. (0.05 *. Float.of_int (row mod 3))
        | 1 -> Float.of_int (row * 5 mod 7)
        | _ -> 2.0)
    |> get_data
  in
  let admitted_names =
    Feature_names.create ~expected_count:3 [| "signal"; "noise"; "constant" |]
    |> get_data
  in
  let y = Target.classification labels in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite
      ~feature_names:admitted_names ~x ~y ()
    |> get_data
  in
  let splitter =
    Stratified_k_fold.create ~folds:3 ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let report =
    Cross_validation.Multiclass_classification.cross_validate
      ~return_models:true ~splitter
      ~scorers:[| Multiclass_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 51)
      (classification_pipeline ())
      dataset
    |> get
  in
  Alcotest.(check int)
    "all classification selector folds fit" 3
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      Alcotest.(check (array string))
        "classification fold schema" [| "signal" |]
        (feature_names
           (Pipeline.output_schema (Option.get fold.Cross_validation.model))))
    (Cross_validation.folds report)

let () =
  Alcotest.run "Univariate feature selection"
    [
      ( "scores",
        [
          Alcotest.test_case "regression ties and schema" `Quick
            test_regression_scores_selection_and_schema;
          Alcotest.test_case "classification and schema" `Quick
            test_classification_scores_and_schema;
        ] );
      ( "validation",
        [
          Alcotest.test_case "declared boundaries" `Quick
            test_validation_boundaries;
        ] );
      ( "integration",
        [
          Alcotest.test_case "fold-local regression pipeline" `Quick
            test_fold_local_regression_pipeline;
          Alcotest.test_case "classification cross-validation" `Quick
            test_classification_cross_validation;
        ] );
    ]
