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

let[@warning "-4"] expect_numerical = function
  | Error error -> (
      match Error.kind error with
      | Error.Numerical _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a numerical error"

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let feature_names schema =
  match Feature_schema.names schema with
  | Some names -> Feature_names.to_array names
  | None -> Alcotest.fail "expected a named feature schema"

let rng () = Rng.create (Seed.of_int 1729)

type static_params = { values : Vector.t; report_wrong_schema : bool }

type static_fitted = {
  params : static_params;
  schema : Feature_schema.t;
  weight_sum : float;
}

module Static_importance = struct
  type t = static_params
  type params = static_params
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = static_fitted
  type rng = Rng.t

  let clone specification = specification
  let params specification = specification

  let fit params ?sample_weight ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    let weight_sum =
      match sample_weight with
      | None -> 0.0
      | Some weights ->
          let total = ref 0.0 in
          for index = 0 to Sample_weight.length weights - 1 do
            total := !total +. Sample_weight.get weights index
          done;
          !total
    in
    let schema =
      if params.report_wrong_schema then
        Feature_schema.anonymous
          ~feature_count:(Feature_schema.feature_count feature_schema + 1)
        |> get_data
      else feature_schema
    in
    Ok { params; schema; weight_sum }

  let predict _ ~feature_schema:_ ~x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide representable predictions"
          error)

  let fitted_params fitted = fitted.params
  let feature_schema fitted = fitted.schema
  let feature_importances fitted = Ok fitted.params.values
end

module Static_selector = Select_from_model.Make (Static_importance)

module Linear_importance = struct
  include Linear_regression

  let feature_importances fitted =
    Feature_importance.absolute_coefficients (coefficients fitted)
end

module Linear_selector = Select_from_model.Make (Linear_importance)

module Ridge_classifier_importance = struct
  include Ridge_classifier

  let feature_importances fitted =
    Feature_importance.coefficient_norms (coefficients fitted)
end

module Ridge_classifier_selector =
  Select_from_model.Make (Ridge_classifier_importance)

let static ?(report_wrong_schema = false) values =
  { values = Vector.of_array values; report_wrong_schema }

let fit_static ?sample_weight specification schema x =
  Static_selector.fit specification ?sample_weight ~rng:(rng ())
    ~feature_schema:schema ~x
    ~y:(Some (regression (Array.init (Matrix.rows x) Float.of_int)))
    ()

let test_coefficient_helpers () =
  let absolute =
    Feature_importance.absolute_coefficients
      (Vector.of_array [| -2.0; 0.0; 3.5 |])
    |> get
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "absolute coefficients" [| 2.0; 0.0; 3.5 |] (Vector.to_array absolute);
  let coefficients = matrix [| [| 3.0; -4.0 |]; [| -4.0; 3.0 |] |] in
  let norms norm =
    Feature_importance.coefficient_norms ~norm coefficients
    |> get |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "L1 rows" [| 7.0; 7.0 |]
    (norms Feature_importance.L1);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "scaled L2 rows" [| 5.0; 5.0 |]
    (norms Feature_importance.L2);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-12))
    "maximum rows" [| 4.0; 4.0 |]
    (norms Feature_importance.Max);
  expect_numerical
    (Feature_importance.absolute_coefficients (Vector.of_array [| Float.nan |]));
  expect_validation
    (Feature_importance.coefficient_norms
       (Matrix.init ~rows:0 ~columns:2 (fun _ _ -> 0.0) |> get_data));
  expect_numerical
    (Feature_importance.coefficient_norms ~norm:Feature_importance.L2
       (matrix [| [| Float.max_float |]; [| Float.max_float |] |]))

let test_thresholds_ties_schema_and_weights () =
  let x =
    matrix
      [|
        [| 0.0; 10.0; 20.0; 30.0 |];
        [| 1.0; 11.0; 21.0; 31.0 |];
        [| 2.0; 12.0; 22.0; 32.0 |];
      |]
  in
  let schema = named_schema [| "first"; "tied"; "third"; "zero" |] in
  let mean = Static_selector.create (static [| 3.0; 3.0; 1.0; 0.0 |]) |> get in
  let mean_fitted = fit_static mean schema x |> get in
  Alcotest.check (Alcotest.float 0.0) "resolved mean" 1.75
    (Static_selector.threshold_value mean_fitted);
  Alcotest.(check (array int))
    "mean threshold" [| 0; 1 |]
    (Static_selector.selected_indices mean_fitted);
  let median =
    Static_selector.create ~threshold:Select_from_model.Median
      (static [| 3.0; 3.0; 1.0; 0.0 |])
    |> get
    |> fun specification -> fit_static specification schema x |> get
  in
  Alcotest.check (Alcotest.float 0.0) "resolved median" 2.0
    (Static_selector.threshold_value median);
  let capped =
    Static_selector.create ~threshold:(Select_from_model.Value 0.0)
      ~max_features:1
      (static [| 3.0; 3.0; 1.0; 0.0 |])
    |> get
    |> fun specification -> fit_static specification schema x |> get
  in
  Alcotest.(check (array int))
    "lower input index wins a capped tie" [| 0 |]
    (Static_selector.selected_indices capped);
  Alcotest.(check (array string))
    "selected schema" [| "first" |]
    (feature_names (Static_selector.output_schema capped));
  let transformed =
    Static_selector.transform capped ~feature_schema:schema ~x |> get
  in
  Alcotest.(check (pair int int))
    "selected shape" (3, 1) (Matrix.shape transformed);
  let weights =
    Sample_weight.of_array ~expected_length:3 [| 1.0; 2.0; 3.0 |] |> get_data
  in
  let weighted = fit_static ~sample_weight:weights mean schema x |> get in
  Alcotest.check (Alcotest.float 0.0) "weights reach importance estimator" 6.0
    (Static_selector.fitted_estimator weighted).weight_sum

let[@warning "-4"] test_validation_boundaries () =
  expect_validation
    (Static_selector.create ~threshold:(Select_from_model.Value (-0.1))
       (static [| 1.0 |]));
  expect_validation
    (Static_selector.create ~threshold:(Select_from_model.Value Float.nan)
       (static [| 1.0 |]));
  expect_validation (Static_selector.create ~max_features:0 (static [| 1.0 |]));
  let x = matrix [| [| 0.0; 1.0 |]; [| 1.0; 0.0 |]; [| 2.0; 1.0 |] |] in
  let schema = named_schema [| "a"; "b" |] in
  let fit values =
    Static_selector.create ~threshold:(Select_from_model.Value 0.0)
      (static values)
    |> get
    |> fun specification -> fit_static specification schema x
  in
  (match fit [| 1.0 |] with
  | Error error -> (
      match Error.kind error with
      | Error.Shape_mismatch _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected an importance length error");
  expect_validation (fit [| 1.0; -1.0 |]);
  expect_validation (fit [| 1.0; Float.nan |]);
  let none_selected =
    Static_selector.create ~threshold:(Select_from_model.Value 3.0)
      (static [| 1.0; 2.0 |])
    |> get
  in
  expect_validation (fit_static none_selected schema x);
  let missing_target = Static_selector.create (static [| 1.0; 2.0 |]) |> get in
  expect_validation
    (Static_selector.fit missing_target ~rng:(rng ()) ~feature_schema:schema ~x
       ~y:None ());
  let empty_x = Matrix.init ~rows:3 ~columns:0 (fun _ _ -> 0.0) |> get_data in
  let empty_schema = Feature_schema.of_matrix empty_x |> get_data in
  let empty_selector = Static_selector.create (static [||]) |> get in
  expect_validation (fit_static empty_selector empty_schema empty_x);
  let wrong_schema =
    Static_selector.create (static ~report_wrong_schema:true [| 1.0; 2.0 |])
    |> get
  in
  (match fit_static wrong_schema schema x with
  | Error error -> (
      match Error.kind error with
      | Error.Compatibility _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected an estimator schema error");
  let short_weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 1.0 |] |> get_data
  in
  match fit_static ~sample_weight:short_weights missing_target schema x with
  | Error error -> (
      match Error.kind error with
      | Error.Data (Data_error.Length_mismatch _) -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a sample-weight length error"

let regression_pipeline () =
  let selector =
    Linear_selector.create ~threshold:(Select_from_model.Value 0.5)
      (Linear_regression.create ())
    |> get
    |> Pipeline.Supervised.transformer ~name:"select" (module Linear_selector)
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

let test_fold_local_regression_cross_validation () =
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
  let admitted_names =
    Feature_names.create ~expected_count:3 [| "signal"; "wave"; "cycle" |]
    |> get_data
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
      ~seed:(Seed.of_int 29) (regression_pipeline ()) dataset
    |> get
  in
  let original_values =
    Array.init rows (fun row ->
        (5.0 *. Float.of_int row) +. (0.2 *. Float.sin (Float.of_int row)))
  in
  let original = run (regression original_values) in
  Alcotest.(check int)
    "all model-selector folds fit" 3
    (Cross_validation.successful_fold_count original);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      Alcotest.(check (array string))
        "fold output schema" [| "signal" |]
        (feature_names
           (Pipeline.output_schema (Option.get fold.Cross_validation.model))))
    (Cross_validation.folds original);
  let first = (Cross_validation.folds original).(0) in
  let changed_values = Array.copy original_values in
  Array.iter
    (fun row -> changed_values.(row) <- 1e6 *. Matrix.get x row 2)
    (Option.get first.Cross_validation.test_indices);
  let changed_first =
    (Cross_validation.folds (run (regression changed_values))).(0)
  in
  Alcotest.(check (array string))
    "held-out targets cannot alter fitted importances"
    (feature_names
       (Pipeline.output_schema (Option.get first.Cross_validation.model)))
    (feature_names
       (Pipeline.output_schema
          (Option.get changed_first.Cross_validation.model)))

let classification_pipeline () =
  let selector =
    Ridge_classifier_selector.create ~threshold:Select_from_model.Median
      ~max_features:2
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
    |> Pipeline.Supervised.transformer ~name:"select"
         (module Ridge_classifier_selector)
    |> get
  in
  let builder =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty selector
    |> get
  in
  let estimator =
    Pipeline.classifier ~name:"ridge" ~classes:Ridge_classifier.classes
      (module Ridge_classifier)
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
  in
  Pipeline.Supervised.set_estimator builder estimator |> get

let test_classification_cross_validation () =
  let rows = 18 in
  let labels = Array.init rows (fun row -> [| -3; 4; 11 |].(row / 6)) in
  let x =
    Matrix.init ~rows ~columns:4 (fun row column ->
        match column with
        | 0 -> Float.of_int labels.(row) +. (0.05 *. Float.of_int (row mod 3))
        | 1 -> Float.of_int (row * 5 mod 7)
        | 2 -> Float.sin (Float.of_int row)
        | _ -> Float.of_int (row mod 2))
    |> get_data
  in
  let feature_names =
    Feature_names.create ~expected_count:4
      [| "signal"; "noise"; "wave"; "binary" |]
    |> get_data
  in
  let y = Target.classification labels in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~feature_names ~x ~y ()
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
      ~seed:(Seed.of_int 61)
      (classification_pipeline ())
      dataset
    |> get
  in
  Alcotest.(check int)
    "all classification selector folds fit" 3
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      Alcotest.(check int)
        "classification selector width" 2
        (Feature_schema.feature_count
           (Pipeline.output_schema (Option.get fold.Cross_validation.model))))
    (Cross_validation.folds report)

let () =
  Alcotest.run "Model-based feature selection"
    [
      ( "importance",
        [
          Alcotest.test_case "coefficient helpers" `Quick
            test_coefficient_helpers;
        ] );
      ( "selection",
        [
          Alcotest.test_case "thresholds, ties, schemas, and weights" `Quick
            test_thresholds_ties_schema_and_weights;
          Alcotest.test_case "validation boundaries" `Quick
            test_validation_boundaries;
        ] );
      ( "integration",
        [
          Alcotest.test_case "fold-local regression cross-validation" `Quick
            test_fold_local_regression_cross_validation;
          Alcotest.test_case "classification cross-validation" `Quick
            test_classification_cross_validation;
        ] );
    ]
