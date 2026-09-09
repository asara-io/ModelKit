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

let regression values = Target.regression (Vector.of_array values) |> get_data

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let feature_names schema =
  match Feature_schema.names schema with
  | Some names -> Feature_names.to_array names
  | None -> Alcotest.fail "expected a named feature schema"

type dummy_params = unit
type dummy_fitted = { schema : Feature_schema.t; importances : Vector.t }

let fit_count = ref 0

module Dummy_importance = struct
  type t = dummy_params
  type params = dummy_params
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = dummy_fitted
  type rng = Rng.t

  let clone specification = specification
  let params specification = specification

  let fit () ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    match sample_weight with
    | None ->
        Error
          (Error.make
             ~remediation:"route fold-local weights to every estimator fit"
             (Error.Validation
                {
                  name = "dummy importance estimator sample weights";
                  reason = "were not supplied";
                }))
    | Some weights ->
        if Sample_weight.length weights <> Matrix.rows x then
          Error
            (Error.make
               ~remediation:"align fold-local weights with training rows"
               (Error.Validation
                  {
                    name = "dummy importance estimator sample weights";
                    reason = "have the wrong length";
                  }))
        else (
          incr fit_count;
          let names = feature_names feature_schema in
          let importances =
            Array.map
              (fun name ->
                let index =
                  int_of_string (String.sub name 1 (String.length name - 1))
                in
                Float.of_int (index + 1))
              names
            |> Vector.of_array
          in
          Ok { schema = feature_schema; importances })

  let predict _ ~feature_schema:_ ~x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide representable predictions"
          error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
  let feature_importances fitted = Ok fitted.importances
end

module Dummy_rfecv =
  Recursive_feature_elimination_cv.Regression.Make (Dummy_importance)

module Linear_importance = struct
  include Linear_regression

  let feature_importances fitted =
    Feature_importance.absolute_coefficients (coefficients fitted)
end

module Linear_rfecv =
  Recursive_feature_elimination_cv.Regression.Make (Linear_importance)

module Ridge_importance = struct
  include Ridge_classifier

  let feature_importances fitted =
    Feature_importance.coefficient_norms (coefficients fitted)
end

module Binary_rfecv =
  Recursive_feature_elimination_cv.Binary_classification.Make (Ridge_importance)

module Reverse_execution = struct
  type t = { calls : int ref }

  let concurrency _ = 2

  let map configuration ~f values =
    incr configuration.calls;
    let results = Array.make (Array.length values) None in
    for index = Array.length values - 1 downto 0 do
      results.(index) <- Some (f ~index values.(index))
    done;
    let rec collect index reversed =
      if index = Array.length results then
        Ok (Array.of_list (List.rev reversed))
      else
        match Option.get results.(index) with
        | Ok value -> collect (index + 1) (value :: reversed)
        | Error error -> Error error
    in
    collect 0 []
end

let dataset_values () =
  let x =
    Matrix.init ~rows:6 ~columns:4 (fun row column ->
        Float.of_int ((row * 4) + column))
    |> get_data
  in
  let schema = named_schema [| "f0"; "f1"; "f2"; "f3" |] in
  let y = regression (Array.make 6 0.0) in
  let sample_weight =
    Sample_weight.of_array ~expected_length:6 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    |> get_data
  in
  let groups =
    Groups.create ~expected_length:6 [| 0; 0; 1; 1; 2; 2 |] |> get_data
  in
  (x, schema, y, sample_weight, groups)

let group_splitter () =
  Group_k_fold.create ~folds:3 ()
  |> get
  |> Cross_validation.target_independent_splitter (module Group_k_fold)

let fit_dummy ?max_fits ?execution () =
  let x, schema, y, sample_weight, groups = dataset_values () in
  let metadata = Metadata.create ~sample_weight ~groups () in
  let specification =
    Dummy_rfecv.create ~min_feature_count:1
      ~step:(Recursive_feature_elimination.Count 2) ?max_fits ?execution
      ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  Dummy_rfecv.fit specification ~metadata
    ~rng:(Rng.create (Seed.of_int 23))
    ~feature_schema:schema ~x ~y:(Some y) ()

let test_scores_bounds_groups_and_refit () =
  fit_count := 0;
  let calls = ref 0 in
  let execution =
    Execution.of_backend (module Reverse_execution) { Reverse_execution.calls }
  in
  let fitted = fit_dummy ~max_fits:12 ~execution () |> get in
  Alcotest.(check int) "execution receives one bounded fold batch" 1 !calls;
  Alcotest.(check int) "all fold-path and refit fits" 12 !fit_count;
  Alcotest.(check int) "reported fit count" 12 (Dummy_rfecv.fit_count fitted);
  Alcotest.(check int)
    "smallest tied width wins" 1
    (Dummy_rfecv.selected_feature_count fitted);
  Alcotest.(check (array int))
    "selected feature" [| 3 |]
    (Dummy_rfecv.selected_indices fitted);
  Alcotest.(check (array int))
    "full-data ranking" [| 3; 3; 2; 1 |]
    (Dummy_rfecv.ranking fitted);
  let results = Dummy_rfecv.cv_results fitted in
  Alcotest.(check (array int))
    "ascending widths" [| 1; 2; 4 |]
    (Array.map
       (fun score -> score.Recursive_feature_elimination_cv.feature_count)
       results);
  Array.iter
    (fun score ->
      Alcotest.check (Alcotest.float 0.0) "mean score" 0.0
        score.Recursive_feature_elimination_cv.mean_score;
      Alcotest.(check int)
        "one score per fold" 3
        (Array.length score.Recursive_feature_elimination_cv.fold_scores))
    results;
  results.(0).Recursive_feature_elimination_cv.fold_scores.(0) <- 99.0;
  Alcotest.check (Alcotest.float 0.0) "CV results own their fold scores" 0.0
    (Dummy_rfecv.cv_results fitted).(0)
      .Recursive_feature_elimination_cv.fold_scores.(0);
  Alcotest.(check (array string))
    "selected output schema" [| "f3" |]
    (feature_names (Dummy_rfecv.output_schema fitted));
  let transformed =
    let x, schema, _, _, _ = dataset_values () in
    Dummy_rfecv.transform fitted ~metadata:Metadata.empty ~feature_schema:schema
      ~x
    |> get
  in
  Alcotest.(check (pair int int))
    "selected matrix" (6, 1) (Matrix.shape transformed)

let test_bound_preflight_and_validation () =
  fit_count := 0;
  expect_validation (fit_dummy ~max_fits:11 ());
  Alcotest.(check int) "fit bound fails before estimator fitting" 0 !fit_count;
  expect_validation
    (Dummy_rfecv.create ~max_fits:0 ~splitter:(group_splitter ())
       ~scorer:Regression_scorer.neg_mean_squared_error ());
  let x, schema, y, sample_weight, groups = dataset_values () in
  let metadata = Metadata.create ~sample_weight ~groups () in
  let too_wide =
    Dummy_rfecv.create ~min_feature_count:5 ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  expect_validation
    (Dummy_rfecv.fit too_wide ~metadata
       ~rng:(Rng.create (Seed.of_int 0))
       ~feature_schema:schema ~x ~y:(Some y) ());
  let valid =
    Dummy_rfecv.create ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  expect_validation
    (Dummy_rfecv.fit valid ~metadata
       ~rng:(Rng.create (Seed.of_int 0))
       ~feature_schema:schema ~x ~y:None ());
  let classification_splitter =
    K_fold.create ~folds:2 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  expect_validation
    (Binary_rfecv.create ~splitter:classification_splitter
       ~scorer:(Binary_classification_scorer.neg_log_loss ())
       (Ridge_classifier.create () |> get))

let regression_pipeline () =
  let splitter =
    K_fold.create ~folds:2 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let selector =
    Linear_rfecv.create ~min_feature_count:1 ~max_fits:9 ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> Pipeline.Supervised.metadata_transformer ~name:"recursive_select_cv"
         (module Linear_rfecv)
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

let test_nested_fold_leakage () =
  let rows = 18 in
  let original_x =
    Matrix.init ~rows ~columns:3 (fun row column ->
        let value = Float.of_int row in
        match column with
        | 0 -> value
        | 1 -> Float.sin value
        | _ -> Float.of_int (row * 7 mod 5))
    |> get_data
  in
  let names =
    Feature_names.create ~expected_count:3 [| "signal"; "wave"; "cycle" |]
    |> get_data
  in
  let original_y =
    Array.init rows (fun row ->
        (5.0 *. Float.of_int row) +. (0.2 *. Float.sin (Float.of_int row)))
  in
  let outer_splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let run x targets =
    let dataset =
      Dataset.create ~finiteness:Dataset.Require_finite ~feature_names:names ~x
        ~y:(regression targets) ()
      |> get_data
    in
    Cross_validation.Regression.cross_validate ~return_models:true
      ~return_indices:true ~splitter:outer_splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 41) (regression_pipeline ()) dataset
    |> get
  in
  let baseline = run original_x original_y in
  let first = (Cross_validation.folds baseline).(0) in
  let changed_rows = Array.make rows false in
  let changed_y = Array.copy original_y in
  Array.iter
    (fun row ->
      changed_rows.(row) <- true;
      changed_y.(row) <- 1e6 *. Float.of_int (row + 1))
    (Option.get first.Cross_validation.test_indices);
  let changed_x =
    Matrix.init ~rows ~columns:3 (fun row column ->
        if changed_rows.(row) then 1e5 *. Float.of_int ((row + 1) * (column + 1))
        else Matrix.get original_x row column)
    |> get_data
  in
  let changed_first = (Cross_validation.folds (run changed_x changed_y)).(0) in
  let output_schema fold =
    Pipeline.output_schema (Option.get fold.Cross_validation.model)
  in
  Alcotest.(check (array string))
    "outer held-out rows cannot alter inner CV selection"
    (feature_names (output_schema first))
    (feature_names (output_schema changed_first))

let () =
  Alcotest.run "Recursive feature elimination with cross-validation"
    [
      ( "selection",
        [
          Alcotest.test_case "scores, bounds, groups, and refit" `Quick
            test_scores_bounds_groups_and_refit;
          Alcotest.test_case "preflight and validation" `Quick
            test_bound_preflight_and_validation;
        ] );
      ( "integration",
        [
          Alcotest.test_case "nested fold leakage" `Quick
            test_nested_fold_leakage;
        ] );
    ]
