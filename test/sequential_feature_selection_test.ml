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

let expect_error = function
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "expected an error"

let regression values = Target.regression (Vector.of_array values) |> get_data

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let feature_names schema =
  match Feature_schema.names schema with
  | Some names -> Feature_names.to_array names
  | None -> Alcotest.fail "expected a named feature schema"

type dummy_params = unit
type dummy_fitted = { schema : Feature_schema.t; cost : float }

let fit_count = ref 0

module Dummy_estimator = struct
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
                  name = "dummy sequential estimator sample weights";
                  reason = "were not supplied";
                }))
    | Some weights ->
        if Sample_weight.length weights <> Matrix.rows x then
          Error
            (Error.make
               ~remediation:"align weights with candidate training rows"
               (Error.Validation
                  {
                    name = "dummy sequential estimator sample weights";
                    reason = "have the wrong length";
                  }))
        else (
          incr fit_count;
          let has_f3 =
            feature_names feature_schema |> Array.exists (String.equal "f3")
          in
          Ok { schema = feature_schema; cost = (if has_f3 then 0.0 else 1.0) })

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make ~remediation:"predict with the fitted candidate schema"
           (Error.Compatibility
              {
                component = "dummy sequential estimator";
                reason = "received a different feature schema";
              }))
    else
      Target.regression
        (Vector.of_array (Array.make (Matrix.rows x) fitted.cost))
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide representable predictions"
            error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

module Dummy_sfs = Sequential_feature_selection.Regression.Make (Dummy_estimator)

module Linear_sfs =
  Sequential_feature_selection.Regression.Make (Linear_regression)

type column_fitted = { column_schema : Feature_schema.t }

module Column_estimator = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = column_fitted
  type rng = Rng.t

  let clone specification = specification
  let params specification = specification

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    Ok { column_schema = feature_schema }

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.column_schema feature_schema) then
      Error
        (Error.make ~remediation:"predict with the fitted candidate schema"
           (Error.Compatibility
              {
                component = "column estimator";
                reason = "received a different feature schema";
              }))
    else
      Target.regression
        (Vector.of_array
           (Array.init (Matrix.rows x) (fun row -> Matrix.get x row 0)))
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide representable predictions"
            error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted.column_schema
end

module Weighted_sfs =
  Sequential_feature_selection.Regression.Make (Column_estimator)

module Binary_sfs =
  Sequential_feature_selection.Binary_classification.Make (Ridge_classifier)

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

let fit_dummy ?(direction = Sequential_feature_selection.Forward) ?max_fits
    ?execution ?(feature_count = 2) () =
  let x, schema, y, sample_weight, groups = dataset_values () in
  let metadata = Metadata.create ~sample_weight ~groups () in
  let specification =
    Dummy_sfs.create ~direction ?max_fits ?execution ~feature_count
      ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  Dummy_sfs.fit specification ~metadata
    ~rng:(Rng.create (Seed.of_int 71))
    ~feature_schema:schema ~x ~y:(Some y) ()

let check_selection direction expected =
  fit_count := 0;
  let calls = ref 0 in
  let execution =
    Execution.of_backend (module Reverse_execution) { Reverse_execution.calls }
  in
  let fitted = fit_dummy ~direction ~max_fits:21 ~execution () |> get in
  Alcotest.(check int) "one execution batch per round" 2 !calls;
  Alcotest.(check int) "candidate-fold fits" 21 !fit_count;
  Alcotest.(check int) "reported fit count" 21 (Dummy_sfs.fit_count fitted);
  Alcotest.(check (array int))
    "selected indices" expected
    (Dummy_sfs.selected_indices fitted);
  Alcotest.(check (array string))
    "selected schema"
    (Array.map (Printf.sprintf "f%d") expected)
    (feature_names (Dummy_sfs.output_schema fitted));
  let returned = Dummy_sfs.selected_indices fitted in
  returned.(0) <- 99;
  Alcotest.(check (array int))
    "selected indices are defensive" expected
    (Dummy_sfs.selected_indices fitted);
  let x, _, _, _, _ = dataset_values () in
  let wrong_schema = named_schema [| "f0"; "f1"; "f2"; "different" |] in
  expect_error
    (Dummy_sfs.transform fitted ~metadata:Metadata.empty
       ~feature_schema:wrong_schema ~x);
  let non_finite =
    Matrix.init ~rows:6 ~columns:4 (fun row column ->
        if row = 0 && column = 0 then Float.nan else Matrix.get x row column)
    |> get_data
  in
  expect_error
    (Dummy_sfs.transform fitted ~metadata:Metadata.empty
       ~feature_schema:(Dummy_sfs.input_schema fitted)
       ~x:non_finite)

let test_forward_and_backward_ties () =
  check_selection Sequential_feature_selection.Forward [| 0; 3 |];
  check_selection Sequential_feature_selection.Backward [| 2; 3 |]

let test_bounds_and_validation () =
  fit_count := 0;
  expect_validation (fit_dummy ~max_fits:20 ());
  Alcotest.(check int) "bound fails before fitting" 0 !fit_count;
  let identity = fit_dummy ~feature_count:4 () |> get in
  Alcotest.(check int)
    "identity selection requires no fit" 0
    (Dummy_sfs.fit_count identity);
  Alcotest.(check (array int))
    "identity selection retains every feature" [| 0; 1; 2; 3 |]
    (Dummy_sfs.selected_indices identity);
  expect_validation
    (Dummy_sfs.create ~feature_count:0 ~splitter:(group_splitter ())
       ~scorer:Regression_scorer.neg_mean_squared_error ());
  expect_validation
    (Dummy_sfs.create ~feature_count:1 ~max_fits:0 ~splitter:(group_splitter ())
       ~scorer:Regression_scorer.neg_mean_squared_error ());
  let x, schema, y, sample_weight, groups = dataset_values () in
  let metadata = Metadata.create ~sample_weight ~groups () in
  let too_wide =
    Dummy_sfs.create ~feature_count:5 ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  expect_validation
    (Dummy_sfs.fit too_wide ~metadata
       ~rng:(Rng.create (Seed.of_int 0))
       ~feature_schema:schema ~x ~y:(Some y) ());
  let valid =
    Dummy_sfs.create ~feature_count:2 ~splitter:(group_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  expect_validation
    (Dummy_sfs.fit valid ~metadata
       ~rng:(Rng.create (Seed.of_int 0))
       ~feature_schema:schema ~x ~y:None ());
  let classification_splitter =
    K_fold.create ~folds:2 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  expect_validation
    (Binary_sfs.create ~feature_count:1 ~splitter:classification_splitter
       ~scorer:(Binary_classification_scorer.neg_log_loss ())
       (Ridge_classifier.create () |> get))

let test_score_weight_routing () =
  let x =
    Matrix.of_arrays
      [|
        [| 0.0; 4.0 |];
        [| 10.0; 4.0 |];
        [| 0.0; 4.0 |];
        [| 10.0; 4.0 |];
        [| 0.0; 4.0 |];
        [| 10.0; 4.0 |];
      |]
    |> get_data
  in
  let schema = named_schema [| "f0"; "f1" |] in
  let y = regression (Array.make 6 0.0) in
  let splitter =
    K_fold.create ~folds:2 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let specification =
    Weighted_sfs.create ~feature_count:1 ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error ()
    |> get
  in
  let fit metadata =
    Weighted_sfs.fit specification ~metadata
      ~rng:(Rng.create (Seed.of_int 9))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get |> Weighted_sfs.selected_indices
  in
  Alcotest.(check (array int))
    "unweighted scorer prefers constant feature" [| 1 |] (fit Metadata.empty);
  let weights =
    Sample_weight.of_array ~expected_length:6
      [| 100.0; 1.0; 100.0; 1.0; 100.0; 1.0 |]
    |> get_data
  in
  Alcotest.(check (array int))
    "fold-local scorer weights change selection" [| 0 |]
    (fit (Metadata.create ~sample_weight:weights ()))

let regression_pipeline () =
  let splitter =
    K_fold.create ~folds:2 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let selector =
    Linear_sfs.create ~feature_count:2 ~max_fits:10 ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> Pipeline.Supervised.metadata_transformer ~name:"sequential_select"
         (module Linear_sfs)
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
      ~seed:(Seed.of_int 43) (regression_pipeline ()) dataset
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
    "outer held-out rows cannot alter inner selection"
    (feature_names (output_schema first))
    (feature_names (output_schema changed_first))

let () =
  Alcotest.run "Sequential feature selection"
    [
      ( "selection",
        [
          Alcotest.test_case "forward and backward ties" `Quick
            test_forward_and_backward_ties;
          Alcotest.test_case "preflight and validation" `Quick
            test_bounds_and_validation;
          Alcotest.test_case "score weight routing" `Quick
            test_score_weight_routing;
        ] );
      ( "integration",
        [
          Alcotest.test_case "nested fold leakage" `Quick
            test_nested_fold_leakage;
        ] );
    ]
