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

type mode = Valid | Wrong_length | Non_finite | Wrong_schema
type tracking_params = { mode : mode }

type fit_observation = {
  names : string array;
  seed : Seed.t;
  weight_sum : float;
}

type tracking_fitted = {
  params : tracking_params;
  schema : Feature_schema.t;
  importances : Vector.t;
}

let observations = ref []

module Tracking_importance = struct
  type t = tracking_params
  type params = tracking_params
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = tracking_fitted
  type rng = Rng.t

  let clone specification = specification
  let params specification = specification

  let fit params ?sample_weight ~rng ~feature_schema ~x ~y:_ () =
    let names = feature_names feature_schema in
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
    observations :=
      { names; seed = Rng.to_seed rng; weight_sum } :: !observations;
    let importances =
      Array.mapi
        (fun column name ->
          match params.mode with
          | Non_finite when column = 0 -> Float.nan
          | Valid | Wrong_length | Wrong_schema | Non_finite ->
              let original =
                int_of_string (String.sub name 1 (String.length name - 1))
              in
              if original <= 2 then 1.0 else Float.of_int original)
        names
    in
    let importances =
      match params.mode with
      | Wrong_length -> Array.sub importances 0 (Matrix.columns x - 1)
      | Valid | Non_finite | Wrong_schema -> importances
    in
    let schema =
      match params.mode with
      | Wrong_schema ->
          Feature_schema.anonymous
            ~feature_count:(Feature_schema.feature_count feature_schema + 1)
          |> get_data
      | Valid | Wrong_length | Non_finite -> feature_schema
    in
    Ok { params; schema; importances = Vector.of_array importances }

  let predict _ ~feature_schema:_ ~x =
    Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide representable predictions"
          error)

  let fitted_params fitted = fitted.params
  let feature_schema fitted = fitted.schema
  let feature_importances fitted = Ok fitted.importances
end

module Tracking_rfe = Recursive_feature_elimination.Make (Tracking_importance)

module Linear_importance = struct
  include Linear_regression

  let feature_importances fitted =
    Feature_importance.absolute_coefficients (coefficients fitted)
end

module Linear_rfe = Recursive_feature_elimination.Make (Linear_importance)

let training_data () =
  let x =
    Matrix.init ~rows:6 ~columns:6 (fun row column ->
        Float.of_int ((row * 10) + column))
    |> get_data
  in
  let schema = named_schema [| "f0"; "f1"; "f2"; "f3"; "f4"; "f5" |] in
  let y = regression [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  (x, schema, y)

let test_rounds_ties_ranking_weights_and_seeds () =
  observations := [];
  let x, schema, y = training_data () in
  let weights =
    Sample_weight.of_array ~expected_length:6 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    |> get_data
  in
  let fitted =
    Tracking_rfe.create ~step:(Recursive_feature_elimination.Fraction 0.4)
      ~feature_count:2 { mode = Valid }
    |> get
    |> fun specification ->
    Tracking_rfe.fit specification ~sample_weight:weights
      ~rng:(Rng.create (Seed.of_int 19))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  Alcotest.(check (array int))
    "selected columns" [| 4; 5 |]
    (Tracking_rfe.selected_indices fitted);
  Alcotest.(check (array int))
    "elimination ranking" [| 3; 3; 2; 2; 1; 1 |]
    (Tracking_rfe.ranking fitted);
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "final importances" [| 4.0; 5.0 |]
    (Vector.to_array (Tracking_rfe.final_importances fitted));
  Alcotest.(check (array string))
    "output schema" [| "f4"; "f5" |]
    (feature_names (Tracking_rfe.output_schema fitted));
  Alcotest.(check (array string))
    "final estimator schema" [| "f4"; "f5" |]
    (feature_names
       (Tracking_importance.feature_schema
          (Tracking_rfe.fitted_estimator fitted)));
  let transformed =
    Tracking_rfe.transform fitted ~feature_schema:schema ~x |> get
  in
  Alcotest.(check (pair int int))
    "transformed shape" (6, 2) (Matrix.shape transformed);
  Alcotest.check (Alcotest.float 0.0) "first selected value" 4.0
    (Matrix.get transformed 0 0);
  let observed = Array.of_list (List.rev !observations) in
  Alcotest.(check int)
    "one fresh fit per round plus final fit" 3 (Array.length observed);
  Alcotest.(check (array (array string)))
    "round-local schemas"
    [|
      [| "f0"; "f1"; "f2"; "f3"; "f4"; "f5" |];
      [| "f2"; "f3"; "f4"; "f5" |];
      [| "f4"; "f5" |];
    |]
    (Array.map (fun observation -> observation.names) observed);
  Array.iteri
    (fun round observation ->
      Alcotest.check (Alcotest.float 0.0) "weights reach every fit" 21.0
        observation.weight_sum;
      Alcotest.(check bool)
        "logical round seed" true
        (Seed.equal observation.seed
           (Seed.derive (Seed.of_int 19)
              ~operation:"recursive-feature-elimination-round" ~index:round)))
    observed

let[@warning "-4"] test_validation_and_estimator_contract () =
  expect_validation (Tracking_rfe.create ~feature_count:0 { mode = Valid });
  expect_validation
    (Tracking_rfe.create ~step:(Recursive_feature_elimination.Count 0)
       ~feature_count:1 { mode = Valid });
  List.iter
    (fun fraction ->
      expect_validation
        (Tracking_rfe.create
           ~step:(Recursive_feature_elimination.Fraction fraction)
           ~feature_count:1 { mode = Valid }))
    [ 0.0; 1.0; Float.nan ];
  let x, schema, y = training_data () in
  let fit ?sample_weight ?(y = Some y) mode feature_count =
    Tracking_rfe.create ~feature_count { mode } |> get |> fun specification ->
    Tracking_rfe.fit specification ?sample_weight
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:schema ~x ~y ()
  in
  expect_validation (fit Valid 7);
  expect_validation (fit ~y:None Valid 2);
  let short_weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 1.0 |] |> get_data
  in
  (match fit ~sample_weight:short_weights Valid 2 with
  | Error error -> (
      match Error.kind error with
      | Error.Data (Data_error.Length_mismatch _) -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a sample-weight length error");
  (match fit Wrong_length 2 with
  | Error error -> (
      match Error.kind error with
      | Error.Shape_mismatch _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected an importance shape error");
  expect_validation (fit Non_finite 2);
  match fit Wrong_schema 2 with
  | Error error -> (
      match Error.kind error with
      | Error.Compatibility _ -> ()
      | _ -> Alcotest.fail ("unexpected error: " ^ Error.to_string error))
  | Ok _ -> Alcotest.fail "expected a fitted-schema error"

let regression_pipeline () =
  let selector =
    Linear_rfe.create ~feature_count:1 (Linear_regression.create ())
    |> get
    |> Pipeline.Supervised.transformer ~name:"recursive_select"
         (module Linear_rfe)
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

let test_fold_local_cross_validation_leakage () =
  let rows = 15 in
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
  let splitter =
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
      ~return_indices:true ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 29) (regression_pipeline ()) dataset
    |> get
  in
  let baseline = run original_x original_y in
  let first = (Cross_validation.folds baseline).(0) in
  let changed_y = Array.copy original_y in
  let changed_rows = Array.make rows false in
  Array.iter
    (fun row ->
      changed_rows.(row) <- true;
      changed_y.(row) <- 1e6 *. Float.of_int (row + 1))
    (Option.get first.Cross_validation.test_indices);
  let changed_x =
    Matrix.init ~rows ~columns:(Matrix.columns original_x) (fun row column ->
        if changed_rows.(row) then 1e5 *. Float.of_int ((row + 1) * (column + 1))
        else Matrix.get original_x row column)
    |> get_data
  in
  let changed_first = (Cross_validation.folds (run changed_x changed_y)).(0) in
  let output_schema fold =
    Pipeline.output_schema (Option.get fold.Cross_validation.model)
  in
  Alcotest.(check (array string))
    "held-out features and targets cannot alter fitted selection"
    (feature_names (output_schema first))
    (feature_names (output_schema changed_first))

let () =
  Alcotest.run "Recursive feature elimination"
    [
      ( "elimination",
        [
          Alcotest.test_case "rounds, ties, ranking, weights, and seeds" `Quick
            test_rounds_ties_ranking_weights_and_seeds;
          Alcotest.test_case "validation and estimator contract" `Quick
            test_validation_and_estimator_contract;
        ] );
      ( "integration",
        [
          Alcotest.test_case "fold-local cross-validation leakage" `Quick
            test_fold_local_cross_validation_leakage;
        ] );
    ]
