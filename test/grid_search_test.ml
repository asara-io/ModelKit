open Modelkit
open Grid_search

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let classification values = Target.classification values

let regression_dataset () =
  let x = Array.init 12 (fun row -> [| Float.of_int (row - 3) |]) in
  let y = Array.init 12 (fun row -> (2.0 *. Float.of_int (row - 3)) +. 1.0) in
  Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix x)
    ~y:(regression y) ()
  |> get_data

let classification_dataset () =
  Dataset.create ~finiteness:Dataset.Require_finite
    ~x:
      (matrix
         [|
           [| -6.0 |];
           [| -5.0 |];
           [| -4.0 |];
           [| -3.0 |];
           [| -2.0 |];
           [| -1.0 |];
           [| 1.0 |];
           [| 2.0 |];
           [| 3.0 |];
           [| 4.0 |];
           [| 5.0 |];
           [| 6.0 |];
         |])
    ~y:(classification [| 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1 |])
    ()
  |> get_data

let k_fold ?(shuffle = false) folds =
  K_fold.create ~folds ~shuffle ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let stratified_k_fold folds =
  Stratified_k_fold.create ~folds ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

type ridge_configuration = { alpha : float; fit_intercept : bool }

let ridge_pipeline configuration =
  let ( let* ) = Result.bind in
  let* specification =
    Ridge_regression.create ~alpha:configuration.alpha
      ~fit_intercept:configuration.fit_intercept ()
  in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator Pipeline.empty estimator

let ridge_grid alphas intercepts =
  let alpha =
    Grid_search.axis ~name:"alpha" ~values:alphas
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun configuration alpha -> Ok { configuration with alpha })
    |> get
  in
  let fit_intercept =
    Grid_search.axis ~name:"fit_intercept" ~values:intercepts
      ~encode:(fun value -> Grid_search.Bool value)
      ~set:(fun configuration fit_intercept ->
        Ok { configuration with fit_intercept })
    |> get
  in
  Grid_search.create
    ~base:{ alpha = 0.0; fit_intercept = true }
    ~build:ridge_pipeline [| alpha; fit_intercept |]
  |> get

let score_summary candidate index = candidate.scores.(index).test |> get

let test_rank_and_refit () =
  let dataset = regression_dataset () in
  let grid = ridge_grid [| 100.0; 0.0 |] [| false; true |] in
  Alcotest.(check int)
    "Cartesian candidate count" 4
    (Grid_search.candidate_count grid);
  let report =
    Grid_search.Regression.search ~return_train_score:true ~grid
      ~splitter:(k_fold 3)
      ~scorers:
        [| Regression_scorer.neg_mean_squared_error; Regression_scorer.r2 () |]
      ~refit:"r2" ~seed:(Seed.of_int 42) dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check int) "reported candidate count" 4 (Array.length candidates);
  Alcotest.(check bool)
    "first candidate parameters" true
    (candidates.(0).parameters
    = [|
        { parameter_name = "alpha"; parameter_value = Float 100.0 };
        { parameter_name = "fit_intercept"; parameter_value = Bool false };
      |]);
  Alcotest.(check bool)
    "last axis varies fastest" true
    (candidates.(1).parameters.(1).parameter_value = Bool true);
  Alcotest.(check bool)
    "first axis advances after last axis" true
    (candidates.(2).parameters.(0).parameter_value = Float 0.0);
  Array.iteri
    (fun index (candidate : _ Grid_search.candidate) ->
      Alcotest.(check int)
        "stable candidate index" index candidate.candidate_index;
      Alcotest.(check int)
        "parameter count" 2
        (Array.length candidate.parameters);
      Alcotest.(check string)
        "first axis order" "alpha" candidate.parameters.(0).parameter_name;
      Alcotest.(check string)
        "second axis order" "fit_intercept"
        candidate.parameters.(1).parameter_name;
      Alcotest.(check bool)
        "candidate ranked" true
        (Option.is_some candidate.rank);
      Alcotest.(check bool)
        "evaluation retained" true
        (Option.is_some candidate.evaluation);
      Alcotest.(check bool)
        "train summary retained" true
        (Option.is_some candidate.scores.(0).train);
      Alcotest.(check bool)
        "nonnegative fit time" true
        (candidate.mean_fit_time >= 0.0);
      Alcotest.(check bool)
        "nonnegative score time" true
        (candidate.mean_score_time >= 0.0))
    candidates;
  Alcotest.(check int) "best candidate rank" 1 (Option.get candidates.(3).rank);
  Alcotest.check (Alcotest.float 1e-10) "best mean R-squared" 1.0
    (score_summary candidates.(3) 1).Score_aggregation.mean;
  let selected = Grid_search.selection report |> get in
  Alcotest.(check int)
    "best candidate refitted" 3 selected.selected_candidate_index;
  let predictions =
    Pipeline.predict selected.selected_model
      ~feature_schema:(Dataset.feature_schema dataset)
      ~x:(Dataset.features dataset)
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-10))
    "refitted predictions"
    (Array.init 12 (fun row -> (2.0 *. Float.of_int (row - 3)) +. 1.0))
    predictions

let linear_pipeline _configuration =
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Ok (Pipeline.set_estimator Pipeline.empty estimator |> get)

let test_tie_ranking () =
  let token =
    Grid_search.axis ~name:"token" ~values:[| 10; 20 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ token -> Ok token)
    |> get
  in
  let grid =
    Grid_search.create ~base:0 ~build:linear_pipeline [| token |] |> get
  in
  let report =
    Grid_search.Regression.search ~grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.r2 () |]
      ~refit:"r2" ~seed:(Seed.of_int 9) (regression_dataset ())
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check int) "first tie rank" 1 (Option.get candidates.(0).rank);
  Alcotest.(check int) "second tie rank" 1 (Option.get candidates.(1).rank);
  Alcotest.(check int)
    "stable tie winner" 0
    (Grid_search.selection report |> get).selected_candidate_index

let test_failure_aware_selection () =
  let dataset = regression_dataset () in
  let grid = ridge_grid [| -1.0; 0.0 |] [| true |] in
  let report =
    Grid_search.Regression.search ~grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.r2 () |]
      ~refit:"r2" ~seed:(Seed.of_int 7) dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check bool)
    "failed candidate retained" true
    (Option.is_some candidates.(0).build_error);
  Alcotest.(check bool)
    "failed candidate unranked" true
    (Option.is_none candidates.(0).rank);
  Alcotest.(check int) "valid candidate rank" 1 (Option.get candidates.(1).rank);
  Alcotest.(check int)
    "valid candidate selected" 1
    (Grid_search.selection report |> get).selected_candidate_index;
  (match
     Grid_search.Regression.search ~failure_policy:Cross_validation.Abort ~grid
       ~splitter:(k_fold 3)
       ~scorers:[| Regression_scorer.r2 () |]
       ~refit:"r2" ~seed:(Seed.of_int 7) dataset
   with
  | Ok _ -> Alcotest.fail "abort policy accepted a failed candidate"
  | Error error ->
      Alcotest.(check bool)
        "candidate context" true
        (Error.context error = [ Error.Candidate 0 ]));
  let failed_grid = ridge_grid [| -2.0; -1.0 |] [| true |] in
  let failed_report =
    Grid_search.Regression.search ~grid:failed_grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.r2 () |]
      ~refit:"r2" ~seed:(Seed.of_int 7) dataset
    |> get
  in
  match Grid_search.selection failed_report with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "selection succeeded without a valid candidate"

type refit_failure_fitted = { failure_size : int; schema : Feature_schema.t }

module Refit_failure_regressor :
  REGRESSOR
    with type t = int
     and type params = int
     and type fitted = refit_failure_fitted
     and type rng = Rng.t = struct
  type t = int
  type params = int
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = refit_failure_fitted
  type rng = Rng.t

  let clone specification = specification
  let params specification = specification

  let fit failure_size ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:_ () =
    if Matrix.rows x = failure_size then
      Error
        (Error.make ~remediation:"use a refit-capable estimator"
           (Error.Validation
              {
                name = "test refit estimator";
                reason = "full-dataset fitting is disabled";
              }))
    else Ok { failure_size; schema = feature_schema }

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make ~remediation:"provide the fitted feature schema"
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      Target.regression (Vector.of_array (Array.make (Matrix.rows x) 0.0))
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"produce finite predictions" error)

  let fitted_params fitted = fitted.failure_size
  let feature_schema fitted = fitted.schema
end

let refit_failure_pipeline failure_size =
  let ( let* ) = Result.bind in
  let* estimator =
    Pipeline.estimator ~name:"refit_failure"
      (module Refit_failure_regressor)
      failure_size
  in
  Pipeline.set_estimator Pipeline.empty estimator

let test_fold_failure_selection () =
  let dataset = regression_dataset () in
  let failure_size =
    Grid_search.axis ~name:"failure_size" ~values:[| 8; -1 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ failure_size -> Ok failure_size)
    |> get
  in
  let grid =
    Grid_search.create ~base:(-1) ~build:refit_failure_pipeline
      [| failure_size |]
    |> get
  in
  let report =
    Grid_search.Regression.search ~grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19) dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check bool)
    "fold failure was not a build failure" true
    (Option.is_none candidates.(0).build_error);
  Alcotest.(check bool)
    "fold failure retained its evaluation" true
    (Option.is_some candidates.(0).evaluation);
  Alcotest.(check bool)
    "fold failure candidate unranked" true
    (Option.is_none candidates.(0).rank);
  Alcotest.(check int)
    "successful candidate selected" 1
    (Grid_search.selection report |> get).selected_candidate_index;
  match
    Grid_search.Regression.search ~failure_policy:Cross_validation.Abort ~grid
      ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19) dataset
  with
  | Ok _ -> Alcotest.fail "abort policy accepted a fold fit failure"
  | Error error ->
      Alcotest.(check bool)
        "fold fit context" true
        (Error.context error
        = [ Error.Candidate 0; Error.Fold 0; Error.Stage "refit_failure" ])

let test_refit_failure () =
  let dataset = regression_dataset () in
  let grid =
    Grid_search.create
      ~base:(Dataset.sample_count dataset)
      ~build:refit_failure_pipeline [||]
    |> get
  in
  let report =
    Grid_search.Regression.search ~grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 17) dataset
    |> get
  in
  Alcotest.(check int)
    "base-only candidate count" 1
    (Grid_search.candidate_count grid);
  (match Grid_search.selection report with
  | Ok _ -> Alcotest.fail "record policy hid a refit failure"
  | Error error ->
      Alcotest.(check bool)
        "refit candidate context" true
        (Error.context error
        = [ Error.Candidate 0; Error.Stage "refit_failure" ]));
  match
    Grid_search.Regression.search ~failure_policy:Cross_validation.Abort ~grid
      ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 17) dataset
  with
  | Ok _ -> Alcotest.fail "abort policy hid a refit failure"
  | Error error ->
      Alcotest.(check bool)
        "aborted refit candidate context" true
        (Error.context error
        = [ Error.Candidate 0; Error.Stage "refit_failure" ])

type logistic_configuration = { c : float }

let logistic_pipeline configuration =
  let specification = Logistic_regression.create ~c:configuration.c () |> get in
  let estimator =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~predict_proba:Logistic_regression.predict_proba
      ~classes:Logistic_regression.classes specification
    |> get
  in
  Ok (Pipeline.set_estimator Pipeline.empty estimator |> get)

let test_binary_search () =
  let c =
    Grid_search.axis ~name:"c" ~values:[| 0.1; 10.0 |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ c -> Ok { c })
    |> get
  in
  let grid =
    Grid_search.create ~base:{ c = 1.0 } ~build:logistic_pipeline [| c |] |> get
  in
  let report =
    Grid_search.Binary_classification.search ~grid
      ~splitter:(stratified_k_fold 3)
      ~scorers:
        [|
          Binary_classification_scorer.accuracy;
          Binary_classification_scorer.roc_auc ();
        |]
      ~refit:"roc_auc" ~seed:(Seed.of_int 11)
      (classification_dataset ())
    |> get
  in
  Alcotest.(check int)
    "binary candidates" 2
    (Array.length (Grid_search.candidates report));
  ignore (Grid_search.selection report |> get)

let test_grid_validation () =
  let expect_error = function
    | Error _ -> ()
    | Ok _ -> Alcotest.fail "invalid grid was accepted"
  in
  Grid_search.axis ~name:"empty" ~values:[||]
    ~encode:(fun value -> Grid_search.Int value)
    ~set:(fun _ token -> Ok token)
  |> expect_error;
  let token =
    Grid_search.axis ~name:"token" ~values:[| 1 |]
      ~encode:(fun value -> Grid_search.Int value)
      ~set:(fun _ token -> Ok token)
    |> get
  in
  Grid_search.create ~base:0 ~build:linear_pipeline [| token; token |]
  |> expect_error;
  let grid =
    Grid_search.create ~base:0 ~build:linear_pipeline [| token |] |> get
  in
  match
    Grid_search.Regression.search ~grid ~splitter:(k_fold 3)
      ~scorers:[| Regression_scorer.r2 () |]
      ~refit:"missing" ~seed:(Seed.of_int 1) (regression_dataset ())
  with
  | Error _ -> ()
  | Ok _ -> Alcotest.fail "unknown refit scorer was accepted"

let test_fit_seed_does_not_change_splits () =
  let dataset = regression_dataset () in
  let run fit_seed =
    Cross_validation.Regression.cross_validate ~return_indices:true ~fit_seed
      ~splitter:(k_fold ~shuffle:true 3)
      ~scorers:[| Regression_scorer.r2 () |]
      ~seed:(Seed.of_int 123)
      (linear_pipeline 0 |> get)
      dataset
    |> get |> Cross_validation.folds
  in
  let first = run (Seed.of_int 1) in
  let second = run (Seed.of_int 2) in
  Array.iteri
    (fun index (fold : _ Cross_validation.fold) ->
      Alcotest.(check (array int))
        "fit seed preserves training split"
        (Option.get fold.Cross_validation.train_indices)
        (Option.get second.(index).Cross_validation.train_indices);
      Alcotest.(check (array int))
        "fit seed preserves test split"
        (Option.get fold.Cross_validation.test_indices)
        (Option.get second.(index).Cross_validation.test_indices))
    first

let () =
  Alcotest.run "grid search"
    [
      ( "selection",
        [
          Alcotest.test_case "rank and refit" `Quick test_rank_and_refit;
          Alcotest.test_case "tie ranking" `Quick test_tie_ranking;
          Alcotest.test_case "binary classification" `Quick test_binary_search;
        ] );
      ( "failures",
        [
          Alcotest.test_case "failure-aware selection" `Quick
            test_failure_aware_selection;
          Alcotest.test_case "fold failure selection" `Quick
            test_fold_failure_selection;
          Alcotest.test_case "refit failure" `Quick test_refit_failure;
          Alcotest.test_case "grid validation" `Quick test_grid_validation;
        ] );
      ( "determinism",
        [
          Alcotest.test_case "separate fit seed" `Quick
            test_fit_seed_does_not_change_splits;
        ] );
    ]
