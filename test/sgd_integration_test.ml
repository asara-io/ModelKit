open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail (Error.to_string error)
  | Ok _ -> Alcotest.fail "expected a typed error"

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_data_length = function
  | Error.Data (Data_error.Length_mismatch _) -> true
  | _ -> false

let[@warning "-4"] is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | _ -> false

let rng seed = Rng.create (Seed.of_int seed)
let rows = 24

let features =
  Matrix.init ~rows ~columns:2 (fun row column ->
      let base = Float.of_int (row - 12) /. 4.0 in
      if column = 0 then base else Float.of_int (row * 7 mod 11) -. 5.0)
  |> get_data

let feature_schema = Feature_schema.of_matrix features |> get_data

let regression_target =
  Vector.init ~length:rows (fun row ->
      1.5
      +. (2.0 *. Matrix.get features row 0)
      -. (0.5 *. Matrix.get features row 1)
      +. (Float.of_int (row * 5 mod 3) *. 0.05))
  |> get_data |> Target.regression |> get_data

let classification_target =
  Target.classification
    (Array.init rows (fun row ->
         if
           Matrix.get features row 0 +. (0.2 *. Matrix.get features row 1) > 0.0
         then 1
         else 0))

let regression_dataset =
  Dataset.create ~finiteness:Dataset.Require_finite ~x:features
    ~y:regression_target ()
  |> get_data

let classification_dataset =
  Dataset.create ~finiteness:Dataset.Require_finite ~x:features
    ~y:classification_target ()
  |> get_data

let scaled_pipeline estimator =
  let scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty scaler |> get in
  Pipeline.set_estimator builder estimator |> get

let regressor ?(eta0 = 0.05) () =
  Sgd_regressor.create ~penalty:Sgd_regressor.L2 ~alpha:0.0001
    ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.25 })
    ~eta0 ~max_epochs:30 ~shuffle:true ()
  |> get

let classifier ?(loss = Sgd_classifier.Log_loss) ?(eta0 = 0.05) () =
  Sgd_classifier.create ~loss ~penalty:Sgd_classifier.L2 ~alpha:0.0001
    ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = 0.25 })
    ~eta0 ~max_epochs:30 ~shuffle:true ()
  |> get

let regression_pipeline specification =
  scaled_pipeline
    (Pipeline.estimator ~name:"sgd regressor"
       (module Sgd_regressor)
       specification
    |> get)

let classification_pipeline specification =
  scaled_pipeline
    (Pipeline.estimator ~name:"sgd classifier"
       ~decision_function:Sgd_classifier.binary_decision_function
       ~predict_proba:Sgd_classifier.predict_proba
       ~classes:Sgd_classifier.classes
       (module Sgd_classifier)
       specification
    |> get)

let k_fold =
  K_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let stratified =
  Stratified_k_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let test_scores report =
  Cross_validation.folds report
  |> Array.map (fun (fold : _ Cross_validation.fold) ->
      Array.map
        (fun (score : Cross_validation.score) ->
          match score.Cross_validation.test_score with
          | Some (Ok value) -> value
          | Some (Error error) -> Alcotest.fail (Error.to_string error)
          | None -> Alcotest.fail "missing test score")
        fold.Cross_validation.scores)

let check_finite label scores =
  Alcotest.(check bool)
    label true
    (Array.for_all (Array.for_all Float.is_finite) scores)

let test_regression_cross_validation () =
  let pipeline = regression_pipeline (regressor ()) in
  let run () =
    Cross_validation.Regression.cross_validate ~return_train_score:true
      ~return_models:true ~splitter:k_fold
      ~scorers:
        [| Regression_scorer.neg_mean_squared_error; Regression_scorer.r2 () |]
      ~seed:(Seed.of_int 42) pipeline regression_dataset
    |> get
  in
  let report = run () in
  Alcotest.(check int)
    "regression folds succeed" 4
    (Cross_validation.successful_fold_count report);
  let scores = test_scores report in
  check_finite "regression scores are finite" scores;
  Alcotest.(check bool)
    "scaled SGD explains the linear target" true
    (Array.for_all (fun fold -> fold.(1) > 0.9) scores);
  Alcotest.(check bool)
    "cross-validation is deterministic" true
    (test_scores (run ()) = scores);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let model = Option.get fold.Cross_validation.model in
      Alcotest.(check int)
        "fold model predicts" rows
        (Pipeline.predict model ~feature_schema ~x:features
        |> get |> Target.length))
    (Cross_validation.folds report)

let test_classification_cross_validation () =
  let report =
    Cross_validation.Binary_classification.cross_validate ~splitter:stratified
      ~scorers:
        [|
          Binary_classification_scorer.accuracy;
          Binary_classification_scorer.roc_auc ();
          Binary_classification_scorer.average_precision ();
          Binary_classification_scorer.neg_log_loss ();
        |]
      ~seed:(Seed.of_int 42)
      (classification_pipeline (classifier ()))
      classification_dataset
    |> get
  in
  Alcotest.(check int)
    "log-loss folds succeed" 4
    (Cross_validation.successful_fold_count report);
  let scores = test_scores report in
  check_finite "log-loss scores are finite" scores;
  Alcotest.(check bool)
    "probabilities rank the separable classes" true
    (Array.for_all (fun fold -> fold.(1) > 0.9) scores);
  let hinge =
    Cross_validation.Binary_classification.cross_validate ~splitter:stratified
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 42)
      (classification_pipeline (classifier ~loss:Sgd_classifier.Hinge ()))
      classification_dataset
    |> get
  in
  Alcotest.(check int)
    "hinge folds succeed with label scorers" 4
    (Cross_validation.successful_fold_count hinge);
  let hinge_probability =
    Cross_validation.Binary_classification.cross_validate ~splitter:stratified
      ~failure_policy:Cross_validation.Record
      ~scorers:[| Binary_classification_scorer.roc_auc () |]
      ~seed:(Seed.of_int 42)
      (classification_pipeline (classifier ~loss:Sgd_classifier.Hinge ()))
      classification_dataset
    |> get
  in
  Alcotest.(check int)
    "hinge cannot serve probability scorers" 0
    (Cross_validation.successful_fold_count hinge_probability);
  Alcotest.(check bool)
    "hinge probability failures are recorded per fold" true
    (Array.for_all
       (fun (fold : _ Cross_validation.fold) ->
         Array.length fold.Cross_validation.failures > 0)
       (Cross_validation.folds hinge_probability))

type regressor_configuration = { eta0 : float }

let test_grid_search () =
  let axis =
    Grid_search.axis ~name:"eta0" ~values:[| 0.5; 0.05; 0.005 |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ eta0 -> Ok { eta0 })
    |> get
  in
  let grid =
    Grid_search.create ~base:{ eta0 = 0.05 }
      ~build:(fun configuration ->
        Result.map regression_pipeline
          (Sgd_regressor.create ~penalty:Sgd_regressor.L2 ~alpha:0.0001
             ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.25 })
             ~eta0:configuration.eta0 ~max_epochs:30 ~shuffle:true ()))
      [| axis |]
    |> get
  in
  let report =
    Grid_search.Regression.search ~grid ~splitter:k_fold
      ~scorers:[| Regression_scorer.r2 () |]
      ~refit:(Regression_scorer.name (Regression_scorer.r2 ()))
      ~seed:(Seed.of_int 42) regression_dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check int) "three candidates" 3 (Array.length candidates);
  Alcotest.(check bool)
    "every candidate evaluated" true
    (Array.for_all
       (fun (candidate : _ Grid_search.candidate) ->
         Option.is_some candidate.Grid_search.rank
         && Option.is_none candidate.Grid_search.build_error)
       candidates);
  let selected = Grid_search.selection report |> get in
  Alcotest.(check int)
    "refit model predicts" rows
    (Pipeline.predict selected.Grid_search.selected_model ~feature_schema
       ~x:features
    |> get |> Target.length);
  let binary_axis =
    Grid_search.axis ~name:"eta0" ~values:[| 0.5; 0.05 |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ eta0 -> Ok { eta0 })
    |> get
  in
  let binary_grid =
    Grid_search.create ~base:{ eta0 = 0.05 }
      ~build:(fun configuration ->
        Result.map classification_pipeline
          (Sgd_classifier.create ~loss:Sgd_classifier.Hinge
             ~penalty:Sgd_classifier.L2 ~alpha:0.0001
             ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = 0.25 })
             ~eta0:configuration.eta0 ~max_epochs:30 ~shuffle:true ()))
      [| binary_axis |]
    |> get
  in
  let binary_report =
    Grid_search.Binary_classification.search ~grid:binary_grid
      ~splitter:stratified
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~refit:
        (Binary_classification_scorer.name Binary_classification_scorer.accuracy)
      ~seed:(Seed.of_int 42) classification_dataset
    |> get
  in
  let selected = Grid_search.selection binary_report |> get in
  Alcotest.(check (array int))
    "refit classifier keeps class metadata" [| 0; 1 |]
    (Pipeline.classes selected.Grid_search.selected_model |> get)

let halves () =
  let split from count =
    Matrix.init ~rows:count ~columns:2 (fun row column ->
        Matrix.get features (from + row) column)
    |> get_data
  in
  let first = split 0 12 in
  let second = split 12 12 in
  let regression from count =
    Vector.init ~length:count (fun row ->
        Vector.get (Target.regression_values regression_target) (from + row))
    |> get_data |> Target.regression |> get_data
  in
  let classification from count =
    Target.classification
      (Array.sub
         (Target.classification_values classification_target)
         from count)
  in
  ( (first, regression 0 12, classification 0 12),
    (second, regression 12 12, classification 12 12) )

let regressor_state fitted =
  ( Vector.to_array (Sgd_regressor.coefficients fitted),
    Sgd_regressor.intercept fitted,
    (Sgd_regressor.report fitted).Sgd_regressor.updates )

let classifier_state fitted =
  ( Matrix.to_arrays (Sgd_classifier.coefficients fitted),
    Vector.to_array (Sgd_classifier.intercepts fitted),
    (Sgd_classifier.report fitted).Sgd_classifier.updates )

let ordered ?(eta0 = 0.05) ?(learning_rate = Sgd_regressor.Constant) () =
  Sgd_regressor.create ~penalty:Sgd_regressor.Elastic_net ~alpha:0.001
    ~l1_ratio:0.4 ~learning_rate ~eta0 ~max_epochs:1 ~shuffle:false ()
  |> get

let ordered_classifier ?(eta0 = 0.05) ?(learning_rate = Sgd_classifier.Constant)
    () =
  Sgd_classifier.create ~loss:Sgd_classifier.Log_loss
    ~penalty:Sgd_classifier.Elastic_net ~alpha:0.001 ~l1_ratio:0.4
    ~learning_rate ~eta0 ~max_epochs:1 ~shuffle:false ()
  |> get

let test_regressor_continuation () =
  let (first_x, first_y, _), (second_x, second_y, _) = halves () in
  List.iter
    (fun learning_rate ->
      let specification = ordered ~learning_rate () in
      let whole =
        Sgd_regressor.fit specification ~rng:(rng 5) ~feature_schema ~x:features
          ~y:regression_target ()
        |> get |> regressor_state
      in
      let parent =
        Sgd_regressor.start specification ~rng:(rng 5) ~feature_schema
      in
      let after_first =
        Sgd_regressor.partial_fit parent ~feature_schema ~x:first_x ~y:first_y
          ()
        |> get
      in
      let chained =
        Sgd_regressor.partial_fit after_first ~feature_schema ~x:second_x
          ~y:second_y ()
        |> get |> Sgd_regressor.to_fitted |> get |> regressor_state
      in
      Alcotest.(check bool)
        "ordered half batches equal one whole batch" true (whole = chained);
      let snapshot = Sgd_regressor.to_fitted after_first |> get in
      let resumed =
        Sgd_regressor.partial_fit
          (Sgd_regressor.checkpoint snapshot)
          ~feature_schema ~x:second_x ~y:second_y ()
        |> get |> Sgd_regressor.to_fitted |> get |> regressor_state
      in
      Alcotest.(check bool)
        "snapshot and resume equal the uninterrupted stream" true
        (resumed = whole);
      let sibling =
        Sgd_regressor.partial_fit after_first ~feature_schema ~x:second_x
          ~y:second_y ()
        |> get |> Sgd_regressor.to_fitted |> get |> regressor_state
      in
      Alcotest.(check bool)
        "branches from one parent are identical" true (sibling = chained);
      Alcotest.(check bool)
        "the parent snapshot is unchanged by its children" true
        (regressor_state (Sgd_regressor.to_fitted after_first |> get)
        = regressor_state snapshot))
    [ Sgd_regressor.Constant; Sgd_regressor.Inverse_scaling { power_t = 0.3 } ];
  let specification = ordered () in
  let parent = Sgd_regressor.start specification ~rng:(rng 5) ~feature_schema in
  let wrong_schema = Feature_schema.anonymous ~feature_count:3 |> get_data in
  Sgd_regressor.partial_fit parent ~feature_schema:wrong_schema ~x:first_x
    ~y:first_y ()
  |> expect_error is_schema_mismatch;
  Sgd_regressor.partial_fit parent ~feature_schema ~x:first_x ~y:second_y
    ~sample_weight:
      (Sample_weight.of_array ~expected_length:3 [| 1.0; 1.0; 1.0 |] |> get_data)
    ()
  |> expect_error is_data_length;
  let empty_x = Matrix.create ~rows:0 ~columns:2 0.0 |> get_data in
  let empty_y = Target.regression (Vector.of_array [||]) |> get_data in
  Sgd_regressor.partial_fit parent ~feature_schema ~x:empty_x ~y:empty_y ()
  |> expect_error is_validation;
  let clean =
    Sgd_regressor.partial_fit parent ~feature_schema ~x:first_x ~y:first_y ()
    |> get |> Sgd_regressor.to_fitted |> get |> regressor_state
  in
  let fresh =
    Sgd_regressor.partial_fit
      (Sgd_regressor.start specification ~rng:(rng 5) ~feature_schema)
      ~feature_schema ~x:first_x ~y:first_y ()
    |> get |> Sgd_regressor.to_fitted |> get |> regressor_state
  in
  Alcotest.(check bool)
    "rejected batches leave the checkpoint reusable" true (clean = fresh);
  let weights =
    Sample_weight.of_array ~expected_length:rows
      (Array.init rows (fun row -> if row < 12 then 1.0 else 0.0))
    |> get_data
  in
  let unpenalized =
    Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
      ~learning_rate:Sgd_regressor.Constant ~eta0:0.05 ~max_epochs:1
      ~shuffle:false ()
    |> get
  in
  let kept =
    Sgd_regressor.fit unpenalized ~sample_weight:weights ~rng:(rng 5)
      ~feature_schema ~x:features ~y:regression_target ()
    |> get |> regressor_state
  in
  let dropped =
    Sgd_regressor.fit unpenalized ~rng:(rng 5) ~feature_schema ~x:first_x
      ~y:first_y ()
    |> get |> regressor_state
  in
  let parameters (coefficients, intercept, _) = (coefficients, intercept) in
  Alcotest.(check bool)
    "zero-weight rows do not change unpenalized constant-rate parameters" true
    (parameters kept = parameters dropped);
  let penalized_kept =
    Sgd_regressor.fit specification ~sample_weight:weights ~rng:(rng 5)
      ~feature_schema ~x:features ~y:regression_target ()
    |> get |> regressor_state
  in
  let penalized_dropped =
    Sgd_regressor.fit specification ~rng:(rng 5) ~feature_schema ~x:first_x
      ~y:first_y ()
    |> get |> regressor_state
  in
  Alcotest.(check bool)
    "zero-weight rows still apply penalty steps" true
    (parameters penalized_kept <> parameters penalized_dropped);
  let updates (_, _, updates) = updates in
  Alcotest.(check (pair int int))
    "zero-weight rows still count as updates" (24, 12)
    (updates kept, updates dropped);
  let shuffled seed =
    Sgd_regressor.fit
      (Sgd_regressor.create ~penalty:Sgd_regressor.L2 ~eta0:0.05
         ~learning_rate:Sgd_regressor.Constant ~max_epochs:2 ~shuffle:true ()
      |> get)
      ~rng:(rng seed) ~feature_schema ~x:features ~y:regression_target ()
    |> get |> regressor_state
  in
  Alcotest.(check bool)
    "shuffled streams depend only on the seed" true
    (shuffled 9 = shuffled 9);
  Alcotest.(check bool)
    "different seeds visit rows in different orders" true
    (shuffled 9 <> shuffled 10)

let test_classifier_continuation () =
  let (first_x, _, first_y), (second_x, _, second_y) = halves () in
  List.iter
    (fun learning_rate ->
      let specification = ordered_classifier ~learning_rate () in
      let whole =
        Sgd_classifier.fit specification ~rng:(rng 5) ~feature_schema
          ~x:features ~y:classification_target ()
        |> get |> classifier_state
      in
      let parent =
        Sgd_classifier.start specification ~rng:(rng 5) ~feature_schema
          ~classes:[| 1; 0 |]
        |> get
      in
      let after_first =
        Sgd_classifier.partial_fit parent ~feature_schema ~x:first_x ~y:first_y
          ()
        |> get
      in
      let chained =
        Sgd_classifier.partial_fit after_first ~feature_schema ~x:second_x
          ~y:second_y ()
        |> get |> Sgd_classifier.to_fitted |> get |> classifier_state
      in
      Alcotest.(check bool)
        "ordered half batches equal one whole batch" true (whole = chained);
      let snapshot = Sgd_classifier.to_fitted after_first |> get in
      let resumed =
        Sgd_classifier.partial_fit
          (Sgd_classifier.checkpoint snapshot)
          ~feature_schema ~x:second_x ~y:second_y ()
        |> get |> Sgd_classifier.to_fitted |> get |> classifier_state
      in
      Alcotest.(check bool)
        "snapshot and resume equal the uninterrupted stream" true
        (resumed = whole);
      Alcotest.(check bool)
        "the parent snapshot is unchanged by its children" true
        (classifier_state (Sgd_classifier.to_fitted after_first |> get)
        = classifier_state snapshot))
    [
      Sgd_classifier.Constant; Sgd_classifier.Inverse_scaling { power_t = 0.3 };
    ];
  let specification = ordered_classifier () in
  let parent =
    Sgd_classifier.start specification ~rng:(rng 5) ~feature_schema
      ~classes:[| 0; 1 |]
    |> get
  in
  let one_class_y = Target.classification (Array.make 12 1) in
  let partial =
    Sgd_classifier.partial_fit parent ~feature_schema ~x:first_x ~y:one_class_y
      ()
    |> get
  in
  Alcotest.(check (array int))
    "a batch may omit registered classes" [| 0; 1 |]
    (Sgd_classifier.checkpoint_classes partial);
  let foreign_y =
    Target.classification (Array.init 12 (fun row -> if row = 7 then 3 else 0))
  in
  Sgd_classifier.partial_fit parent ~feature_schema ~x:first_x ~y:foreign_y ()
  |> expect_error is_validation;
  let wrong_schema = Feature_schema.anonymous ~feature_count:3 |> get_data in
  Sgd_classifier.partial_fit parent ~feature_schema:wrong_schema ~x:first_x
    ~y:first_y ()
  |> expect_error is_schema_mismatch;
  let clean =
    Sgd_classifier.partial_fit parent ~feature_schema ~x:first_x ~y:first_y ()
    |> get |> Sgd_classifier.to_fitted |> get |> classifier_state
  in
  let reordered =
    Sgd_classifier.partial_fit
      (Sgd_classifier.start specification ~rng:(rng 5) ~feature_schema
         ~classes:[| 1; 0 |]
      |> get)
      ~feature_schema ~x:first_x ~y:first_y ()
    |> get |> Sgd_classifier.to_fitted |> get |> classifier_state
  in
  Alcotest.(check bool)
    "rejected batches leave the checkpoint reusable and registration order is \
     irrelevant"
    true (clean = reordered);
  let multiclass_y =
    Target.classification (Array.init rows (fun row -> row mod 3))
  in
  let joint =
    Sgd_classifier.fit specification ~rng:(rng 5) ~feature_schema ~x:features
      ~y:multiclass_y ()
    |> get
  in
  let scores =
    Sgd_classifier.decision_function joint ~feature_schema ~x:features |> get
  in
  Alcotest.(check (pair int int))
    "multiclass scores have one column per class" (rows, 3)
    (Matrix.shape scores);
  let per_class_updates =
    (Sgd_classifier.report joint).Sgd_classifier.updates
  in
  Alcotest.(check int)
    "one shared update counter across one-versus-rest models" rows
    per_class_updates

let () =
  Alcotest.run "SGD integration"
    [
      ( "supported workflows",
        [
          Alcotest.test_case "regression pipeline and cross-validation" `Quick
            test_regression_cross_validation;
          Alcotest.test_case "classification pipeline and cross-validation"
            `Quick test_classification_cross_validation;
          Alcotest.test_case "grid search over immutable SGD configurations"
            `Quick test_grid_search;
        ] );
      ( "continuation equivalence",
        [
          Alcotest.test_case "regressor adversaries" `Quick
            test_regressor_continuation;
          Alcotest.test_case "classifier adversaries" `Quick
            test_classifier_continuation;
        ] );
    ]
