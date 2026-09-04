open Modelkit
open Cross_validation

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let regression_pipeline name estimator specification =
  Pipeline.set_estimator Pipeline.empty
    (Pipeline.estimator ~name estimator specification |> get)
  |> get

let regression_fixture () =
  let x =
    Matrix.init ~rows:15 ~columns:1 (fun row _ -> Float.of_int (row - 5) /. 2.0)
    |> get_data
  in
  let y =
    Vector.init ~length:15 (fun row ->
        Float.exp (0.4 +. (0.15 *. Matrix.get x row 0)))
    |> get_data |> Target.regression |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  (x, y, dataset)

let k_fold =
  K_fold.create ~folds:3 () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let successful_score (fold : _ Cross_validation.fold) =
  match fold.scores.(0).test_score with
  | Some (Ok score) -> Float.is_finite score
  | None | Some (Error _) -> false

let test_regression_workflows () =
  let x, y, dataset = regression_fixture () in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let pipelines =
    [|
      ( "lasso",
        regression_pipeline "lasso"
          (module Lasso_regression)
          (Lasso_regression.create ~alpha:0.05 () |> get) );
      ( "elastic net",
        regression_pipeline "elastic net"
          (module Elastic_net_regression)
          (Elastic_net_regression.create ~alpha:0.05 ~l1_ratio:0.5 () |> get) );
      ( "Poisson",
        regression_pipeline "Poisson"
          (module Poisson_regression)
          (Poisson_regression.create ~alpha:0.05 () |> get) );
      ( "Tweedie",
        regression_pipeline "Tweedie"
          (module Tweedie_regression)
          (Tweedie_regression.create ~power:1.5 ~alpha:0.05 () |> get) );
    |]
  in
  Array.iter
    (fun (name, pipeline) ->
      let report =
        Cross_validation.Regression.cross_validate ~splitter:k_fold
          ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
          ~seed:(Seed.of_int 42) pipeline dataset
        |> get
      in
      Alcotest.(check int)
        (name ^ " successful folds")
        3
        (Cross_validation.successful_fold_count report);
      Alcotest.(check bool)
        (name ^ " finite scores") true
        (Array.for_all successful_score (Cross_validation.folds report));
      let fitted =
        Pipeline.fit pipeline
          ~rng:(Rng.create (Seed.of_int 42))
          ~feature_schema ~x ~y ()
        |> get
      in
      Alcotest.(check int)
        (name ^ " prediction length")
        15
        (Pipeline.predict fitted ~feature_schema ~x |> get |> Target.length))
    pipelines

let test_ridge_classifier_workflow () =
  let x =
    Matrix.init ~rows:12 ~columns:1 (fun row _ -> Float.of_int (row - 6))
    |> get_data
  in
  let y =
    Target.classification (Array.init 12 (fun row -> if row < 6 then -3 else 8))
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let terminal =
    Pipeline.estimator ~name:"ridge classifier"
      ~classes:Ridge_classifier.classes
      (module Ridge_classifier)
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let splitter =
    Stratified_k_fold.create ~folds:3 ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let report =
    Cross_validation.Binary_classification.cross_validate ~splitter
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 42) pipeline dataset
    |> get
  in
  Alcotest.(check int)
    "ridge successful folds" 3
    (Cross_validation.successful_fold_count report);
  Alcotest.(check bool)
    "ridge finite scores" true
    (Array.for_all successful_score (Cross_validation.folds report))

let test_multinomial_pipeline_capabilities () =
  let x =
    Matrix.of_arrays
      [|
        [| -3.0; 0.0 |];
        [| -2.0; 0.5 |];
        [| 0.0; 3.0 |];
        [| 0.5; 2.0 |];
        [| 3.0; -3.0 |];
        [| 2.0; -2.0 |];
      |]
    |> get_data
  in
  let y = Target.classification [| -4; -4; 2; 2; 9; 9 |] in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let terminal =
    Pipeline.estimator ~name:"multinomial logistic"
      ~predict_proba:Multinomial_logistic_regression.predict_proba
      ~classes:Multinomial_logistic_regression.classes
      (module Multinomial_logistic_regression)
      (Multinomial_logistic_regression.create ~c:10.0 () |> get)
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty terminal |> get in
  let fitted =
    Pipeline.fit pipeline
      ~rng:(Rng.create (Seed.of_int 42))
      ~feature_schema ~x ~y ()
    |> get
  in
  Alcotest.(check (pair int int))
    "probability shape" (6, 3)
    (Pipeline.predict_proba fitted ~feature_schema ~x |> get |> Matrix.shape);
  Alcotest.(check (array int))
    "class order" [| -4; 2; 9 |]
    (Pipeline.classes fitted |> get)

let () =
  Alcotest.run "linear workbench integration"
    [
      ( "supported workflows",
        [
          Alcotest.test_case "regression pipeline and cross-validation" `Quick
            test_regression_workflows;
          Alcotest.test_case "ridge pipeline and cross-validation" `Quick
            test_ridge_classifier_workflow;
          Alcotest.test_case "multinomial pipeline capabilities" `Quick
            test_multinomial_pipeline_capabilities;
        ] );
    ]
