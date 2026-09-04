open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let rows = 30

let features =
  Matrix.init ~rows ~columns:2 (fun row column ->
      let angle = Float.of_int row *. 2.0 *. Float.pi /. Float.of_int rows in
      let radius = 3.0 +. (Float.of_int (row mod 4) *. 0.3) in
      if column = 0 then radius *. Float.cos angle
      else radius *. Float.sin angle)
  |> get_data

let feature_schema = Feature_schema.of_matrix features |> get_data
let labels = Array.init rows (fun row -> [| -4; 2; 9 |].(row * 3 / rows))
let target = Target.classification labels

let dataset =
  Dataset.create ~finiteness:Dataset.Require_finite ~x:features ~y:target ()
  |> get_data

let splitter =
  Stratified_k_fold.create ~folds:3 ~shuffle:true ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let label_scorers =
  Multiclass_classification_scorer.
    [|
      accuracy;
      balanced_accuracy ();
      f1 ~average:Multiclass_classification_metrics.Macro ();
      precision ~undefined:Undefined_metric_policy.Use_fallback
        ~average:Multiclass_classification_metrics.Weighted ();
    |]

let probability_scorers =
  Array.append label_scorers [| Multiclass_classification_scorer.neg_log_loss |]

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

let cross_validate ?(scorers = label_scorers) pipeline =
  Cross_validation.Multiclass_classification.cross_validate
    ~return_train_score:true ~splitter ~scorers ~seed:(Seed.of_int 42) pipeline
    dataset
  |> get

let check_report name report =
  Alcotest.(check int)
    (name ^ " folds succeed") 3
    (Cross_validation.successful_fold_count report);
  let scores = test_scores report in
  Alcotest.(check bool)
    (name ^ " scores are finite")
    true
    (Array.for_all (Array.for_all Float.is_finite) scores);
  Alcotest.(check bool)
    (name ^ " separates the classes")
    true
    (Array.for_all (fun fold -> fold.(0) > 0.8) scores)

let ridge_pipeline ?class_weight () =
  Pipeline.set_estimator Pipeline.empty
    (Pipeline.classifier ?class_weight ~name:"ridge"
       ~classes:Ridge_classifier.classes
       (module Ridge_classifier)
       (Ridge_classifier.create ~alpha:0.1 () |> get)
    |> get)
  |> get

let multinomial_pipeline () =
  Pipeline.set_estimator Pipeline.empty
    (Pipeline.classifier ~name:"multinomial"
       ~predict_proba:Multinomial_logistic_regression.predict_proba
       ~classes:Multinomial_logistic_regression.classes
       (module Multinomial_logistic_regression)
       (Multinomial_logistic_regression.create ~c:10.0 () |> get)
    |> get)
  |> get

let sgd_pipeline loss =
  Pipeline.set_estimator Pipeline.empty
    (Pipeline.classifier ~name:"sgd" ~predict_proba:Sgd_classifier.predict_proba
       ~classes:Sgd_classifier.classes
       (module Sgd_classifier)
       (Sgd_classifier.create ~loss ~penalty:Sgd_classifier.No_penalty
          ~learning_rate:Sgd_classifier.Constant ~eta0:0.1 ~max_epochs:50
          ~shuffle:true ()
       |> get)
    |> get)
  |> get

let test_cross_validation () =
  check_report "ridge" (cross_validate (ridge_pipeline ()));
  check_report "balanced ridge"
    (cross_validate (ridge_pipeline ~class_weight:Class_weight.balanced ()));
  check_report "multinomial"
    (cross_validate ~scorers:probability_scorers (multinomial_pipeline ()));
  check_report "sgd log loss"
    (cross_validate ~scorers:probability_scorers
       (sgd_pipeline Sgd_classifier.Log_loss));
  check_report "sgd hinge" (cross_validate (sgd_pipeline Sgd_classifier.Hinge));
  let hinge_probability =
    Cross_validation.Multiclass_classification.cross_validate ~splitter
      ~failure_policy:Cross_validation.Record
      ~scorers:[| Multiclass_classification_scorer.neg_log_loss |]
      ~seed:(Seed.of_int 42)
      (sgd_pipeline Sgd_classifier.Hinge)
      dataset
    |> get
  in
  Alcotest.(check int)
    "hinge records probability failures" 0
    (Cross_validation.successful_fold_count hinge_probability);
  let binary_labels =
    Target.classification
      (Array.map (fun label -> if label = 9 then 1 else 0) labels)
  in
  let binary_dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x:features
      ~y:binary_labels ()
    |> get_data
  in
  let binary_report =
    Cross_validation.Multiclass_classification.cross_validate ~splitter
      ~scorers:probability_scorers ~seed:(Seed.of_int 42)
      ( multinomial_pipeline () |> fun _ ->
        Pipeline.set_estimator Pipeline.empty
          (Pipeline.classifier ~name:"logistic"
             ~predict_proba:Logistic_regression.predict_proba
             ~classes:Logistic_regression.classes
             (module Logistic_regression)
             (Logistic_regression.create () |> get)
          |> get)
        |> get )
      binary_dataset
    |> get
  in
  Alcotest.(check int)
    "binary pipelines score under multiclass scorers" 3
    (Cross_validation.successful_fold_count binary_report)

type configuration = { c : float }

let test_grid_search () =
  let axis =
    Grid_search.axis ~name:"c" ~values:[| 0.1; 10.0 |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ c -> Ok { c })
    |> get
  in
  let grid =
    Grid_search.create ~base:{ c = 1.0 }
      ~build:(fun configuration ->
        let ( let* ) = Result.bind in
        let* specification =
          Multinomial_logistic_regression.create ~c:configuration.c ()
        in
        let* estimator =
          Pipeline.classifier ~name:"multinomial"
            ~predict_proba:Multinomial_logistic_regression.predict_proba
            ~classes:Multinomial_logistic_regression.classes
            (module Multinomial_logistic_regression)
            specification
        in
        Pipeline.set_estimator Pipeline.empty estimator)
      [| axis |]
    |> get
  in
  let report =
    Grid_search.Multiclass_classification.search ~grid ~splitter
      ~scorers:probability_scorers ~refit:"neg_log_loss" ~seed:(Seed.of_int 42)
      dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  Alcotest.(check int) "two candidates" 2 (Array.length candidates);
  Alcotest.(check bool)
    "every candidate ranked" true
    (Array.for_all
       (fun (candidate : _ Grid_search.candidate) ->
         Option.is_some candidate.Grid_search.rank)
       candidates);
  let selected = Grid_search.selection report |> get in
  Alcotest.(check (array int))
    "refit model keeps class metadata" [| -4; 2; 9 |]
    (Pipeline.classes selected.Grid_search.selected_model |> get);
  Alcotest.(check (pair int int))
    "refit model predicts probabilities" (rows, 3)
    (Pipeline.predict_proba selected.Grid_search.selected_model ~feature_schema
       ~x:features
    |> get |> Matrix.shape)

let () =
  Alcotest.run "multiclass integration"
    [
      ( "supported workflows",
        [
          Alcotest.test_case "cross-validation across multiclass estimators"
            `Quick test_cross_validation;
          Alcotest.test_case "grid search with probability refit" `Quick
            test_grid_search;
        ] );
    ]
