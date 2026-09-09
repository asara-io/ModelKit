open Modelkit

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

type configuration = { alpha : float }

let pipeline configuration =
  let ( let* ) = Result.bind in
  let* scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* pipeline = Pipeline.add_transformer Pipeline.empty scaler in
  let* specification = Ridge_regression.create ~alpha:configuration.alpha () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator pipeline estimator

let grid () =
  let alpha =
    Grid_search.axis ~name:"alpha" ~values:[| 0.0; 0.1; 1.0 |]
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ alpha -> Ok { alpha })
    |> get
  in
  Grid_search.create ~base:{ alpha = 0.1 } ~build:pipeline [| alpha |] |> get

let splitter folds =
  K_fold.create ~folds () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let search ~seed dataset =
  Grid_search.Regression.search ~grid:(grid ()) ~splitter:(splitter 3)
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~refit:"neg_mean_squared_error" ~seed dataset
  |> get

let score model dataset =
  let prediction =
    Pipeline.predict model
      ~feature_schema:(Dataset.feature_schema dataset)
      ~x:(Dataset.features dataset)
    |> get
  in
  Regression_scorer.score Regression_scorer.neg_mean_squared_error
    ?sample_weight:(Dataset.sample_weight dataset)
    ~truth:(Dataset.target dataset) ~prediction ()
  |> get

let dataset () =
  let samples = 72 in
  let x =
    Matrix.init ~rows:samples ~columns:3 (fun row column ->
        let value = Float.of_int row in
        match column with
        | 0 -> value /. 10.0
        | 1 -> sin value
        | _ -> cos (value /. 3.0))
    |> get_data
  in
  let y =
    Target.regression
      (Vector.of_array
         (Array.init samples (fun row ->
              let value = Float.of_int row in
              1.5
              +. (0.8 *. (value /. 10.0))
              -. (0.4 *. sin value)
              +. (0.1 *. Float.of_int ((row mod 5) - 2)))))
    |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let () =
  let root_seed = Seed.of_int 42 in
  let holdout_rng =
    Seed.derive root_seed ~operation:"nested-cv-final-holdout" ~index:0
    |> Rng.create
  in
  let development, final_test =
    Train_test_split.split ~test_size:(Split_size.Fraction 0.25)
      ~rng:holdout_rng (dataset ()) ()
    |> get
  in
  let outer =
    K_fold.create ~folds:3 () |> get |> fun specification ->
    K_fold.split specification
      ~rng:
        (Seed.derive root_seed ~operation:"nested-cv-outer-splitter" ~index:0
        |> Rng.create)
      ~x:(Dataset.features development)
      ~y:None ()
    |> get
  in
  let outer_scores =
    Array.mapi
      (fun outer_index (train_rows, test_rows) ->
        let split = Split.of_views ~train:train_rows ~test:test_rows |> get in
        let outer_train, outer_test =
          Split.materialize development split |> get
        in
        let inner_seed =
          Seed.derive root_seed ~operation:"nested-cv-inner-search"
            ~index:outer_index
        in
        let selected =
          search ~seed:inner_seed outer_train |> Grid_search.selection |> get
        in
        score selected.Grid_search.selected_model outer_test)
      outer
  in
  let nested_estimate = Score_aggregation.summarize outer_scores |> get in
  let final_seed =
    Seed.derive root_seed ~operation:"nested-cv-final-search" ~index:0
  in
  let final_model =
    search ~seed:final_seed development |> Grid_search.selection |> get
  in
  let final_score = score final_model.Grid_search.selected_model final_test in
  Printf.printf
    "nested CV neg-MSE: %.6f across %d outer folds\n\
     final holdout neg-MSE: %.6f\n"
    nested_estimate.Score_aggregation.mean
    nested_estimate.Score_aggregation.count final_score
