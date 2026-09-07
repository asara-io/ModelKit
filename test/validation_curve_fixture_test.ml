open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let fields () =
  In_channel.with_open_text (Sys.getenv "MODELKIT_VALIDATION_CURVE_FIXTURE")
    (fun channel ->
      In_channel.input_lines channel
      |> List.filter_map (fun line ->
          match String.split_on_char '\t' line with
          | [ key; values ] -> Some (key, String.split_on_char ',' values)
          | _ -> None))

let floats fields name = List.assoc name fields |> List.map float_of_string

type configuration = { alpha : float }

let pipeline configuration =
  let ( let* ) = Result.bind in
  let* scale =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* staged = Pipeline.add_transformer Pipeline.empty scale in
  let* specification = Ridge_regression.create ~alpha:configuration.alpha () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator staged estimator

let score name = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (name ^ ": " ^ Error.to_string error)
  | None -> Alcotest.fail (name ^ " is absent")

let test_fixture () =
  let fields = fields () in
  let feature_0 = floats fields "feature_0" |> Array.of_list in
  let feature_1 = floats fields "feature_1" |> Array.of_list in
  let target = floats fields "target" |> Array.of_list in
  let rows = Array.length target in
  let x =
    Matrix.init ~rows ~columns:2 (fun row column ->
        if column = 0 then feature_0.(row) else feature_1.(row))
    |> get_data
  in
  let y = Target.regression (Vector.of_array target) |> get_data in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let alphas = floats fields "alphas" |> Array.of_list in
  let specification =
    Validation_curve.create ~name:"alpha" ~base:{ alpha = 1.0 } ~values:alphas
      ~encode:(fun alpha -> Grid_search.Float alpha)
      ~set:(fun _ alpha -> Ok { alpha })
      ~build:pipeline ()
    |> get
  in
  let splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let points =
    Validation_curve.Regression.evaluate ~specification ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 101) dataset
    |> get |> Validation_curve.points
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "parameter values" alphas
    (Array.map (fun point -> point.Validation_curve.parameter_value) points);
  Array.iteri
    (fun point_index point ->
      let folds =
        Cross_validation.folds (Option.get point.Validation_curve.evaluation)
      in
      let expected_train =
        floats fields ("train_scores_" ^ string_of_int point_index)
      in
      let expected_test =
        floats fields ("test_scores_" ^ string_of_int point_index)
      in
      Array.iteri
        (fun fold_index fold ->
          let observed = fold.Cross_validation.scores.(0) in
          Alcotest.check (Alcotest.float 1e-7) "training score"
            (List.nth expected_train fold_index)
            (score "training score" observed.Cross_validation.train_score);
          Alcotest.check (Alcotest.float 1e-7) "validation score"
            (List.nth expected_test fold_index)
            (score "validation score" observed.Cross_validation.test_score))
        folds)
    points

let () =
  Alcotest.run "Validation-curve reference"
    [ ("sklearn", [ ("scaled ridge scores", `Quick, test_fixture) ]) ]
