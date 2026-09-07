open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let fields () =
  In_channel.with_open_text (Sys.getenv "MODELKIT_LEARNING_CURVE_FIXTURE")
    (fun channel ->
      In_channel.input_lines channel
      |> List.filter_map (fun line ->
          match String.split_on_char '\t' line with
          | [ key; values ] -> Some (key, String.split_on_char ',' values)
          | _ -> None))

let floats fields name = List.assoc name fields |> List.map float_of_string
let ints fields name = List.assoc name fields |> List.map int_of_string

let score name = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (name ^ ": " ^ Error.to_string error)
  | None -> Alcotest.fail (name ^ " is absent")

let test_fixture () =
  let fields = fields () in
  let x_values = floats fields "x" |> Array.of_list in
  let y_values = floats fields "y" |> Array.of_list in
  let x =
    Matrix.init ~rows:(Array.length x_values) ~columns:1 (fun row _ ->
        x_values.(row))
    |> get_data
  in
  let y = Target.regression (Vector.of_array y_values) |> get_data in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let pipeline =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
    |> Pipeline.set_estimator Pipeline.empty
    |> get
  in
  let splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let schedule =
    Learning_curve.schedule
      [|
        Learning_curve.Fraction 0.25;
        Learning_curve.Fraction 0.5;
        Learning_curve.Fraction 1.0;
      |]
    |> get
  in
  let points =
    Learning_curve.Regression.evaluate ~schedule ~splitter
      ~scorers:[| Regression_scorer.neg_mean_absolute_error |]
      ~seed:(Seed.of_int 101) pipeline dataset
    |> get |> Learning_curve.points
  in
  Alcotest.(check (array int))
    "resolved training sizes"
    (ints fields "training_sizes" |> Array.of_list)
    (Array.map (fun point -> point.Learning_curve.training_samples) points);
  Array.iteri
    (fun point_index point ->
      let folds = Cross_validation.folds point.Learning_curve.evaluation in
      let expected_train =
        floats fields ("train_scores_" ^ string_of_int point_index)
      in
      let expected_test =
        floats fields ("test_scores_" ^ string_of_int point_index)
      in
      Array.iteri
        (fun fold_index fold ->
          let observed = fold.Cross_validation.scores.(0) in
          Alcotest.check (Alcotest.float 1e-9) "training score"
            (List.nth expected_train fold_index)
            (score "training score" observed.Cross_validation.train_score);
          Alcotest.check (Alcotest.float 1e-9) "validation score"
            (List.nth expected_test fold_index)
            (score "validation score" observed.Cross_validation.test_score))
        folds)
    points

let () =
  Alcotest.run "Learning-curve reference"
    [ ("sklearn", [ ("scores and sizes", `Quick, test_fixture) ]) ]
