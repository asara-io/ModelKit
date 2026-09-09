open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let fields () =
  In_channel.with_open_text (Sys.getenv "MODELKIT_PERMUTATION_FIXTURE")
    (fun channel ->
      In_channel.input_lines channel
      |> List.filter_map (fun line ->
          match String.split_on_char '\t' line with
          | [ key; values ] -> Some (key, String.split_on_char ',' values)
          | _ -> None))

let floats fields name = List.assoc name fields |> List.map float_of_string
let scalar fields name = floats fields name |> List.hd

let pipeline () =
  let ( let* ) = Result.bind in
  let* scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* pipeline = Pipeline.add_transformer Pipeline.empty scaler in
  let* specification = Ridge_regression.create ~alpha:0.5 () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator pipeline estimator

let test_fixture () =
  let fields = fields () in
  let feature_0 = floats fields "feature_0" |> Array.of_list in
  let feature_1 = floats fields "feature_1" |> Array.of_list in
  let target = floats fields "target" |> Array.of_list in
  let groups =
    List.assoc "groups" fields |> List.map int_of_string |> Array.of_list
  in
  let rows = Array.length target in
  let x =
    Matrix.init ~rows ~columns:2 (fun row column ->
        if column = 0 then feature_0.(row) else feature_1.(row))
    |> get_data
  in
  let y = Target.regression (Vector.of_array target) |> get_data in
  let groups = Groups.create ~expected_length:rows groups |> get_data in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~groups ~x ~y ()
    |> get_data
  in
  let splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let specification = Permutation_test.create ~permutations:5 () |> get in
  let report =
    Permutation_test.Regression.evaluate ~specification ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error ~seed:(Seed.of_int 73)
      (pipeline () |> get)
      dataset
    |> get
  in
  Alcotest.check (Alcotest.float 1e-7) "observed score"
    (scalar fields "observed_score")
    (Permutation_test.observed_score report);
  Alcotest.check
    (Alcotest.array (Alcotest.float 1e-7))
    "within-group permutation scores"
    (floats fields "permutation_scores" |> Array.of_list)
    (Permutation_test.permutation_scores report);
  Alcotest.check (Alcotest.float 0.0) "corrected p-value"
    (scalar fields "p_value")
    (Permutation_test.p_value report)

let () =
  Alcotest.run "Permutation-test reference"
    [ ("sklearn", [ ("scaled ridge significance", `Quick, test_fixture) ]) ]
