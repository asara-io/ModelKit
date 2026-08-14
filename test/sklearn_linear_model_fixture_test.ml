open Modelkit

type fixture = {
  vectors : (string, float array) Hashtbl.t;
  matrices : (string, (int, float array) Hashtbl.t) Hashtbl.t;
}

let parse_floats value =
  if String.equal value "" then [||]
  else
    value |> String.split_on_char ',' |> List.map float_of_string
    |> Array.of_list

let read_fixture path =
  let fixture = { vectors = Hashtbl.create 16; matrices = Hashtbl.create 8 } in
  In_channel.with_open_text path (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if String.length line > 0 && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ name; values ] ->
                Hashtbl.replace fixture.vectors name (parse_floats values)
            | [ name; row; values ] ->
                let rows =
                  match Hashtbl.find_opt fixture.matrices name with
                  | Some rows -> rows
                  | None ->
                      let rows = Hashtbl.create 8 in
                      Hashtbl.add fixture.matrices name rows;
                      rows
                in
                Hashtbl.replace rows (int_of_string row) (parse_floats values)
            | fields ->
                Alcotest.failf "invalid fixture row with %d fields"
                  (List.length fields)));
  fixture

let vector fixture name =
  match Hashtbl.find_opt fixture.vectors name with
  | Some values -> values
  | None -> Alcotest.failf "fixture vector %S is missing" name

let matrix fixture name =
  let rows =
    match Hashtbl.find_opt fixture.matrices name with
    | Some rows -> rows
    | None -> Alcotest.failf "fixture matrix %S is missing" name
  in
  Array.init (Hashtbl.length rows) (fun row ->
      match Hashtbl.find_opt rows row with
      | Some values -> values
      | None -> Alcotest.failf "fixture matrix %S row %d is missing" name row)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let agrees expected observed =
  let scale = Float.max (Float.abs expected) (Float.abs observed) in
  Float.abs (expected -. observed) <= 1e-7 *. Float.max 1.0 scale

let check_float label expected observed =
  Alcotest.(check bool) label true (agrees expected observed)

let check_vector label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      check_float
        (Format.sprintf "%s[%d]" label index)
        expected observed.(index))
    expected

let check_matrix label expected observed =
  let rows, columns = Matrix.shape observed in
  Alcotest.(check int) (label ^ " rows") (Array.length expected) rows;
  let expected_columns =
    if Array.length expected = 0 then 0 else Array.length expected.(0)
  in
  Alcotest.(check int) (label ^ " columns") expected_columns columns;
  Array.iteri
    (fun row values ->
      Array.iteri
        (fun column expected ->
          check_float
            (Format.sprintf "%s[%d,%d]" label row column)
            expected
            (Matrix.get observed row column))
        values)
    expected

let rng () = Rng.create (Seed.of_int 1729)

let setup fixture =
  let x_train = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x_train |> get_data in
  let regression_target =
    vector fixture "regression_target"
    |> Vector.of_array |> Target.regression |> get_data
  in
  let classification_target =
    vector fixture "classification_target"
    |> Array.map int_of_float |> Target.classification
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  ( x_train,
    x_predict,
    feature_schema,
    regression_target,
    classification_target,
    sample_weight )

let test_linear fixture () =
  let x_train, x_predict, feature_schema, target, _, sample_weight =
    setup fixture
  in
  let fitted =
    Linear_regression.fit
      (Linear_regression.create ())
      ~sample_weight ~rng:(rng ()) ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  check_vector "linear coefficients"
    (vector fixture "linear_coefficients")
    (Linear_regression.coefficients fitted |> Vector.to_array);
  check_float "linear intercept"
    (vector fixture "linear_intercept").(0)
    (Linear_regression.intercept fitted);
  let prediction =
    Linear_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array
  in
  check_vector "linear prediction"
    (vector fixture "linear_prediction")
    prediction

let test_ridge fixture () =
  let x_train, x_predict, feature_schema, target, _, sample_weight =
    setup fixture
  in
  let alpha = (vector fixture "ridge_alpha").(0) in
  let specification = Ridge_regression.create ~alpha () |> get in
  let fitted =
    Ridge_regression.fit specification ~sample_weight ~rng:(rng ())
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  check_vector "ridge coefficients"
    (vector fixture "ridge_coefficients")
    (Ridge_regression.coefficients fitted |> Vector.to_array);
  check_float "ridge intercept"
    (vector fixture "ridge_intercept").(0)
    (Ridge_regression.intercept fitted);
  let prediction =
    Ridge_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array
  in
  check_vector "ridge prediction" (vector fixture "ridge_prediction") prediction

let test_logistic fixture () =
  let x_train, x_predict, feature_schema, _, target, sample_weight =
    setup fixture
  in
  let c = (vector fixture "logistic_c").(0) in
  let specification =
    Logistic_regression.create ~c ~tolerance:1e-12 ~max_iterations:1000 ()
    |> get
  in
  let fitted =
    Logistic_regression.fit specification ~sample_weight ~rng:(rng ())
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  check_vector "logistic classes"
    (vector fixture "logistic_classes")
    (Logistic_regression.classes fitted |> Array.map Float.of_int);
  check_vector "logistic coefficients"
    (vector fixture "logistic_coefficients")
    (Logistic_regression.coefficients fitted |> Vector.to_array);
  check_float "logistic intercept"
    (vector fixture "logistic_intercept").(0)
    (Logistic_regression.intercept fitted);
  let decision =
    Logistic_regression.decision_function fitted ~feature_schema ~x:x_predict
    |> get |> Vector.to_array
  in
  check_vector "logistic decision" (vector fixture "logistic_decision") decision;
  let probabilities =
    Logistic_regression.predict_proba fitted ~feature_schema ~x:x_predict |> get
  in
  check_matrix "logistic probabilities"
    (matrix fixture "logistic_probabilities")
    probabilities;
  let prediction =
    Logistic_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.classification_values |> Array.map Float.of_int
  in
  check_vector "logistic prediction"
    (vector fixture "logistic_prediction")
    prediction

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_LINEAR_MODEL_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_LINEAR_MODEL_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn linear-model fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "ordinary least squares" `Quick
            (test_linear fixture);
          Alcotest.test_case "ridge" `Quick (test_ridge fixture);
          Alcotest.test_case "binary logistic regression" `Quick
            (test_logistic fixture);
        ] );
    ]
