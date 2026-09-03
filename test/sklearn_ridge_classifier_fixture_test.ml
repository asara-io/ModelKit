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
  Alcotest.(check int)
    (label ^ " rows") (Array.length expected) (Matrix.rows observed);
  let columns =
    if Array.length expected = 0 then 0 else Array.length expected.(0)
  in
  Alcotest.(check int) (label ^ " columns") columns (Matrix.columns observed);
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

let int_vector fixture name = vector fixture name |> Array.map int_of_float
let rng () = Rng.create (Seed.of_int 1729)

let setup fixture target_name =
  let x_train = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x_train |> get_data in
  let target = int_vector fixture target_name |> Target.classification in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  let specification =
    Ridge_classifier.create ~alpha:(vector fixture "alpha").(0) () |> get
  in
  let fitted =
    Ridge_classifier.fit specification ~sample_weight ~rng:(rng ())
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  (x_predict, feature_schema, fitted)

let test_binary fixture () =
  let x_predict, feature_schema, fitted = setup fixture "binary_target" in
  Alcotest.(check (array int))
    "binary classes"
    (int_vector fixture "binary_classes")
    (Ridge_classifier.classes fitted);
  let expected_coefficients = matrix fixture "binary_coefficients" in
  let coefficients = Ridge_classifier.coefficients fitted in
  Alcotest.(check (pair int int))
    "expanded binary coefficient shape"
    (2, Array.length expected_coefficients.(0))
    (Matrix.shape coefficients);
  Array.iteri
    (fun feature expected ->
      check_float
        (Format.sprintf "positive coefficient[%d]" feature)
        expected
        (Matrix.get coefficients 1 feature);
      check_float
        (Format.sprintf "negative coefficient[%d]" feature)
        (-.expected)
        (Matrix.get coefficients 0 feature))
    expected_coefficients.(0);
  let expected_intercept = (vector fixture "binary_intercepts").(0) in
  let intercepts = Ridge_classifier.intercepts fitted in
  check_float "positive intercept" expected_intercept (Vector.get intercepts 1);
  check_float "negative intercept" (-.expected_intercept)
    (Vector.get intercepts 0);
  let expected_decisions = vector fixture "binary_decisions" in
  let decisions =
    Ridge_classifier.decision_function fitted ~feature_schema ~x:x_predict
    |> get
  in
  Array.iteri
    (fun row expected ->
      check_float
        (Format.sprintf "positive decision[%d]" row)
        expected
        (Matrix.get decisions row 1);
      check_float
        (Format.sprintf "negative decision[%d]" row)
        (-.expected)
        (Matrix.get decisions row 0))
    expected_decisions;
  let predictions =
    Ridge_classifier.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int))
    "binary predictions"
    (int_vector fixture "binary_predictions")
    predictions

let test_multiclass fixture () =
  let x_predict, feature_schema, fitted = setup fixture "multiclass_target" in
  Alcotest.(check (array int))
    "multiclass classes"
    (int_vector fixture "multiclass_classes")
    (Ridge_classifier.classes fitted);
  check_matrix "multiclass coefficients"
    (matrix fixture "multiclass_coefficients")
    (Ridge_classifier.coefficients fitted);
  check_vector "multiclass intercepts"
    (vector fixture "multiclass_intercepts")
    (Ridge_classifier.intercepts fitted |> Vector.to_array);
  let decisions =
    Ridge_classifier.decision_function fitted ~feature_schema ~x:x_predict
    |> get
  in
  check_matrix "multiclass decisions"
    (matrix fixture "multiclass_decisions")
    decisions;
  let predictions =
    Ridge_classifier.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.classification_values
  in
  Alcotest.(check (array int))
    "multiclass predictions"
    (int_vector fixture "multiclass_predictions")
    predictions

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_RIDGE_CLASSIFIER_FIXTURE" with
    | Some path -> path
    | None ->
        Alcotest.fail "MODELKIT_SKLEARN_RIDGE_CLASSIFIER_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn ridge-classifier fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "binary" `Quick (test_binary fixture);
          Alcotest.test_case "multiclass" `Quick (test_multiclass fixture);
        ] );
    ]
