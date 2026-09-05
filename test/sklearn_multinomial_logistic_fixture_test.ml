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

let test_parity fixture () =
  let x_train = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x_train |> get_data in
  let target =
    vector fixture "target" |> Array.map int_of_float |> Target.classification
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  let specification =
    Multinomial_logistic_regression.create
      ~c:(vector fixture "c").(0)
      ~tolerance:(vector fixture "tolerance").(0)
      ~max_iterations:(int_of_float (vector fixture "max_iterations").(0))
      ()
    |> get
  in
  let fitted =
    Multinomial_logistic_regression.fit specification ~sample_weight
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  Alcotest.(check (array int))
    "classes"
    (vector fixture "classes" |> Array.map int_of_float)
    (Multinomial_logistic_regression.classes fitted);
  check_matrix "coefficients"
    (matrix fixture "coefficients")
    (Multinomial_logistic_regression.coefficients fitted);
  check_vector "intercepts"
    (vector fixture "intercepts")
    (Multinomial_logistic_regression.intercepts fitted |> Vector.to_array);
  check_matrix "decisions"
    (matrix fixture "decisions")
    (Multinomial_logistic_regression.decision_function fitted ~feature_schema
       ~x:x_predict
    |> get);
  check_matrix "probabilities"
    (matrix fixture "probabilities")
    (Multinomial_logistic_regression.predict_proba fitted ~feature_schema
       ~x:x_predict
    |> get);
  Alcotest.(check (array int))
    "predictions"
    (vector fixture "predictions" |> Array.map int_of_float)
    (Multinomial_logistic_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.classification_values)

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_MULTINOMIAL_LOGISTIC_FIXTURE" with
    | Some path -> path
    | None ->
        Alcotest.fail "MODELKIT_SKLEARN_MULTINOMIAL_LOGISTIC_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn multinomial-logistic fixture"
    [
      ( "parity",
        [ Alcotest.test_case "weighted fit" `Quick (test_parity fixture) ] );
    ]
