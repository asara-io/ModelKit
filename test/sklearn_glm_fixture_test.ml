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
  let fixture = { vectors = Hashtbl.create 16; matrices = Hashtbl.create 4 } in
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

let check_vector label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      Alcotest.(check bool)
        (Format.sprintf "%s[%d]" label index)
        true
        (agrees expected observed.(index)))
    expected

let fixture_path () =
  match Sys.getenv_opt "MODELKIT_SKLEARN_GLM_FIXTURE" with
  | Some path -> path
  | None -> Alcotest.fail "MODELKIT_SKLEARN_GLM_FIXTURE is not set"

let setup fixture =
  let x_train = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let target =
    vector fixture "target" |> Vector.of_array |> Target.regression |> get_data
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:(Matrix.rows x_train)
      (vector fixture "sample_weight")
    |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x_train |> get_data in
  (x_train, x_predict, target, sample_weight, feature_schema)

let test_poisson fixture =
  let x_train, x_predict, target, sample_weight, feature_schema =
    setup fixture
  in
  let fitted =
    Poisson_regression.fit
      (Poisson_regression.create
         ~alpha:(vector fixture "alpha").(0)
         ~tolerance:(vector fixture "tolerance").(0)
         ~max_iterations:(int_of_float (vector fixture "max_iterations").(0))
         ()
      |> get)
      ~sample_weight
      ~rng:(Rng.create (Seed.of_int 2026))
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  check_vector "Poisson coefficients"
    (vector fixture "poisson_coefficients")
    (Poisson_regression.coefficients fitted |> Vector.to_array);
  check_vector "Poisson intercept"
    (vector fixture "poisson_intercept")
    [| Poisson_regression.intercept fitted |];
  check_vector "Poisson predictions"
    (vector fixture "poisson_predictions")
    (Poisson_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array)

let test_tweedie fixture =
  let x_train, x_predict, target, sample_weight, feature_schema =
    setup fixture
  in
  let fitted =
    Tweedie_regression.fit
      (Tweedie_regression.create
         ~power:(vector fixture "tweedie_power").(0)
         ~alpha:(vector fixture "alpha").(0)
         ~link:Tweedie_regression.Log
         ~tolerance:(vector fixture "tolerance").(0)
         ~max_iterations:(int_of_float (vector fixture "max_iterations").(0))
         ()
      |> get)
      ~sample_weight
      ~rng:(Rng.create (Seed.of_int 2026))
      ~feature_schema ~x:x_train ~y:target ()
    |> get
  in
  check_vector "Tweedie coefficients"
    (vector fixture "tweedie_coefficients")
    (Tweedie_regression.coefficients fitted |> Vector.to_array);
  check_vector "Tweedie intercept"
    (vector fixture "tweedie_intercept")
    [| Tweedie_regression.intercept fitted |];
  check_vector "Tweedie predictions"
    (vector fixture "tweedie_predictions")
    (Tweedie_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array)

let () =
  let fixture = read_fixture (fixture_path ()) in
  Alcotest.run "sklearn generalized-linear-model fixture"
    [
      ( "parity",
        [
          Alcotest.test_case "Poisson" `Quick (fun () -> test_poisson fixture);
          Alcotest.test_case "Tweedie" `Quick (fun () -> test_tweedie fixture);
        ] );
    ]
