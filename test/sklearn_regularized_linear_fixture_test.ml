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

let rng () = Rng.create (Seed.of_int 1729)

let setup fixture =
  let x_train = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x_train |> get_data in
  let target =
    vector fixture "target" |> Vector.of_array |> Target.regression |> get_data
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  (x_train, x_predict, feature_schema, target, sample_weight)

let test_lasso fixture () =
  let x, x_predict, feature_schema, y, sample_weight = setup fixture in
  let specification =
    Lasso_regression.create
      ~alpha:(vector fixture "lasso_alpha").(0)
      ~tolerance:1e-12 ~max_iterations:100_000 ()
    |> get
  in
  let fitted =
    Lasso_regression.fit specification ~sample_weight ~rng:(rng ())
      ~feature_schema ~x ~y ()
    |> get
  in
  check_vector "lasso coefficients"
    (vector fixture "lasso_coefficients")
    (Lasso_regression.coefficients fitted |> Vector.to_array);
  check_float "lasso intercept"
    (vector fixture "lasso_intercept").(0)
    (Lasso_regression.intercept fitted);
  let prediction =
    Lasso_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array
  in
  check_vector "lasso prediction" (vector fixture "lasso_prediction") prediction

let test_elastic_net fixture () =
  let x, x_predict, feature_schema, y, sample_weight = setup fixture in
  let specification =
    Elastic_net_regression.create
      ~alpha:(vector fixture "elastic_alpha").(0)
      ~l1_ratio:(vector fixture "elastic_l1_ratio").(0)
      ~tolerance:1e-12 ~max_iterations:100_000 ()
    |> get
  in
  let fitted =
    Elastic_net_regression.fit specification ~sample_weight ~rng:(rng ())
      ~feature_schema ~x ~y ()
    |> get
  in
  check_vector "elastic-net coefficients"
    (vector fixture "elastic_coefficients")
    (Elastic_net_regression.coefficients fitted |> Vector.to_array);
  check_float "elastic-net intercept"
    (vector fixture "elastic_intercept").(0)
    (Elastic_net_regression.intercept fitted);
  let prediction =
    Elastic_net_regression.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array
  in
  check_vector "elastic-net prediction"
    (vector fixture "elastic_prediction")
    prediction

let test_paths fixture () =
  let x = matrix fixture "centered_x" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "centered_target"
    |> Vector.of_array |> Target.regression |> get_data
  in
  let lasso =
    Lasso_path.fit
      (Lasso_path.create ~fit_intercept:false ~tolerance:1e-12
         ~max_iterations:100_000 ()
      |> get)
      ~alphas:(Vector.of_array (vector fixture "lasso_path_alphas"))
      ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  check_vector "lasso path alphas"
    (vector fixture "lasso_path_alphas")
    (Lasso_path.alphas lasso |> Vector.to_array);
  check_matrix "lasso path coefficients"
    (matrix fixture "lasso_path_coefficients")
    (Lasso_path.coefficients lasso);
  let elastic =
    Elastic_net_path.fit
      (Elastic_net_path.create
         ~l1_ratio:(vector fixture "elastic_l1_ratio").(0)
         ~fit_intercept:false ~tolerance:1e-12 ~max_iterations:100_000 ()
      |> get)
      ~alphas:(Vector.of_array (vector fixture "elastic_path_alphas"))
      ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  check_vector "elastic path alphas"
    (vector fixture "elastic_path_alphas")
    (Elastic_net_path.alphas elastic |> Vector.to_array);
  check_matrix "elastic path coefficients"
    (matrix fixture "elastic_path_coefficients")
    (Elastic_net_path.coefficients elastic)

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_REGULARIZED_LINEAR_FIXTURE" with
    | Some path -> path
    | None ->
        Alcotest.fail "MODELKIT_SKLEARN_REGULARIZED_LINEAR_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn regularized-linear fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "lasso" `Quick (test_lasso fixture);
          Alcotest.test_case "elastic net" `Quick (test_elastic_net fixture);
          Alcotest.test_case "regularization paths" `Quick (test_paths fixture);
        ] );
    ]
