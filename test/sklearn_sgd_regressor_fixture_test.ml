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

let check_vector label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      let observed = observed.(index) in
      let scale = Float.max (Float.abs expected) (Float.abs observed) in
      Alcotest.(check bool)
        (Format.sprintf "%s[%d]" label index)
        true
        (Float.abs (expected -. observed) <= 1e-12 *. Float.max 1.0 scale))
    expected

let setup fixture =
  let x = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "target" |> Vector.of_array |> Target.regression |> get_data
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  let eta0 = (vector fixture "eta0").(0) in
  let epochs = int_of_float (vector fixture "epochs").(0) in
  (x, x_predict, feature_schema, y, sample_weight, eta0, epochs)

let specification ~eta0 ~epochs =
  Sgd_regressor.create ~penalty:Sgd_regressor.No_penalty
    ~learning_rate:Sgd_regressor.Constant ~eta0 ~max_epochs:epochs
    ~shuffle:false ()
  |> get

let check_model fixture prefix fitted feature_schema x_predict =
  check_vector (prefix ^ " coefficients")
    (vector fixture (prefix ^ "_coefficients"))
    (Sgd_regressor.coefficients fitted |> Vector.to_array);
  check_vector (prefix ^ " intercept")
    (vector fixture (prefix ^ "_intercept"))
    [| Sgd_regressor.intercept fitted |];
  check_vector (prefix ^ " prediction")
    (vector fixture (prefix ^ "_prediction"))
    (Sgd_regressor.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.regression_values |> Vector.to_array)

let test_fit fixture () =
  let x, x_predict, feature_schema, y, sample_weight, eta0, epochs =
    setup fixture
  in
  let fitted =
    Sgd_regressor.fit
      (specification ~eta0 ~epochs)
      ~sample_weight
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x ~y ()
    |> get
  in
  check_model fixture "fit" fitted feature_schema x_predict

let test_partial_fit fixture () =
  let x, x_predict, feature_schema, y, sample_weight, eta0, epochs =
    setup fixture
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_regressor.partial_fit checkpoint ~sample_weight ~feature_schema ~x ~y
        ()
      |> get
      |> train (remaining - 1)
  in
  let fitted =
    Sgd_regressor.start
      (specification ~eta0 ~epochs)
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema
    |> train epochs |> Sgd_regressor.to_fitted |> get
  in
  check_model fixture "partial" fitted feature_schema x_predict

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_SGD_REGRESSOR_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_SGD_REGRESSOR_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn SGD-regressor fixture"
    [
      ( "parity",
        [
          Alcotest.test_case "fit" `Quick (test_fit fixture);
          Alcotest.test_case "partial_fit" `Quick (test_partial_fit fixture);
        ] );
    ]
