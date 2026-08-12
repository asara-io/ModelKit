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
  let row_count = Hashtbl.length rows in
  Array.init row_count (fun row ->
      match Hashtbl.find_opt rows row with
      | Some values -> values
      | None -> Alcotest.failf "fixture matrix %S row %d is missing" name row)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let check_float label expected observed =
  let agrees =
    if Float.is_nan expected then Float.is_nan observed
    else
      let tolerance = 1e-12 *. Float.max 1.0 (Float.abs expected) in
      Float.abs (expected -. observed) <= tolerance
  in
  Alcotest.(check bool) label true agrees

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

let test_fixture path () =
  let fixture = read_fixture path in
  let input = matrix fixture "input" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix input |> get_data in
  let fit_imputer specification =
    Simple_imputer.fit specification ~rng:(rng ()) ~feature_schema:schema
      ~x:input ~y:None ()
    |> get
  in
  let check_imputer strategy_name specification =
    let fitted = fit_imputer specification in
    check_vector
      (strategy_name ^ " statistics")
      (vector fixture (strategy_name ^ "_statistics"))
      (Simple_imputer.statistics fitted |> Vector.to_array);
    let transformed =
      Simple_imputer.transform fitted ~feature_schema:schema ~x:input |> get
    in
    check_matrix
      (strategy_name ^ " output")
      (matrix fixture (strategy_name ^ "_output"))
      transformed;
    transformed
  in
  let mean_output = check_imputer "mean" (Simple_imputer.mean ()) in
  ignore (check_imputer "median" (Simple_imputer.median ()));
  ignore (check_imputer "constant" (Simple_imputer.constant (-2.0) |> get));
  let scaler =
    Standard_scaler.fit
      (Standard_scaler.create ())
      ~rng:(rng ()) ~feature_schema:schema ~x:mean_output ~y:None ()
    |> get
  in
  check_vector "scaler mean"
    (vector fixture "scaler_mean")
    (Standard_scaler.mean scaler |> Vector.to_array);
  check_vector "scaler variance"
    (vector fixture "scaler_variance")
    (Standard_scaler.variance scaler |> Vector.to_array);
  check_vector "scaler scale"
    (vector fixture "scaler_scale")
    (Standard_scaler.scale scaler |> Vector.to_array);
  check_matrix "scaled output"
    (matrix fixture "scaled_output")
    (Standard_scaler.transform scaler ~feature_schema:schema ~x:mean_output
    |> get);
  let threshold = (vector fixture "variance_threshold").(0) in
  let selector =
    Variance_threshold.fit
      (Variance_threshold.create ~threshold () |> get)
      ~rng:(rng ()) ~feature_schema:schema ~x:mean_output ~y:None ()
    |> get
  in
  check_vector "feature variances"
    (vector fixture "feature_variances")
    (Variance_threshold.variances selector |> Vector.to_array);
  let expected_indices =
    vector fixture "selected_indices" |> Array.map int_of_float
  in
  Alcotest.(check (array int))
    "selected indices" expected_indices
    (Variance_threshold.selected_indices selector);
  check_matrix "selected output"
    (matrix fixture "selected_output")
    (Variance_threshold.transform selector ~feature_schema:schema ~x:mean_output
    |> get)

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_PREPROCESSING_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_PREPROCESSING_FIXTURE is not set"
  in
  Alcotest.run "sklearn preprocessing fixtures"
    [
      ( "preprocessing",
        [ Alcotest.test_case "v1" `Quick (test_fixture fixture_path) ] );
    ]
