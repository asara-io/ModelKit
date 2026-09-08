open Modelkit

type fixture = {
  vectors : (string, float array) Hashtbl.t;
  matrices : (string, (int, float array) Hashtbl.t) Hashtbl.t;
}

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let parse_floats value =
  value |> String.split_on_char ',' |> List.map float_of_string |> Array.of_list

let read_fixture () =
  let fixture = { vectors = Hashtbl.create 10; matrices = Hashtbl.create 6 } in
  In_channel.with_open_text (Sys.getenv "MODELKIT_MODEL_SELECTOR_FIXTURE")
    (fun input ->
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
                      let rows = Hashtbl.create 16 in
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
  Array.init (Hashtbl.length rows) (fun row -> Hashtbl.find rows row)

let check_float label expected observed =
  let tolerance = 1e-8 *. Float.max 1.0 (Float.abs expected) in
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed) <= tolerance)

let check_vector label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Vector.length observed);
  Array.iteri
    (fun index expected ->
      check_float
        (Printf.sprintf "%s[%d]" label index)
        expected
        (Vector.get observed index))
    expected

let check_matrix label expected observed =
  let rows, columns = Matrix.shape observed in
  Alcotest.(check int) (label ^ " rows") (Array.length expected) rows;
  Alcotest.(check int) (label ^ " columns") (Array.length expected.(0)) columns;
  Array.iteri
    (fun row values ->
      Array.iteri
        (fun column expected ->
          check_float
            (Printf.sprintf "%s[%d,%d]" label row column)
            expected
            (Matrix.get observed row column))
        values)
    expected

module Linear_importance = struct
  include Linear_regression

  let feature_importances fitted =
    Feature_importance.absolute_coefficients (coefficients fitted)
end

module Linear_selector = Select_from_model.Make (Linear_importance)

module Ridge_importance = struct
  include Ridge_classifier

  let feature_importances fitted =
    Feature_importance.coefficient_norms (coefficients fitted)
end

module Ridge_selector = Select_from_model.Make (Ridge_importance)

let test_fixture () =
  let fixture = read_fixture () in
  let regression_x =
    matrix fixture "regression_x" |> Matrix.of_arrays |> get_data
  in
  let regression_schema = Feature_schema.of_matrix regression_x |> get_data in
  let regression_y =
    vector fixture "regression_y"
    |> Vector.of_array |> Target.regression |> get_data
  in
  let regression =
    Linear_selector.create ~max_features:2 (Linear_regression.create ()) |> get
    |> fun specification ->
    Linear_selector.fit specification
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:regression_schema ~x:regression_x ~y:(Some regression_y)
      ()
    |> get
  in
  check_vector "regression importances"
    (vector fixture "regression_importances")
    (Linear_selector.importances regression);
  check_float "regression threshold"
    (vector fixture "regression_threshold").(0)
    (Linear_selector.threshold_value regression);
  Alcotest.(check (array int))
    "regression indices"
    (vector fixture "regression_selected" |> Array.map int_of_float)
    (Linear_selector.selected_indices regression);
  check_matrix "regression output"
    (matrix fixture "regression_output")
    (Linear_selector.transform regression ~feature_schema:regression_schema
       ~x:regression_x
    |> get);
  let classification_x =
    matrix fixture "classification_x" |> Matrix.of_arrays |> get_data
  in
  let classification_schema =
    Feature_schema.of_matrix classification_x |> get_data
  in
  let classification_y =
    vector fixture "classification_y"
    |> Array.map int_of_float |> Target.classification
  in
  let classification =
    Ridge_selector.create ~threshold:Select_from_model.Median ~max_features:2
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
    |> fun specification ->
    Ridge_selector.fit specification
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:classification_schema ~x:classification_x
      ~y:(Some classification_y) ()
    |> get
  in
  check_vector "classification importances"
    (vector fixture "classification_importances")
    (Ridge_selector.importances classification);
  check_float "classification threshold"
    (vector fixture "classification_threshold").(0)
    (Ridge_selector.threshold_value classification);
  Alcotest.(check (array int))
    "classification indices"
    (vector fixture "classification_selected" |> Array.map int_of_float)
    (Ridge_selector.selected_indices classification);
  check_matrix "classification output"
    (matrix fixture "classification_output")
    (Ridge_selector.transform classification
       ~feature_schema:classification_schema ~x:classification_x
    |> get)

let () =
  Alcotest.run "Model-based selection reference"
    [ ("sklearn", [ ("coefficient selectors", `Quick, test_fixture) ]) ]
