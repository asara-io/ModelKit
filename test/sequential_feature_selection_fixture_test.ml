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
  let fixture = { vectors = Hashtbl.create 12; matrices = Hashtbl.create 12 } in
  In_channel.with_open_text (Sys.getenv "MODELKIT_SFS_FIXTURE") (fun input ->
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

module Linear_sfs =
  Sequential_feature_selection.Regression.Make (Linear_regression)

module Ridge_sfs =
  Sequential_feature_selection.Multiclass_classification.Make (Ridge_classifier)

let regression_splitter () =
  K_fold.create ~folds:3 () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let classification_splitter () =
  Stratified_k_fold.create ~folds:3 ()
  |> get
  |> Cross_validation.target_aware_splitter (module Stratified_k_fold)

let test_regression fixture direction name expected_fits =
  let x = matrix fixture "regression_x" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "regression_y"
    |> Vector.of_array |> Target.regression |> get_data
  in
  let fitted =
    Linear_sfs.create ~direction ~max_fits:expected_fits ~feature_count:2
      ~splitter:(regression_splitter ())
      ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> fun specification ->
    Linear_sfs.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  Alcotest.(check int)
    (name ^ " fit count") expected_fits
    (Linear_sfs.fit_count fitted);
  Alcotest.(check (array int))
    (name ^ " selected")
    (vector fixture (name ^ "_selected") |> Array.map int_of_float)
    (Linear_sfs.selected_indices fitted);
  check_matrix (name ^ " output")
    (matrix fixture (name ^ "_output"))
    (Linear_sfs.transform fitted ~metadata:Metadata.empty ~feature_schema:schema
       ~x
    |> get)

let test_classification fixture direction name expected_fits =
  let x = matrix fixture "classification_x" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "classification_y"
    |> Array.map int_of_float |> Target.classification
  in
  let fitted =
    Ridge_sfs.create ~direction ~max_fits:expected_fits ~feature_count:2
      ~splitter:(classification_splitter ())
      ~scorer:Multiclass_classification_scorer.accuracy
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
    |> fun specification ->
    Ridge_sfs.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  Alcotest.(check int)
    (name ^ " fit count") expected_fits
    (Ridge_sfs.fit_count fitted);
  Alcotest.(check (array int))
    (name ^ " selected")
    (vector fixture (name ^ "_selected") |> Array.map int_of_float)
    (Ridge_sfs.selected_indices fitted);
  check_matrix (name ^ " output")
    (matrix fixture (name ^ "_output"))
    (Ridge_sfs.transform fitted ~metadata:Metadata.empty ~feature_schema:schema
       ~x
    |> get)

let test_fixture () =
  let fixture = read_fixture () in
  test_regression fixture Sequential_feature_selection.Forward
    "regression_forward" 27;
  test_regression fixture Sequential_feature_selection.Backward
    "regression_backward" 36;
  test_classification fixture Sequential_feature_selection.Forward
    "classification_forward" 27;
  test_classification fixture Sequential_feature_selection.Backward
    "classification_backward" 36

let () =
  Alcotest.run "Sequential feature selection reference"
    [ ("sklearn", [ ("forward and backward", `Quick, test_fixture) ]) ]
