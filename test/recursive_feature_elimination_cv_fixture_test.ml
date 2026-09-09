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
  let fixture = { vectors = Hashtbl.create 24; matrices = Hashtbl.create 8 } in
  In_channel.with_open_text (Sys.getenv "MODELKIT_RFECV_FIXTURE") (fun input ->
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

let check_float_array label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      check_float
        (Printf.sprintf "%s[%d]" label index)
        expected observed.(index))
    expected

let check_vector label expected observed =
  check_float_array label expected (Vector.to_array observed)

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

module Linear_rfecv =
  Recursive_feature_elimination_cv.Regression.Make (Linear_importance)

module Ridge_importance = struct
  include Ridge_classifier

  let feature_importances fitted =
    Feature_importance.coefficient_norms (coefficients fitted)
end

module Ridge_rfecv =
  Recursive_feature_elimination_cv.Multiclass_classification.Make
    (Ridge_importance)

let check_scores fixture prefix results =
  Alcotest.(check (array int))
    (prefix ^ " feature counts")
    (vector fixture (prefix ^ "_feature_counts") |> Array.map int_of_float)
    (Array.map
       (fun result -> result.Recursive_feature_elimination_cv.feature_count)
       results);
  check_float_array (prefix ^ " mean scores")
    (vector fixture (prefix ^ "_mean_scores"))
    (Array.map
       (fun result -> result.Recursive_feature_elimination_cv.mean_score)
       results);
  check_float_array
    (prefix ^ " standard deviations")
    (vector fixture (prefix ^ "_standard_deviations"))
    (Array.map
       (fun result ->
         result.Recursive_feature_elimination_cv.standard_deviation)
       results);
  for fold = 0 to 2 do
    check_float_array
      (Printf.sprintf "%s fold %d scores" prefix fold)
      (vector fixture (Printf.sprintf "%s_fold_%d_scores" prefix fold))
      (Array.map
         (fun result ->
           result.Recursive_feature_elimination_cv.fold_scores.(fold))
         results)
  done

let test_regression fixture =
  let x = matrix fixture "regression_x" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "regression_y"
    |> Vector.of_array |> Target.regression |> get_data
  in
  let splitter =
    K_fold.create ~folds:3 () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let fitted =
    Linear_rfecv.create ~min_feature_count:1
      ~step:(Recursive_feature_elimination.Count 1) ~splitter
      ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> fun specification ->
    Linear_rfecv.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  check_scores fixture "regression" (Linear_rfecv.cv_results fitted);
  Alcotest.(check (array int))
    "regression selected"
    (vector fixture "regression_selected" |> Array.map int_of_float)
    (Linear_rfecv.selected_indices fitted);
  Alcotest.(check (array int))
    "regression ranking"
    (vector fixture "regression_ranking" |> Array.map int_of_float)
    (Linear_rfecv.ranking fitted);
  check_vector "regression final importances"
    (vector fixture "regression_importances")
    (Linear_rfecv.final_importances fitted);
  check_matrix "regression output"
    (matrix fixture "regression_output")
    (Linear_rfecv.transform fitted ~metadata:Metadata.empty
       ~feature_schema:schema ~x
    |> get)

let test_classification fixture =
  let x = matrix fixture "classification_x" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture "classification_y"
    |> Array.map int_of_float |> Target.classification
  in
  let splitter =
    Stratified_k_fold.create ~folds:3 ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let fitted =
    Ridge_rfecv.create ~min_feature_count:1
      ~step:(Recursive_feature_elimination.Fraction 0.4) ~splitter
      ~scorer:Multiclass_classification_scorer.accuracy
      (Ridge_classifier.create ~alpha:0.5 () |> get)
    |> get
    |> fun specification ->
    Ridge_rfecv.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 0))
      ~feature_schema:schema ~x ~y:(Some y) ()
    |> get
  in
  check_scores fixture "classification" (Ridge_rfecv.cv_results fitted);
  Alcotest.(check (array int))
    "classification selected"
    (vector fixture "classification_selected" |> Array.map int_of_float)
    (Ridge_rfecv.selected_indices fitted);
  Alcotest.(check (array int))
    "classification ranking"
    (vector fixture "classification_ranking" |> Array.map int_of_float)
    (Ridge_rfecv.ranking fitted);
  check_vector "classification final importances"
    (vector fixture "classification_importances")
    (Ridge_rfecv.final_importances fitted);
  check_matrix "classification output"
    (matrix fixture "classification_output")
    (Ridge_rfecv.transform fitted ~metadata:Metadata.empty
       ~feature_schema:schema ~x
    |> get)

let () =
  let fixture = read_fixture () in
  Alcotest.run "Recursive feature elimination CV reference"
    [
      ( "sklearn",
        [
          ("regression RFECV", `Quick, fun () -> test_regression fixture);
          ("classification RFECV", `Quick, fun () -> test_classification fixture);
        ] );
    ]
