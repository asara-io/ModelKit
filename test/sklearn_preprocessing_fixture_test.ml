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

type passthrough_fitted = { schema : Feature_schema.t }

module Passthrough_estimator :
  ESTIMATOR
    with type t = unit
     and type params = unit
     and type target = unit
     and type prediction = Matrix.t
     and type fitted = passthrough_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = unit
  type prediction = Matrix.t
  type fitted = passthrough_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:() () =
    match Feature_schema.validate_matrix feature_schema x with
    | Ok () -> Ok { schema = feature_schema }
    | Error error ->
        Error
          (Error.of_data_error ~remediation:"provide aligned fixture data" error)

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make ~remediation:"provide the fitted fixture schema"
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      match Feature_schema.validate_matrix feature_schema x with
      | Ok () -> Ok x
      | Error error ->
          Error
            (Error.of_data_error ~remediation:"provide aligned fixture data"
               error)

  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

let test_pipeline_fixture path () =
  let fixture = read_fixture path in
  let input = matrix fixture "input" |> Matrix.of_arrays |> get_data in
  let schema = Feature_schema.of_matrix input |> get_data in
  let imputer =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty imputer |> get in
  let builder = Pipeline.add_transformer builder scaler |> get in
  let estimator =
    Pipeline.estimator ~name:"passthrough" (module Passthrough_estimator) ()
    |> get
  in
  let specification = Pipeline.set_estimator builder estimator |> get in
  let fitted =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema ~x:input
      ~y:() ()
    |> get
  in
  let expected = matrix fixture "scaled_output" in
  check_matrix "pipeline transformed output" expected
    (Pipeline.transform fitted ~feature_schema:schema ~x:input |> get);
  check_matrix "pipeline prediction input" expected
    (Pipeline.predict fitted ~feature_schema:schema ~x:input |> get)

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

let test_transform_fixture path () =
  let fixture = read_fixture path in
  let numeric =
    matrix fixture "numeric_input" |> Matrix.of_arrays |> get_data
  in
  let numeric_schema = Feature_schema.of_matrix numeric |> get_data in
  let min_max =
    Min_max_scaler.fit
      (Min_max_scaler.create ~feature_range:(-1.0, 2.0) () |> get)
      ~rng:(rng ()) ~feature_schema:numeric_schema ~x:numeric ~y:None ()
    |> get
  in
  check_vector "min-max minimum"
    (vector fixture "min_max_data_min")
    (Min_max_scaler.data_min min_max |> Vector.to_array);
  check_vector "min-max maximum"
    (vector fixture "min_max_data_max")
    (Min_max_scaler.data_max min_max |> Vector.to_array);
  check_vector "min-max range"
    (vector fixture "min_max_data_range")
    (Min_max_scaler.data_range min_max |> Vector.to_array);
  check_vector "min-max scale"
    (vector fixture "min_max_scale")
    (Min_max_scaler.scale min_max |> Vector.to_array);
  check_vector "min-max offset"
    (vector fixture "min_max_offset")
    (Min_max_scaler.offset min_max |> Vector.to_array);
  check_matrix "min-max output"
    (matrix fixture "min_max_output")
    (Min_max_scaler.transform min_max ~feature_schema:numeric_schema ~x:numeric
    |> get);
  let max_abs =
    Max_abs_scaler.fit (Max_abs_scaler.create ()) ~rng:(rng ())
      ~feature_schema:numeric_schema ~x:numeric ~y:None ()
    |> get
  in
  check_vector "max-absolute maximum"
    (vector fixture "max_abs_max")
    (Max_abs_scaler.max_abs max_abs |> Vector.to_array);
  check_vector "max-absolute scale"
    (vector fixture "max_abs_scale")
    (Max_abs_scaler.scale max_abs |> Vector.to_array);
  check_matrix "max-absolute output"
    (matrix fixture "max_abs_output")
    (Max_abs_scaler.transform max_abs ~feature_schema:numeric_schema ~x:numeric
    |> get);
  let robust =
    Robust_scaler.fit
      (Robust_scaler.create () |> get)
      ~rng:(rng ()) ~feature_schema:numeric_schema ~x:numeric ~y:None ()
    |> get
  in
  check_vector "robust center"
    (vector fixture "robust_center")
    (Robust_scaler.center robust |> Vector.to_array);
  check_vector "robust scale"
    (vector fixture "robust_scale")
    (Robust_scaler.scale robust |> Vector.to_array);
  check_matrix "robust output"
    (matrix fixture "robust_output")
    (Robust_scaler.transform robust ~feature_schema:numeric_schema ~x:numeric
    |> get);
  let normalizer_input =
    matrix fixture "normalizer_input" |> Matrix.of_arrays |> get_data
  in
  let normalizer_schema =
    Feature_schema.of_matrix normalizer_input |> get_data
  in
  let check_normalizer name norm =
    let fitted =
      Normalizer.fit
        (Normalizer.create ~norm ())
        ~rng:(rng ()) ~feature_schema:normalizer_schema ~x:normalizer_input
        ~y:None ()
      |> get
    in
    check_matrix
      (name ^ " normalizer output")
      (matrix fixture ("normalizer_" ^ name ^ "_output"))
      (Normalizer.transform fitted ~feature_schema:normalizer_schema
         ~x:normalizer_input
      |> get)
  in
  check_normalizer "l1" Normalizer.L1;
  check_normalizer "l2" Normalizer.L2;
  check_normalizer "max" Normalizer.Max;
  let categorical =
    matrix fixture "categorical_input" |> Matrix.of_arrays |> get_data
  in
  let categorical_predict =
    matrix fixture "categorical_predict" |> Matrix.of_arrays |> get_data
  in
  let categorical_schema = Feature_schema.of_matrix categorical |> get_data in
  let one_hot =
    One_hot_encoder.fit
      (One_hot_encoder.create ~unknown_category:One_hot_encoder.Ignore () |> get)
      ~rng:(rng ()) ~feature_schema:categorical_schema ~x:categorical ~y:None ()
    |> get
  in
  Array.iteri
    (fun column observed ->
      check_vector
        (Format.sprintf "one-hot categories %d" column)
        (vector fixture (Format.sprintf "one_hot_categories_%d" column))
        (Vector.to_array observed))
    (One_hot_encoder.categories one_hot);
  check_matrix "one-hot output"
    (matrix fixture "one_hot_output")
    (One_hot_encoder.transform one_hot ~feature_schema:categorical_schema
       ~x:categorical_predict
    |> get);
  let ordinal =
    Ordinal_encoder.fit
      (Ordinal_encoder.create
         ~unknown_category:(Ordinal_encoder.Use_encoded_value (-1.0)) ()
      |> get)
      ~rng:(rng ()) ~feature_schema:categorical_schema ~x:categorical ~y:None ()
    |> get
  in
  Array.iteri
    (fun column observed ->
      check_vector
        (Format.sprintf "ordinal categories %d" column)
        (vector fixture (Format.sprintf "ordinal_categories_%d" column))
        (Vector.to_array observed))
    (Ordinal_encoder.categories ordinal);
  check_matrix "ordinal output"
    (matrix fixture "ordinal_output")
    (Ordinal_encoder.transform ordinal ~feature_schema:categorical_schema
       ~x:categorical_predict
    |> get);
  let labels =
    vector fixture "label_input"
    |> Array.map int_of_float |> Target.classification
  in
  let label = Label_encoder.fit (Label_encoder.create ()) ~y:labels |> get in
  Alcotest.(check (array int))
    "label classes"
    (vector fixture "label_classes" |> Array.map int_of_float)
    (Label_encoder.classes label);
  let encoded = Label_encoder.transform label labels |> get in
  Alcotest.(check (array int))
    "encoded labels"
    (vector fixture "label_encoded" |> Array.map int_of_float)
    (Target.classification_values encoded);
  Alcotest.(check (array int))
    "decoded labels"
    (vector fixture "label_decoded" |> Array.map int_of_float)
    (Label_encoder.inverse_transform label encoded
    |> get |> Target.classification_values);
  let polynomial =
    matrix fixture "polynomial_input" |> Matrix.of_arrays |> get_data
  in
  let polynomial_schema = Feature_schema.of_matrix polynomial |> get_data in
  let check_polynomial name specification =
    let fitted =
      Polynomial_features.fit specification ~rng:(rng ())
        ~feature_schema:polynomial_schema ~x:polynomial ~y:None ()
      |> get
    in
    check_matrix (name ^ " output")
      (matrix fixture (name ^ "_output"))
      (Polynomial_features.transform fitted ~feature_schema:polynomial_schema
         ~x:polynomial
      |> get)
  in
  check_polynomial "polynomial" (Polynomial_features.create () |> get);
  check_polynomial "interaction"
    (Polynomial_features.create ~interaction_only:true () |> get);
  let missing =
    matrix fixture "missing_input" |> Matrix.of_arrays |> get_data
  in
  let missing_schema = Feature_schema.of_matrix missing |> get_data in
  let missing_only =
    Missing_indicator.fit
      (Missing_indicator.create ~error_on_new:false ())
      ~rng:(rng ()) ~feature_schema:missing_schema ~x:missing ~y:None ()
    |> get
  in
  Alcotest.(check (array int))
    "missing-only features"
    (vector fixture "missing_only_features" |> Array.map int_of_float)
    (Missing_indicator.selected_features missing_only);
  check_matrix "missing-only output"
    (matrix fixture "missing_only_output")
    (Missing_indicator.transform missing_only ~feature_schema:missing_schema
       ~x:missing
    |> get);
  let missing_all =
    Missing_indicator.fit
      (Missing_indicator.create ~features:Missing_indicator.All ())
      ~rng:(rng ()) ~feature_schema:missing_schema ~x:missing ~y:None ()
    |> get
  in
  check_matrix "missing-all output"
    (matrix fixture "missing_all_output")
    (Missing_indicator.transform missing_all ~feature_schema:missing_schema
       ~x:missing
    |> get)

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_PREPROCESSING_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_PREPROCESSING_FIXTURE is not set"
  in
  let transform_fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_TRANSFORM_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_TRANSFORM_FIXTURE is not set"
  in
  Alcotest.run "sklearn preprocessing fixtures"
    [
      ( "preprocessing",
        [
          Alcotest.test_case "v1" `Quick (test_fixture fixture_path);
          Alcotest.test_case "pipeline v1" `Quick
            (test_pipeline_fixture fixture_path);
          Alcotest.test_case "additional transforms v1" `Quick
            (test_transform_fixture transform_fixture_path);
        ] );
    ]
