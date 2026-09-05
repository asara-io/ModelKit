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
  let fixture = { vectors = Hashtbl.create 64; matrices = Hashtbl.create 32 } in
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

let check_float label expected observed =
  let scale = Float.max (Float.abs expected) (Float.abs observed) in
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed) <= 1e-12 *. Float.max 1.0 scale)

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

type case = { name : string; target : string; loss : Sgd_classifier.loss }

let cases =
  [
    { name = "binary_hinge"; target = "binary"; loss = Sgd_classifier.Hinge };
    { name = "binary_log"; target = "binary"; loss = Sgd_classifier.Log_loss };
    {
      name = "multiclass_hinge";
      target = "multiclass";
      loss = Sgd_classifier.Hinge;
    };
    {
      name = "multiclass_log";
      target = "multiclass";
      loss = Sgd_classifier.Log_loss;
    };
  ]

let setup fixture case =
  let x = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let y =
    vector fixture (case.target ^ "_target")
    |> Array.map int_of_float |> Target.classification
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  let epochs = int_of_float (vector fixture "epochs").(0) in
  let specification =
    Sgd_classifier.create ~loss:case.loss ~penalty:Sgd_classifier.No_penalty
      ~learning_rate:Sgd_classifier.Constant
      ~eta0:(vector fixture "eta0").(0)
      ~max_epochs:epochs ~shuffle:false ()
    |> get
  in
  (x, x_predict, feature_schema, y, sample_weight, epochs, specification)

let check_model fixture case prefix fitted feature_schema x_predict =
  let prefix = case.name ^ "_" ^ prefix in
  Alcotest.(check (array int))
    (prefix ^ " classes")
    (vector fixture (case.name ^ "_classes") |> Array.map int_of_float)
    (Sgd_classifier.classes fitted);
  check_matrix (prefix ^ " coefficients")
    (matrix fixture (prefix ^ "_coefficients"))
    (Sgd_classifier.coefficients fitted);
  check_vector (prefix ^ " intercepts")
    (vector fixture (prefix ^ "_intercepts"))
    (Sgd_classifier.intercepts fitted |> Vector.to_array);
  check_matrix (prefix ^ " decisions")
    (matrix fixture (prefix ^ "_decisions"))
    (Sgd_classifier.decision_function fitted ~feature_schema ~x:x_predict |> get);
  (match case.loss with
  | Sgd_classifier.Log_loss ->
      check_matrix
        (prefix ^ " probabilities")
        (matrix fixture (prefix ^ "_probabilities"))
        (Sgd_classifier.predict_proba fitted ~feature_schema ~x:x_predict |> get)
  | Sgd_classifier.Hinge -> ());
  Alcotest.(check (array int))
    (prefix ^ " predictions")
    (vector fixture (prefix ^ "_predictions") |> Array.map int_of_float)
    (Sgd_classifier.predict fitted ~feature_schema ~x:x_predict
    |> get |> Target.classification_values)

let test_fit fixture case () =
  let x, x_predict, feature_schema, y, sample_weight, _, specification =
    setup fixture case
  in
  let fitted =
    Sgd_classifier.fit specification ~sample_weight
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x ~y ()
    |> get
  in
  check_model fixture case "fit" fitted feature_schema x_predict

let test_partial_fit fixture case () =
  let x, x_predict, feature_schema, y, sample_weight, epochs, specification =
    setup fixture case
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_classifier.partial_fit checkpoint ~sample_weight ~feature_schema ~x ~y
        ()
      |> get
      |> train (remaining - 1)
  in
  let fitted =
    Sgd_classifier.start specification
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema
      ~classes:
        (vector fixture (case.name ^ "_classes") |> Array.map int_of_float)
    |> get |> train epochs |> Sgd_classifier.to_fitted |> get
  in
  check_model fixture case "partial" fitted feature_schema x_predict

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_SGD_CLASSIFIER_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_SGD_CLASSIFIER_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn SGD-classifier fixture"
    (List.map
       (fun case ->
         ( case.name,
           [
             Alcotest.test_case "fit" `Quick (test_fit fixture case);
             Alcotest.test_case "partial_fit" `Quick
               (test_partial_fit fixture case);
           ] ))
       cases)
