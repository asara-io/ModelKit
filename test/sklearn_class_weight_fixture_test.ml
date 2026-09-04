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
  let fixture = { vectors = Hashtbl.create 32; matrices = Hashtbl.create 16 } in
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

let check_float ?(tolerance = 1e-7) label expected observed =
  let scale = Float.max (Float.abs expected) (Float.abs observed) in
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed) <= tolerance *. Float.max 1.0 scale)

let check_vector ?tolerance label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index expected ->
      check_float ?tolerance
        (Format.sprintf "%s[%d]" label index)
        expected observed.(index))
    expected

let check_matrix ?tolerance label expected observed =
  Alcotest.(check int)
    (label ^ " rows") (Array.length expected) (Matrix.rows observed);
  Array.iteri
    (fun row values ->
      Array.iteri
        (fun column expected ->
          check_float ?tolerance
            (Format.sprintf "%s[%d,%d]" label row column)
            expected
            (Matrix.get observed row column))
        values)
    expected

let setup fixture =
  let x = matrix fixture "x_train" |> Matrix.of_arrays |> get_data in
  let x_predict = matrix fixture "x_predict" |> Matrix.of_arrays |> get_data in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let labels name =
    vector fixture name |> Array.map int_of_float |> Target.classification
  in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  let explicit =
    Class_weight.explicit
      (Array.to_list
         (Array.map2
            (fun label weight -> (int_of_float label, weight))
            (vector fixture "explicit_classes")
            (vector fixture "explicit_weights")))
    |> get
  in
  ( x,
    x_predict,
    feature_schema,
    labels "binary_target",
    labels "multiclass_target",
    sample_weight,
    explicit )

let test_class_weights fixture () =
  let _, _, _, binary, _, sample_weight, _ = setup fixture in
  check_vector ~tolerance:1e-12 "weighted balanced class weights"
    (vector fixture "balanced_class_weights")
    (Class_weight.class_weights Class_weight.balanced ~sample_weight binary
    |> get |> Array.map snd);
  check_vector ~tolerance:1e-12 "unweighted balanced class weights"
    (vector fixture "unweighted_balanced_class_weights")
    (Class_weight.class_weights Class_weight.balanced binary
    |> get |> Array.map snd)

let logistic fixture =
  Logistic_regression.create
    ~c:(vector fixture "c").(0)
    ~tolerance:(vector fixture "tolerance").(0)
    ~max_iterations:(int_of_float (vector fixture "max_iterations").(0))
    ()
  |> get

let test_binary_logistic fixture () =
  let x, x_predict, feature_schema, binary, _, sample_weight, explicit =
    setup fixture
  in
  List.iter
    (fun (prefix, class_weight) ->
      let pipeline =
        Pipeline.set_estimator Pipeline.empty
          (Pipeline.classifier ~class_weight ~name:"logistic"
             ~predict_proba:Logistic_regression.predict_proba
             (module Logistic_regression)
             (logistic fixture)
          |> get)
        |> get
      in
      let fitted =
        Pipeline.fit pipeline ~sample_weight
          ~rng:(Rng.create (Seed.of_int 1729))
          ~feature_schema ~x ~y:binary ()
        |> get
      in
      check_matrix
        (prefix ^ " logistic probabilities")
        (matrix fixture (prefix ^ "_logistic_probabilities"))
        (Pipeline.predict_proba fitted ~feature_schema ~x:x_predict |> get);
      let direct =
        Logistic_regression.fit (logistic fixture)
          ~sample_weight:
            (Class_weight.resolve class_weight ~sample_weight binary |> get)
          ~rng:(Rng.create (Seed.of_int 1729))
          ~feature_schema ~x ~y:binary ()
        |> get
      in
      check_matrix
        (prefix ^ " logistic coefficients")
        (matrix fixture (prefix ^ "_logistic_coefficients"))
        (Matrix.of_arrays
           [| Logistic_regression.coefficients direct |> Vector.to_array |]
        |> get_data);
      check_vector
        (prefix ^ " logistic intercept")
        (vector fixture (prefix ^ "_logistic_intercept"))
        [| Logistic_regression.intercept direct |])
    [ ("balanced", Class_weight.balanced); ("explicit", explicit) ]

let test_multinomial fixture () =
  let x, x_predict, feature_schema, _, multiclass, sample_weight, _ =
    setup fixture
  in
  let specification =
    Multinomial_logistic_regression.create
      ~c:(vector fixture "c").(0)
      ~tolerance:(vector fixture "tolerance").(0)
      ~max_iterations:(int_of_float (vector fixture "max_iterations").(0))
      ()
    |> get
  in
  let pipeline =
    Pipeline.set_estimator Pipeline.empty
      (Pipeline.classifier ~class_weight:Class_weight.balanced
         ~name:"multinomial"
         ~predict_proba:Multinomial_logistic_regression.predict_proba
         (module Multinomial_logistic_regression)
         specification
      |> get)
    |> get
  in
  let fitted =
    Pipeline.fit pipeline ~sample_weight
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x ~y:multiclass ()
    |> get
  in
  check_matrix "balanced multinomial probabilities"
    (matrix fixture "balanced_multinomial_probabilities")
    (Pipeline.predict_proba fitted ~feature_schema ~x:x_predict |> get);
  let direct =
    Multinomial_logistic_regression.fit specification
      ~sample_weight:
        (Class_weight.resolve Class_weight.balanced ~sample_weight multiclass
        |> get)
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x ~y:multiclass ()
    |> get
  in
  check_matrix "balanced multinomial coefficients"
    (matrix fixture "balanced_multinomial_coefficients")
    (Multinomial_logistic_regression.coefficients direct);
  check_vector "balanced multinomial intercepts"
    (vector fixture "balanced_multinomial_intercepts")
    (Multinomial_logistic_regression.intercepts direct |> Vector.to_array)

let test_ridge fixture () =
  let x, x_predict, feature_schema, _, multiclass, _, _ = setup fixture in
  let fitted =
    Ridge_classifier.fit
      (Ridge_classifier.create ~alpha:(vector fixture "alpha").(0) () |> get)
      ~sample_weight:
        (Class_weight.resolve Class_weight.balanced multiclass |> get)
      ~rng:(Rng.create (Seed.of_int 1729))
      ~feature_schema ~x ~y:multiclass ()
    |> get
  in
  check_matrix "balanced ridge coefficients"
    (matrix fixture "balanced_ridge_coefficients")
    (Ridge_classifier.coefficients fitted);
  check_vector "balanced ridge intercepts"
    (vector fixture "balanced_ridge_intercepts")
    (Ridge_classifier.intercepts fitted |> Vector.to_array);
  check_matrix "balanced ridge decisions"
    (matrix fixture "balanced_ridge_decisions")
    (Ridge_classifier.decision_function fitted ~feature_schema ~x:x_predict
    |> get)

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_CLASS_WEIGHT_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_CLASS_WEIGHT_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn class-weight fixture"
    [
      ( "parity",
        [
          Alcotest.test_case "class weights" `Quick (test_class_weights fixture);
          Alcotest.test_case "binary logistic" `Quick
            (test_binary_logistic fixture);
          Alcotest.test_case "multinomial logistic" `Quick
            (test_multinomial fixture);
          Alcotest.test_case "ridge classifier" `Quick (test_ridge fixture);
        ] );
    ]
