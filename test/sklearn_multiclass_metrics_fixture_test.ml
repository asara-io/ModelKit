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
  let fixture = { vectors = Hashtbl.create 32; matrices = Hashtbl.create 8 } in
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

let scalar fixture name = (vector fixture name).(0)

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
    (Float.abs (expected -. observed) <= 1e-7 *. Float.max 1.0 scale)

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

let labels fixture name =
  vector fixture name |> Array.map int_of_float |> Target.classification

let setup fixture =
  let truth = labels fixture "truth" in
  let prediction = labels fixture "prediction" in
  let sample_weight =
    let values = vector fixture "sample_weight" in
    Sample_weight.of_array ~expected_length:(Array.length values) values
    |> get_data
  in
  (truth, prediction, sample_weight)

let averages =
  Multiclass_classification_metrics.
    [ ("micro", Micro); ("macro", Macro); ("weighted", Weighted) ]

let test_confusion fixture () =
  let truth, prediction, sample_weight = setup fixture in
  let confusion =
    Multiclass_classification_metrics.confusion_matrix ~sample_weight ~truth
      ~prediction ()
    |> get
  in
  Alcotest.(check (array int))
    "ascending labels" [| 0; 1; 2 |]
    confusion.Multiclass_classification_metrics.labels;
  check_matrix "weighted confusion"
    (matrix fixture "confusion_matrix")
    confusion.Multiclass_classification_metrics.counts;
  let reordered =
    Multiclass_classification_metrics.confusion_matrix ~labels:[| 2; 0; 1 |]
      ~truth ~prediction ()
    |> get
  in
  Alcotest.(check (array int))
    "explicit label order" [| 2; 0; 1 |]
    reordered.Multiclass_classification_metrics.labels;
  check_matrix "unweighted reordered confusion"
    (matrix fixture "unweighted_confusion_matrix")
    reordered.Multiclass_classification_metrics.counts

let test_scalars fixture () =
  let truth, prediction, sample_weight = setup fixture in
  check_float "accuracy"
    (scalar fixture "accuracy")
    (Multiclass_classification_metrics.accuracy ~sample_weight ~truth
       ~prediction ()
    |> get);
  check_float "balanced accuracy"
    (scalar fixture "balanced_accuracy")
    (Multiclass_classification_metrics.balanced_accuracy ~sample_weight ~truth
       ~prediction ()
    |> get);
  List.iter
    (fun (name, average) ->
      check_float ("precision " ^ name)
        (scalar fixture ("precision_" ^ name))
        (Multiclass_classification_metrics.precision ~average ~sample_weight
           ~truth ~prediction ()
        |> get);
      check_float ("recall " ^ name)
        (scalar fixture ("recall_" ^ name))
        (Multiclass_classification_metrics.recall ~average ~sample_weight ~truth
           ~prediction ()
        |> get);
      check_float ("f1 " ^ name)
        (scalar fixture ("f1_" ^ name))
        (Multiclass_classification_metrics.f1 ~average ~sample_weight ~truth
           ~prediction ()
        |> get))
    averages;
  let scores =
    Multiclass_classification_metrics.class_scores ~sample_weight ~truth
      ~prediction ()
    |> get
  in
  check_vector "class precision"
    (vector fixture "class_precision")
    (Vector.to_array scores.Multiclass_classification_metrics.precisions);
  check_vector "class recall"
    (vector fixture "class_recall")
    (Vector.to_array scores.Multiclass_classification_metrics.recalls);
  check_vector "class f1"
    (vector fixture "class_f1")
    (Vector.to_array scores.Multiclass_classification_metrics.f1_scores);
  check_vector "class support"
    (vector fixture "class_support")
    (Vector.to_array scores.Multiclass_classification_metrics.supports)

let test_zero_division fixture () =
  let truth = labels fixture "sparse_truth" in
  let prediction = labels fixture "sparse_prediction" in
  let undefined = Undefined_metric_policy.Use_fallback in
  check_float "sparse precision macro"
    (scalar fixture "sparse_precision_macro")
    (Multiclass_classification_metrics.precision ~undefined
       ~average:Multiclass_classification_metrics.Macro ~truth ~prediction ()
    |> get);
  check_float "sparse recall macro"
    (scalar fixture "sparse_recall_macro")
    (Multiclass_classification_metrics.recall ~undefined
       ~average:Multiclass_classification_metrics.Macro ~truth ~prediction ()
    |> get);
  check_float "sparse f1 weighted"
    (scalar fixture "sparse_f1_weighted")
    (Multiclass_classification_metrics.f1 ~undefined
       ~average:Multiclass_classification_metrics.Weighted ~truth ~prediction ()
    |> get);
  check_float "sparse balanced accuracy"
    (scalar fixture "sparse_balanced_accuracy")
    (Multiclass_classification_metrics.balanced_accuracy ~truth ~prediction ()
    |> get)

let test_log_loss fixture () =
  let truth, _, sample_weight = setup fixture in
  let probabilities =
    matrix fixture "probabilities" |> Matrix.of_arrays |> get_data
  in
  let classes = vector fixture "classes" |> Array.map int_of_float in
  check_float "log loss"
    (scalar fixture "log_loss")
    (Multiclass_classification_metrics.log_loss ~sample_weight ~truth ~classes
       ~probabilities ()
    |> get);
  let prediction =
    Multiclass_prediction.create ~classes ~probabilities () |> get
  in
  check_float "scorer negates log loss"
    (-.scalar fixture "log_loss")
    (Multiclass_classification_scorer.score
       Multiclass_classification_scorer.neg_log_loss ~sample_weight ~truth
       ~prediction ()
    |> get)

let () =
  let path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_MULTICLASS_METRICS_FIXTURE" with
    | Some path -> path
    | None ->
        Alcotest.fail "MODELKIT_SKLEARN_MULTICLASS_METRICS_FIXTURE is not set"
  in
  let fixture = read_fixture path in
  Alcotest.run "sklearn multiclass metric fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "confusion matrices" `Quick
            (test_confusion fixture);
          Alcotest.test_case "averaged and per-class scores" `Quick
            (test_scalars fixture);
          Alcotest.test_case "zero-division fallbacks" `Quick
            (test_zero_division fixture);
          Alcotest.test_case "log loss" `Quick (test_log_loss fixture);
        ] );
    ]
