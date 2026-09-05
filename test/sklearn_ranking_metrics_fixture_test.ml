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
  |> Matrix.of_arrays
  |> function
  | Ok matrix -> matrix
  | Error error -> Alcotest.fail (Data_error.to_string error)

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

let labels fixture name =
  vector fixture name |> Array.map int_of_float |> Target.classification

let weights fixture name =
  let values = vector fixture name in
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let test_average_precision fixture () =
  check_float "average precision"
    (scalar fixture "average_precision")
    (Binary_classification_metrics.average_precision
       ~sample_weight:(weights fixture "binary_weight")
       ~truth:(labels fixture "binary_truth")
       ~positive_probabilities:
         (Vector.of_array (vector fixture "binary_probabilities"))
       ()
    |> get)

let test_multiclass_auc fixture () =
  let truth = labels fixture "truth" in
  let classes = vector fixture "classes" |> Array.map int_of_float in
  let probabilities = matrix fixture "probabilities" in
  let sample_weight = weights fixture "sample_weight" in
  let open Multiclass_classification_metrics in
  List.iter
    (fun (name, average) ->
      check_float ("ovr " ^ name)
        (scalar fixture ("roc_auc_ovr_" ^ name))
        (Multiclass_ranking.roc_auc ~strategy:Multiclass_ranking.One_vs_rest
           ~average ~sample_weight ~truth ~classes ~probabilities ()
        |> get))
    [ ("macro", Macro); ("weighted", Weighted); ("micro", Micro) ];
  List.iter
    (fun (name, average) ->
      check_float ("ovo " ^ name)
        (scalar fixture ("roc_auc_ovo_" ^ name))
        (Multiclass_ranking.roc_auc ~strategy:Multiclass_ranking.One_vs_one
           ~average ~truth ~classes ~probabilities ()
        |> get))
    [ ("macro", Macro); ("weighted", Weighted) ];
  List.iter
    (fun k ->
      check_float
        (Format.sprintf "top %d accuracy" k)
        (scalar fixture (Format.sprintf "top_%d_accuracy" k))
        (Multiclass_ranking.top_k_accuracy ~k ~sample_weight ~truth ~classes
           ~probabilities ()
        |> get))
    [ 1; 2 ]

let test_gains fixture () =
  let relevance = matrix fixture "relevance" in
  let scores = matrix fixture "ranking_scores" in
  let sample_weight = weights fixture "ranking_weight" in
  List.iter
    (fun (label, k) ->
      List.iter
        (fun (ties, ignore_ties) ->
          let suffix = Format.sprintf "_%s_%s" label ties in
          check_float ("dcg" ^ suffix)
            (scalar fixture ("dcg" ^ suffix))
            (Ranking_metrics.dcg ?k ~ignore_ties ~sample_weight ~relevance
               ~scores ()
            |> get);
          check_float ("ndcg" ^ suffix)
            (scalar fixture ("ndcg" ^ suffix))
            (Ranking_metrics.ndcg ?k ~ignore_ties ~sample_weight ~relevance
               ~scores ()
            |> get))
        [ ("averaged", false); ("ignored", true) ])
    [ ("all", None); ("3", Some 3) ]

let () =
  let path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_RANKING_METRICS_FIXTURE" with
    | Some path -> path
    | None ->
        Alcotest.fail "MODELKIT_SKLEARN_RANKING_METRICS_FIXTURE is not set"
  in
  let fixture = read_fixture path in
  Alcotest.run "sklearn ranking metric fixtures"
    [
      ( "parity",
        [
          Alcotest.test_case "average precision" `Quick
            (test_average_precision fixture);
          Alcotest.test_case "multiclass ROC AUC and top-k" `Quick
            (test_multiclass_auc fixture);
          Alcotest.test_case "discounted cumulative gain" `Quick
            (test_gains fixture);
        ] );
    ]
