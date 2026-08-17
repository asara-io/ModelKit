open Modelkit
open Grid_search

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let value ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.0
  -. 5.0

let target ~seed ~features row =
  let signal = ref 1.25 in
  for column = 0 to features - 1 do
    let coefficient = Float.of_int ((column mod 5) - 2) *. 0.2 in
    signal := !signal +. (coefficient *. value ~seed row column)
  done;
  let noise = Float.of_int ((((row * 13) + 1729) mod 11) - 5) *. 0.01 in
  !signal +. noise

type configuration = { alpha : float; fit_intercept : bool }

let pipeline configuration =
  let ( let* ) = Result.bind in
  let* ridge =
    Ridge_regression.create ~alpha:configuration.alpha
      ~fit_intercept:configuration.fit_intercept ()
  in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) ridge
  in
  Pipeline.set_estimator Pipeline.empty estimator

let score candidate index =
  match candidate.scores.(index).test with
  | Ok summary -> summary.Score_aggregation.mean
  | Error error -> fail (Error.to_string error)

let train_score candidate index =
  match candidate.scores.(index).train with
  | Some (Ok summary) -> summary.Score_aggregation.mean
  | Some (Error error) -> fail (Error.to_string error)
  | None -> fail "grid-search benchmark train score is missing"

let parameter_float candidate index =
  match candidate.parameters.(index).parameter_value with
  | Float value -> value
  | Bool _ | Int _ | String _ -> fail "expected a float grid parameter"

let parameter_bool candidate index =
  match candidate.parameters.(index).parameter_value with
  | Bool value -> value
  | Float _ | Int _ | String _ -> fail "expected a bool grid parameter"

let candidate_signature candidate =
  let rank = Option.value candidate.rank ~default:0 |> Float.of_int in
  [|
    parameter_float candidate 0;
    (if parameter_bool candidate 1 then 1.0 else 0.0);
    train_score candidate 0;
    score candidate 0;
    train_score candidate 1;
    score candidate 1;
    rank;
  |]

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: grid_search_worker SAMPLES FEATURES SEED FOLDS ALPHAS_CSV \
       FIT_INTERCEPT_CSV";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let folds = int_of_string Sys.argv.(4) in
  let alphas =
    String.split_on_char ',' Sys.argv.(5) |> List.map float_of_string
  in
  let intercepts =
    String.split_on_char ',' Sys.argv.(6)
    |> List.map (function
      | "true" -> true
      | "false" -> false
      | value -> fail ("invalid Boolean grid value: " ^ value))
  in
  if samples < folds || features < 1 then
    fail "grid-search benchmark dimensions are invalid";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (value ~seed) |> get_data
  in
  let y =
    Array.init samples (target ~seed ~features)
    |> Vector.of_array |> Target.regression |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let alpha =
    Grid_search.axis ~name:"alpha" ~values:(Array.of_list alphas)
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun configuration alpha -> Ok { configuration with alpha })
    |> get
  in
  let fit_intercept =
    Grid_search.axis ~name:"fit_intercept" ~values:(Array.of_list intercepts)
      ~encode:(fun value -> Grid_search.Bool value)
      ~set:(fun configuration fit_intercept ->
        Ok { configuration with fit_intercept })
    |> get
  in
  let grid =
    Grid_search.create
      ~base:{ alpha = 0.0; fit_intercept = true }
      ~build:pipeline [| alpha; fit_intercept |]
    |> get
  in
  let splitter =
    K_fold.create ~folds () |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let report =
    Grid_search.Regression.search ~return_train_score:true ~grid ~splitter
      ~scorers:
        [| Regression_scorer.neg_mean_squared_error; Regression_scorer.r2 () |]
      ~refit:"r2" ~seed:(Seed.of_int seed) dataset
    |> get
  in
  let candidates = Grid_search.candidates report in
  let selected = Grid_search.selection report |> get in
  let predictions =
    Pipeline.predict selected.selected_model
      ~feature_schema:(Dataset.feature_schema dataset)
      ~x:(Dataset.features dataset)
    |> get |> Target.regression_values |> Vector.to_array
  in
  let signature =
    Array.concat
      [
        Array.concat (Array.to_list (Array.map candidate_signature candidates));
        [|
          Float.of_int selected.selected_candidate_index;
          predictions.(0);
          predictions.(samples - 1);
        |];
      ]
  in
  let signature_text =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"candidates":%d,"checksum":%S,"features":%d,"folds":%d,"ocaml":%S,"operations":["finite_grid_expansion","cross_validate","candidate_ranking","best_model_refit"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (Array.length candidates) (checksum signature) features
    folds Sys.ocaml_version samples signature_text
