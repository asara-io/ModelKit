open Modelkit
open Evaluation_metadata_support

let test_fixture () =
  let fields =
    In_channel.with_open_text (Sys.getenv "MODELKIT_RANDOMIZED_FIXTURE")
      (fun channel ->
        In_channel.input_lines channel
        |> List.filter_map (fun line ->
            match String.split_on_char '\t' line with
            | [ name; values ] ->
                Some
                  ( name,
                    String.split_on_char ',' values
                    |> List.map float_of_string |> Array.of_list )
            | _ -> None))
  in
  let values name = List.assoc name fields in
  let matrix name =
    values name
    |> Array.map (fun value -> [| value |])
    |> Matrix.of_arrays |> data
  in
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix "x")
      ~y:(regression (values "y"))
      ()
    |> data
  in
  let build fit_intercept =
    let* estimator =
      Pipeline.estimator ~name:"linear"
        (module Linear_regression)
        (Linear_regression.create ~fit_intercept ())
    in
    Pipeline.set_estimator Pipeline.empty estimator
  in
  let axis =
    Randomized_search.axis ~name:"fit_intercept"
      ~distribution:(Parameter_distribution.choice [| false; true |] |> get)
      ~encode:(fun value -> Grid_search.Bool value)
      ~set:(fun _ value -> Ok value)
    |> get
  in
  let space =
    Randomized_search.create ~iterations:2 ~base:true ~build [| axis |] |> get
  in
  let splitter =
    Cross_validation.target_independent_splitter
      (module K_fold)
      (K_fold.create ~folds:3 () |> get)
  in
  let report =
    Randomized_search.Regression.search ~space ~splitter
      ~return_train_score:true
      ~scorers:
        [|
          Regression_scorer.neg_mean_squared_error;
          Regression_scorer.neg_mean_absolute_error;
        |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 19) source
    |> get
  in
  let candidates = Randomized_search.candidates report in
  Alcotest.(check int)
    "exhaustive finite candidate set" 2 (Array.length candidates);
  let intercept candidate =
    match candidate.Grid_search.parameters.(0).Grid_search.parameter_value with
    | Grid_search.Bool value -> value
    | Grid_search.Int _ | Grid_search.Float _ | Grid_search.String _ ->
        Alcotest.fail "invalid parameter"
  in
  Array.iter
    (fun candidate ->
      let prefix =
        if intercept candidate then "intercept" else "no_intercept"
      in
      Array.iter
        (fun score ->
          let compare partition summary =
            let key =
              prefix ^ "_" ^ partition ^ "_" ^ score.Grid_search.scorer_name
            in
            Alcotest.check (Alcotest.float 1e-10) key
              (values key).(0)
              (get summary).Score_aggregation.mean
          in
          compare "train" (Option.get score.Grid_search.train);
          compare "test" score.Grid_search.test)
        candidate.Grid_search.scores)
    candidates;
  let selected = Randomized_search.selection report |> get in
  Alcotest.(check bool)
    "named refit selects matching configuration"
    ((values "selected_intercept").(0) = 1.)
    (intercept candidates.(selected.Grid_search.selected_candidate_index));
  let prediction =
    Pipeline.predict selected.Grid_search.selected_model
      ~feature_schema:(Dataset.feature_schema source)
      ~x:(matrix "test_x")
    |> get
  in
  Alcotest.(check (array (Alcotest.float 1e-10)))
    "full-data refit predictions" (values "prediction")
    (Target.regression_values prediction |> Vector.to_array)

let () =
  Alcotest.run "Randomized search reference"
    [
      ("sklearn", [ ("finite search scores and refit", `Quick, test_fixture) ]);
    ]
