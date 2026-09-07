open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let dataset () =
  let rows = 24 in
  let x =
    Matrix.init ~rows ~columns:2 (fun row column ->
        let value = Float.of_int row in
        if column = 0 then value else sin value)
    |> get_data
  in
  let y =
    Target.regression
      (Vector.of_array
         (Array.init rows (fun row ->
              let value = Float.of_int row in
              (1.75 *. value) +. cos value)))
    |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let pipeline () =
  let ( let* ) = Result.bind in
  let* specification = Ridge_regression.create ~alpha:0.5 () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator Pipeline.empty estimator

let splitter () =
  K_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let signature execution =
  let specification = Permutation_test.create ~permutations:16 () |> get in
  let report =
    Permutation_test.Regression.evaluate ~execution ~specification
      ~splitter:(splitter ()) ~scorer:Regression_scorer.neg_mean_squared_error
      ~seed:(Seed.of_int 505)
      (pipeline () |> get)
      (dataset ())
    |> get
  in
  ( Permutation_test.observed_score report,
    Permutation_test.permutation_scores report,
    Permutation_test.p_value report )

let test_domain_count_invariance () =
  let expected = signature Execution.sequential in
  List.iter
    (fun domains ->
      let execution =
        Modelkit_parallel.create ~inner_threads:1 ~domains ()
        |> get |> Modelkit_parallel.execution
      in
      Alcotest.(check bool)
        "same observed score, null scores, and p-value" true
        (expected = signature execution))
    [ 1; 2; 4 ]

let () =
  Alcotest.run "parallel permutation significance tests"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
        ] );
    ]
