open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let dataset () =
  let x =
    Matrix.init ~rows:24 ~columns:2 (fun row column ->
        let value = Float.of_int row in
        if column = 0 then value else value *. value)
    |> get_data
  in
  let y =
    Target.regression
      (Vector.of_array
         (Array.init 24 (fun row ->
              let value = Float.of_int row in
              (0.25 *. value *. value) +. value +. 2.0)))
    |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let pipeline () =
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let splitter () =
  K_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let schedule () =
  Learning_curve.schedule ~shuffle:true
    [|
      Learning_curve.Fraction 0.25;
      Learning_curve.Fraction 0.5;
      Learning_curve.Fraction 1.0;
    |]
  |> get

let score = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (Error.to_string error)
  | None -> Alcotest.fail "expected score"

let signature execution =
  Learning_curve.Regression.evaluate ~return_indices:true ~execution
    ~schedule:(schedule ()) ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~seed:(Seed.of_int 404) (pipeline ()) (dataset ())
  |> get |> Learning_curve.points
  |> Array.map (fun point ->
      ( point.Learning_curve.training_samples,
        Cross_validation.folds point.Learning_curve.evaluation
        |> Array.map (fun fold ->
            ( Option.get fold.Cross_validation.train_indices,
              Option.get fold.Cross_validation.test_indices,
              score
                fold.Cross_validation.scores.(0).Cross_validation.train_score,
              score fold.Cross_validation.scores.(0).Cross_validation.test_score
            )) ))

let test_domain_count_invariance () =
  let expected = signature Execution.sequential in
  List.iter
    (fun domains ->
      let execution =
        Modelkit_parallel.create ~inner_threads:1 ~domains ()
        |> get |> Modelkit_parallel.execution
      in
      Alcotest.(check bool)
        "same curve report" true
        (expected = signature execution))
    [ 1; 2; 4 ]

let () =
  Alcotest.run "parallel learning curves"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
        ] );
    ]
