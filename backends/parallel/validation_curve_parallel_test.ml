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
              (0.25 *. value *. value) +. value +. Float.of_int (row mod 3))))
    |> get_data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let pipeline alpha =
  let ( let* ) = Result.bind in
  let* specification = Ridge_regression.create ~alpha () in
  let* estimator =
    Pipeline.estimator ~name:"ridge" (module Ridge_regression) specification
  in
  Pipeline.set_estimator Pipeline.empty estimator

let specification () =
  Validation_curve.create ~name:"alpha" ~base:1.0 ~values:[| 0.0; 0.5; 5.0 |]
    ~encode:(fun value -> Grid_search.Float value)
    ~set:(fun _ value -> Ok value)
    ~build:pipeline ()
  |> get

let splitter () =
  K_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let score = function
  | Some (Ok value) -> value
  | Some (Error error) -> Alcotest.fail (Error.to_string error)
  | None -> Alcotest.fail "expected score"

let signature execution =
  Validation_curve.Regression.evaluate ~return_indices:true ~execution
    ~specification:(specification ()) ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~seed:(Seed.of_int 505) (dataset ())
  |> get |> Validation_curve.points
  |> Array.map (fun point ->
      ( point.Validation_curve.point_index,
        point.Validation_curve.parameter_value,
        Cross_validation.folds (Option.get point.Validation_curve.evaluation)
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
  Alcotest.run "parallel validation curves"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
        ] );
    ]
