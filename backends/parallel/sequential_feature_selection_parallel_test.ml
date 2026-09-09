open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

module Selector =
  Sequential_feature_selection.Regression.Make (Linear_regression)

let dataset () =
  let x =
    Matrix.init ~rows:24 ~columns:4 (fun row column ->
        let value = Float.of_int (row + 1) in
        match column with
        | 0 -> value
        | 1 -> Float.sin value
        | 2 -> Float.of_int (row * 7 mod 5)
        | _ -> Float.cos (value *. 0.5))
    |> get_data
  in
  let y =
    Array.init 24 (fun row ->
        let value = Float.of_int (row + 1) in
        (3.0 *. value)
        +. (0.25 *. Float.sin value)
        +. (0.01 *. Float.of_int (row mod 3)))
    |> Vector.of_array |> Target.regression |> get_data
  in
  (x, Feature_schema.of_matrix x |> get_data, y)

let signature direction execution =
  let x, feature_schema, y = dataset () in
  let splitter =
    K_fold.create ~folds:3 ~shuffle:true ()
    |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let fitted =
    Selector.create ~direction ~feature_count:2 ~max_fits:21 ~execution
      ~splitter ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> fun specification ->
    Selector.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 906))
      ~feature_schema ~x ~y:(Some y) ()
    |> get
  in
  (Selector.selected_indices fitted, Selector.fit_count fitted)

let test_domain_count_invariance () =
  List.iter
    (fun direction ->
      let expected = signature direction Execution.sequential in
      List.iter
        (fun domains ->
          let execution =
            Modelkit_parallel.create ~inner_threads:1 ~domains ()
            |> get |> Modelkit_parallel.execution
          in
          Alcotest.(check bool)
            "same sequential-selection result" true
            (expected = signature direction execution))
        [ 1; 2; 4 ])
    [
      Sequential_feature_selection.Forward;
      Sequential_feature_selection.Backward;
    ]

let () =
  Alcotest.run "parallel sequential feature selection"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
        ] );
    ]
