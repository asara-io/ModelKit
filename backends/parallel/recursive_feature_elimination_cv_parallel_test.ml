open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

module Linear_importance = struct
  include Linear_regression

  let feature_importances fitted =
    Feature_importance.absolute_coefficients (coefficients fitted)
end

module Rfecv =
  Recursive_feature_elimination_cv.Regression.Make (Linear_importance)

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

let signature execution =
  let x, feature_schema, y = dataset () in
  let splitter =
    K_fold.create ~folds:3 ~shuffle:true ()
    |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let fitted =
    Rfecv.create ~min_feature_count:1
      ~step:(Recursive_feature_elimination.Count 1) ~max_fits:20 ~execution
      ~splitter ~scorer:Regression_scorer.neg_mean_squared_error
      (Linear_regression.create ())
    |> get
    |> fun specification ->
    Rfecv.fit specification ~metadata:Metadata.empty
      ~rng:(Rng.create (Seed.of_int 904))
      ~feature_schema ~x ~y:(Some y) ()
    |> get
  in
  ( Rfecv.cv_results fitted,
    Rfecv.selected_indices fitted,
    Rfecv.ranking fitted,
    Vector.to_array (Rfecv.final_importances fitted),
    Rfecv.fit_count fitted )

let test_domain_count_invariance () =
  let expected = signature Execution.sequential in
  List.iter
    (fun domains ->
      let execution =
        Modelkit_parallel.create ~inner_threads:1 ~domains ()
        |> get |> Modelkit_parallel.execution
      in
      Alcotest.(check bool)
        "same recursive-elimination CV result" true
        (expected = signature execution))
    [ 1; 2; 4 ]

let () =
  Alcotest.run "parallel recursive feature elimination CV"
    [
      ( "execution",
        [
          Alcotest.test_case "domain-count invariance" `Quick
            test_domain_count_invariance;
        ] );
    ]
