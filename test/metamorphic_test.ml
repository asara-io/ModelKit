open Modelkit

let get_ok = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let check_float message expected observed =
  Alcotest.check (Alcotest.float 1e-12) message expected observed

let test_sum_permutation () =
  let values = [| 2.0; -4.0; 8.0; 16.0; -1.0 |] in
  let reversed = [| -1.0; 16.0; 8.0; -4.0; 2.0 |] in
  check_float "reversing exact inputs preserves their sum"
    (Reference_backend.sum (Vector.of_array values))
    (Reference_backend.sum (Vector.of_array reversed))

let test_dot_scaling () =
  let left = Vector.of_array [| 1.0; -2.0; 3.0 |] in
  let scaled_left = Vector.of_array [| 4.0; -8.0; 12.0 |] in
  let right = Vector.of_array [| 5.0; 6.0; -7.0 |] in
  let original = get_ok (Reference_backend.dot left right) in
  let scaled = get_ok (Reference_backend.dot scaled_left right) in
  check_float "scaling an operand scales the dot product" (4.0 *. original)
    scaled

let test_row_permutation () =
  let matrix =
    Result.get_ok
      (Matrix.of_arrays [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |])
  in
  let permuted =
    Result.get_ok
      (Matrix.of_arrays [| [| 5.0; 6.0 |]; [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |])
  in
  let vector = Vector.of_array [| 2.0; -1.0 |] in
  let original =
    get_ok (Reference_backend.matrix_vector_product matrix vector)
    |> Vector.to_array
  in
  let observed =
    get_ok (Reference_backend.matrix_vector_product permuted vector)
    |> Vector.to_array
  in
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "row permutation permutes outputs"
    [| original.(2); original.(0); original.(1) |]
    observed

let test_transpose_identity () =
  let matrix =
    Result.get_ok
      (Matrix.of_arrays [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |]; [| 5.0; 6.0 |] |])
  in
  let x = Vector.of_array [| 0.5; -2.0 |] in
  let y = Vector.of_array [| 3.0; -1.0; 4.0 |] in
  let matrix_x = get_ok (Reference_backend.matrix_vector_product matrix x) in
  let transposed_matrix_y =
    get_ok (Reference_backend.transposed_matrix_vector_product matrix y)
  in
  let left = get_ok (Reference_backend.dot matrix_x y) in
  let right = get_ok (Reference_backend.dot x transposed_matrix_y) in
  check_float "<Xx,y> equals <x,X^T y>" left right

let test_preprocessing_row_permutation () =
  let original =
    Result.get_ok
      (Matrix.of_arrays
         [| [| 1.0; Float.nan |]; [| 3.0; 8.0 |]; [| 5.0; 4.0 |] |])
  in
  let permuted =
    Result.get_ok
      (Matrix.of_arrays
         [| [| 5.0; 4.0 |]; [| 1.0; Float.nan |]; [| 3.0; 8.0 |] |])
  in
  let schema = Result.get_ok (Feature_schema.of_matrix original) in
  let fit_imputer x =
    get_ok
      (Simple_imputer.fit (Simple_imputer.mean ())
         ~rng:(Rng.create (Seed.of_int 0))
         ~feature_schema:schema ~x ~y:None ())
  in
  let original_imputer = fit_imputer original in
  let permuted_imputer = fit_imputer permuted in
  Array.iter2
    (check_float "row permutation preserves imputation statistics")
    (Vector.to_array (Simple_imputer.statistics original_imputer))
    (Vector.to_array (Simple_imputer.statistics permuted_imputer));
  let complete_original =
    get_ok
      (Simple_imputer.transform original_imputer ~feature_schema:schema
         ~x:original)
  in
  let complete_permuted =
    get_ok
      (Simple_imputer.transform permuted_imputer ~feature_schema:schema
         ~x:permuted)
  in
  let fit_scaler x =
    get_ok
      (Standard_scaler.fit
         (Standard_scaler.create ())
         ~rng:(Rng.create (Seed.of_int 0))
         ~feature_schema:schema ~x ~y:None ())
  in
  let original_scaler = fit_scaler complete_original in
  let permuted_scaler = fit_scaler complete_permuted in
  Array.iter2
    (check_float "row permutation preserves scaling means")
    (Vector.to_array (Standard_scaler.mean original_scaler))
    (Vector.to_array (Standard_scaler.mean permuted_scaler));
  Array.iter2
    (check_float "row permutation preserves scaling variances")
    (Vector.to_array (Standard_scaler.variance original_scaler))
    (Vector.to_array (Standard_scaler.variance permuted_scaler))

let () =
  Alcotest.run "metamorphic invariants"
    [
      ( "reference backend",
        [
          Alcotest.test_case "sum permutation" `Quick test_sum_permutation;
          Alcotest.test_case "dot scaling" `Quick test_dot_scaling;
          Alcotest.test_case "row permutation" `Quick test_row_permutation;
          Alcotest.test_case "transpose identity" `Quick test_transpose_identity;
          Alcotest.test_case "preprocessing row permutation" `Quick
            test_preprocessing_row_permutation;
        ] );
    ]
