open Modelkit

module type CONFIG = sig
  module Backend : NUMERICAL_BACKEND

  val absolute_tolerance : float
  val relative_tolerance : float
end

module Make (Config : CONFIG) = struct
  module Backend = Config.Backend

  let close expected observed =
    if Float.is_nan expected then Float.is_nan observed
    else if Float.is_infinite expected then expected = observed
    else
      let difference = Float.abs (expected -. observed) in
      let scale = Float.max (Float.abs expected) (Float.abs observed) in
      difference
      <= Config.absolute_tolerance +. (Config.relative_tolerance *. scale)

  let check_float message expected observed =
    if not (close expected observed) then
      Alcotest.failf "%s: expected %.17g, observed %.17g" message expected
        observed

  let get_ok = function
    | Ok value -> value
    | Error error -> Alcotest.fail (Error.to_string error)

  let check_length_mismatch ~expected ~observed = function
    | Error error -> (
        match Error.kind error with
        | Error.Data
            (Data_error.Length_mismatch
               { expected = actual_expected; observed = actual_observed; _ }) ->
            Alcotest.(check int) "expected dimension" expected actual_expected;
            Alcotest.(check int) "observed dimension" observed actual_observed
        | Error.Data
            ( Data_error.Negative_dimension _ | Data_error.Ragged_matrix _
            | Data_error.Index_out_of_bounds _ | Data_error.Non_finite _
            | Data_error.Negative_weight _ | Data_error.All_zero_weights
            | Data_error.Empty_feature_name _
            | Data_error.Duplicate_feature_name _
            | Data_error.Csr_row_offset_mismatch _
            | Data_error.Invalid_csr_row_offset _
            | Data_error.Invalid_csr_column_order _ )
        | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
        | Error.Validation _ | Error.Numerical _ | Error.Convergence _
        | Error.Compatibility _ | Error.Artifact _ | Error.Callback_failure _
        | Error.Cancelled ->
            Alcotest.fail "dimension failure used the wrong error kind")
    | Ok _ -> Alcotest.fail "misaligned operands were accepted"

  let check_vector message expected observed =
    let observed = Vector.to_array observed in
    Alcotest.(check int)
      (message ^ " length") (Array.length expected) (Array.length observed);
    Array.iteri
      (fun index expected ->
        check_float
          (Format.asprintf "%s at index %d" message index)
          expected observed.(index))
      expected

  let test_sum () =
    check_float "empty sum" 0.0 (Backend.sum (Vector.of_array [||]));
    check_float "ordinary sum" 6.0
      (Backend.sum (Vector.of_array [| 1.0; 2.0; 3.0 |]));
    check_float "cancellation-safe sum" 1.0
      (Backend.sum (Vector.of_array [| 1e16; 1.0; -1e16 |]));
    check_float "NaN propagation" Float.nan
      (Backend.sum (Vector.of_array [| 1.0; Float.nan |]));
    check_float "infinity propagation" Float.infinity
      (Backend.sum (Vector.of_array [| Float.infinity; 1.0 |]))

  let test_dot () =
    let observed =
      get_ok
        (Backend.dot
           (Vector.of_array [| 1.0; 2.0; 3.0 |])
           (Vector.of_array [| 4.0; 5.0; 6.0 |]))
    in
    check_float "dot product" 32.0 observed;
    check_length_mismatch ~expected:1 ~observed:0
      (Backend.dot (Vector.of_array [| 1.0 |]) (Vector.of_array [||]))

  let test_matrix_vector_product () =
    let matrix =
      Result.get_ok
        (Matrix.of_arrays [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |])
    in
    let observed =
      get_ok
        (Backend.matrix_vector_product matrix
           (Vector.of_array [| 1.0; 1.0; 1.0 |]))
    in
    check_vector "matrix-vector product" [| 6.0; 15.0 |] observed;
    check_length_mismatch ~expected:3 ~observed:1
      (Backend.matrix_vector_product matrix (Vector.of_array [| 1.0 |]))

  let test_transposed_matrix_vector_product () =
    let matrix =
      Result.get_ok
        (Matrix.of_arrays [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |])
    in
    let observed =
      get_ok
        (Backend.transposed_matrix_vector_product matrix
           (Vector.of_array [| 2.0; -1.0 |]))
    in
    check_vector "transposed matrix-vector product" [| -2.0; -1.0; 0.0 |]
      observed;
    check_length_mismatch ~expected:2 ~observed:1
      (Backend.transposed_matrix_vector_product matrix
         (Vector.of_array [| 1.0 |]))

  let test_feature_matrix_dispatch () =
    let dense =
      Result.get_ok
        (Matrix.of_arrays [| [| 1.0; 0.0; 3.0 |]; [| 0.0; -2.0; 0.0 |] |])
    in
    let csr = Csr_matrix.of_dense dense in
    let operand = Vector.of_array [| 2.0; 3.0; -1.0 |] in
    let dense_product =
      get_ok
        (Backend.feature_matrix_vector_product
           (Feature_matrix.dense dense)
           operand)
    in
    let csr_product =
      get_ok
        (Backend.feature_matrix_vector_product (Feature_matrix.csr csr) operand)
    in
    check_vector "dense feature-matrix product" [| -1.0; -6.0 |] dense_product;
    check_vector "CSR feature-matrix product" [| -1.0; -6.0 |] csr_product;
    let transposed_operand = Vector.of_array [| 2.0; -1.0 |] in
    let dense_transposed =
      get_ok
        (Backend.transposed_feature_matrix_vector_product
           (Feature_matrix.dense dense)
           transposed_operand)
    in
    let csr_transposed =
      get_ok
        (Backend.transposed_feature_matrix_vector_product
           (Feature_matrix.csr csr) transposed_operand)
    in
    check_vector "dense transposed feature-matrix product" [| 2.0; 2.0; 6.0 |]
      dense_transposed;
    check_vector "CSR transposed feature-matrix product" [| 2.0; 2.0; 6.0 |]
      csr_transposed;
    check_length_mismatch ~expected:3 ~observed:1
      (Backend.feature_matrix_vector_product (Feature_matrix.csr csr)
         (Vector.of_array [| 1.0 |]));
    check_length_mismatch ~expected:2 ~observed:1
      (Backend.transposed_feature_matrix_vector_product (Feature_matrix.csr csr)
         (Vector.of_array [| 1.0 |]))

  let tests =
    [
      Alcotest.test_case "stable reductions" `Quick test_sum;
      Alcotest.test_case "dot product" `Quick test_dot;
      Alcotest.test_case "matrix-vector product" `Quick
        test_matrix_vector_product;
      Alcotest.test_case "transposed matrix-vector product" `Quick
        test_transposed_matrix_vector_product;
      Alcotest.test_case "dense and CSR feature-matrix dispatch" `Quick
        test_feature_matrix_dispatch;
    ]
end
