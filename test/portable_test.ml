open Modelkit

let fail message = raise (Failure message)
let check condition message = if not condition then fail message

let check_float expected observed message =
  if expected <> observed then
    fail (Format.asprintf "%s: expected %.17g, observed %.17g" message expected observed)

let check_int64 expected observed message =
  if not (Int64.equal expected observed) then
    fail
      (Format.asprintf "%s: expected %Ld, observed %Ld" message expected observed)

let get_ok = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let[@warning "-4"] check_length_mismatch ~name ~expected ~observed = function
  | Error error -> (
      match Error.kind error with
      | Error.Data
          (Data_error.Length_mismatch
            {
              name = observed_name;
              expected = observed_expected;
              observed = observed_observed;
            }) ->
          check (observed_name = name) "length error named the wrong operand";
          check (observed_expected = expected) "length error has the wrong expectation";
          check (observed_observed = observed) "length error has the wrong observation"
      | _ -> fail "dimension failure used the wrong error kind")
  | Ok _ -> fail "misaligned operands were accepted"

let test_stable_reductions () =
  let cancellation = Vector.of_array [| 1e16; 1.0; -1e16 |] in
  check_float 1.0 (Reference_backend.sum cancellation)
    "compensated sum lost a small term";
  let dot =
    get_ok
      (Reference_backend.dot
         (Vector.of_array [| 1e16; 1.0; 1e16 |])
         (Vector.of_array [| 1.0; 1.0; -1.0 |]))
  in
  check_float 1.0 dot "compensated dot product lost a small term";
  check_float 0.0 (Reference_backend.sum (Vector.of_array [||]))
    "empty sum is not zero";
  check
    (Float.is_nan
       (Reference_backend.sum
          (Vector.of_array [| Float.infinity; Float.neg_infinity |])))
    "opposing infinities did not produce NaN";
  check
    (Reference_backend.sum (Vector.of_array [| Float.infinity; 1.0 |])
    = Float.infinity)
    "positive infinity was not preserved";
  check
    (Float.is_nan (Reference_backend.sum (Vector.of_array [| Float.nan |])))
    "NaN was not preserved"

let test_matrix_kernels () =
  let matrix =
    Result.get_ok
      (Matrix.of_arrays
         [| [| 1e16; 1.0; -1e16 |]; [| 1.0; 2.0; 3.0 |] |])
  in
  let product =
    get_ok
      (Reference_backend.matrix_vector_product matrix
         (Vector.of_array [| 1.0; 1.0; 1.0 |]))
  in
  check (Vector.to_array product = [| 1.0; 6.0 |])
    "matrix-vector product is incorrect";
  let transposed_matrix =
    Result.get_ok
      (Matrix.of_arrays
         [| [| 1e16; 1.0 |]; [| 1.0; 2.0 |]; [| -1e16; 3.0 |] |])
  in
  let transposed_product =
    get_ok
      (Reference_backend.transposed_matrix_vector_product transposed_matrix
         (Vector.of_array [| 1.0; 1.0; 1.0 |]))
  in
  check (Vector.to_array transposed_product = [| 1.0; 6.0 |])
    "transposed matrix-vector product is incorrect";
  check_length_mismatch ~name:"dot-product right operand" ~expected:1 ~observed:0
    (Reference_backend.dot (Vector.of_array [| 1.0 |])
       (Vector.of_array [||]));
  check_length_mismatch ~name:"matrix-vector operand" ~expected:3 ~observed:1
    (Reference_backend.matrix_vector_product matrix
       (Vector.of_array [| 1.0 |]));
  check_length_mismatch ~name:"transposed-matrix-vector operand" ~expected:3
    ~observed:1
    (Reference_backend.transposed_matrix_vector_product transposed_matrix
       (Vector.of_array [| 1.0 |]))

let test_sequential_execution () =
  check (Sequential_execution.concurrency Sequential_execution.default = 1)
    "sequential concurrency is not one";
  let ordered =
    get_ok
      (Sequential_execution.map Sequential_execution.default
         ~f:(fun ~index value -> Ok (index, value))
         [| "a"; "b"; "c" |])
  in
  check (ordered = [| (0, "a"); (1, "b"); (2, "c") |])
    "sequential output order changed";
  check
    (get_ok
       (Sequential_execution.map Sequential_execution.default
          ~f:(fun ~index:_ value -> Ok value)
          [||])
    = [||])
    "sequential execution rejected empty work";
  let visited = ref [] in
  let failed =
    Sequential_execution.map Sequential_execution.default
      ~f:(fun ~index value ->
        visited := index :: !visited;
        if index = 1 then Error "stop" else Ok value)
      [| 10; 20; 30; 40 |]
  in
  check (failed = Error "stop") "sequential execution returned the wrong error";
  check (!visited = [ 1; 0 ]) "work continued after cancellation";
  let propagated =
    try
      ignore
        (Sequential_execution.map Sequential_execution.default
           ~f:(fun ~index:_ _ -> raise Exit)
           [| () |]);
      false
    with Exit -> true
  in
  check propagated "sequential execution swallowed a programmer exception"

let test_seed_derivation () =
  let root = Seed.of_int 42 in
  check (Seed.equal root (Seed.of_int64 42L)) "seed constructors disagree";
  let fold_zero = Seed.derive root ~operation:"cross_validate" ~index:0 in
  let fold_one = Seed.derive root ~operation:"cross_validate" ~index:1 in
  let candidate = Seed.derive root ~operation:"candidate" ~index:3 in
  let nested = Seed.derive candidate ~operation:"fold" ~index:7 in
  check_int64 (-655211088011813383L) (Seed.to_int64 fold_zero)
    "fold-zero seed changed";
  check_int64 (-3872047652920971549L) (Seed.to_int64 fold_one)
    "fold-one seed changed";
  check_int64 2923168321759935908L (Seed.to_int64 nested)
    "nested seed changed";
  check (not (Seed.equal fold_zero fold_one)) "logical indices share a seed";
  check
    (Seed.equal fold_zero
       (Seed.derive root ~operation:"cross_validate" ~index:0))
    "seed derivation depends on call order";
  let state = Rng.create root in
  let first, state = Rng.next_int64 state in
  let second, state = Rng.next_int64 state in
  let sample, _ = Rng.next_float state in
  check_int64 (-4767286540954276203L) first "first RNG output changed";
  check_int64 2949826092126892291L second "second RNG output changed";
  check_int64 4598690451703514086L (Int64.bits_of_float sample)
    "floating RNG output changed";
  check (sample >= 0.0 && sample < 1.0) "floating RNG output is out of range"

let () =
  test_stable_reductions ();
  test_matrix_kernels ();
  test_sequential_execution ();
  test_seed_derivation ()
