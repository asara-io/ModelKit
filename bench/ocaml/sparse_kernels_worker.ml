open Modelkit

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let cell_value ~seed row column =
  ((Float.of_int (((row * 17) + (column * 31) + seed) mod 1000) +. 0.5) /. 100.0)
  -. 5.0

let stored ~seed ~threshold row column =
  ((row * 101) + (column * 53) + seed) mod 10000 < threshold

type timing = { elapsed_ns : int64; allocated_words : float }

let timed ~repeats f =
  let allocated_before = Gc.allocated_bytes () in
  let started = Unix.gettimeofday () in
  let result = ref (f ()) in
  for _ = 2 to repeats do
    result := f ()
  done;
  let elapsed = Unix.gettimeofday () -. started in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  (!result, { elapsed_ns = Int64.of_float (elapsed *. 1e9); allocated_words })

let timing_json { elapsed_ns; allocated_words } =
  Printf.sprintf {|{"allocated_words":%.0f,"elapsed_ns":%Ld}|} allocated_words
    elapsed_ns

let vector_sum vector =
  let total = ref 0.0 in
  for index = 0 to Vector.length vector - 1 do
    total := !total +. Vector.get vector index
  done;
  !total

let build_csr ~seed ~rows ~columns ~density =
  let threshold = Float.to_int (Float.round (density *. 10000.0)) in
  let stored = stored ~seed ~threshold in
  let row_offsets = Array.make (rows + 1) 0 in
  for row = 0 to rows - 1 do
    let count = ref 0 in
    for column = 0 to columns - 1 do
      if stored row column then incr count
    done;
    row_offsets.(row + 1) <- row_offsets.(row) + !count
  done;
  let nonzeros = row_offsets.(rows) in
  let column_indices = Array.make nonzeros 0 in
  let values = Array.make nonzeros 0.0 in
  let position = ref 0 in
  for row = 0 to rows - 1 do
    for column = 0 to columns - 1 do
      if stored row column then (
        column_indices.(!position) <- column;
        values.(!position) <- cell_value ~seed row column;
        incr position)
    done
  done;
  Csr_matrix.of_arrays ~rows ~columns ~row_offsets ~column_indices ~values
  |> get_data

let memory_json (memory : Matrix_memory.t) =
  Printf.sprintf {|{"dense_equivalent_bytes":%s,"total_bytes":%Ld}|}
    (match memory.Matrix_memory.dense_equivalent_bytes with
    | Some bytes -> Int64.to_string bytes
    | None -> "null")
    memory.Matrix_memory.total_bytes

let density_case ~seed ~rows ~columns ~repeats ~operand ~transposed_operand
    density =
  let csr = build_csr ~seed ~rows ~columns ~density in
  let dense = Csr_matrix.to_dense csr in
  let csr_features = Feature_matrix.csr csr in
  let dense_features = Feature_matrix.dense dense in
  let csr_product, csr_timing =
    timed ~repeats (fun () ->
        Reference_backend.feature_matrix_vector_product csr_features operand
        |> get)
  in
  let dense_product, dense_timing =
    timed ~repeats (fun () ->
        Reference_backend.feature_matrix_vector_product dense_features operand
        |> get)
  in
  let csr_transposed, csr_transposed_timing =
    timed ~repeats (fun () ->
        Reference_backend.transposed_feature_matrix_vector_product csr_features
          transposed_operand
        |> get)
  in
  let dense_transposed, dense_transposed_timing =
    timed ~repeats (fun () ->
        Reference_backend.transposed_feature_matrix_vector_product
          dense_features transposed_operand
        |> get)
  in
  let signature =
    [|
      Float.of_int (Csr_matrix.nonzero_count csr);
      vector_sum csr_product;
      vector_sum dense_product;
      vector_sum csr_transposed;
      vector_sum dense_transposed;
    |]
  in
  let json =
    Printf.sprintf
      {|{"csr_memory":%s,"csr_product":%s,"csr_transposed_product":%s,"dense_product":%s,"dense_transposed_product":%s,"density":%.17g,"nonzeros":%d}|}
      (memory_json (Csr_matrix.memory csr))
      (timing_json csr_timing)
      (timing_json csr_transposed_timing)
      (timing_json dense_timing)
      (timing_json dense_transposed_timing)
      density
      (Csr_matrix.nonzero_count csr)
  in
  (csr, signature, json)

let weighted_column_sum_csr csr =
  let total = ref 0.0 in
  for row = 0 to Csr_matrix.rows csr - 1 do
    Csr_matrix.iter_row csr ~row ~f:(fun ~column ~value ->
        total := !total +. (Float.of_int column *. value))
  done;
  !total

let weighted_column_sum_dense matrix =
  let total = ref 0.0 in
  let rows, columns = Matrix.shape matrix in
  for row = 0 to rows - 1 do
    for column = 0 to columns - 1 do
      total := !total +. (Float.of_int column *. Matrix.get matrix row column)
    done
  done;
  !total

let () =
  if Array.length Sys.argv <> 9 then
    fail
      "usage: sparse_kernels_worker SAMPLES FEATURES SEED DENSITIES REPEATS \
       ONE_HOT_SAMPLES ONE_HOT_FEATURES ONE_HOT_CARDINALITY";
  let rows = int_of_string Sys.argv.(1) in
  let columns = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let densities =
    String.split_on_char ',' Sys.argv.(4) |> List.map float_of_string
  in
  let repeats = int_of_string Sys.argv.(5) in
  let one_hot_samples = int_of_string Sys.argv.(6) in
  let one_hot_features = int_of_string Sys.argv.(7) in
  let cardinality = int_of_string Sys.argv.(8) in
  let allocated_before = Gc.allocated_bytes () in
  let operand =
    Vector.init ~length:columns (fun column ->
        (Float.of_int (column * 7 mod 13) /. 13.0) -. 0.5)
    |> get_data
  in
  let transposed_operand =
    Vector.init ~length:rows (fun row ->
        (Float.of_int (row * 3 mod 11) /. 11.0) -. 0.5)
    |> get_data
  in
  let cases =
    List.map
      (density_case ~seed ~rows ~columns ~repeats ~operand ~transposed_operand)
      densities
  in
  (* One-hot encoding: dense output versus direct CSR output. *)
  let one_hot_input =
    Matrix.init ~rows:one_hot_samples ~columns:one_hot_features
      (fun row feature ->
        Float.of_int (((row * 13) + (feature * 7)) mod cardinality))
    |> get_data
  in
  let schema = Feature_schema.of_matrix one_hot_input |> get_data in
  let rng = Rng.create (Seed.of_int seed) in
  let encoder =
    One_hot_encoder.create
      ~max_output_features:(one_hot_features * cardinality)
      ()
    |> get
  in
  let fitted =
    One_hot_encoder.fit encoder ~rng ~feature_schema:schema ~x:one_hot_input
      ~y:None ()
    |> get
  in
  let one_hot_dense, one_hot_dense_timing =
    timed ~repeats:1 (fun () ->
        One_hot_encoder.transform fitted ~feature_schema:schema ~x:one_hot_input
        |> get)
  in
  let one_hot_csr, one_hot_csr_timing =
    timed ~repeats:1 (fun () ->
        One_hot_encoder.transform_csr fitted ~feature_schema:schema
          ~x:one_hot_input
        |> get)
  in
  (* Row-view materialization of every other row of the middle density. *)
  let source, _, _ = List.nth cases (List.length cases / 2) in
  let even_rows = Array.init (rows / 2) (fun index -> 2 * index) in
  let row_view = Row_view.create ~source_size:rows even_rows |> get_data in
  let view = Csr_matrix.view source row_view |> get_data in
  let view_memory = Csr_matrix.view_memory view in
  let materialized, materialize_timing =
    timed ~repeats:1 (fun () -> Csr_matrix.materialize view)
  in
  let signature =
    Array.concat
      (List.map (fun (_, signature, _) -> signature) cases
      @ [
          [|
            Float.of_int (Matrix.columns one_hot_dense);
            Float.of_int (Csr_matrix.nonzero_count one_hot_csr);
            weighted_column_sum_dense one_hot_dense;
            weighted_column_sum_csr one_hot_csr;
            Float.of_int (Csr_matrix.rows materialized);
            Float.of_int (Csr_matrix.nonzero_count materialized);
            vector_sum (Csr_matrix.values materialized);
          |];
        ])
  in
  let formatted =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  let one_hot_dense_memory =
    let rows, columns = Matrix.shape one_hot_dense in
    Int64.mul 8L (Int64.of_int (rows * columns))
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"densities":[%s],"features":%d,"materialization":{"allocated_bytes":%Ld,"materialize":%s,"materialized_bytes":%Ld,"shared_bytes":%Ld,"view_rows":%d},"ocaml":%S,"one_hot":{"csr_memory":%s,"csr_transform":%s,"dense_bytes":%Ld,"dense_transform":%s,"output_columns":%d},"operations":["csr_and_dense_feature_matrix_vector_products","one_hot_dense_and_csr_transform","csr_row_view_materialization"],"repeats":%d,"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words
    (Digest.to_hex (Digest.string formatted))
    (String.concat "," (List.map (fun (_, _, json) -> json) cases))
    columns view_memory.Csr_matrix.allocated_bytes
    (timing_json materialize_timing)
    view_memory.Csr_matrix.materialized_bytes
    view_memory.Csr_matrix.shared_bytes (Row_view.length row_view)
    Sys.ocaml_version
    (memory_json (Csr_matrix.memory one_hot_csr))
    (timing_json one_hot_csr_timing)
    one_hot_dense_memory
    (timing_json one_hot_dense_timing)
    (Matrix.columns one_hot_dense)
    repeats rows formatted
