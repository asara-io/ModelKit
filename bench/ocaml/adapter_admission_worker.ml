open Modelkit
open Admission

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let hash ~seed ~row ~column ~row_factor ~column_factor =
  Int64.add
    (Int64.add
       (Int64.mul (Int64.of_int row) row_factor)
       (Int64.mul (Int64.of_int column) column_factor))
    (Int64.of_int seed)

let missing ~seed ~missing_modulus row column =
  column > 0
  && Int64.rem
       (hash ~seed ~row ~column ~row_factor:101L ~column_factor:53L)
       (Int64.of_int missing_modulus)
     = 0L

let value ~seed row column =
  Int64.rem (hash ~seed ~row ~column ~row_factor:17L ~column_factor:31L) 1000L
  |> Int64.to_float
  |> fun value -> value /. 100.0

let label row = Int64.of_int (row mod 3)
let weight row = 1.0 +. (Float.of_int (row mod 5) *. 0.25)
let group row = Int64.of_int (row / 10)

type measurement = {
  elapsed_ns : int64;
  allocated_words : float;
  retained_payload_bytes : int64;
  temporary_payload_bytes : int64;
  report_count : int;
  signature : float array;
}

let measure admit =
  let allocated_before = Gc.allocated_bytes () in
  let started = Unix.gettimeofday () in
  let admitted : Target.classification Admission.dataset = admit () in
  let elapsed = Unix.gettimeofday () -. started in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  let dataset = admitted.dataset in
  let matrix = Dataset.features dataset in
  let rows, columns = Matrix.shape matrix in
  let sum = ref 0.0 in
  for row = 0 to rows - 1 do
    for column = 0 to columns - 1 do
      let value = Matrix.get matrix row column in
      if not (Float.is_nan value) then sum := !sum +. value
    done
  done;
  let nulls =
    match admitted.feature_null_mask with
    | Some mask -> Null_mask.null_count mask
    | None -> 0
  in
  let labels = Target.classification_values (Dataset.target dataset) in
  let label_sum = Array.fold_left (fun total label -> total + label) 0 labels in
  let weights = Option.get (Dataset.sample_weight dataset) in
  let weight_sum = ref 0.0 in
  for row = 0 to rows - 1 do
    weight_sum := !weight_sum +. Sample_weight.get weights row
  done;
  let groups = Option.get (Dataset.groups dataset) in
  let group_sum = ref 0 in
  for row = 0 to rows - 1 do
    group_sum := !group_sum + Groups.get groups row
  done;
  {
    elapsed_ns = Int64.of_float (elapsed *. 1e9);
    allocated_words;
    retained_payload_bytes =
      Admission.retained_payload_bytes admitted.dataset_reports;
    temporary_payload_bytes =
      Admission.temporary_payload_bytes admitted.dataset_reports;
    report_count = List.length admitted.dataset_reports;
    signature =
      [|
        Float.of_int rows;
        Float.of_int columns;
        !sum;
        Float.of_int nulls;
        Float.of_int label_sum;
        !weight_sum;
        Float.of_int !group_sum;
      |];
  }

let measurement_json name measurement =
  let allocated_bytes =
    measurement.allocated_words *. Float.of_int (Sys.word_size / 8)
  in
  let retained = Int64.to_float measurement.retained_payload_bytes in
  Printf.sprintf
    {|%S:{"allocated_bytes_per_retained_payload_byte":%.6f,"allocated_words":%.0f,"elapsed_ns":%Ld,"report_count":%d,"retained_payload_bytes":%Ld,"temporary_payload_bytes":%Ld}|}
    name
    (if retained > 0.0 then allocated_bytes /. retained else 0.0)
    measurement.allocated_words measurement.elapsed_ns measurement.report_count
    measurement.retained_payload_bytes measurement.temporary_payload_bytes

let () =
  if Array.length Sys.argv <> 5 then
    fail "usage: adapter_admission_worker SAMPLES FEATURES SEED MISSING_MODULUS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let missing_modulus = int_of_string Sys.argv.(4) in
  let missing = missing ~seed ~missing_modulus in
  let value = value ~seed in
  let names = Array.init features (Printf.sprintf "feature_%d") in
  let x =
    Nx.init Nx.float64 [| samples; features |] (fun index ->
        let row = index.(0) and column = index.(1) in
        if missing row column then Float.nan else value row column)
  in
  let mask =
    Nx.init Nx.bool [| samples; features |] (fun index ->
        missing index.(0) index.(1))
  in
  let y = Nx.init Nx.int64 [| samples |] (fun index -> label index.(0)) in
  let w = Nx.init Nx.float64 [| samples |] (fun index -> weight index.(0)) in
  let g = Nx.init Nx.int64 [| samples |] (fun index -> group index.(0)) in
  let frame =
    Talon.create
      (Array.to_list
         (Array.mapi
            (fun column name ->
              ( name,
                Talon.Col.float64_opt
                  (Array.init samples (fun row ->
                       if missing row column then None
                       else Some (value row column))) ))
            names)
      @ [
          ("target", Talon.Col.int64 (Array.init samples label));
          ("weight", Talon.Col.float64 (Array.init samples weight));
          ("group", Talon.Col.int64 (Array.init samples group));
        ])
  in
  let nx =
    measure (fun () ->
        Modelkit_nx.classification_dataset ~names ~feature_null_mask:mask
          ~sample_weight:w ~groups:g ~x ~y ()
        |> get)
  in
  let talon =
    measure (fun () ->
        Modelkit_talon.classification_dataset ~sample_weight:"weight"
          ~groups:"group" ~features:(Array.to_list names) ~target:"target" frame
        |> get)
  in
  let signature = Array.append nx.signature talon.signature in
  let formatted =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  Printf.printf
    {|{"adapters":{%s,%s},"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["nx_classification_dataset_admission","talon_classification_dataset_admission"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    (measurement_json "nx" nx)
    (measurement_json "talon" talon)
    (nx.allocated_words +. talon.allocated_words)
    (Digest.to_hex (Digest.string formatted))
    features Sys.ocaml_version samples formatted
