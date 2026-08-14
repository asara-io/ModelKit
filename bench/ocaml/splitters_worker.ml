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

let statistics splits =
  let train_total = ref 0 in
  let test_total = ref 0 in
  let smallest_test = ref max_int in
  let largest_test = ref 0 in
  Array.iter
    (fun (train, test) ->
      train_total := !train_total + Row_view.length train;
      let test_size = Row_view.length test in
      test_total := !test_total + test_size;
      smallest_test := Int.min !smallest_test test_size;
      largest_test := Int.max !largest_test test_size)
    splits;
  [|
    Array.length splits;
    !train_total;
    !test_total;
    !smallest_test;
    !largest_test;
  |]

let () =
  if Array.length Sys.argv <> 7 then
    fail
      "usage: splitters_worker SAMPLES FOLDS CLASSES GROUP_SIZE TEST_SIZE GAP";
  let samples = int_of_string Sys.argv.(1) in
  let folds = int_of_string Sys.argv.(2) in
  let classes = int_of_string Sys.argv.(3) in
  let group_size = int_of_string Sys.argv.(4) in
  let test_size = int_of_string Sys.argv.(5) in
  let gap = int_of_string Sys.argv.(6) in
  let allocated_before = Gc.allocated_bytes () in
  let x = Matrix.init ~rows:samples ~columns:1 (fun _ _ -> 0.0) |> get_data in
  let rng = Rng.create (Seed.of_int 1729) in
  let k_fold =
    K_fold.split (K_fold.create ~folds () |> get) ~rng ~x ~y:None () |> get
  in
  let target =
    Target.classification (Array.init samples (fun row -> row mod classes))
  in
  let stratified =
    Stratified_k_fold.split
      (Stratified_k_fold.create ~folds () |> get)
      ~rng ~x ~y:(Some target) ()
    |> get
  in
  let groups =
    Groups.create ~expected_length:samples
      (Array.init samples (fun row -> row / group_size))
    |> get_data
  in
  let grouped =
    Group_k_fold.split
      (Group_k_fold.create ~folds () |> get)
      ~rng ~groups ~x ~y:None ()
    |> get
  in
  let time_series =
    Time_series_split.split
      (Time_series_split.create ~folds ~test_size ~gap () |> get)
      ~rng ~x ~y:None ()
    |> get
  in
  let signature =
    [|
      statistics k_fold;
      statistics stratified;
      statistics grouped;
      statistics time_series;
    |]
    |> Array.to_list |> Array.concat
  in
  let signature_text =
    signature |> Array.map string_of_int |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"folds":%d,"ocaml":%S,"operations":["k_fold","stratified_k_fold","group_k_fold","time_series_split"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words signature_text folds Sys.ocaml_version samples
    signature_text
