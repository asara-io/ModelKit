open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix rows =
  Matrix.init ~rows ~columns:1 (fun row _ -> Float.of_int row) |> get_data

let rng seed = Rng.create (Seed.of_int seed)
let indices (train, test) = (Row_view.indices train, Row_view.indices test)

let check_indices message expected observed =
  Alcotest.(check (array int)) message expected observed

let check_partition sample_count (train, test) =
  let seen = Array.make sample_count 0 in
  Array.iter (fun row -> seen.(row) <- seen.(row) + 1) (Row_view.indices train);
  Array.iter (fun row -> seen.(row) <- seen.(row) + 1) (Row_view.indices test);
  Array.iter (Alcotest.(check int) "row occurs once in fold partition" 1) seen

let test_k_fold () =
  let x = matrix 10 in
  let specification = K_fold.create ~folds:3 () |> get in
  let splits = K_fold.split specification ~rng:(rng 1) ~x ~y:None () |> get in
  Alcotest.(check int) "fold count" 3 (Array.length splits);
  let expected = [| [| 0; 1; 2; 3 |]; [| 4; 5; 6 |]; [| 7; 8; 9 |] |] in
  Array.iteri
    (fun fold split ->
      check_partition 10 split;
      let _, test = indices split in
      check_indices "balanced test rows" expected.(fold) test)
    splits

let test_k_fold_shuffle () =
  let x = matrix 24 in
  let specification = K_fold.create ~folds:4 ~shuffle:true () |> get in
  let split seed =
    K_fold.split specification ~rng:(rng seed) ~x ~y:None ()
    |> get
    |> Array.map (fun (_, test) -> Row_view.indices test)
  in
  let first = split 29 in
  let repeated = split 29 in
  let different = split 30 in
  Alcotest.(check (array (array int)))
    "fixed seed reproduces membership" first repeated;
  Alcotest.(check bool)
    "different seed changes membership" false (first = different);
  Array.iter
    (fun rows ->
      let sorted = Array.copy rows in
      Array.sort Int.compare sorted;
      check_indices "emitted test rows retain source order" sorted rows)
    first

let class_counts labels rows =
  let counts = Hashtbl.create 4 in
  Array.iter
    (fun row ->
      let label = labels.(row) in
      let count = Option.value (Hashtbl.find_opt counts label) ~default:0 in
      Hashtbl.replace counts label (count + 1))
    rows;
  counts

let test_stratified_k_fold () =
  let labels = [| 10; 10; 10; 10; 10; 10; 20; 20; 20; 20; 20; 20 |] in
  let x = matrix (Array.length labels) in
  let specification = Stratified_k_fold.create ~folds:3 () |> get in
  let splits =
    Stratified_k_fold.split specification ~rng:(rng 7) ~x
      ~y:(Some (Target.classification labels))
      ()
    |> get
  in
  let expected = [| [| 0; 1; 6; 7 |]; [| 2; 3; 8; 9 |]; [| 4; 5; 10; 11 |] |] in
  Array.iteri
    (fun fold split ->
      check_partition 12 split;
      let _, test = indices split in
      check_indices "stratified test rows" expected.(fold) test;
      let counts = class_counts labels test in
      Alcotest.(check int) "first class count" 2 (Hashtbl.find counts 10);
      Alcotest.(check int) "second class count" 2 (Hashtbl.find counts 20))
    splits

let test_stratified_shuffle () =
  let labels = Array.init 30 (fun row -> row mod 3) in
  let x = matrix 30 in
  let specification =
    Stratified_k_fold.create ~folds:5 ~shuffle:true () |> get
  in
  let split seed =
    Stratified_k_fold.split specification ~rng:(rng seed) ~x
      ~y:(Some (Target.classification labels))
      ()
    |> get
    |> Array.map (fun (_, test) -> Row_view.indices test)
  in
  Alcotest.(check (array (array int)))
    "stratified fixed seed" (split 41) (split 41);
  Alcotest.(check bool) "stratified different seed" false (split 41 = split 42)

let test_group_k_fold () =
  let group_values = [| 0; 0; 0; 1; 1; 2; 2; 3; 4 |] in
  let x = matrix (Array.length group_values) in
  let groups =
    Groups.create ~expected_length:(Array.length group_values) group_values
    |> get_data
  in
  let specification = Group_k_fold.create ~folds:3 () |> get in
  let splits =
    Group_k_fold.split specification ~rng:(rng 0) ~groups ~x ~y:None () |> get
  in
  let expected = [| [| 0; 1; 2 |]; [| 5; 6; 8 |]; [| 3; 4; 7 |] |] in
  Array.iteri
    (fun fold (train, test) ->
      check_partition 9 (train, test);
      check_indices "balanced group test rows" expected.(fold)
        (Row_view.indices test);
      let train_groups = Hashtbl.create 5 in
      Array.iter
        (fun row -> Hashtbl.replace train_groups group_values.(row) ())
        (Row_view.indices train);
      Array.iter
        (fun row ->
          Alcotest.(check bool)
            "test group is absent from training" false
            (Hashtbl.mem train_groups group_values.(row)))
        (Row_view.indices test))
    splits

let test_time_series_split () =
  let x = matrix 12 in
  let specification =
    Time_series_split.create ~folds:3 ~test_size:2 ~gap:1 () |> get
  in
  let splits =
    Time_series_split.split specification ~rng:(rng 0) ~x ~y:None () |> get
  in
  let expected_train =
    [|
      [| 0; 1; 2; 3; 4 |];
      [| 0; 1; 2; 3; 4; 5; 6 |];
      [| 0; 1; 2; 3; 4; 5; 6; 7; 8 |];
    |]
  in
  let expected_test = [| [| 6; 7 |]; [| 8; 9 |]; [| 10; 11 |] |] in
  Array.iteri
    (fun fold split ->
      let train, test = indices split in
      check_indices "expanding training rows" expected_train.(fold) train;
      check_indices "contiguous test rows" expected_test.(fold) test;
      Alcotest.(check bool)
        "training precedes test with a gap" true
        (train.(Array.length train - 1) + 1 < test.(0)))
    splits

let test_time_series_default_window () =
  let x = matrix 12 in
  let specification = Time_series_split.create ~folds:3 () |> get in
  let splits =
    Time_series_split.split specification ~rng:(rng 0) ~x ~y:None () |> get
  in
  let sizes =
    Array.map
      (fun (train, test) -> (Row_view.length train, Row_view.length test))
      splits
  in
  Alcotest.(check (array (pair int int)))
    "default window sizes"
    [| (3, 3); (6, 3); (9, 3) |]
    sizes

let is_validation = function
  | Error.Validation _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_shape = function
  | Error.Shape_mismatch _ -> true
  | Error.Data _ | Error.Feature_schema_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let test_split_validation () =
  Split.create ~source_size:4 ~train:[| 0; 0 |] ~test:[| 1 |]
  |> expect_error is_validation;
  Split.create ~source_size:4 ~train:[| 0; 1 |] ~test:[| 1; 2 |]
  |> expect_error is_validation;
  Split.create ~source_size:4 ~train:[||] ~test:[| 1 |]
  |> expect_error is_validation;
  let train = Row_view.create ~source_size:4 [| 0; 1 |] |> get_data in
  let test = Row_view.create ~source_size:5 [| 2; 3 |] |> get_data in
  Split.of_views ~train ~test |> expect_error is_validation

let test_splitter_errors () =
  K_fold.create ~folds:1 () |> expect_error is_validation;
  Time_series_split.create ~test_size:0 () |> expect_error is_validation;
  Time_series_split.create ~gap:(-1) () |> expect_error is_validation;
  let x = matrix 3 in
  K_fold.split (K_fold.create ~folds:4 () |> get) ~rng:(rng 0) ~x ~y:None ()
  |> expect_error is_validation;
  Stratified_k_fold.split
    (Stratified_k_fold.create ~folds:2 () |> get)
    ~rng:(rng 0) ~x ~y:None ()
  |> expect_error is_validation;
  let short_target = Target.classification [| 0; 1 |] in
  Stratified_k_fold.split
    (Stratified_k_fold.create ~folds:2 () |> get)
    ~rng:(rng 0) ~x ~y:(Some short_target) ()
  |> expect_error is_shape;
  Group_k_fold.split
    (Group_k_fold.create ~folds:2 () |> get)
    ~rng:(rng 0) ~x ~y:None ()
  |> expect_error is_validation;
  Time_series_split.split
    (Time_series_split.create ~folds:3 ~test_size:2 () |> get)
    ~rng:(rng 0) ~x:(matrix 6) ~y:None ()
  |> expect_error is_validation;
  Time_series_split.split
    (Time_series_split.create ~folds:2 ~test_size:max_int () |> get)
    ~rng:(rng 0) ~x:(matrix 6) ~y:None ()
  |> expect_error is_validation

let test_materialization () =
  let x = matrix 6 in
  let names =
    Feature_names.create ~expected_count:1 [| "position" |] |> get_data
  in
  let target =
    Target.regression (Vector.of_array [| 10.0; 11.0; 12.0; 13.0; 14.0; 15.0 |])
    |> get_data
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:6 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    |> get_data
  in
  let groups =
    Groups.create ~expected_length:6 [| 0; 0; 1; 1; 2; 2 |] |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~feature_names:names
      ~sample_weight ~groups ~x ~y:target ()
    |> get_data
  in
  let split =
    Split.create ~source_size:6 ~train:[| 0; 2; 4 |] ~test:[| 1; 3; 5 |] |> get
  in
  let train, test = Split.materialize dataset split |> get in
  Alcotest.(check int) "materialized train rows" 3 (Dataset.sample_count train);
  Alcotest.(check int) "materialized test rows" 3 (Dataset.sample_count test);
  check_indices "materialized train targets" [| 10; 12; 14 |]
    (Dataset.target train |> Target.regression_values |> Vector.to_array
   |> Array.map int_of_float);
  check_indices "materialized test groups" [| 0; 1; 2 |]
    (Dataset.groups test |> Option.get |> Groups.to_array);
  Alcotest.check
    (Alcotest.array (Alcotest.float 0.0))
    "materialized test weights" [| 2.0; 4.0; 6.0 |]
    (Dataset.sample_weight test |> Option.get |> Sample_weight.to_vector
   |> Vector.to_array);
  Alcotest.(check bool)
    "feature schema is retained" true
    (Feature_schema.equal
       (Dataset.feature_schema dataset)
       (Dataset.feature_schema train));
  let report = Dataset.access_report train in
  Alcotest.(check bool)
    "features are copied" true
    (report.Dataset.feature_access = Dataset.Copy)

let () =
  Alcotest.run "splitters"
    [
      ( "cross-validation",
        [
          Alcotest.test_case "K-fold" `Quick test_k_fold;
          Alcotest.test_case "K-fold shuffle" `Quick test_k_fold_shuffle;
          Alcotest.test_case "stratified K-fold" `Quick test_stratified_k_fold;
          Alcotest.test_case "stratified shuffle" `Quick test_stratified_shuffle;
          Alcotest.test_case "group K-fold" `Quick test_group_k_fold;
          Alcotest.test_case "time-series" `Quick test_time_series_split;
          Alcotest.test_case "time-series default" `Quick
            test_time_series_default_window;
        ] );
      ( "validation",
        [
          Alcotest.test_case "split invariants" `Quick test_split_validation;
          Alcotest.test_case "splitter errors" `Quick test_splitter_errors;
          Alcotest.test_case "materialization" `Quick test_materialization;
        ] );
    ]
