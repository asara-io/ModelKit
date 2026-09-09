open Modelkit
open Evaluation_metadata_support

let rng n = Rng.create (Seed.of_int n)

let matrix n =
  Matrix.init ~rows:n ~columns:1 (fun row _ -> float_of_int row) |> data

let rows (train, test) = (Row_view.indices train, Row_view.indices test)
let signature result = Array.map rows result
let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let check_coverage ?groups n result =
  let tested = Array.make n 0 in
  Array.iter
    (fun (train, test) ->
      let train = Row_view.indices train and test = Row_view.indices test in
      Alcotest.(check bool)
        "nonempty partitions" true
        (Array.length train > 0 && Array.length test > 0);
      let all = Array.append train test in
      Array.sort Int.compare all;
      Alcotest.(check (array int))
        "disjoint complete partition" (Array.init n Fun.id) all;
      Array.iter (fun row -> tested.(row) <- tested.(row) + 1) test;
      Option.iter
        (fun groups ->
          let training_groups = Array.map (Array.get groups) train in
          Array.iter
            (fun row ->
              Alcotest.(check bool)
                "group exclusion" false
                (Array.mem groups.(row) training_groups))
            test)
        groups;
      List.iter
        (fun actual ->
          let sorted = Array.copy actual in
          Array.sort Int.compare sorted;
          Alcotest.(check (array int)) "source row order" sorted actual)
        [ train; test ])
    result;
  Alcotest.(check (array int)) "each row tested once" (Array.make n 1) tested

let test_predefined () =
  let source = [| -1; 8; 8; 2; 2; -1; 99; 99 |] in
  let spec = Predefined_split.create ~test_folds:source () |> get in
  source.(0) <- 2;
  let exposed = Predefined_split.params spec in
  exposed.Predefined_split.test_folds.(0) <- 99;
  let ids = Predefined_split.fold_ids spec in
  ids.(0) <- 100;
  Alcotest.(check (array int))
    "copied fold IDs" [| 2; 8; 99 |]
    (Predefined_split.fold_ids spec);
  let run seed =
    Predefined_split.split spec ~rng:(rng seed) ~x:(matrix 8) ~y:None () |> get
  in
  let result = run 1 in
  Alcotest.(check bool) "RNG ignored" true (signature result = signature (run 2));
  let expected = [| [| 3; 4 |]; [| 1; 2 |]; [| 6; 7 |] |] in
  Array.iteri
    (fun i (train, test) ->
      Alcotest.(check (array int))
        "ascending sparse fold IDs" expected.(i) (Row_view.indices test);
      let train = Row_view.indices train in
      Alcotest.(check bool)
        "always-training rows" true
        (Array.mem 0 train && Array.mem 5 train);
      Alcotest.(check int) "complete complement" 6 (Array.length train))
    result;
  List.iter
    (fun test_folds ->
      expect_error "invalid predefined assignments"
        (Predefined_split.create ~test_folds ()))
    [ [||]; [| -1; -1 |]; [| -2; 0 |]; [| 0; 0 |] ];
  let one =
    Predefined_split.create ~test_folds:[| -1; max_int; max_int |] () |> get
  in
  let result =
    Predefined_split.split one ~rng:(rng 0) ~x:(matrix 3) ~y:None () |> get
  in
  Alcotest.(check int)
    "single fold with training-only rows" 1 (Array.length result);
  expect_error "misaligned assignments"
    (Predefined_split.split one ~rng:(rng 0) ~x:(matrix 2) ~y:None ())

let test_leave_out () =
  let spec = Leave_one_out.create () in
  for n = 2 to 15 do
    let result =
      Leave_one_out.split spec ~rng:(rng 0) ~x:(matrix n) ~y:None () |> get
    in
    check_coverage n result;
    Array.iteri
      (fun i (_, test) ->
        Alcotest.(check (array int))
          "one row per fold" [| i |] (Row_view.indices test))
      result
  done;
  List.iter
    (fun n ->
      expect_error "insufficient rows"
        (Leave_one_out.split spec ~rng:(rng 0) ~x:(matrix n) ~y:None ()))
    [ 0; 1 ];
  let labels = [| 10; -3; 10; 2; -3; 2 |] in
  let groups = Groups.create ~expected_length:6 labels |> data in
  let spec = Leave_one_group_out.create () in
  let result =
    Leave_one_group_out.split spec ~rng:(rng 0) ~groups ~x:(matrix 6) ~y:None ()
    |> get
  in
  check_coverage ~groups:labels 6 result;
  Array.iteri
    (fun i (_, test) ->
      Alcotest.(check (array int))
        "ascending group IDs"
        [| [| 1; 4 |]; [| 3; 5 |]; [| 0; 2 |] |].(i)
        (Row_view.indices test))
    result;
  expect_error "missing groups"
    (Leave_one_group_out.split spec ~rng:(rng 0) ~x:(matrix 6) ~y:None ());
  expect_error "misaligned groups"
    (Leave_one_group_out.split spec ~rng:(rng 0) ~groups ~x:(matrix 5) ~y:None
       ());
  let groups = Groups.create ~expected_length:6 (Array.make 6 1) |> data in
  expect_error "only one group"
    (Leave_one_group_out.split spec ~rng:(rng 0) ~groups ~x:(matrix 6) ~y:None
       ())

let stratified ?(shuffle = false) ~folds ~seed labels group_labels =
  let n = Array.length labels in
  Stratified_group_k_fold.split
    (Stratified_group_k_fold.create ~folds ~shuffle () |> get)
    ~rng:(rng seed)
    ~groups:(Groups.create ~expected_length:n group_labels |> data)
    ~x:(matrix n)
    ~y:(Some (Target.classification labels))
    ()
  |> get

let test_group_invariants () =
  for group_count = 2 to 10 do
    let n = (2 * group_count) + 1 in
    let groups = Array.init n (fun row -> (row mod group_count) - 3) in
    let labels = Array.init n (fun row -> ((row * row) + (row / 2)) mod 4) in
    for folds = 2 to group_count do
      check_coverage ~groups n
        (stratified ~shuffle:true ~folds ~seed:group_count labels groups)
    done
  done;
  let groups = Array.init 120 (fun row -> row / 2)
  and labels = Array.init 120 Fun.id in
  check_coverage ~groups 120 (stratified ~folds:4 ~seed:0 labels groups);
  let groups = [| 0; 0; 1; 1; 2; 2; 3; 3 |]
  and labels = [| 9; 9; 0; 0; 0; 0; 0; 0 |] in
  check_coverage ~groups 8 (stratified ~folds:4 ~seed:0 labels groups)

let test_group_determinism () =
  let labels = Array.init 24 (fun row -> row mod 2)
  and groups = Array.init 24 (fun row -> row / 2) in
  let run seed = stratified ~shuffle:true ~folds:3 ~seed labels groups in
  let baseline = run 17 in
  Alcotest.(check bool)
    "same shuffled seed" true
    (signature baseline = signature (run 17));
  Alcotest.(check bool)
    "seed changes tied group order" false
    (signature baseline = signature (run 18));
  Array.iter
    (fun (_, test) ->
      let counts = Array.make 2 0 in
      Array.iter
        (fun row -> counts.(labels.(row)) <- counts.(labels.(row)) + 1)
        (Row_view.indices test);
      Alcotest.(check (array int))
        "balanced groups preserve class balance" [| 4; 4 |] counts)
    baseline;
  let plain seed = stratified ~folds:3 ~seed labels groups in
  Alcotest.(check bool)
    "unshuffled ignores RNG" true
    (signature (plain 1) = signature (plain 2));
  let group_signature group_labels result =
    Array.map
      (fun (_, test) ->
        Row_view.indices test |> Array.to_list
        |> List.map (Array.get group_labels)
        |> List.sort_uniq Int.compare)
      result
  in
  let reverse a =
    Array.init (Array.length a) (fun i -> a.(Array.length a - 1 - i))
  in
  let reversed_groups = reverse groups in
  Alcotest.(check bool)
    "row permutation preserves group membership" true
    (group_signature groups baseline
    = group_signature reversed_groups
        (stratified ~shuffle:true ~folds:3 ~seed:17 (reverse labels)
           reversed_groups))

let test_group_errors () =
  expect_error "fold count" (Stratified_group_k_fold.create ~folds:1 ());
  let spec = Stratified_group_k_fold.create ~folds:3 () |> get in
  let x = matrix 6 and y = Target.classification [| 0; 1; 0; 1; 0; 1 |] in
  let groups =
    Groups.create ~expected_length:6 [| 0; 0; 1; 1; 2; 2 |] |> data
  in
  let fail ?groups y =
    expect_error "invalid grouped split"
      (Stratified_group_k_fold.split spec ~rng:(rng 0) ?groups ~x ~y ())
  in
  fail (Some y);
  fail ~groups None;
  fail ~groups (Some (Target.classification [| 0 |]));
  fail ~groups:(Groups.create ~expected_length:1 [| 0 |] |> data) (Some y);
  fail
    ~groups:(Groups.create ~expected_length:6 [| 0; 0; 0; 1; 1; 1 |] |> data)
    (Some y)

let check_execution execution =
  let source = dataset () in
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let splitters =
    [
      Cross_validation.target_independent_splitter
        (module Leave_one_out)
        (Leave_one_out.create ());
      Cross_validation.target_independent_splitter
        (module Leave_one_group_out)
        (Leave_one_group_out.create ());
      Cross_validation.target_independent_splitter
        (module Predefined_split)
        (Predefined_split.create
           ~test_folds:(Array.init 16 (fun i -> if i < 2 then -1 else i mod 3))
           ()
        |> get);
    ]
  in
  let summarize report =
    Cross_validation.folds report
    |> Array.map (fun fold ->
        ( Option.get fold.Cross_validation.train_indices,
          Option.get fold.Cross_validation.test_indices,
          score fold ))
  in
  List.iter
    (fun splitter ->
      let run execution =
        Cross_validation.Regression.cross_validate ~execution
          ~return_indices:true ~splitter
          ~scorers:[| Regression_scorer.neg_mean_squared_error |]
          ~seed:(Seed.of_int 19) pipeline source
        |> get |> summarize
      in
      Alcotest.(check bool)
        "regression CV matches sequential" true
        (run execution = run Execution.sequential))
    splitters;
  let groups =
    Groups.create ~expected_length:24 (Array.init 24 (fun i -> i / 2)) |> data
  in
  let y = Target.classification (Array.init 24 (fun i -> i mod 2)) in
  let x =
    Matrix.init ~rows:24 ~columns:1 (fun row _ ->
        float_of_int (row mod 2) +. (0.01 *. float_of_int row))
    |> data
  in
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y ~groups () |> data
  in
  let estimator =
    Pipeline.classifier ~name:"logistic"
      (module Logistic_regression)
      (Logistic_regression.create () |> get)
    |> get
  in
  let pipeline = Pipeline.set_estimator Pipeline.empty estimator |> get in
  let splitter =
    Cross_validation.target_aware_splitter
      (module Stratified_group_k_fold)
      (Stratified_group_k_fold.create ~folds:3 ~shuffle:true () |> get)
  in
  let run execution =
    Cross_validation.Binary_classification.cross_validate ~execution
      ~return_indices:true ~splitter
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed:(Seed.of_int 19) pipeline source
    |> get |> summarize
  in
  Alcotest.(check bool)
    "stratified-group CV matches sequential" true
    (run execution = run Execution.sequential)

let tests =
  [
    ("predefined ownership, coverage and feasibility", `Quick, test_predefined);
    ("exhaustive row and group exclusion", `Quick, test_leave_out);
    ("stratified-group coverage adversaries", `Quick, test_group_invariants);
    ( "stratified-group balance and reproducibility",
      `Quick,
      test_group_determinism );
    ("stratified-group input failures", `Quick, test_group_errors);
    ("CV integration", `Quick, fun () -> check_execution Execution.sequential);
  ]
