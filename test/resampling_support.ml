open Modelkit
open Evaluation_metadata_support

let rng n = Rng.create (Seed.of_int n)

let matrix n =
  Matrix.init ~rows:n ~columns:1 (fun i _ -> float_of_int i) |> data

let rows (train, test) = (Row_view.indices train, Row_view.indices test)
let signature splits = Array.map rows splits
let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let check_pair n train_count test_count (train, test) =
  Alcotest.(check int) "training size" train_count (Row_view.length train);
  Alcotest.(check int) "test size" test_count (Row_view.length test);
  let seen = Array.make n false in
  Array.iter
    (fun view ->
      Array.iter
        (fun row ->
          Alcotest.(check bool) "in bounds" true (row >= 0 && row < n);
          Alcotest.(check bool) "unique and disjoint" false seen.(row);
          seen.(row) <- true)
        (Row_view.indices view))
    [| train; test |]

let test_sizes () =
  let x = matrix 11 in
  let split ?train_size ?test_size () =
    let spec = Holdout.create ?train_size ?test_size ~shuffle:false () |> get in
    Holdout.split spec ~rng:(rng 7) ~x ~y:None () |> get |> fun a -> a.(0)
  in
  let first = split () in
  check_pair 11 8 3 first;
  Alcotest.(check (array int))
    "ordered training" (Array.init 8 Fun.id)
    (fst (rows first));
  Alcotest.(check (array int)) "ordered test" [| 8; 9; 10 |] (snd (rows first));
  check_pair 11 3 8 (split ~train_size:(Split_size.Fraction 0.3) ());
  check_pair 11 7 4 (split ~test_size:(Split_size.Fraction 0.3) ());
  let partial =
    split ~train_size:(Split_size.Count 3) ~test_size:(Split_size.Count 2) ()
  in
  check_pair 11 3 2 partial;
  Alcotest.(check (array int))
    "ordered partial test" [| 3; 4 |]
    (snd (rows partial));
  List.iter
    (fun size ->
      expect_error "invalid size" (Holdout.create ~test_size:size ()))
    [
      Split_size.Count 0;
      Split_size.Count (-1);
      Split_size.Fraction nan;
      Split_size.Fraction infinity;
      Split_size.Fraction 0.;
      Split_size.Fraction 1.;
    ];
  expect_error "fraction sum"
    (Shuffle_split.create ~train_size:(Split_size.Fraction 0.8)
       ~test_size:(Split_size.Fraction 0.3) ());
  List.iter
    (fun spec ->
      expect_error "infeasible partitions"
        (Holdout.split spec ~rng:(rng 0) ~x ~y:None ()))
    [
      Holdout.create ~train_size:(Split_size.Count max_int) () |> get;
      Holdout.create ~train_size:(Split_size.Count 8)
        ~test_size:(Split_size.Count 4) ()
      |> get;
      Holdout.create ~train_size:(Split_size.Fraction 0.01) () |> get;
    ];
  List.iter
    (fun n ->
      expect_error "insufficient rows"
        (Holdout.split
           (Holdout.create () |> get)
           ~rng:(rng 1) ~x:(matrix n) ~y:None ()))
    [ 0; 1 ]

let test_shuffle () =
  for n = 2 to 50 do
    let spec =
      Shuffle_split.create ~splits:3 ~test_size:(Split_size.Count 1) () |> get
    in
    let result =
      Shuffle_split.split spec ~rng:(rng n) ~x:(matrix n) ~y:None () |> get
    in
    Array.iter (check_pair n (n - 1) 1) result
  done;
  let x = matrix 31 in
  let split splits seed =
    Shuffle_split.split
      (Shuffle_split.create ~splits () |> get)
      ~rng:(rng seed) ~x ~y:None ()
    |> get |> signature
  in
  let first = split 3 17 in
  Alcotest.(check bool) "same seed" true (first = split 3 17);
  Alcotest.(check bool) "different seed" false (first = split 3 18);
  Alcotest.(check bool) "prefix stable" true (first = Array.sub (split 5 17) 0 3);
  Alcotest.(check bool) "independent draws" false (first.(0) = first.(1));
  expect_error "zero splits" (Shuffle_split.create ~splits:0 ());
  expect_error "split array limit" (Shuffle_split.create ~splits:max_int ())

let test_stratification () =
  let labels =
    Array.init 40 (fun i -> if i < 20 then 70 else if i < 32 then -4 else 19)
  in
  let y = Target.classification labels and x = matrix 40 in
  let spec =
    Stratified_shuffle_split.create ~splits:3 ~train_size:(Split_size.Count 20)
      ~test_size:(Split_size.Count 10) ()
    |> get
  in
  let split y =
    Stratified_shuffle_split.split spec ~rng:(rng 19) ~x ~y:(Some y) () |> get
  in
  let result = split y in
  let counts view =
    Array.fold_left
      (fun (a, b, c) row ->
        match labels.(row) with
        | 70 -> (a + 1, b, c)
        | -4 -> (a, b + 1, c)
        | _ -> (a, b, c + 1))
      (0, 0, 0) (Row_view.indices view)
  in
  Array.iter
    (fun ((train, test) as pair) ->
      check_pair 40 20 10 pair;
      Alcotest.(check bool)
        "proportional training" true
        (counts train = (10, 6, 4));
      Alcotest.(check bool) "proportional test" true (counts test = (5, 3, 2)))
    result;
  Alcotest.(check bool)
    "class renaming invariance" true
    (signature result
    = signature
        (split
           (Target.classification (Array.map (fun label -> -label + 5) labels)))
    );
  for n = 4 to 60 do
    let x = matrix n in
    let y = Target.classification (Array.init n (fun i -> i mod 2)) in
    let spec =
      Stratified_shuffle_split.create ~splits:2 ~test_size:(Split_size.Count 2)
        ()
      |> get
    in
    Stratified_shuffle_split.split spec ~rng:(rng n) ~x ~y:(Some y) ()
    |> get
    |> Array.iter (check_pair n (n - 2) 2)
  done;
  let bad y x =
    expect_error "infeasible stratification"
      (Stratified_shuffle_split.split spec ~rng:(rng 0) ~x ~y ())
  in
  bad None x;
  bad (Some (Target.classification [| 1; 2 |])) x;
  bad (Some (Target.classification (Array.init 40 Fun.id))) x;
  let small =
    Stratified_shuffle_split.create ~test_size:(Split_size.Count 1) () |> get
  in
  expect_error "test rows below class count"
    (Stratified_shuffle_split.split small ~rng:(rng 0) ~x ~y:(Some y) ());
  ignore
    (Cross_validation.target_aware_splitter
       (module Stratified_shuffle_split)
       spec)

let test_repeated () =
  let n = 23 and folds = 4 and repeats = 3 in
  let x = matrix n in
  let verify result =
    Alcotest.(check int)
      "total fold count" (folds * repeats) (Array.length result);
    for repeat = 0 to repeats - 1 do
      let tested = Array.make n 0 in
      for fold = 0 to folds - 1 do
        let train, test = result.((repeat * folds) + fold) in
        check_pair n (Row_view.length train) (Row_view.length test) (train, test);
        Alcotest.(check int)
          "complete partition" n
          (Row_view.length train + Row_view.length test);
        Array.iter
          (fun row -> tested.(row) <- tested.(row) + 1)
          (Row_view.indices test)
      done;
      Alcotest.(check (array int))
        "one test occurrence per repetition" (Array.make n 1) tested
    done;
    Alcotest.(check bool)
      "repetitions change membership" false
      (signature (Array.sub result 0 folds)
      = signature (Array.sub result folds folds))
  in
  let split repeats seed =
    Repeated_k_fold.split
      (Repeated_k_fold.create ~folds ~repeats () |> get)
      ~rng:(rng seed) ~x ~y:None ()
    |> get
  in
  let first = split repeats 11 in
  verify first;
  Alcotest.(check bool)
    "repeated seed stability" true
    (signature first = signature (split repeats 11));
  Alcotest.(check bool)
    "repetition prefix" true
    (signature first = signature (Array.sub (split 4 11) 0 12));
  let y = Target.classification (Array.init n (fun i -> i mod 3)) in
  let spec = Repeated_stratified_k_fold.create ~folds ~repeats () |> get in
  let result =
    Repeated_stratified_k_fold.split spec ~rng:(rng 11) ~x ~y:(Some y) () |> get
  in
  verify result;
  for repeat = 0 to repeats - 1 do
    for label = 0 to 2 do
      let counts =
        Array.init folds (fun fold ->
            snd result.((repeat * folds) + fold)
            |> Row_view.indices
            |> Array.fold_left
                 (fun total row -> total + if row mod 3 = label then 1 else 0)
                 0)
      in
      Alcotest.(check bool)
        "per-class fold balance" true
        (Array.fold_left max 0 counts - Array.fold_left min max_int counts <= 1)
    done
  done;
  ignore
    (Cross_validation.target_aware_splitter
       (module Repeated_stratified_k_fold)
       spec);
  expect_error "repeat overflow"
    (Repeated_k_fold.create ~folds:2 ~repeats:Sys.max_array_length ());
  expect_error "no repeats" (Repeated_k_fold.create ~repeats:0 ());
  expect_error "one fold" (Repeated_stratified_k_fold.create ~folds:1 ());
  expect_error "too many folds"
    (Repeated_k_fold.split
       (Repeated_k_fold.create ~folds:24 () |> get)
       ~rng:(rng 0) ~x ~y:None ())

let test_alignment () =
  let source = dataset () in
  let verify part =
    Alcotest.(check bool)
      "schema preserved" true
      (Feature_schema.equal
         (Dataset.feature_schema source)
         (Dataset.feature_schema part));
    let x = Dataset.features part in
    let y = Target.regression_values (Dataset.target part) in
    let weights = Option.get (Dataset.sample_weight part)
    and groups = Option.get (Dataset.groups part) in
    for i = 0 to Dataset.sample_count part - 1 do
      let row = Matrix.get x i 0 |> int_of_float in
      Alcotest.check (Alcotest.float 0.) "target aligned"
        (float_of_int ((3 * row) + 5))
        (Vector.get y i);
      Alcotest.check (Alcotest.float 0.) "weight aligned"
        (float_of_int (row + 1))
        (Sample_weight.get weights i);
      Alcotest.(check int)
        "group aligned"
        (100 + (row / 2))
        (Groups.get groups i)
    done
  in
  let train, test = Train_test_split.split ~rng:(rng 17) source () |> get in
  verify train;
  verify test;
  Alcotest.(check int) "default test fraction" 4 (Dataset.sample_count test);
  let stratify = Target.classification (Array.init 16 (fun i -> i mod 2)) in
  let train, test =
    Train_test_split.split ~stratify ~rng:(rng 17) source () |> get
  in
  verify train;
  verify test;
  expect_error "unshuffled stratification"
    (Train_test_split.split ~shuffle:false ~stratify ~rng:(rng 0) source ());
  expect_error "misaligned stratification"
    (Train_test_split.split
       ~stratify:(Target.classification [||])
       ~rng:(rng 0) source ());
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~x:(matrix 4)
      ~y:(regression [| 0.; 1.; 2.; 3. |])
      ~sample_weight:
        (Sample_weight.of_array ~expected_length:4 [| 1.; 0.; 0.; 0. |] |> data)
      ()
    |> data
  in
  expect_error "zero selected total weight"
    (Train_test_split.split ~shuffle:false ~rng:(rng 0) source ())

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
        (module Shuffle_split)
        (Shuffle_split.create ~splits:3 () |> get);
      Cross_validation.target_independent_splitter
        (module Holdout)
        (Holdout.create () |> get);
      Cross_validation.target_independent_splitter
        (module Repeated_k_fold)
        (Repeated_k_fold.create ~folds:4 ~repeats:2 () |> get);
    ]
  in
  List.iter
    (fun splitter ->
      let run execution =
        Cross_validation.Regression.cross_validate ~execution
          ~return_indices:true ~splitter
          ~scorers:[| Regression_scorer.neg_mean_squared_error |]
          ~seed:(Seed.of_int 17) pipeline source
        |> get |> Cross_validation.folds
        |> Array.map (fun fold ->
            ( Option.get fold.Cross_validation.train_indices,
              Option.get fold.Cross_validation.test_indices,
              score fold ))
      in
      Alcotest.(check bool)
        "identical CV indices and scores across execution backends" true
        (run execution = run Execution.sequential))
    splitters

let tests =
  [
    ("sizes, rounding, holdout and invalid inputs", `Quick, test_sizes);
    ("shuffle invariants and deterministic streams", `Quick, test_shuffle);
    ("stratified allocation and feasibility", `Quick, test_stratification);
    ("repeated coverage and class balance", `Quick, test_repeated);
    ("aligned materialization", `Quick, test_alignment);
    ("CV integration", `Quick, fun () -> check_execution Execution.sequential);
  ]
