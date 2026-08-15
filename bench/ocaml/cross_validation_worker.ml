open Modelkit
open Cross_validation

let fail message =
  prerr_endline message;
  exit 1

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let raw_value ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.0
  -. 5.0

let feature ~seed ~missing_modulus row column =
  if column > 0 && ((row * 101) + (column * 53) + seed) mod missing_modulus = 0
  then Float.nan
  else raw_value ~seed row column

let target ~seed row =
  let score =
    raw_value ~seed row 0
    +. (0.25 *. raw_value ~seed row 1)
    -. (0.1 *. raw_value ~seed row 2)
  in
  if score > 0.0 then 7 else -3

let pipeline ~c ~tolerance ~max_iterations =
  let imputer =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
    |> get
  in
  let logistic =
    Logistic_regression.create ~c ~tolerance ~max_iterations () |> get
  in
  let estimator =
    Pipeline.estimator ~name:"logistic"
      (module Logistic_regression)
      ~decision_function:Logistic_regression.decision_function
      ~predict_proba:Logistic_regression.predict_proba
      ~classes:Logistic_regression.classes logistic
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty imputer |> get in
  let builder = Pipeline.add_transformer builder scaler |> get in
  Pipeline.set_estimator builder estimator |> get

let score_value = function
  | Some (Ok value) -> value
  | Some (Error error) -> fail (Error.to_string error)
  | None -> fail "cross-validation benchmark score is missing"

let index_sum indices =
  Array.fold_left (fun sum index -> sum + index) 0 indices |> Float.of_int

let fold_signature fold =
  if fold.fit_time < 0.0 || fold.score_time < 0.0 then
    fail "cross-validation benchmark observed a negative fold timing";
  if Option.is_none fold.model then
    fail "cross-validation benchmark model was not retained";
  let scores = fold.scores in
  let train_indices = Option.get fold.train_indices in
  let test_indices = Option.get fold.test_indices in
  [|
    score_value scores.(0).train_score;
    score_value scores.(0).test_score;
    score_value scores.(1).train_score;
    score_value scores.(1).test_score;
    score_value scores.(2).train_score;
    score_value scores.(2).test_score;
    score_value scores.(3).train_score;
    score_value scores.(3).test_score;
    Float.of_int (Array.length train_indices);
    Float.of_int (Array.length test_indices);
    index_sum train_indices;
    index_sum test_indices;
    1.0;
  |]

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 9 then
    fail
      "usage: cross_validation_worker SAMPLES FEATURES SEED FOLDS \
       MISSING_MODULUS LOGISTIC_C LOGISTIC_TOLERANCE LOGISTIC_MAX_ITERATIONS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let folds = int_of_string Sys.argv.(4) in
  let missing_modulus = int_of_string Sys.argv.(5) in
  let c = float_of_string Sys.argv.(6) in
  let tolerance = float_of_string Sys.argv.(7) in
  let max_iterations = int_of_string Sys.argv.(8) in
  if features < 3 then
    fail "cross-validation benchmark requires at least three features";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (feature ~seed ~missing_modulus)
    |> get_data
  in
  let y = Target.classification (Array.init samples (target ~seed)) in
  let dataset =
    Dataset.create ~finiteness:Dataset.Allow_nan ~x ~y () |> get_data
  in
  let splitter =
    Stratified_k_fold.create ~folds ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let report =
    Cross_validation.Binary_classification.cross_validate
      ~return_train_score:true ~return_models:true ~return_indices:true
      ~splitter
      ~scorers:
        [|
          Binary_classification_scorer.accuracy;
          Binary_classification_scorer.balanced_accuracy ~positive_label:7 ();
          Binary_classification_scorer.neg_log_loss ~positive_label:7 ();
          Binary_classification_scorer.roc_auc ~positive_label:7 ();
        |]
      ~seed:(Seed.of_int seed)
      (pipeline ~c ~tolerance ~max_iterations)
      dataset
    |> get
  in
  let report_folds = Cross_validation.folds report in
  let signature =
    report_folds |> Array.map fold_signature |> Array.to_list |> Array.concat
  in
  let signature_text =
    signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ","
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"folds":%d,"ocaml":%S,"operations":["mean_imputation","standard_scaling","binary_logistic_regression","cross_validate","multiple_scorers","train_scores","models","indices"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features folds Sys.ocaml_version
    samples signature_text
