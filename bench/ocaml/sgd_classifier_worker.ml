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

let feature ~seed row column =
  Float.of_int (((row * (17 + (column * 12))) + (column * 31) + seed) mod 1000)
  /. 100.0
  -. 5.0

let binary_target x row =
  if Matrix.get x row 0 +. (0.25 *. Matrix.get x row 1) > 1.0 then 9 else -4

let multiclass_target x row =
  if Matrix.get x row 0 +. (0.25 *. Matrix.get x row 1) > 1.0 then 9
  else if Matrix.get x row 2 -. (0.2 *. Matrix.get x row 3) > 0.0 then 2
  else -4

let checksum values =
  values
  |> Array.map (Printf.sprintf "%.17g")
  |> Array.to_list |> String.concat ":"

let () =
  if Array.length Sys.argv <> 6 then
    fail "usage: sgd_classifier_worker SAMPLES FEATURES SEED ETA0 EPOCHS";
  let samples = int_of_string Sys.argv.(1) in
  let features = int_of_string Sys.argv.(2) in
  let seed = int_of_string Sys.argv.(3) in
  let eta0 = float_of_string Sys.argv.(4) in
  let epochs = int_of_string Sys.argv.(5) in
  if features < 4 then
    fail "SGD classification benchmark requires at least 4 features";
  let allocated_before = Gc.allocated_bytes () in
  let x =
    Matrix.init ~rows:samples ~columns:features (feature ~seed) |> get_data
  in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  let binary = Target.classification (Array.init samples (binary_target x)) in
  let multiclass =
    Target.classification (Array.init samples (multiclass_target x))
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:samples
      (Array.init samples (fun row -> 1.0 +. (Float.of_int (row mod 5) *. 0.25)))
    |> get_data
  in
  let specification loss =
    Sgd_classifier.create ~loss ~penalty:Sgd_classifier.No_penalty
      ~learning_rate:Sgd_classifier.Constant ~eta0 ~max_epochs:epochs
      ~shuffle:false ()
    |> get
  in
  let rng = Rng.create (Seed.of_int seed) in
  let hinge =
    Sgd_classifier.fit
      (specification Sgd_classifier.Hinge)
      ~sample_weight ~rng ~feature_schema ~x ~y:binary ()
    |> get
  in
  let log_loss =
    Sgd_classifier.fit
      (specification Sgd_classifier.Log_loss)
      ~sample_weight ~rng ~feature_schema ~x ~y:multiclass ()
    |> get
  in
  let rec train remaining checkpoint =
    if remaining = 0 then checkpoint
    else
      Sgd_classifier.partial_fit checkpoint ~sample_weight ~feature_schema ~x
        ~y:multiclass ()
      |> get
      |> train (remaining - 1)
  in
  let incremental =
    Sgd_classifier.start
      (specification Sgd_classifier.Log_loss)
      ~rng ~feature_schema ~classes:[| -4; 2; 9 |]
    |> get |> train epochs |> Sgd_classifier.to_fitted |> get
  in
  let hinge_decisions =
    Sgd_classifier.binary_decision_function hinge ~feature_schema ~x |> get
  in
  let hinge_predictions =
    Sgd_classifier.predict hinge ~feature_schema ~x
    |> get |> Target.classification_values
  in
  let probabilities =
    Sgd_classifier.predict_proba log_loss ~feature_schema ~x |> get
  in
  let predictions =
    Sgd_classifier.predict log_loss ~feature_schema ~x
    |> get |> Target.classification_values
  in
  let incremental_probabilities =
    Sgd_classifier.predict_proba incremental ~feature_schema ~x |> get
  in
  let last = samples - 1 in
  let signature =
    [|
      Matrix.get (Sgd_classifier.coefficients hinge) 0 0;
      Vector.get (Sgd_classifier.intercepts hinge) 0;
      Vector.get hinge_decisions 0;
      Vector.get hinge_decisions last;
      Float.of_int hinge_predictions.(0);
      Float.of_int hinge_predictions.(last);
      Matrix.get (Sgd_classifier.coefficients log_loss) 0 0;
      Vector.get (Sgd_classifier.intercepts log_loss) 0;
      Matrix.get probabilities 0 0;
      Matrix.get probabilities 0 1;
      Matrix.get probabilities 0 2;
      Matrix.get probabilities last 0;
      Matrix.get probabilities last 1;
      Matrix.get probabilities last 2;
      Float.of_int predictions.(0);
      Float.of_int predictions.(last);
      Matrix.get incremental_probabilities last 2;
      Float.of_int (Sgd_classifier.report log_loss).Sgd_classifier.updates;
      Float.of_int (Sgd_classifier.report incremental).Sgd_classifier.updates;
    |]
  in
  let allocated_words =
    (Gc.allocated_bytes () -. allocated_before)
    /. Float.of_int (Sys.word_size / 8)
  in
  Printf.printf
    {|{"allocated_words":%.0f,"checksum":%S,"features":%d,"ocaml":%S,"operations":["weighted_binary_hinge_sgd_fit_decision_predict","weighted_multiclass_log_loss_sgd_fit_proba_predict","weighted_multiclass_log_loss_sgd_partial_fit_proba"],"samples":%d,"signature":[%s],"threadpools":[{"architecture":null,"internal_api":"native","num_threads":1,"prefix":"modelkit","user_api":"ocaml","version":null}]}|}
    allocated_words (checksum signature) features Sys.ocaml_version samples
    (signature
    |> Array.map (Printf.sprintf "%.17g")
    |> Array.to_list |> String.concat ",")
