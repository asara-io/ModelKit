open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail (Error.to_string error)
  | Ok _ -> Alcotest.fail "expected a typed error"

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_data = function Error.Data _ -> true | _ -> false

let check_close label expected observed =
  Alcotest.(check bool)
    label true
    (Float.abs (expected -. observed)
    <= 1e-12 *. Float.max 1.0 (Float.abs expected))

let check_floats label expected observed =
  Alcotest.(check int)
    (label ^ " length") (Array.length expected) (Array.length observed);
  Array.iteri
    (fun index value ->
      check_close (Format.sprintf "%s[%d]" label index) value observed.(index))
    expected

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let test_balanced () =
  let y = Target.classification [| 1; 1; 1; 1; 0; 0 |] in
  let resolved = Class_weight.class_weights Class_weight.balanced y |> get in
  Alcotest.(check (array int))
    "ascending labels" [| 0; 1 |] (Array.map fst resolved);
  check_floats "sklearn balanced example" [| 1.5; 0.75 |]
    (Array.map snd resolved);
  let rows = Class_weight.resolve Class_weight.balanced y |> get in
  check_floats "row weights"
    [| 0.75; 0.75; 0.75; 0.75; 1.5; 1.5 |]
    (Sample_weight.to_vector rows |> Vector.to_array);
  let sample_weight = weights [| 1.0; 1.0; 0.0; 0.0; 2.0; 2.0 |] in
  let weighted =
    Class_weight.class_weights Class_weight.balanced ~sample_weight y |> get
  in
  check_floats "weighted balanced frequencies" [| 0.75; 1.5 |]
    (Array.map snd weighted);
  let rows =
    Class_weight.resolve Class_weight.balanced ~sample_weight y |> get
  in
  check_floats "zero-weight rows stay zero and others multiply"
    [| 1.5; 1.5; 0.0; 0.0; 1.5; 1.5 |]
    (Sample_weight.to_vector rows |> Vector.to_array);
  let absent = weights [| 1.0; 1.0; 1.0; 1.0; 0.0; 0.0 |] in
  let only_ones =
    Class_weight.class_weights Class_weight.balanced ~sample_weight:absent y
    |> get
  in
  Alcotest.(check (array int))
    "classes without weight are absent" [| 1 |] (Array.map fst only_ones);
  check_floats "single effective class weighs one" [| 1.0 |]
    (Array.map snd only_ones)

let test_explicit () =
  let y = Target.classification [| 5; 5; -1; 9 |] in
  let specification =
    Class_weight.explicit [ (9, 4.0); (-1, 0.5); (42, 7.0) ] |> get
  in
  let resolved = Class_weight.class_weights specification y |> get in
  Alcotest.(check (array int)) "labels" [| -1; 5; 9 |] (Array.map fst resolved);
  check_floats "listed weights apply and unlisted default to one"
    [| 0.5; 1.0; 4.0 |] (Array.map snd resolved);
  let rows =
    Class_weight.resolve specification
      ~sample_weight:(weights [| 2.0; 1.0; 1.0; 0.5 |])
      y
    |> get
  in
  check_floats "explicit row weights" [| 2.0; 1.0; 0.5; 2.0 |]
    (Sample_weight.to_vector rows |> Vector.to_array);
  List.iter
    (expect_error is_validation)
    [
      Class_weight.explicit [];
      Class_weight.explicit [ (1, 1.0); (1, 2.0) ];
      Class_weight.explicit [ (1, -1.0) ];
      Class_weight.explicit [ (1, Float.nan) ];
    ];
  let one_row = Target.classification [| 5 |] in
  let effective_zero = Target.classification [| 5; 5; 5; 5 |] in
  Class_weight.class_weights Class_weight.balanced one_row
  |> get |> Array.map snd
  |> check_floats "single class balances to one" [| 1.0 |];
  Class_weight.resolve specification
    ~sample_weight:(weights [| 0.0; 0.0; 1.0; 0.0 |])
    effective_zero
  |> get |> Sample_weight.to_vector |> Vector.to_array
  |> check_floats "zero-weight rows stay zero under explicit weights"
       [| 0.0; 0.0; 1.0; 0.0 |];
  Class_weight.resolve Class_weight.balanced ~sample_weight:(weights [| 1.0 |])
    y
  |> expect_error is_data

let x =
  Matrix.of_arrays
    [|
      [| -3.0 |];
      [| -2.0 |];
      [| -1.0 |];
      [| 1.0 |];
      [| 2.0 |];
      [| 3.0 |];
      [| 4.0 |];
      [| 5.0 |];
    |]
  |> get_data

let feature_schema = Feature_schema.of_matrix x |> get_data
let y = Target.classification [| 0; 0; 0; 0; 0; 0; 1; 1 |]
let rng () = Rng.create (Seed.of_int 42)

let logistic_pipeline ?class_weight () =
  Pipeline.set_estimator Pipeline.empty
    (Pipeline.classifier ?class_weight ~name:"logistic"
       ~predict_proba:Logistic_regression.predict_proba
       ~classes:Logistic_regression.classes
       (module Logistic_regression)
       (Logistic_regression.create ~c:0.5 () |> get)
    |> get)
  |> get

let test_pipeline_resolution () =
  let plain =
    Pipeline.fit (logistic_pipeline ()) ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  let balanced =
    Pipeline.fit
      (logistic_pipeline ~class_weight:Class_weight.balanced ())
      ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  let manual =
    Logistic_regression.fit
      (Logistic_regression.create ~c:0.5 () |> get)
      ~sample_weight:(Class_weight.resolve Class_weight.balanced y |> get)
      ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  let probabilities fitted =
    Pipeline.predict_proba fitted ~feature_schema ~x |> get
  in
  let manual_probabilities =
    Logistic_regression.predict_proba manual ~feature_schema ~x |> get
  in
  Alcotest.(check bool)
    "pipeline resolution equals manual resolution" true
    (Matrix.to_arrays (probabilities balanced)
    = Matrix.to_arrays manual_probabilities);
  Alcotest.(check bool)
    "balanced weights change the fit" true
    (Matrix.to_arrays (probabilities plain)
    <> Matrix.to_arrays manual_probabilities);
  Alcotest.(check bool)
    "minority probability rises under balanced weights" true
    (Matrix.get (probabilities balanced) 5 1
    > Matrix.get (probabilities plain) 5 1);
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let splitter =
    Stratified_k_fold.create ~folds:2 ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let report =
    Cross_validation.Binary_classification.cross_validate ~return_models:true
      ~return_indices:true ~splitter
      ~scorers:[| Binary_classification_scorer.balanced_accuracy () |]
      ~seed:(Seed.of_int 42)
      (logistic_pipeline ~class_weight:Class_weight.balanced ())
      dataset
    |> get
  in
  Alcotest.(check int)
    "balanced folds succeed" 2
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let train = Option.get fold.Cross_validation.train_indices in
      let view = Row_view.create ~source_size:8 train |> get_data in
      let fold_y = Target.select y view |> get_data in
      let fold_x =
        Matrix.init ~rows:(Array.length train) ~columns:1 (fun row column ->
            Matrix.get x train.(row) column)
        |> get_data
      in
      let expected =
        Logistic_regression.fit
          (Logistic_regression.create ~c:0.5 () |> get)
          ~sample_weight:
            (Class_weight.resolve Class_weight.balanced fold_y |> get)
          ~rng:(Rng.create (Seed.of_int 0))
          ~feature_schema ~x:fold_x ~y:fold_y ()
        |> get
      in
      let model = Option.get fold.Cross_validation.model in
      Alcotest.(check bool)
        "fold-local balanced resolution matches the fold's own rows" true
        (Matrix.to_arrays
           (Pipeline.predict_proba model ~feature_schema ~x |> get)
        = Matrix.to_arrays
            (Logistic_regression.predict_proba expected ~feature_schema ~x
            |> get)))
    (Cross_validation.folds report)

let test_transformer_routing () =
  let sample_weight = weights [| 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 4.0; 4.0 |] in
  let build ?route_sample_weight () =
    let scaler =
      Pipeline.transformer ?route_sample_weight ~name:"scale"
        (module Standard_scaler)
        (Standard_scaler.create ())
      |> get
    in
    Pipeline.set_estimator
      (Pipeline.add_transformer Pipeline.empty scaler |> get)
      (Pipeline.estimator ~name:"logistic"
         (module Logistic_regression)
         (Logistic_regression.create () |> get)
      |> get)
    |> get
  in
  let transform pipeline =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
    |> fun fitted ->
    Pipeline.transform fitted ~feature_schema ~x |> get |> Matrix.to_arrays
  in
  let unrouted = transform (build ()) in
  let routed = transform (build ~route_sample_weight:true ()) in
  let unweighted_scaler =
    Standard_scaler.fit
      (Standard_scaler.create ())
      ~rng:(rng ()) ~feature_schema ~x ~y:None ()
    |> get
  in
  let weighted_scaler =
    Standard_scaler.fit
      (Standard_scaler.create ())
      ~sample_weight ~rng:(rng ()) ~feature_schema ~x ~y:None ()
    |> get
  in
  let expected scaler =
    Standard_scaler.transform scaler ~feature_schema ~x
    |> get |> Matrix.to_arrays
  in
  Alcotest.(check bool)
    "default stages fit unweighted" true
    (unrouted = expected unweighted_scaler);
  Alcotest.(check bool)
    "routed stages fit weighted" true
    (routed = expected weighted_scaler);
  Alcotest.(check bool) "the two differ" true (routed <> unrouted);
  let imputer =
    Pipeline.transformer ~route_sample_weight:true ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let rejecting =
    Pipeline.set_estimator
      (Pipeline.add_transformer Pipeline.empty imputer |> get)
      (Pipeline.estimator ~name:"logistic"
         (module Logistic_regression)
         (Logistic_regression.create () |> get)
      |> get)
    |> get
  in
  Pipeline.fit rejecting ~sample_weight ~rng:(rng ()) ~feature_schema ~x ~y ()
  |> expect_error is_validation

let test_artifact_round_trip () =
  let pipeline =
    Pipeline.set_estimator
      (Pipeline.add_transformer Pipeline.empty
         (Artifact.standard_scaler_stage ~route_sample_weight:true ~name:"scale"
            (Standard_scaler.create ())
         |> get)
      |> get)
      (Artifact.logistic_regression_estimator
         ~class_weight:Class_weight.balanced ~name:"model"
         (Logistic_regression.create () |> get)
      |> get)
    |> get
  in
  let sample_weight = weights [| 1.0; 2.0; 1.0; 2.0; 1.0; 2.0; 1.0; 2.0 |] in
  let fitted =
    Pipeline.fit pipeline ~sample_weight ~rng:(rng ()) ~feature_schema ~x ~y ()
    |> get
  in
  let loaded =
    Artifact.encode_binary_classification fitted
    |> get |> Artifact.decode_binary_classification |> get |> Artifact.model
  in
  Alcotest.(check bool)
    "weighted, class-weighted pipeline survives the artifact round trip" true
    (Matrix.to_arrays (Pipeline.predict_proba fitted ~feature_schema ~x |> get)
    = Matrix.to_arrays (Pipeline.predict_proba loaded ~feature_schema ~x |> get)
    )

let () =
  Alcotest.run "class weights"
    [
      ( "resolution",
        [
          Alcotest.test_case "balanced" `Quick test_balanced;
          Alcotest.test_case "explicit and errors" `Quick test_explicit;
        ] );
      ( "propagation",
        [
          Alcotest.test_case "pipeline and fold-local cross-validation" `Quick
            test_pipeline_resolution;
          Alcotest.test_case "transformer routing" `Quick
            test_transformer_routing;
          Alcotest.test_case "artifact round trip" `Quick
            test_artifact_round_trip;
        ] );
    ]
