open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let ( let* ) = Result.bind
let regression values = Target.regression (Vector.of_array values) |> get_data
let matrix values = Matrix.of_arrays values |> get_data
let rng () = Rng.create (Seed.of_int 42)

let invalid reason =
  Error
    (Error.make ~remediation:"provide aligned training inputs"
       (Error.Validation { name = "test transformer"; reason }))

let data_result result =
  Result.map_error
    (fun error ->
      Error.of_data_error ~remediation:"provide valid transformer output" error)
    result

module Target_summary = struct
  type t = bool
  type params = bool
  type target = Target.regression Target.t
  type rng = Rng.t

  type fitted = {
    schema : Feature_schema.t;
    mean : float;
    random : float;
    alignment : bool;
  }

  let clone t = t
  let params t = t

  let fit alignment ?sample_weight ~rng ~feature_schema ~x ~y () =
    match y with
    | None -> invalid "target was not routed"
    | Some y ->
        let values = Target.regression_values y in
        let numerator = ref 0. in
        let denominator = ref 0. in
        let aligned = ref true in
        for row = 0 to Matrix.rows x - 1 do
          let value = Vector.get values row in
          let weight =
            match sample_weight with
            | None -> 1.
            | Some weights -> Sample_weight.get weights row
          in
          if alignment then (
            let id = Matrix.get x row 0 in
            aligned := !aligned && value = (10. *. id) +. 3.;
            if Option.is_some sample_weight then
              aligned := !aligned && weight = id +. 1.);
          numerator := !numerator +. (weight *. value);
          denominator := !denominator +. weight
        done;
        if not !aligned then invalid "features, targets, or weights misaligned"
        else
          let random, _ = Rng.next_float rng in
          Ok
            {
              schema = feature_schema;
              mean = !numerator /. !denominator;
              random;
              alignment;
            }

  let transform fitted ~feature_schema:_ ~x =
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
      (fun _ column -> if column = 0 then fitted.mean else fitted.random)
    |> data_result

  let fitted_params fitted = fitted.alignment
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
end

module First_column = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = target
  type fitted = Feature_schema.t
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    Ok feature_schema

  let predict _ ~feature_schema:_ ~x =
    let* values =
      Vector.init ~length:(Matrix.rows x) (fun row -> Matrix.get x row 0)
      |> data_result
    in
    Target.regression values |> data_result

  let fitted_params _ = ()
  let feature_schema fitted = fitted
end

module Identity = struct
  type t = (unit -> unit) * bool
  type params = t
  type target = unit
  type fitted = Feature_schema.t * t
  type rng = Rng.t

  let clone t = t
  let params t = t

  let fit ((observe, weighted) as spec) ?sample_weight ~rng:_ ~feature_schema
      ~x:_ ~y () =
    observe ();
    if Option.is_some y then invalid "unsupervised stage received targets"
    else if Option.is_some sample_weight <> weighted then
      invalid "unexpected unsupervised weight routing"
    else Ok (feature_schema, spec)

  let transform _ ~feature_schema:_ ~x = Ok x
  let fitted_params (_, spec) = spec
  let input_schema (schema, _) = schema
  let output_schema (schema, _) = schema
end

module Class_summary = struct
  include Target_summary

  type target = Target.classification Target.t

  let fit alignment ?sample_weight ~rng ~feature_schema ~x ~y () =
    let y =
      Option.map
        (fun target ->
          Target.classification_values target
          |> Array.map Float.of_int |> regression)
        y
    in
    Target_summary.fit alignment ?sample_weight ~rng ~feature_schema ~x ~y ()
end

module Dropping_summary = struct
  include Target_summary

  let transform _ ~feature_schema:_ ~x =
    Matrix.init
      ~rows:(Matrix.rows x - 1)
      ~columns:(Matrix.columns x)
      (fun row column -> Matrix.get x row column)
    |> data_result
end

let x = Array.init 12 (fun row -> [| Float.of_int row; 0. |]) |> matrix
let schema = Feature_schema.of_matrix x |> get_data
let y = Array.init 12 (fun row -> Float.of_int ((10 * row) + 3)) |> regression

let weights =
  Sample_weight.of_array ~expected_length:12
    (Array.init 12 (fun row -> Float.of_int (row + 1)))
  |> get_data

let terminal () =
  Pipeline.estimator ~name:"model" (module First_column) () |> get

let stage ?route_sample_weight ?(alignment = true) ?(name = "summary") () =
  Pipeline.Supervised.transformer ?route_sample_weight ~name
    (module Target_summary)
    alignment
  |> get

let pipeline ?route_sample_weight ?(observe = fun () -> ()) () =
  let prefix =
    Pipeline.transformer ~route_sample_weight:true ~name:"prefix"
      (module Identity)
      (observe, true)
    |> get |> Pipeline.Supervised.unsupervised
  in
  let suffix =
    Pipeline.transformer ~name:"suffix" (module Identity) ((fun () -> ()), false)
    |> get |> Pipeline.Supervised.unsupervised
  in
  let open Pipeline.Supervised in
  let builder = add_transformer empty prefix |> get in
  let builder =
    add_transformer builder (stage ?route_sample_weight ()) |> get
  in
  let builder = add_transformer builder suffix |> get in
  set_estimator builder (terminal ()) |> get

let fit specification =
  Pipeline.fit specification ~sample_weight:weights ~rng:(rng ())
    ~feature_schema:schema ~x ~y ()
  |> get

let predict fitted =
  Pipeline.predict fitted ~feature_schema:schema ~x
  |> get |> Target.regression_values |> Vector.to_array

let mean indices weighted =
  let numerator, denominator =
    Array.fold_left
      (fun (numerator, denominator) row ->
        let weight = if weighted then Float.of_int (row + 1) else 1. in
        ( numerator +. (weight *. Float.of_int ((10 * row) + 3)),
          denominator +. weight ))
      (0., 0.) indices
  in
  numerator /. denominator

let test_routing () =
  List.iter
    (fun weighted ->
      let specification = pipeline ~route_sample_weight:weighted () in
      let fitted = fit specification in
      Alcotest.(check (array (Alcotest.float 1e-12)))
        "training-only weighted target mean"
        (Array.make 12 (mean (Array.init 12 Fun.id) weighted))
        (predict fitted);
      let clone = fit (Pipeline.clone specification) in
      Alcotest.(check (array (array (Alcotest.float 0.))))
        "clone and stage RNG repeat"
        (Pipeline.transform fitted ~feature_schema:schema ~x
        |> get |> Matrix.to_arrays)
        (Pipeline.transform clone ~feature_schema:schema ~x
        |> get |> Matrix.to_arrays);
      let changed_x = matrix [| [| -1000.; 1. |] |] in
      let prediction =
        Pipeline.predict fitted ~feature_schema:schema ~x:changed_x
        |> get |> Target.regression_values
      in
      Alcotest.(check (Alcotest.float 1e-12))
        "inference needs no training metadata"
        (mean (Array.init 12 Fun.id) weighted)
        (Vector.get prediction 0))
    [ false; true ];
  let default = fit (pipeline ()) in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "weights are opt-in by default"
    (Array.make 12 (mean (Array.init 12 Fun.id) false))
    (predict default)

let test_alignment_validation () =
  let calls = ref 0 in
  let specification = pipeline ~observe:(fun () -> incr calls) () in
  let[@warning "-4"] check_error expected_context = function
    | Ok _ -> Alcotest.fail "expected alignment error"
    | Error error ->
        Alcotest.(check bool)
          "typed length error" true
          (match Error.kind error with
          | Error.Data (Data_error.Length_mismatch _) -> true
          | _ -> false);
        Alcotest.(check bool)
          "error context" true
          (Error.context error = expected_context);
        Alcotest.(check int) "no transformer fitted" 0 !calls
  in
  Pipeline.fit specification ~sample_weight:weights ~rng:(rng ())
    ~feature_schema:schema ~x ~y:(regression [| 3. |]) ()
  |> check_error [ Error.Stage "summary" ];
  let short_weights =
    Sample_weight.of_array ~expected_length:1 [| 1. |] |> get_data
  in
  Pipeline.fit specification ~sample_weight:short_weights ~rng:(rng ())
    ~feature_schema:schema ~x ~y ()
  |> check_error []

let dataset =
  Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights ~x ~y
    ()
  |> get_data

let splitter () =
  K_fold.create ~folds:3 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let test_cross_validation () =
  let specification = pipeline ~route_sample_weight:true () in
  let run dataset =
    Cross_validation.Regression.cross_validate ~return_models:true
      ~return_indices:true ~failure_policy:Cross_validation.Record
      ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 2026) specification dataset
    |> get
  in
  let report = run dataset in
  Alcotest.(check int)
    "all folds fit" 3
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let train = Option.get fold.Cross_validation.train_indices in
      let fitted = Option.get fold.Cross_validation.model in
      Alcotest.(check (array (Alcotest.float 1e-12)))
        "fold-local target and weight alignment"
        (Array.make 12 (mean train true))
        (predict fitted))
    (Cross_validation.folds report);
  let first = (Cross_validation.folds report).(0) in
  let held_out = Option.get first.Cross_validation.test_indices in
  let changed_targets = Target.regression_values y |> Vector.to_array in
  Array.iter (fun row -> changed_targets.(row) <- 1e6) held_out;
  let changed =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights ~x
      ~y:(regression changed_targets)
      ()
    |> get_data
  in
  let changed_first = (Cross_validation.folds (run changed)).(0) in
  Alcotest.(check (array (Alcotest.float 0.)))
    "held-out targets cannot change training"
    (predict (Option.get first.Cross_validation.model))
    (predict (Option.get changed_first.Cross_validation.model))

let test_grid_search () =
  let grid =
    Grid_search.create ~base:()
      ~build:(fun () -> Ok (pipeline ~route_sample_weight:true ()))
      [||]
    |> get
  in
  let report =
    Grid_search.Regression.search ~grid ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 7) dataset
    |> get
  in
  let selected = Grid_search.selection report |> get in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "refit receives all aligned training rows"
    (Array.make 12 (mean (Array.init 12 Fun.id) true))
    (predict selected.Grid_search.selected_model)

let test_classification () =
  let class_y =
    Target.classification (Array.init 12 (fun row -> if row < 9 then 0 else 1))
  in
  let stage =
    Pipeline.Supervised.transformer ~route_sample_weight:true ~name:"classes"
      (module Class_summary)
      false
    |> get
  in
  let builder =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty stage |> get
  in
  let terminal =
    Pipeline.classifier ~class_weight:Class_weight.balanced ~name:"logistic"
      (module Logistic_regression)
      ~predict_proba:Logistic_regression.predict_proba
      ~decision_function:Logistic_regression.decision_function
      ~classes:Logistic_regression.classes
      (Logistic_regression.create () |> get)
    |> get
  in
  let specification =
    Pipeline.Supervised.set_estimator builder terminal |> get
  in
  let class_dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights ~x
      ~y:class_y ()
    |> get_data
  in
  let splitter =
    Stratified_k_fold.create ~folds:3 ~shuffle:true ()
    |> get
    |> Cross_validation.target_aware_splitter (module Stratified_k_fold)
  in
  let report =
    Cross_validation.Binary_classification.cross_validate ~return_models:true
      ~return_indices:true ~splitter
      ~scorers:[| Binary_classification_scorer.neg_log_loss () |]
      ~seed:(Seed.of_int 21) specification class_dataset
    |> get
  in
  Alcotest.(check int)
    "classification folds succeed" 3
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let train = Option.get fold.Cross_validation.train_indices in
      let numerator, denominator =
        Array.fold_left
          (fun (n, d) row ->
            let w = Sample_weight.get weights row in
            ((n +. if row < 9 then 0. else w), d +. w))
          (0., 0.) train
      in
      let fitted = Option.get fold.Cross_validation.model in
      let transformed =
        Pipeline.transform fitted ~feature_schema:schema ~x |> get
      in
      Alcotest.check (Alcotest.float 1e-12) "raw fold weights reach transformer"
        (numerator /. denominator)
        (Matrix.get transformed 0 0);
      let proba =
        Pipeline.predict_proba fitted ~feature_schema:schema ~x |> get
      in
      Alcotest.check (Alcotest.float 1e-7)
        "terminal alone resolves balanced weights" 0.5 (Matrix.get proba 0 1);
      Alcotest.(check (array int))
        "class dispatch" [| 0; 1 |]
        (Pipeline.classes fitted |> get);
      Alcotest.(check int)
        "decision dispatch" 12
        (Pipeline.decision_function fitted ~feature_schema:schema ~x
        |> get |> Vector.length))
    (Cross_validation.folds report)

let[@warning "-4"] test_output_failure () =
  let stage =
    Pipeline.Supervised.transformer ~name:"drop-row"
      (module Dropping_summary)
      true
    |> get
  in
  let specification =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty stage |> get
    |> fun builder ->
    Pipeline.Supervised.set_estimator builder (terminal ()) |> get
  in
  let report =
    Cross_validation.Regression.cross_validate
      ~failure_policy:Cross_validation.Record ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 0) specification dataset
    |> get
  in
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      Alcotest.(check int)
        "one fitting failure" 1
        (Array.length fold.Cross_validation.failures);
      let failure = fold.Cross_validation.failures.(0) in
      Alcotest.(check bool)
        "shape mismatch remains typed" true
        (match Error.kind failure.Cross_validation.error with
        | Error.Shape_mismatch _ -> true
        | _ -> false);
      Alcotest.(check bool)
        "fold and supervised stage context" true
        (Error.context failure.Cross_validation.error
        = [
            Error.Fold fold.Cross_validation.fold_index; Error.Stage "drop-row";
          ]))
    (Cross_validation.folds report)

let[@warning "-4"] test_names_and_artifacts () =
  let open Pipeline.Supervised in
  let builder = add_transformer empty (stage ()) |> get in
  let expect_error = function
    | Ok _ -> Alcotest.fail "expected typed error"
    | Error _ -> ()
  in
  transformer ~name:" " (module Target_summary) false |> expect_error;
  add_transformer builder (stage ()) |> expect_error;
  let duplicate =
    Pipeline.estimator ~name:"summary" (module First_column) () |> get
  in
  set_estimator builder duplicate |> expect_error;
  let linear =
    Artifact.linear_regression_estimator ~name:"linear"
      (Linear_regression.create ())
    |> get
  in
  let fitted = fit (set_estimator builder linear |> get) in
  Artifact.encode_regression fitted |> fun result ->
  match result with
  | Ok _ -> Alcotest.fail "supervised stage has no reviewed codec"
  | Error error ->
      Alcotest.(check bool)
        "unsupported codec is an artifact error" true
        (match Error.kind error with Error.Artifact _ -> true | _ -> false)

let test_unsupervised_compatibility () =
  let scaler =
    Artifact.standard_scaler_stage ~route_sample_weight:true ~name:"scale"
      (Standard_scaler.create ())
    |> get
  in
  let linear =
    Artifact.linear_regression_estimator ~name:"linear"
      (Linear_regression.create ())
    |> get
  in
  let legacy =
    Pipeline.add_transformer Pipeline.empty scaler |> get |> fun builder ->
    Pipeline.set_estimator builder linear |> get
  in
  let lifted =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty
      (Pipeline.Supervised.unsupervised scaler)
    |> get
    |> fun builder -> Pipeline.Supervised.set_estimator builder linear |> get
  in
  Alcotest.(check bytes)
    "lifting preserves fitted artifact bytes"
    (Artifact.encode_regression (fit legacy) |> get)
    (Artifact.encode_regression (fit lifted) |> get);
  let supervised =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty
      (stage ~route_sample_weight:true ())
    |> get
    |> fun builder ->
    Pipeline.Supervised.set_estimator builder (terminal ()) |> get
  in
  let fitted =
    Pipeline.fit supervised ~rng:(rng ()) ~feature_schema:schema ~x ~y () |> get
  in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "absent optional weights remain absent"
    (Array.make 12 (mean (Array.init 12 Fun.id) false))
    (predict fitted)

let () =
  Alcotest.run "supervised pipelines"
    [
      ( "routing",
        [
          Alcotest.test_case "mixed stages, weights, clone, inference" `Quick
            test_routing;
          Alcotest.test_case "pre-fit alignment validation" `Quick
            test_alignment_validation;
          Alcotest.test_case "fold alignment and held-out target leakage" `Quick
            test_cross_validation;
          Alcotest.test_case "grid-search refit" `Quick test_grid_search;
          Alcotest.test_case "classification and terminal class weights" `Quick
            test_classification;
          Alcotest.test_case "row preservation and contextual failures" `Quick
            test_output_failure;
          Alcotest.test_case "names and unsupported artifacts" `Quick
            test_names_and_artifacts;
          Alcotest.test_case "unsupervised codecs and absent optional weights"
            `Quick test_unsupervised_compatibility;
        ] );
    ]
