open Modelkit

let fail message = raise (Failure message)
let check condition message = if not condition then fail message

let get_data = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> fail (Error.to_string error)

let error kind remediation = Error.make ~remediation kind

let validate_schema expected observed =
  if Feature_schema.equal expected observed then Ok ()
  else
    Error
      (error
         (Error.Feature_schema_mismatch { expected; observed })
         "provide features with the fitted schema")

let validate_matrix schema matrix =
  match Feature_schema.validate_matrix schema matrix with
  | Ok () -> Ok ()
  | Error data_error ->
      Error
        (Error.of_data_error ~remediation:"provide aligned features" data_error)

let validate_target x target =
  let expected = Matrix.rows x in
  let observed = Target.length target in
  if expected = observed then Ok ()
  else
    Error
      (error
         (Error.Shape_mismatch
            {
              name = "target";
              expected = [ expected ];
              observed = [ observed ];
            })
         "provide one target per sample")

let regression values = Target.regression (Vector.of_array values) |> get_data

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let rng () = Rng.create (Seed.of_int 1729)

type first_column_fitted = { schema : Feature_schema.t }

module First_column_regressor :
  REGRESSOR
    with type t = unit
     and type params = unit
     and type fitted = first_column_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = first_column_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y () =
    match validate_matrix feature_schema x with
    | Error _ as error -> error
    | Ok () -> (
        match validate_target x y with
        | Error _ as error -> error
        | Ok () -> Ok { schema = feature_schema })

  let predict fitted ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () -> (
        match validate_matrix feature_schema x with
        | Error _ as error -> error
        | Ok () ->
            Target.regression
              (Vector.init ~length:(Matrix.rows x) (fun row ->
                   Matrix.get x row 0)
              |> get_data)
            |> Result.map_error (fun data_error ->
                Error.of_data_error
                  ~remediation:"produce finite first-column predictions"
                  data_error))

  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

let first_column_decision_function fitted ~feature_schema ~x =
  First_column_regressor.predict fitted ~feature_schema ~x
  |> Result.map Target.regression_values

let first_column_predict_proba fitted ~feature_schema ~x =
  match first_column_decision_function fitted ~feature_schema ~x with
  | Error _ as error -> error
  | Ok decision ->
      Matrix.init ~rows:(Vector.length decision) ~columns:2 (fun row column ->
          let positive =
            1.0 /. (1.0 +. Float.exp (-.Vector.get decision row))
          in
          if column = 0 then 1.0 -. positive else positive)
      |> Result.map_error (fun data_error ->
          Error.of_data_error
            ~remediation:"provide representable probability dimensions"
            data_error)

type weighted_mean_fitted = { schema : Feature_schema.t; weighted_mean : float }

module Weighted_mean_regressor :
  REGRESSOR
    with type t = unit
     and type params = unit
     and type fitted = weighted_mean_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = weighted_mean_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    match
      (validate_matrix feature_schema x, validate_target x y, sample_weight)
    with
    | Error error, _, _ | _, Error error, _ -> Error error
    | Ok (), Ok (), None ->
        Error
          (error
             (Error.Validation
                {
                  name = "weighted mean estimator";
                  reason = "sample weights are required";
                })
             "provide sample weights")
    | Ok (), Ok (), Some weights ->
        let values = Target.regression_values y in
        let numerator = ref 0.0 in
        let denominator = ref 0.0 in
        for index = 0 to Vector.length values - 1 do
          let weight = Sample_weight.get weights index in
          numerator := !numerator +. (weight *. Vector.get values index);
          denominator := !denominator +. weight
        done;
        Ok
          {
            schema = feature_schema;
            weighted_mean = !numerator /. !denominator;
          }

  let predict (fitted : weighted_mean_fitted) ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () ->
        Target.regression
          (Vector.of_array (Array.make (Matrix.rows x) fitted.weighted_mean))
        |> Result.map_error (fun data_error ->
            Error.of_data_error ~remediation:"produce finite predictions"
              data_error)

  let fitted_params _ = ()
  let feature_schema (fitted : weighted_mean_fitted) = fitted.schema
end

type random_fitted = { schema : Feature_schema.t; value : float }

module Random_regressor :
  REGRESSOR
    with type t = unit
     and type params = unit
     and type fitted = random_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = random_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng ~feature_schema ~x ~y () =
    match (validate_matrix feature_schema x, validate_target x y) with
    | Error error, _ | _, Error error -> Error error
    | Ok (), Ok () ->
        let value, _ = Rng.next_float rng in
        Ok { schema = feature_schema; value }

  let predict (fitted : random_fitted) ~feature_schema ~x =
    match validate_schema fitted.schema feature_schema with
    | Error _ as error -> error
    | Ok () ->
        Target.regression
          (Vector.of_array (Array.make (Matrix.rows x) fitted.value))
        |> Result.map_error (fun data_error ->
            Error.of_data_error ~remediation:"produce finite predictions"
              data_error)

  let fitted_params _ = ()
  let feature_schema (fitted : random_fitted) = fitted.schema
end

type dropping_fitted = { dropping_schema : Feature_schema.t }

module Dropping_transformer :
  TRANSFORMER
    with type t = unit
     and type params = unit
     and type target = unit
     and type fitted = dropping_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = unit
  type fitted = dropping_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    Ok { dropping_schema = feature_schema }

  let transform _fitted ~feature_schema:_ ~x =
    Matrix.init
      ~rows:(Int.max 0 (Matrix.rows x - 1))
      ~columns:(Matrix.columns x)
      (fun row column -> Matrix.get x row column)
    |> Result.map_error (fun data_error ->
        Error.of_data_error ~remediation:"provide representable dimensions"
          data_error)

  let fitted_params _ = ()
  let input_schema fitted = fitted.dropping_schema
  let output_schema fitted = fitted.dropping_schema
end

let add_transformer builder ~name transformer_module specification =
  let transformer =
    Pipeline.transformer ~name transformer_module specification |> get
  in
  Pipeline.add_transformer builder transformer |> get

let first_column_estimator ?(capabilities = true) name =
  if capabilities then
    Pipeline.estimator ~name
      (module First_column_regressor)
      ~decision_function:first_column_decision_function
      ~predict_proba:first_column_predict_proba ()
    |> get
  else Pipeline.estimator ~name (module First_column_regressor) () |> get

let fitted_preprocessing_pipeline () =
  let schema = named_schema [| "value" |] in
  let training = Matrix.of_arrays [| [| 1.0 |]; [| 3.0 |] |] |> get_data in
  let target = regression [| 0.0; 1.0 |] in
  let builder =
    add_transformer Pipeline.empty ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
  in
  let builder =
    add_transformer builder ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let specification =
    Pipeline.set_estimator builder (first_column_estimator "model") |> get
  in
  let fitted =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema ~x:training
      ~y:target ()
    |> get
  in
  (schema, specification, fitted)

let check_float message expected observed =
  Alcotest.check (Alcotest.float 1e-12) message expected observed

let is_data_length_mismatch = function
  | Error.Data (Data_error.Length_mismatch _) -> true
  | Error.Data
      ( Data_error.Negative_dimension _ | Data_error.Ragged_matrix _
      | Data_error.Index_out_of_bounds _ | Data_error.Non_finite _
      | Data_error.Negative_weight _ | Data_error.All_zero_weights
      | Data_error.Empty_feature_name _ | Data_error.Duplicate_feature_name _ )
  | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Validation _ | Error.Numerical _ | Error.Convergence _
  | Error.Compatibility _ | Error.Artifact _ | Error.Cancelled ->
      false

let test_fit_transform_predict () =
  let schema, specification, fitted = fitted_preprocessing_pipeline () in
  check
    (Pipeline.transformer_names specification = [| "impute"; "scale" |])
    "pipeline stage order changed";
  check
    (String.equal (Pipeline.estimator_name specification) "model")
    "pipeline estimator name changed";
  let capabilities = Pipeline.capabilities specification in
  check capabilities.Pipeline.decision_function
    "decision capability was not recorded";
  check capabilities.Pipeline.predict_proba
    "probability capability was not recorded";
  check
    (Feature_schema.equal (Pipeline.input_schema fitted) schema)
    "pipeline input schema changed";
  check
    (Feature_schema.equal (Pipeline.output_schema fitted) schema)
    "pipeline output schema changed";
  let production = Matrix.of_arrays [| [| Float.nan |] |] |> get_data in
  let transformed =
    Pipeline.transform fitted ~feature_schema:schema ~x:production |> get
  in
  check_float "fitted training mean is reused" 0.0 (Matrix.get transformed 0 0);
  let prediction =
    Pipeline.predict fitted ~feature_schema:schema ~x:production
    |> get |> Target.regression_values
  in
  check_float "prediction follows fitted preprocessing" 0.0
    (Vector.get prediction 0);
  let decision =
    Pipeline.decision_function fitted ~feature_schema:schema ~x:production
    |> get
  in
  check_float "decision dispatch" 0.0 (Vector.get decision 0);
  let probabilities =
    Pipeline.predict_proba fitted ~feature_schema:schema ~x:production |> get
  in
  check_float "negative probability" 0.5 (Matrix.get probabilities 0 0);
  check_float "positive probability" 0.5 (Matrix.get probabilities 0 1)

let test_feature_name_propagation () =
  let schema = named_schema [| "constant"; "signal" |] in
  let x =
    Matrix.of_arrays [| [| 2.0; 1.0 |]; [| 2.0; 3.0 |]; [| 2.0; 5.0 |] |]
    |> get_data
  in
  let threshold = Variance_threshold.create () |> get in
  let builder =
    add_transformer Pipeline.empty ~name:"select"
      (module Variance_threshold)
      threshold
  in
  let specification =
    Pipeline.set_estimator builder (first_column_estimator "model") |> get
  in
  let fitted =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| 0.0; 0.0; 0.0 |])
      ()
    |> get
  in
  let names =
    Pipeline.output_schema fitted
    |> Feature_schema.names |> Option.get |> Feature_names.to_array
  in
  check (names = [| "signal" |]) "selected feature name was not propagated"

let test_target_and_weight_routing () =
  let schema = named_schema [| "value" |] in
  let x = Matrix.of_arrays [| [| Float.nan |]; [| 4.0 |] |] |> get_data in
  let builder =
    add_transformer Pipeline.empty ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.constant 0.0 |> get)
  in
  let estimator =
    Pipeline.estimator ~name:"weighted" (module Weighted_mean_regressor) ()
    |> get
  in
  let specification = Pipeline.set_estimator builder estimator |> get in
  let weights =
    Sample_weight.of_array ~expected_length:2 [| 1.0; 3.0 |] |> get_data
  in
  let fitted =
    Pipeline.fit specification ~sample_weight:weights ~rng:(rng ())
      ~feature_schema:schema ~x
      ~y:(regression [| 2.0; 10.0 |])
      ()
    |> get
  in
  let prediction =
    Pipeline.predict fitted ~feature_schema:schema ~x
    |> get |> Target.regression_values
  in
  check_float "sample weights reached only the estimator" 8.0
    (Vector.get prediction 0);
  let misaligned_weights =
    Sample_weight.of_array ~expected_length:1 [| 1.0 |] |> get_data
  in
  check
    (match
       Pipeline.fit specification ~sample_weight:misaligned_weights
         ~rng:(rng ()) ~feature_schema:schema ~x
         ~y:(regression [| 2.0; 10.0 |])
         ()
     with
    | Error error -> is_data_length_mismatch (Error.kind error)
    | Ok _ -> false)
    "pipeline accepted sample weights for a different training row count"

let test_deterministic_child_rng () =
  let schema = named_schema [| "value" |] in
  let x = Matrix.of_arrays [| [| 1.0 |] |] |> get_data in
  let builder =
    add_transformer Pipeline.empty ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let estimator =
    Pipeline.estimator ~name:"random" (module Random_regressor) () |> get
  in
  let specification = Pipeline.set_estimator builder estimator |> get in
  let root = Seed.of_int 41 in
  let fit () =
    Pipeline.fit specification ~rng:(Rng.create root) ~feature_schema:schema ~x
      ~y:(regression [| 0.0 |]) ()
    |> get
  in
  let predict fitted =
    Pipeline.predict fitted ~feature_schema:schema ~x
    |> get |> Target.regression_values
    |> fun values -> Vector.get values 0
  in
  let first = predict (fit ()) in
  let second = predict (fit ()) in
  check_float "fixed root RNG reproduces fitted state" first second;
  let expected_seed =
    Seed.derive root ~operation:"pipeline-estimator:random" ~index:1
  in
  let expected, _ = Rng.next_float (Rng.create expected_seed) in
  check_float "estimator RNG follows logical stage identity" expected first

let is_validation = function
  | Error.Validation _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_compatibility = function
  | Error.Compatibility _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
  | Error.Validation _ | Error.Numerical _ | Error.Convergence _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_schema_mismatch = function
  | Error.Feature_schema_mismatch _ -> true
  | Error.Data _ | Error.Shape_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let is_shape_mismatch = function
  | Error.Shape_mismatch _ -> true
  | Error.Data _ | Error.Feature_schema_mismatch _ | Error.Validation _
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Cancelled ->
      false

let expect_error predicate result =
  match result with
  | Error error when predicate (Error.kind error) -> error
  | Error error -> fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> fail "expected an error"

let test_construction_errors () =
  ignore
    (expect_error is_validation
       (Pipeline.transformer ~name:"  "
          (module Simple_imputer)
          (Simple_imputer.mean ())));
  let transformer =
    Pipeline.transformer ~name:"same"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty transformer |> get in
  ignore
    (expect_error is_validation (Pipeline.add_transformer builder transformer));
  let estimator = first_column_estimator "same" in
  ignore (expect_error is_validation (Pipeline.set_estimator builder estimator))

let test_stage_and_schema_errors () =
  let schema = named_schema [| "value" |] in
  let x = Matrix.of_arrays [| [| Float.nan |] |] |> get_data in
  let builder =
    add_transformer Pipeline.empty ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
  in
  let specification =
    Pipeline.set_estimator builder (first_column_estimator "model") |> get
  in
  let error =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| 0.0 |]) ()
    |> expect_error is_validation
  in
  check
    (match Error.context error with
    | Error.Stage "impute" :: _ -> true
    | (Error.Stage _ | Error.Fold _ | Error.Candidate _ | Error.Feature _) :: _
    | [] ->
        false)
    "transformer failure lacks stage context";
  let _, _, fitted = fitted_preprocessing_pipeline () in
  let other_schema = named_schema [| "other" |] in
  ignore
    (Pipeline.predict fitted ~feature_schema:other_schema ~x
    |> expect_error is_schema_mismatch);
  let dropping =
    Pipeline.transformer ~name:"drop-rows" (module Dropping_transformer) ()
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty dropping |> get in
  let specification =
    Pipeline.set_estimator builder (first_column_estimator "model") |> get
  in
  let error =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema
      ~x:(Matrix.of_arrays [| [| 1.0 |]; [| 2.0 |] |] |> get_data)
      ~y:(regression [| 0.0; 1.0 |])
      ()
    |> expect_error is_shape_mismatch
  in
  check
    (match Error.context error with
    | Error.Stage "drop-rows" :: _ -> true
    | (Error.Stage _ | Error.Fold _ | Error.Candidate _ | Error.Feature _) :: _
    | [] ->
        false)
    "row-count contract failure lacks transformer stage context"

let test_unsupported_capabilities () =
  let schema = named_schema [| "value" |] in
  let x = Matrix.of_arrays [| [| 1.0 |] |] |> get_data in
  let specification =
    Pipeline.set_estimator Pipeline.empty
      (first_column_estimator ~capabilities:false "plain")
    |> get
  in
  let fitted =
    Pipeline.fit specification ~rng:(rng ()) ~feature_schema:schema ~x
      ~y:(regression [| 0.0 |]) ()
    |> get
  in
  let error =
    Pipeline.predict_proba fitted ~feature_schema:schema ~x
    |> expect_error is_compatibility
  in
  check
    (Error.context error = [ Error.Stage "plain" ])
    "capability error lacks estimator stage context";
  ignore
    (Pipeline.decision_function fitted ~feature_schema:schema ~x
    |> expect_error is_compatibility);
  ignore (Pipeline.classes fitted |> expect_error is_compatibility)

let () =
  Alcotest.run "pipeline"
    [
      ( "orchestration",
        [
          Alcotest.test_case "fit transform and dispatch" `Quick
            test_fit_transform_predict;
          Alcotest.test_case "feature names" `Quick
            test_feature_name_propagation;
          Alcotest.test_case "target and sample weights" `Quick
            test_target_and_weight_routing;
          Alcotest.test_case "logical child RNG" `Quick
            test_deterministic_child_rng;
        ] );
      ( "errors",
        [
          Alcotest.test_case "construction" `Quick test_construction_errors;
          Alcotest.test_case "stage and schema" `Quick
            test_stage_and_schema_errors;
          Alcotest.test_case "unsupported capabilities" `Quick
            test_unsupported_capabilities;
        ] );
    ]
