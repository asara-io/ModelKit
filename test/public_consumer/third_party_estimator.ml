open Modelkit

let ( let* ) = Result.bind

let data_result result =
  Result.map_error
    (Error.of_data_error ~context:[]
       ~remediation:"supply finite, row-aligned third-party estimator data")
    result

module Weighted_mean_regressor = struct
  type params = { offset : float }
  type t = params

  type fitted = {
    specification : t;
    schema : Feature_schema.t;
    prediction : float;
  }

  type target = Target.regression Target.t
  type prediction = target
  type rng = Rng.t

  let create ?(offset = 0.) () =
    if Float.is_finite offset then Ok { offset }
    else
      Error
        (Error.make ~remediation:"choose a finite prediction offset"
           (Error.Validation
              { name = "weighted mean offset"; reason = "must be finite" }))

  let clone specification = specification
  let params specification = specification

  let fit_request _ =
    Metadata.Request.create ~sample_weight:Metadata.Request.Required
      ~groups:Metadata.Request.Required ~callback:Metadata.Request.Optional ()

  let fit specification ~metadata ~rng:_ ~feature_schema ~x ~y () =
    let* () = data_result (Feature_schema.validate_matrix feature_schema x) in
    let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
    match (Metadata.sample_weight metadata, Metadata.groups metadata) with
    | Some sample_weight, Some groups ->
        let values = Target.regression_values y in
        let numerator = ref 0. and denominator = ref 0. in
        let rec accumulate row =
          if row = Matrix.rows x then Ok ()
          else
            let weight = Sample_weight.get sample_weight row in
            let group = Float.of_int (Groups.get groups row) in
            let target = Vector.get values row in
            if target <> (3. *. weight) +. group then
              Error
                (Error.make
                   ~remediation:
                     "select targets and metadata with identical row views"
                   (Error.Compatibility
                      {
                        component = "third-party weighted mean regressor";
                        reason = "target, weight, and group rows are misaligned";
                      }))
            else (
              numerator := !numerator +. (weight *. target);
              denominator := !denominator +. weight;
              accumulate (row + 1))
        in
        let* () = accumulate 0 in
        let* () =
          match Metadata.callback metadata with
          | None -> Ok ()
          | Some callback ->
              Modelkit.Callback.progress callback ~completed:1 ~total:1 ()
        in
        Ok
          {
            specification;
            schema = feature_schema;
            prediction = (!numerator /. !denominator) +. specification.offset;
          }
    | _ ->
        Error
          (Error.make
             ~remediation:"route the estimator's required weights and groups"
             (Error.Validation
                {
                  name = "third-party estimator metadata";
                  reason = "required metadata is absent";
                }))

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      Error
        (Error.make ~remediation:"predict with the fitted feature schema"
           (Error.Feature_schema_mismatch
              { expected = fitted.schema; observed = feature_schema }))
    else
      Array.make (Matrix.rows x) fitted.prediction
      |> Vector.of_array |> Target.regression |> data_result

  let fitted_params fitted = fitted.specification
  let feature_schema fitted = fitted.schema
end

let matrix values = Matrix.of_arrays values |> data_result

let dataset () =
  let rows = 24 in
  let* x =
    matrix
      (Array.init rows (fun row ->
           [| Float.of_int row; Float.of_int ((row mod 5) - 2) |]))
  in
  let weights = Array.init rows (fun row -> Float.of_int (row + 1)) in
  let group_values = Array.init rows (fun row -> row / 3) in
  let* y =
    Array.init rows (fun row ->
        (3. *. weights.(row)) +. Float.of_int group_values.(row))
    |> Vector.of_array |> Target.regression |> data_result
  in
  let* sample_weight =
    Sample_weight.of_array ~expected_length:rows weights |> data_result
  in
  let* groups =
    Groups.create ~expected_length:rows group_values |> data_result
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight ~groups ~x ~y
    ()
  |> data_result

let provenance () =
  Pipeline.provenance ~package:"third-party-estimator" ~version:"1.0.0"
    ~implementation:"Weighted_mean_regressor"

let preprocessing () =
  let* imputer =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
  in
  let* scaler =
    Pipeline.transformer ~name:"scale"
      (module Standard_scaler)
      (Standard_scaler.create ())
  in
  let* chain = Transformer_pipeline.create [| imputer; scaler |] in
  Transformer_pipeline.stage ~name:"preprocess" chain

let pipeline ?(nested = true) specification =
  let* provenance = provenance () in
  let* estimator =
    Pipeline.metadata_estimator ~provenance ~name:"weighted_mean"
      (module Weighted_mean_regressor)
      specification
  in
  if nested then
    let* transformer = preprocessing () in
    let* builder = Pipeline.add_transformer Pipeline.empty transformer in
    Pipeline.set_estimator builder estimator
  else Pipeline.set_estimator Pipeline.empty estimator

let equal_params (left : Weighted_mean_regressor.params)
    (right : Weighted_mean_regressor.params) =
  Float.equal left.Weighted_mean_regressor.offset
    right.Weighted_mean_regressor.offset

let equal_prediction left right =
  Vector.to_array (Target.regression_values left)
  = Vector.to_array (Target.regression_values right)

let conformance specification =
  let* source = dataset () in
  let fixture : (_, _, _, _, _, _) Conformance.Metadata_estimator.fixture =
    {
      Conformance.Metadata_estimator.specification;
      rng = (fun () -> Rng.create (Seed.of_int 31));
      feature_schema = Dataset.feature_schema source;
      x = Dataset.features source;
      y = Dataset.target source;
      metadata = Metadata.of_dataset source;
      equal_params;
      prediction_length = Target.length;
      equal_prediction;
    }
  in
  Ok
    (Conformance.Metadata_estimator.check
       (module Weighted_mean_regressor)
       fixture)
