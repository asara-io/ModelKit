open Modelkit_data
open Modelkit_protocols

module Pipeline = struct
  type capabilities = { decision_function : bool; predict_proba : bool }

  type encoded_component = {
    component_tag : int;
    component_version : int;
    component_payload : bytes;
  }

  type fitted_transformer = {
    stage_name : string;
    transform_input_schema : Feature_schema.t;
    transform_output_schema : Feature_schema.t;
    apply_transform :
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result;
    encode_transformer : (unit -> (encoded_component, Error.t) result) option;
  }

  type transformer = {
    transformer_name : string;
    fit_transform :
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (fitted_transformer * Matrix.t * Feature_schema.t, Error.t) result;
  }

  type builder = {
    reversed_transformers : transformer list;
    names : string list;
  }

  type 'prediction fitted_estimator = {
    terminal_name : string;
    terminal_predict :
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      ('prediction, Error.t) result;
    terminal_decision_function :
      (feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result)
      option;
    terminal_predict_proba :
      (feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result)
      option;
    terminal_classes : (unit -> int array) option;
    encode_estimator : (unit -> (encoded_component, Error.t) result) option;
  }

  type ('target, 'prediction) estimator = {
    estimator_name : string;
    estimator_capabilities : capabilities;
    fit_estimator :
      ?sample_weight:Sample_weight.t ->
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      y:'target ->
      unit ->
      ('prediction fitted_estimator, Error.t) result;
  }

  type ('target, 'prediction) t = {
    transformers : transformer array;
    estimator : ('target, 'prediction) estimator;
  }

  type ('target, 'prediction) fitted = {
    fitted_transformers : fitted_transformer array;
    fitted_estimator : 'prediction fitted_estimator;
    pipeline_input_schema : Feature_schema.t;
    pipeline_output_schema : Feature_schema.t;
  }

  let ( let* ) = Result.bind

  let validation_error ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let validate_name name =
    if String.length (String.trim name) = 0 then
      Error
        (validation_error ~name:"pipeline stage name"
           ~reason:"must not be blank"
           ~remediation:"choose a non-empty stage name")
    else Ok ()

  let duplicate_name name =
    validation_error ~name:"pipeline stage names"
      ~reason:(Format.sprintf "stage name %S is duplicated" name)
      ~remediation:"choose a unique name for every pipeline stage"

  let validate_schema ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide features with the fitted schema"
           (Error.Feature_schema_mismatch { expected; observed }))

  let validate_matrix schema matrix =
    match Feature_schema.validate_matrix schema matrix with
    | Ok () -> Ok ()
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"provide a matrix matching the current stage schema"
             error)

  let validate_transform_output ~input ~output_schema output =
    let expected_rows = Matrix.rows input in
    let observed_rows = Matrix.rows output in
    if expected_rows <> observed_rows then
      Error
        (Error.make
           ~remediation:"preserve sample count in every transformer stage"
           (Error.Shape_mismatch
              {
                name = "transformer output rows";
                expected = [ expected_rows ];
                observed = [ observed_rows ];
              }))
    else validate_matrix output_schema output

  let validate_sample_weight x = function
    | None -> Ok ()
    | Some weights ->
        let expected = Matrix.rows x in
        let observed = Sample_weight.length weights in
        if expected = observed then Ok ()
        else
          Error
            (Error.of_data_error
               ~remediation:"provide one sample weight per training row"
               (Data_error.Length_mismatch
                  { name = "pipeline sample weights"; expected; observed }))

  let validate_fitted_schema ~stage ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make
           ~remediation:
             "fix the component so its fitted schema matches its fit input"
           (Error.Compatibility
              {
                component = stage;
                reason = "component returned an inconsistent fitted schema";
              }))

  let with_stage name result =
    Result.map_error (Error.with_context (Error.Stage name)) result

  let transformer_internal (type specification fitted) ?encode ~name
      (module Transformer : TRANSFORMER
        with type t = specification
         and type target = unit
         and type fitted = fitted
         and type rng = Rng.t) (specification : specification) =
    let* () = validate_name name in
    let fit_transform ~rng ~feature_schema ~x =
      let* fitted =
        Transformer.fit specification ~rng ~feature_schema ~x ~y:None ()
      in
      let* () =
        validate_fitted_schema ~stage:name ~expected:feature_schema
          (Transformer.input_schema fitted)
      in
      let output_schema = Transformer.output_schema fitted in
      let* transformed = Transformer.transform fitted ~feature_schema ~x in
      let* () = validate_transform_output ~input:x ~output_schema transformed in
      let fitted_transformer : fitted_transformer =
        {
          stage_name = name;
          transform_input_schema = feature_schema;
          transform_output_schema = output_schema;
          apply_transform = Transformer.transform fitted;
          encode_transformer =
            Option.map (fun encode () -> encode fitted) encode;
        }
      in
      Ok (fitted_transformer, transformed, output_schema)
    in
    Ok { transformer_name = name; fit_transform }

  let transformer ~name transformer specification =
    transformer_internal ~name transformer specification

  let estimator_internal (type specification target prediction fitted) ?encode
      ~name
      (module Estimator : ESTIMATOR
        with type t = specification
         and type target = target
         and type prediction = prediction
         and type fitted = fitted
         and type rng = Rng.t) ?decision_function ?predict_proba ?classes
      (specification : specification) =
    let* () = validate_name name in
    let capabilities : capabilities =
      {
        decision_function = Option.is_some decision_function;
        predict_proba = Option.is_some predict_proba;
      }
    in
    let fit ?sample_weight ~rng ~feature_schema ~x ~y () =
      let* fitted =
        Estimator.fit specification ?sample_weight ~rng ~feature_schema ~x ~y ()
      in
      let fitted_schema = Estimator.feature_schema fitted in
      let* () =
        validate_fitted_schema ~stage:name ~expected:feature_schema
          fitted_schema
      in
      let fitted_estimator : prediction fitted_estimator =
        {
          terminal_name = name;
          terminal_predict = Estimator.predict fitted;
          terminal_decision_function =
            Option.map (fun dispatch -> dispatch fitted) decision_function;
          terminal_predict_proba =
            Option.map (fun dispatch -> dispatch fitted) predict_proba;
          terminal_classes =
            Option.map (fun dispatch () -> dispatch fitted) classes;
          encode_estimator = Option.map (fun encode () -> encode fitted) encode;
        }
      in
      Ok fitted_estimator
    in
    Ok
      {
        estimator_name = name;
        estimator_capabilities = capabilities;
        fit_estimator = fit;
      }

  let estimator ~name estimator ?decision_function ?predict_proba ?classes
      specification =
    estimator_internal ~name estimator ?decision_function ?predict_proba
      ?classes specification

  let empty = { reversed_transformers = []; names = [] }

  let add_transformer (builder : builder) (transformer : transformer) =
    if List.exists (String.equal transformer.transformer_name) builder.names
    then Error (duplicate_name transformer.transformer_name)
    else
      Ok
        {
          reversed_transformers = transformer :: builder.reversed_transformers;
          names = transformer.transformer_name :: builder.names;
        }

  let set_estimator (builder : builder)
      (estimator : ('target, 'prediction) estimator) =
    if List.exists (String.equal estimator.estimator_name) builder.names then
      Error (duplicate_name estimator.estimator_name)
    else
      Ok
        {
          transformers = Array.of_list (List.rev builder.reversed_transformers);
          estimator;
        }

  let clone specification = specification

  let transformer_names specification =
    Array.map
      (fun transformer -> transformer.transformer_name)
      specification.transformers

  let estimator_name specification = specification.estimator.estimator_name

  let capabilities specification =
    specification.estimator.estimator_capabilities

  let child_rng root ~kind ~name ~index =
    let seed =
      Seed.derive (Rng.to_seed root) ~operation:(kind ^ ":" ^ name) ~index
    in
    Rng.create seed

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = validate_matrix feature_schema x in
    let* () = validate_sample_weight x sample_weight in
    let rec fit_transformers index current_schema current_x reversed_fitted =
      if index = Array.length specification.transformers then
        Ok (Array.of_list (List.rev reversed_fitted), current_schema, current_x)
      else
        let transformer = specification.transformers.(index) in
        let stage_rng =
          child_rng rng ~kind:"pipeline-transformer"
            ~name:transformer.transformer_name ~index
        in
        let* fitted, transformed, output_schema =
          with_stage transformer.transformer_name
            (transformer.fit_transform ~rng:stage_rng
               ~feature_schema:current_schema ~x:current_x)
        in
        fit_transformers (index + 1) output_schema transformed
          (fitted :: reversed_fitted)
    in
    let* fitted_transformers, output_schema, transformed_x =
      fit_transformers 0 feature_schema x []
    in
    let estimator_index = Array.length specification.transformers in
    let estimator_rng =
      child_rng rng ~kind:"pipeline-estimator"
        ~name:specification.estimator.estimator_name ~index:estimator_index
    in
    let* fitted_estimator =
      with_stage specification.estimator.estimator_name
        (specification.estimator.fit_estimator ?sample_weight ~rng:estimator_rng
           ~feature_schema:output_schema ~x:transformed_x ~y ())
    in
    Ok
      {
        fitted_transformers;
        fitted_estimator;
        pipeline_input_schema = feature_schema;
        pipeline_output_schema = output_schema;
      }

  let transform fitted ~feature_schema ~x =
    let* () =
      validate_schema ~expected:fitted.pipeline_input_schema feature_schema
    in
    let* () = validate_matrix feature_schema x in
    let rec apply index current_schema current_x =
      if index = Array.length fitted.fitted_transformers then Ok current_x
      else
        let transformer = fitted.fitted_transformers.(index) in
        let* () =
          with_stage transformer.stage_name
            (validate_schema ~expected:transformer.transform_input_schema
               current_schema)
        in
        let* transformed =
          with_stage transformer.stage_name
            (transformer.apply_transform ~feature_schema:current_schema
               ~x:current_x)
        in
        let* () =
          with_stage transformer.stage_name
            (validate_transform_output ~input:current_x
               ~output_schema:transformer.transform_output_schema transformed)
        in
        apply (index + 1) transformer.transform_output_schema transformed
    in
    apply 0 feature_schema x

  let predict fitted ~feature_schema ~x =
    let* transformed = transform fitted ~feature_schema ~x in
    with_stage fitted.fitted_estimator.terminal_name
      (fitted.fitted_estimator.terminal_predict
         ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let unsupported fitted capability =
    Error.make
      ~context:[ Error.Stage fitted.fitted_estimator.terminal_name ]
      ~remediation:("configure a terminal estimator that supports " ^ capability)
      (Error.Compatibility
         {
           component = "pipeline terminal estimator";
           reason = capability ^ " is unavailable";
         })

  let decision_function fitted ~feature_schema ~x =
    match fitted.fitted_estimator.terminal_decision_function with
    | None -> Error (unsupported fitted "decision_function")
    | Some dispatch ->
        let* transformed = transform fitted ~feature_schema ~x in
        with_stage fitted.fitted_estimator.terminal_name
          (dispatch ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let predict_proba fitted ~feature_schema ~x =
    match fitted.fitted_estimator.terminal_predict_proba with
    | None -> Error (unsupported fitted "predict_proba")
    | Some dispatch ->
        let* transformed = transform fitted ~feature_schema ~x in
        with_stage fitted.fitted_estimator.terminal_name
          (dispatch ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let classes fitted =
    match fitted.fitted_estimator.terminal_classes with
    | None -> Error (unsupported fitted "probability class order")
    | Some dispatch -> Ok (Array.copy (dispatch ()))

  let input_schema fitted = fitted.pipeline_input_schema
  let output_schema fitted = fitted.pipeline_output_schema
end
