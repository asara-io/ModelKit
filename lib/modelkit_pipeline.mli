open Modelkit_data
open Modelkit_protocols

module Pipeline : sig
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

  val transformer_internal :
    ?encode:('fitted -> (encoded_component, Error.t) result) ->
    name:string ->
    (module TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result

  val transformer :
    name:string ->
    (module TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result

  val estimator_internal :
    ?encode:('fitted -> (encoded_component, Error.t) result) ->
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = 'target
        and type prediction = 'prediction
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    ?decision_function:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result) ->
    ?predict_proba:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result) ->
    ?classes:('fitted -> int array) ->
    'specification ->
    (('target, 'prediction) estimator, Error.t) result

  val estimator :
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = 'target
        and type prediction = 'prediction
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    ?decision_function:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result) ->
    ?predict_proba:
      ('fitted ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result) ->
    ?classes:('fitted -> int array) ->
    'specification ->
    (('target, 'prediction) estimator, Error.t) result

  val empty : builder
  val add_transformer : builder -> transformer -> (builder, Error.t) result

  val set_estimator :
    builder ->
    ('target, 'prediction) estimator ->
    (('target, 'prediction) t, Error.t) result

  val clone : ('target, 'prediction) t -> ('target, 'prediction) t
  val transformer_names : ('target, 'prediction) t -> string array
  val estimator_name : ('target, 'prediction) t -> string
  val capabilities : ('target, 'prediction) t -> capabilities

  val fit :
    ('target, 'prediction) t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:'target ->
    unit ->
    (('target, 'prediction) fitted, Error.t) result

  val transform :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val predict :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    ('prediction, Error.t) result

  val decision_function :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Vector.t, Error.t) result

  val predict_proba :
    ('target, 'prediction) fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val classes : ('target, 'prediction) fitted -> (int array, Error.t) result
  val input_schema : ('target, 'prediction) fitted -> Feature_schema.t
  val output_schema : ('target, 'prediction) fitted -> Feature_schema.t
end
