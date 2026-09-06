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
      sample_weight:Sample_weight.t option ->
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (fitted_transformer * Matrix.t * Feature_schema.t, Error.t) result;
  }

  type builder = {
    reversed_transformers : transformer list;
    names : string list;
  }

  type 'target stage = {
    name : string;
    validate_target : x:Matrix.t -> y:'target -> (unit, Error.t) result;
    fit_stage :
      sample_weight:Sample_weight.t option ->
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      y:'target ->
      (fitted_transformer * Matrix.t * Feature_schema.t, Error.t) result;
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
    transformers : 'target stage array;
    estimator : ('target, 'prediction) estimator;
  }

  type ('target, 'prediction) fitted = {
    fitted_transformers : fitted_transformer array;
    fitted_estimator : 'prediction fitted_estimator;
    pipeline_input_schema : Feature_schema.t;
    pipeline_output_schema : Feature_schema.t;
  }

  val validate_transform_output :
    input:Matrix.t ->
    output_schema:Feature_schema.t ->
    Matrix.t ->
    (unit, Error.t) result

  val transformer_internal :
    ?encode:('fitted -> (encoded_component, Error.t) result) ->
    ?route_sample_weight:bool ->
    name:string ->
    (module TRANSFORMER
       with type t = 'specification
        and type target = unit
        and type fitted = 'fitted
        and type rng = Rng.t) ->
    'specification ->
    (transformer, Error.t) result

  val transformer :
    ?route_sample_weight:bool ->
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
    ?resolve_weights:
      (?sample_weight:Sample_weight.t ->
      'target ->
      (Sample_weight.t option, Error.t) result) ->
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

  val classifier_internal :
    ?encode:('fitted -> (encoded_component, Error.t) result) ->
    ?class_weight:Modelkit_class_weight.Class_weight.t ->
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = Target.classification Target.t
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
    ((Target.classification Target.t, 'prediction) estimator, Error.t) result

  val classifier :
    ?class_weight:Modelkit_class_weight.Class_weight.t ->
    name:string ->
    (module ESTIMATOR
       with type t = 'specification
        and type target = Target.classification Target.t
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
    ((Target.classification Target.t, 'prediction) estimator, Error.t) result

  val empty : builder
  val add_transformer : builder -> transformer -> (builder, Error.t) result

  val set_estimator :
    builder ->
    ('target, 'prediction) estimator ->
    (('target, 'prediction) t, Error.t) result

  module Supervised : sig
    type nonrec 'kind stage = 'kind Target.t stage
    type 'kind builder

    val transformer :
      ?route_sample_weight:bool ->
      name:string ->
      (module TRANSFORMER
         with type t = 'specification
          and type target = 'kind Target.t
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      'specification ->
      ('kind stage, Error.t) result
    (** Packages a supervised transformer. Each fit receives [Some y] from the
        pipeline's training rows; weights reach it only when
        [route_sample_weight] is true. Targets and weights must have one entry
        per row. Length errors are rejected before any stage fits. Class weights
        remain a terminal-estimator policy and do not alter the sample weights
        routed to transformers. *)

    val unsupervised : transformer -> 'kind stage
    (** Adapts an existing unsupervised stage, retaining its weight-routing and
        artifact-codec policies. Its fit still receives [y:None]. *)

    val empty : 'kind builder

    val add_transformer :
      'kind builder -> 'kind stage -> ('kind builder, Error.t) result

    val set_estimator :
      'kind builder ->
      ('kind Target.t, 'prediction) estimator ->
      (('kind Target.t, 'prediction) t, Error.t) result
    (** The terminal and supervised stages share the same target kind. The
        resulting pipeline uses the ordinary fit, prediction, CV, and search
        APIs. Targets are used only during fitting; inference reuses learned
        transforms without requiring targets or weights.

        A supervised stage fits on and transforms the same training rows. This
        is suitable for feature selection; target encoders needing internal
        cross-fitting require a separate fit-transform contract. Supervised
        stages currently have no artifact codec. *)
  end

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
