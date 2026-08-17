open Modelkit_data
open Modelkit_protocols
open Modelkit_pipeline
open Modelkit_preprocessing
open Modelkit_linear_models

module Artifact : sig
  type limits
  type metadata
  type 'model loaded

  type regression_model =
    (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

  type binary_classification_model =
    ( Target.classification Target.t,
      Target.classification Target.t )
    Pipeline.fitted

  val default_limits : limits

  val limits :
    ?max_bytes:int ->
    ?max_components:int ->
    ?max_features:int ->
    ?max_string_bytes:int ->
    ?max_metadata_entries:int ->
    unit ->
    (limits, Error.t) result

  val empty_metadata : metadata

  val metadata :
    ?training_rows:int ->
    ?root_seed:Seed.t ->
    ?sample_weighted:bool ->
    ?labels:(string * string) array ->
    unit ->
    (metadata, Error.t) result

  val model : 'model loaded -> 'model
  val metadata_of_loaded : _ loaded -> metadata
  val producer_version : _ loaded -> string
  val producer_ocaml_version : _ loaded -> string option
  val training_rows : metadata -> int option
  val root_seed : metadata -> Seed.t option
  val sample_weighted : metadata -> bool option
  val labels : metadata -> (string * string) array

  val simple_imputer_stage :
    name:string -> Simple_imputer.t -> (Pipeline.transformer, Error.t) result

  val standard_scaler_stage :
    name:string -> Standard_scaler.t -> (Pipeline.transformer, Error.t) result

  val variance_threshold_stage :
    name:string ->
    Variance_threshold.t ->
    (Pipeline.transformer, Error.t) result

  val linear_regression_estimator :
    name:string ->
    Linear_regression.t ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result

  val ridge_regression_estimator :
    name:string ->
    Ridge_regression.t ->
    ( (Target.regression Target.t, Target.regression Target.t) Pipeline.estimator,
      Error.t )
    result

  val logistic_regression_estimator :
    name:string ->
    Logistic_regression.t ->
    ( ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.estimator,
      Error.t )
    result

  val encode_regression :
    ?metadata:metadata ->
    ?limits:limits ->
    regression_model ->
    (bytes, Error.t) result

  val encode_binary_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    binary_classification_model ->
    (bytes, Error.t) result

  val decode_regression :
    ?limits:limits -> bytes -> (regression_model loaded, Error.t) result

  val decode_binary_classification :
    ?limits:limits ->
    bytes ->
    (binary_classification_model loaded, Error.t) result

  val save_regression :
    ?metadata:metadata ->
    ?limits:limits ->
    path:string ->
    regression_model ->
    (unit, Error.t) result
  (** Encodes completely before opening [path], then writes the destination
      directly. Atomic replacement, signing, and encryption are transport-level
      responsibilities. *)

  val save_binary_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    path:string ->
    binary_classification_model ->
    (unit, Error.t) result
  (** The binary-classification counterpart to [save_regression]. *)

  val load_regression :
    ?limits:limits ->
    path:string ->
    unit ->
    (regression_model loaded, Error.t) result

  val load_binary_classification :
    ?limits:limits ->
    path:string ->
    unit ->
    (binary_classification_model loaded, Error.t) result
end
