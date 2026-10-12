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

  type classification_model = binary_classification_model
  (** The OCaml target type is shared. Binary byte/file APIs additionally
      require a binary task tag and exactly two classes. *)

  (** Explicit external terminal-estimator codecs, bound to the public protocol.
      No artifact can register a decoder, choose code, or load a library.
      Applications select trusted codec implementations on each load call;
      omitting [decoders] admits built-in components only. External transformer
      and nested-composition codecs remain unsupported. *)
  module Codec : sig
    type ('specification, 'kind) t

    type 'kind decoder
    (** Regression and classification decoder selections have different types. A
        codec's decoder is identified by its package, name, and exact format
        version. Multiple versions can be selected; duplicate identities and
        versions fail before any callback. *)

    val regression :
      package:string ->
      name:string ->
      version:int ->
      (module REGRESSOR
         with type t = 'specification
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      encode:('fitted -> (bytes, Error.t) result) ->
      decode:
        (provenance:Pipeline.provenance ->
        feature_schema:Feature_schema.t ->
        bytes ->
        ('fitted, Error.t) result) ->
      (('specification, Target.regression) t, Error.t) result
    (** Creates a regression codec. Package/name identifiers are nonempty ASCII
        letters, digits, hyphens, underscores, or dots; [modelkit] is reserved.
        Format versions range from 1 to 65535, independently of package
        versions. [encode] produces a versioned data-only payload, without
        training data, closures, commands, or [Marshal] values. [decode]
        restores fitted state without refitting and receives validated
        provenance and feature schema.

        Framing, checksum, envelope bounds, schemas, task, component identity,
        and version are checked before [decode]. The callback owns validation
        and bounds inside its opaque payload, any native resource acquisition,
        and cleanup on errors/exceptions. The returned fitted schema must match
        the supplied schema. Callbacks return typed routine failures; unrelated
        programmer exceptions propagate. *)

    val classification :
      package:string ->
      name:string ->
      version:int ->
      (module CLASSIFIER
         with type t = 'specification
          and type fitted = 'fitted
          and type rng = Rng.t) ->
      classes:('fitted -> int array) ->
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
      encode:('fitted -> (bytes, Error.t) result) ->
      decode:
        (provenance:Pipeline.provenance ->
        feature_schema:Feature_schema.t ->
        classes:int array ->
        bytes ->
        ('fitted, Error.t) result) ->
      unit ->
      (('specification, Target.classification) t, Error.t) result
    (** Classification counterpart. [classes] is mandatory; optional probability
        columns follow that exact order. The callback receives a fresh copy of
        the validated nonempty, distinct, bounded class mapping and must restore
        it unchanged. Class order need not be sorted. General classification
        permits one class; individual estimators may impose stricter rules.
        Restored prediction capabilities match the callbacks attached here. *)

    val estimator :
      ?name:string ->
      provenance:Pipeline.provenance ->
      ('specification, 'kind) t ->
      'specification ->
      (('kind Target.t, 'kind Target.t) Pipeline.estimator, Error.t) result
    (** Packages the estimator with its codec and explicit producer provenance.
        The provenance package must match the codec package; its version is the
        component's producer version. The default stage name is [estimator].
        Pipeline artifact reporting identifies [External_artifact], preserving
        the built-in portable-codec guarantee. *)

    val decoder : ('specification, 'kind) t -> 'kind decoder
    (** Supplies an explicit per-call reader selection. Load-time configuration,
        such as inference thread limits, can be captured by the codec's decode
        callback when the caller constructs this selection. *)
  end

  val default_limits : limits
  (** 64 MiB total bytes, 1024 components, one million features/classes, one MiB
      strings, and 128 metadata entries. Opaque external payloads remain bounded
      by total bytes; their internal limits belong to the decoder. *)

  val limits :
    ?max_bytes:int ->
    ?max_components:int ->
    ?max_features:int ->
    ?max_classes:int ->
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
    ?route_sample_weight:bool ->
    name:string ->
    Standard_scaler.t ->
    (Pipeline.transformer, Error.t) result

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
    ?class_weight:Modelkit_class_weight.Class_weight.t ->
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
    ?decoders:Target.regression Codec.decoder array ->
    ?limits:limits ->
    bytes ->
    (regression_model loaded, Error.t) result

  val decode_binary_classification :
    ?decoders:Target.classification Codec.decoder array ->
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
    ?decoders:Target.regression Codec.decoder array ->
    ?limits:limits ->
    path:string ->
    unit ->
    (regression_model loaded, Error.t) result

  val load_binary_classification :
    ?decoders:Target.classification Codec.decoder array ->
    ?limits:limits ->
    path:string ->
    unit ->
    (binary_classification_model loaded, Error.t) result

  val encode_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    classification_model ->
    (bytes, Error.t) result
  (** Writes the generalized classification task in container version two.
      Built-in regression/binary writers retain version one; external terminals
      use version two with package-qualified codecs and component provenance. *)

  val decode_classification :
    ?decoders:Target.classification Codec.decoder array ->
    ?limits:limits ->
    bytes ->
    (classification_model loaded, Error.t) result
  (** Accepts generalized classification and legacy binary artifacts. The
      binary-specific loader accepts only the binary task, so a generalized
      artifact must use this loader even when its fitted vocabulary has two
      labels. Unknown/unselected codecs and incompatible tasks fail typed. *)

  val save_classification :
    ?metadata:metadata ->
    ?limits:limits ->
    path:string ->
    classification_model ->
    (unit, Error.t) result

  val load_classification :
    ?decoders:Target.classification Codec.decoder array ->
    ?limits:limits ->
    path:string ->
    unit ->
    (classification_model loaded, Error.t) result
end
