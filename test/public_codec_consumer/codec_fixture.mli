open Modelkit

module Mean :
  REGRESSOR with type t = float and type params = float and type rng = Rng.t

module Classifier : sig
  include
    CLASSIFIER with type t = int and type params = int and type rng = Rng.t

  val labels : int -> int array
  val classes : fitted -> int array

  val predict_proba :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result
end

val package : string
val provenance : string -> (Pipeline.provenance, Error.t) result
val failure : string -> ('a, Error.t) result

val mean_codec :
  ?calls:int ref ->
  ?bad_schema:bool ->
  ?version:int ->
  ?name:string ->
  unit ->
  ((float, Target.regression) Artifact.Codec.t, Error.t) result

val classifier_codec :
  ?calls:int ref ->
  ?bad_classes:bool ->
  unit ->
  ((int, Target.classification) Artifact.Codec.t, Error.t) result
