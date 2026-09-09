module Callback = Modelkit.Callback

val get : ('a, Modelkit.Error.t) result -> 'a
val data : ('a, Modelkit.Data_error.t) result -> 'a
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
val regression : float array -> Modelkit.Target.regression Modelkit.Target.t
val fits : int Atomic.t
val transforms : int Atomic.t
val invalid : string -> ('a, Modelkit.Error.t) result
val report : Modelkit.Metadata.t -> (unit, Modelkit.Error.t) result
val dataset : unit -> Modelkit.Target.regression Modelkit.Dataset.t

module Consumer : sig
  type t = {
    align : bool;
    offset : float;
    callbacks : Modelkit.Metadata.Request.policy;
  }

  type params = t
  type target = Modelkit.Target.regression Modelkit.Target.t

  type fitted = {
    specification : t;
    input : Modelkit.Feature_schema.t;
    output : Modelkit.Feature_schema.t;
    mean : float;
  }

  type rng = Modelkit.Rng.t

  val clone : 'a -> 'a
  val params : 'a -> 'a
  val fit_request : t -> Modelkit.Metadata.Request.t
  val transform_request : t -> Modelkit.Metadata.Request.t

  val fit :
    t ->
    metadata:Modelkit.Metadata.t ->
    rng:'a ->
    feature_schema:Modelkit.Feature_schema.t ->
    x:Modelkit.Matrix.t ->
    y:Modelkit.Target.regression Modelkit.Target.t option ->
    unit ->
    (fitted, Modelkit.Error.t) result

  val transform :
    fitted ->
    metadata:Modelkit.Metadata.t ->
    feature_schema:'a ->
    x:Modelkit.Matrix.t ->
    (Modelkit.Matrix.t, Modelkit.Error.t) result

  val fitted_params : fitted -> t
  val input_schema : fitted -> Modelkit.Feature_schema.t
  val output_schema : fitted -> Modelkit.Feature_schema.t
end

module Terminal : sig
  type t = unit
  type params = unit
  type target = Modelkit.Target.regression Modelkit.Target.t
  type prediction = target
  type fitted = Modelkit.Feature_schema.t
  type rng = Modelkit.Rng.t

  val clone : unit -> unit
  val params : unit -> unit
  val fit_request : unit -> Modelkit.Metadata.Request.t

  val fit :
    unit ->
    metadata:Modelkit.Metadata.t ->
    rng:'a ->
    feature_schema:'b ->
    x:'c ->
    y:'d ->
    unit ->
    ('b, Modelkit.Error.t) result

  val predict :
    'a ->
    feature_schema:'b ->
    x:Modelkit.Matrix.t ->
    (Modelkit.Target.regression Modelkit.Target.t, 'c) result

  val fitted_params : 'a -> unit
  val feature_schema : 'a -> 'a
end

val nest :
  'a Modelkit.Pipeline.Supervised.stage -> 'a Modelkit.Pipeline.Supervised.stage

val pipeline :
  ?align:bool ->
  ?offset:float ->
  ?callbacks:Modelkit.Metadata.Request.policy ->
  unit ->
  ( Modelkit.Target.regression Modelkit.Target.t,
    Terminal.prediction )
  Modelkit.Pipeline.t

val splitter : unit -> 'a Modelkit.Cross_validation.splitter

val run :
  ?metadata:Modelkit.Metadata.t ->
  ?execution:Modelkit.Execution.t ->
  ?failure_policy:Modelkit.Cross_validation.failure_policy ->
  ( Modelkit.Target.regression Modelkit.Target.t,
    Modelkit.Target.regression Modelkit.Target.t )
  Modelkit.Pipeline.t ->
  Modelkit.Target.regression Modelkit.Dataset.t ->
  ( Modelkit.Cross_validation.Regression.model Modelkit.Cross_validation.report,
    Modelkit.Error.t )
  result

val predictions :
  Modelkit.Metadata.t ->
  'a Modelkit.Dataset.t ->
  ('b, Modelkit.Target.regression Modelkit.Target.t) Modelkit.Pipeline.fitted ->
  float array

val mean : Modelkit.Target.regression Modelkit.Dataset.t -> int array -> float

val expected :
  Modelkit.Target.regression Modelkit.Dataset.t ->
  Modelkit.Metadata.t ->
  int array ->
  float array

val grid :
  unit ->
  ( float,
    Modelkit.Target.regression Modelkit.Target.t,
    Terminal.prediction )
  Modelkit.Grid_search.grid

val search :
  ?metadata:Modelkit.Metadata.t ->
  ?execution:Modelkit.Execution.t ->
  Modelkit.Target.regression Modelkit.Dataset.t ->
  ( Modelkit.Grid_search.Regression.model Modelkit.Grid_search.report,
    Modelkit.Error.t )
  result

val score : 'a Modelkit.Cross_validation.fold -> float
