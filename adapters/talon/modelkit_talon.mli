(** Checked admission from explicitly selected Talon dataframe columns into
    immutable ModelKit data.

    Every function reads the typed tensor behind a named column without Talon's
    convenience conversion, so no column is cast, widened, or narrowed
    implicitly. Selected columns keep their names and selection order as the
    feature schema. Feature columns may carry Talon null masks; target, weight,
    and group columns must not contain nulls. *)

type 'a conversion = 'a Modelkit.Admission.conversion = {
  value : 'a;
  report : Modelkit.Conversion_report.t;
}

type features = Modelkit.Admission.features = {
  matrix : Modelkit.Matrix.t;
  schema : Modelkit.Feature_schema.t;
  null_mask : Modelkit.Null_mask.t option;
  feature_reports : Modelkit.Conversion_report.t list;
}

type 'kind admitted_dataset = 'kind Modelkit.Admission.dataset = {
  dataset : 'kind Modelkit.Dataset.t;
  feature_null_mask : Modelkit.Null_mask.t option;
  dataset_reports : Modelkit.Conversion_report.t list;
}

val features : Talon.t -> string list -> (features, Modelkit.Error.t) result
(** [features frame columns] admits the named float64 columns of [frame] in the
    order given by [columns], which also becomes the feature-name order.

    A column that is missing, selected twice, or not float64 is a validation
    error. Column null masks are merged into the returned {!type:features} null
    mask and stored as NaN in the matrix, while genuine unmasked NaNs are
    retained as data. Unmasked infinities are rejected. The mask is returned
    only when at least one selected column carries an explicit Talon null mask.
*)

val regression_target :
  Talon.t ->
  string ->
  ( Modelkit.Target.regression Modelkit.Target.t conversion,
    Modelkit.Error.t )
  result
(** Admits a float64 column without nulls as a finite regression target. *)

val classification_target :
  Talon.t ->
  string ->
  ( Modelkit.Target.classification Modelkit.Target.t conversion,
    Modelkit.Error.t )
  result
(** Admits an int64 column without nulls whose labels fit OCaml [int] on the
    current platform. *)

val sample_weight :
  Talon.t ->
  string ->
  (Modelkit.Sample_weight.t conversion, Modelkit.Error.t) result
(** Admits a float64 column without nulls holding finite, non-negative weights
    with at least one positive value. *)

val groups :
  Talon.t -> string -> (Modelkit.Groups.t conversion, Modelkit.Error.t) result
(** Admits an int64 column without nulls whose values fit OCaml [int]. *)

val regression_dataset :
  ?sample_weight:string ->
  ?groups:string ->
  features:string list ->
  target:string ->
  Talon.t ->
  (Modelkit.Target.regression admitted_dataset, Modelkit.Error.t) result

val classification_dataset :
  ?sample_weight:string ->
  ?groups:string ->
  features:string list ->
  target:string ->
  Talon.t ->
  (Modelkit.Target.classification admitted_dataset, Modelkit.Error.t) result
(** Dataset admission requires every role to name a distinct column, validates
    column types, null policy, numeric domains, and label ranges, and returns a
    ModelKit dataset whose feature names are the selected column names. Feature
    NaNs are retained under [Dataset.Allow_nan]; downstream estimators must
    impute them or otherwise declare support. *)
