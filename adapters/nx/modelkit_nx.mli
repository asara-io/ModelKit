(** Checked admission from Nx tensors into immutable ModelKit data. *)

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

val features :
  ?names:string array ->
  ?null_mask:Nx.bool_t ->
  Nx.float64_t ->
  (features, Modelkit.Error.t) result
(** Admits a rank-two float64 tensor in logical row-major order.

    Both contiguous tensors and strided views are accepted. Explicit nulls are
    represented by [null_mask] and stored as NaN in [matrix], while the returned
    mask preserves their identity separately from genuine NaNs. Unmasked
    infinities are rejected. *)

val regression_target :
  Nx.float64_t ->
  ( Modelkit.Target.regression Modelkit.Target.t conversion,
    Modelkit.Error.t )
  result
(** Admits a finite rank-one float64 regression target. *)

val classification_target :
  Nx.int64_t ->
  ( Modelkit.Target.classification Modelkit.Target.t conversion,
    Modelkit.Error.t )
  result
(** Admits a rank-one int64 classification target whose labels fit OCaml [int]
    on the current platform. *)

val sample_weight :
  Nx.float64_t -> (Modelkit.Sample_weight.t conversion, Modelkit.Error.t) result
(** Admits finite, non-negative rank-one weights with at least one positive
    value. *)

val groups :
  Nx.int64_t -> (Modelkit.Groups.t conversion, Modelkit.Error.t) result
(** Admits rank-one group labels whose values fit OCaml [int]. *)

val regression_dataset :
  ?names:string array ->
  ?feature_null_mask:Nx.bool_t ->
  ?sample_weight:Nx.float64_t ->
  ?groups:Nx.int64_t ->
  x:Nx.float64_t ->
  y:Nx.float64_t ->
  unit ->
  (Modelkit.Target.regression admitted_dataset, Modelkit.Error.t) result

val classification_dataset :
  ?names:string array ->
  ?feature_null_mask:Nx.bool_t ->
  ?sample_weight:Nx.float64_t ->
  ?groups:Nx.int64_t ->
  x:Nx.float64_t ->
  y:Nx.int64_t ->
  unit ->
  (Modelkit.Target.classification admitted_dataset, Modelkit.Error.t) result
(** Dataset admission validates tensor ranks, metadata alignment, feature names,
    numeric domains, and label ranges before returning a ModelKit dataset.
    Feature NaNs are retained under [Dataset.Allow_nan]; downstream estimators
    must impute them or otherwise declare support. *)
