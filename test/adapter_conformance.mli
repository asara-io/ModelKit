(** Source-neutral admission conformance suite shared by every adapter.

    Adapter test executables instantiate {!Make} with a module that constructs
    the adapter's own source values from plain OCaml arrays. Feature names used
    by the harness never collide with the column roles ["target"], ["weight"],
    and ["group"] that table adapters may reserve. *)

open Modelkit

module type ADAPTER = sig
  val name : string

  val features :
    ?null_mask:bool array array ->
    names:string array ->
    float array array ->
    (Admission.features, Error.t) result

  val regression_target :
    float array ->
    (Target.regression Target.t Admission.conversion, Error.t) result

  val classification_target :
    int64 array ->
    (Target.classification Target.t Admission.conversion, Error.t) result

  val sample_weight :
    float array -> (Sample_weight.t Admission.conversion, Error.t) result

  val groups : int64 array -> (Groups.t Admission.conversion, Error.t) result

  val classification_dataset :
    ?null_mask:bool array array ->
    ?sample_weight:float array ->
    ?groups:int64 array ->
    names:string array ->
    x:float array array ->
    y:int64 array ->
    unit ->
    (Target.classification Admission.dataset, Error.t) result

  val regression_dataset :
    ?null_mask:bool array array ->
    ?sample_weight:float array ->
    ?groups:int64 array ->
    names:string array ->
    x:float array array ->
    y:float array ->
    unit ->
    (Target.regression Admission.dataset, Error.t) result
end

module Make (_ : ADAPTER) : sig
  val tests : unit Alcotest.test_case list
end
