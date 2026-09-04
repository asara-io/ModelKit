open Modelkit_data

module Class_weight : sig
  type t = Balanced | Explicit of (int * float) array

  val balanced : t
  val explicit : (int * float) list -> (t, Error.t) result

  val class_weights :
    t ->
    ?sample_weight:Sample_weight.t ->
    Target.classification Target.t ->
    ((int * float) array, Error.t) result

  val resolve :
    t ->
    ?sample_weight:Sample_weight.t ->
    Target.classification Target.t ->
    (Sample_weight.t, Error.t) result
end
