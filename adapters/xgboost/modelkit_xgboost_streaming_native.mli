type vector = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

type batch = {
  features : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t;
  labels : vector;
  weights : vector option;
}

val train :
  parameters:(string * string) list ->
  rounds:int ->
  threads:int ->
  rows:int ->
  columns:int ->
  cache_prefix:string ->
  next:(unit -> (batch option, Modelkit.Error.t) result) ->
  reset:(unit -> (unit, Modelkit.Error.t) result) ->
  unit ->
  (bytes, Modelkit.Error.t) result
