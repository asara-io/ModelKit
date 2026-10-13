val parameters :
  value:(string -> string) ->
  objective:string ->
  classes:int ->
  (string * string) list

val predict :
  parameters:(string * string) list ->
  rounds:int ->
  x:float array array ->
  y:float array ->
  sample_weight:float array option ->
  test:float array array ->
  float array

val frozen_sampling_reference : unit -> bool
