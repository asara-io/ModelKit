open Modelkit_data

val limit : int
val invalid : unit -> 'a
val token : (string -> unit) -> string -> unit
val int : (string -> unit) -> int -> unit
val float : (string -> unit) -> float -> unit
val bool : (string -> unit) -> bool -> unit

val array :
  ((string -> unit) -> 'a -> unit) -> (string -> unit) -> 'a array -> unit

val option :
  ((string -> unit) -> 'a -> unit) -> (string -> unit) -> 'a option -> unit

val result :
  ((string -> unit) -> 'a -> unit) ->
  (string -> unit) ->
  ('a, Error.t) result ->
  unit

val error : (string -> unit) -> Error.t -> unit
val schema : (string -> unit) -> Feature_schema.t -> unit
val encode : ((string -> unit) -> 'a -> unit) -> 'a -> string
val digest : ((string -> unit) -> 'a -> unit) -> 'a -> string

type reader

val reader : string -> reader
val read_token : reader -> string
val read_int : reader -> int
val read_float : reader -> float
val read_bool : reader -> bool
val read_array : (reader -> 'a) -> reader -> 'a array
val read_option : (reader -> 'a) -> reader -> 'a option
val read_result : (reader -> 'a) -> reader -> ('a, Error.t) result
val read_error : reader -> Error.t
val finish : reader -> unit
val protect : (unit -> 'a) -> ('a, Error.t) result
