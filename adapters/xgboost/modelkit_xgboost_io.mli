open Modelkit

val failure : string -> ('a, Error.t) result

type limits = {
  max_bytes : int;
  max_features : int;
  max_classes : int;
  max_string_bytes : int;
}

val default_limits : limits

val limits :
  ?max_bytes:int ->
  ?max_features:int ->
  ?max_classes:int ->
  ?max_string_bytes:int ->
  unit ->
  (limits, Error.t) result

module Writer : sig
  type t

  val create : unit -> t
  val byte : t -> int -> unit
  val int : t -> int -> unit
  val i64 : t -> int64 -> unit
  val float : t -> float -> unit
  val string : t -> string -> unit
  val bytes : t -> bytes -> unit
  val contents : t -> bytes
end

module Reader : sig
  type t

  val create : limits:limits -> bytes -> t
  val byte : t -> (int, Error.t) result
  val int : t -> (int, Error.t) result
  val i64 : t -> (int64, Error.t) result
  val float : t -> (float, Error.t) result
  val string : t -> (string, Error.t) result
  val bytes : t -> (bytes, Error.t) result
  val length : t -> maximum:int -> (int, Error.t) result
  val require : t -> int -> (unit, Error.t) result
  val ints : t -> maximum:int -> (int array, Error.t) result
  val finish : t -> (unit, Error.t) result
end

val frame : limits:limits -> task:int -> bytes -> (bytes, Error.t) result

val unframe :
  limits:limits -> tasks:int list -> bytes -> (int * bytes, Error.t) result

val save : path:string -> bytes -> (unit, Error.t) result
val read : limits:limits -> path:string -> (bytes, Error.t) result

module Json : sig
  type t

  val parse : max_bytes:int -> string -> (t, Error.t) result
  val version : t -> (int * int * int, Error.t) result

  val check_model :
    columns:int ->
    classes:int ->
    rounds:int ->
    objective:string ->
    version:int * int * int ->
    t ->
    (unit, Error.t) result

  val check_config :
    columns:int ->
    classes:int ->
    objective:string ->
    threads:int ->
    t ->
    (unit, Error.t) result
end
