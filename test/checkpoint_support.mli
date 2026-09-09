val tests : (string * Alcotest.speed_level * (unit -> unit)) list
val check_execution : Modelkit.Execution.t -> unit
val produce : unit -> bytes
val resume : ?execution:Modelkit.Execution.t -> bytes -> unit
