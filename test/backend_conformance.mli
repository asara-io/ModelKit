open Modelkit

module type CONFIG = sig
  module Backend : NUMERICAL_BACKEND

  val absolute_tolerance : float
  val relative_tolerance : float
end

module Make (Config : CONFIG) : sig
  module Backend : module type of Config.Backend

  val tests : unit Alcotest.test_case list
end
