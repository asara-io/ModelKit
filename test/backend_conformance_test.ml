open Modelkit

module Reference_conformance = Backend_conformance.Make (struct
  module Backend = Reference_backend

  let absolute_tolerance = 0.0
  let relative_tolerance = 0.0
end)

let () =
  Alcotest.run "numerical backend conformance"
    [ (Reference_backend.name, Reference_conformance.tests) ]
