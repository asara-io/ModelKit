let () =
  Alcotest.run "Successive halving" [ ("contracts", Halving_support.tests) ]
