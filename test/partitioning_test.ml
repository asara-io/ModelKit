let () =
  Alcotest.run "Partitioning" [ ("contracts", Partitioning_support.tests) ]
