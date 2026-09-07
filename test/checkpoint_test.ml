let () =
  Alcotest.run "Search checkpoints" [ ("contracts", Checkpoint_support.tests) ]
