let () =
  Alcotest.run "Transformed targets"
    [ ("regression", Transformed_target_support.tests) ]
