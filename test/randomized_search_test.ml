let () =
  Alcotest.run "Randomized search"
    [ ("contracts", Randomized_search_support.tests) ]
