open Modelkit
module External = Third_party_estimator

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let[@warning "-4"] expect_artifact_error = function
  | Error error -> (
      match Error.kind error with
      | Error.Artifact _ -> ()
      | _ -> Alcotest.fail ("expected artifact error: " ^ Error.to_string error)
      )
  | Ok _ -> Alcotest.fail "expected unsupported artifact encoding"

let specification offset =
  External.Weighted_mean_regressor.create ~offset () |> get

let splitter () =
  K_fold.create ~folds:4 ~shuffle:true ()
  |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let test_conformance () =
  External.conformance (specification 0.)
  |> get |> Conformance.passed
  |> Alcotest.(check bool) "published checks" true

let test_nested_metadata_parallel_cv () =
  let source = External.dataset () |> get in
  let pipeline = External.pipeline (specification 0.) |> get in
  let execution =
    Modelkit_parallel.create ~inner_threads:1 ~domains:2 ()
    |> get |> Modelkit_parallel.execution
  in
  let report =
    Cross_validation.Regression.cross_validate ~execution
      ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 41) pipeline source
    |> get
  in
  Alcotest.(check int)
    "all parallel folds succeed" 4
    (Cross_validation.successful_fold_count report)

let test_randomized_search () =
  let axis =
    Randomized_search.axis ~name:"offset"
      ~distribution:(Parameter_distribution.choice [| -2.; 0.; 2. |] |> get)
      ~encode:(fun value -> Grid_search.Float value)
      ~set:(fun _ offset -> External.Weighted_mean_regressor.create ~offset ())
    |> get
  in
  let space =
    Randomized_search.create ~iterations:3 ~base:(specification 0.)
      ~build:(fun specification -> External.pipeline specification)
      [| axis |]
    |> get
  in
  let report =
    Randomized_search.Regression.search ~space ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 43)
      (External.dataset () |> get)
    |> get
  in
  Alcotest.(check int)
    "all candidates evaluated" 3
    (Array.length (Randomized_search.candidates report));
  ignore (Randomized_search.refit_result report |> get)

let test_provenance_and_artifact_support () =
  let source = External.dataset () |> get in
  let pipeline = External.pipeline (specification 0.) |> get in
  let fitted =
    Pipeline.fit_with_metadata pipeline
      ~metadata:(Metadata.of_dataset source)
      ~rng:(Rng.create (Seed.of_int 47))
      ~feature_schema:(Dataset.feature_schema source)
      ~x:(Dataset.features source) ~y:(Dataset.target source) ()
    |> get
  in
  let report = Pipeline.artifact_report fitted in
  let terminal = Pipeline.artifact_estimator report in
  Alcotest.(check string)
    "component name" "weighted_mean" terminal.Pipeline.component_name;
  (match terminal.Pipeline.provenance with
  | None -> Alcotest.fail "missing external provenance"
  | Some provenance ->
      Alcotest.(check string)
        "package" "third-party-estimator"
        (Pipeline.provenance_package provenance);
      Alcotest.(check string)
        "version" "1.0.0"
        (Pipeline.provenance_version provenance);
      Alcotest.(check string)
        "implementation" "Weighted_mean_regressor"
        (Pipeline.provenance_implementation provenance));
  Alcotest.(check bool)
    "terminal codec unsupported" true
    (terminal.Pipeline.serialization_support = Pipeline.Unsupported);
  Alcotest.(check bool)
    "pipeline artifact unsupported" false
    (Pipeline.portable_artifact_supported report);
  let bare = External.pipeline ~nested:false (specification 0.) |> get in
  let bare_fitted =
    Pipeline.fit_with_metadata bare
      ~metadata:(Metadata.of_dataset source)
      ~rng:(Rng.create (Seed.of_int 47))
      ~feature_schema:(Dataset.feature_schema source)
      ~x:(Dataset.features source) ~y:(Dataset.target source) ()
    |> get
  in
  Artifact.encode_regression bare_fitted |> expect_artifact_error

let () =
  Alcotest.run "Public estimator consumer"
    [
      ( "extension contract",
        [
          Alcotest.test_case "conformance" `Quick test_conformance;
          Alcotest.test_case "nested metadata parallel CV" `Quick
            test_nested_metadata_parallel_cv;
          Alcotest.test_case "randomized search" `Quick test_randomized_search;
          Alcotest.test_case "provenance and artifacts" `Quick
            test_provenance_and_artifact_support;
        ] );
    ]
