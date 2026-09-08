open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data_result result =
  Result.map_error
    (Error.of_data_error ~context:[]
       ~remediation:"return valid external component data")
    result

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data

let vector_equal left right =
  let left = Target.regression_values left |> Vector.to_array in
  let right = Target.regression_values right |> Vector.to_array in
  left = right

type estimator_fitted = { value : float; schema : Feature_schema.t }

module External_estimator = struct
  type t = float
  type params = float
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type fitted = estimator_fitted
  type rng = Rng.t

  let clone value = value
  let params value = value

  let fit value ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    Ok { value; schema = feature_schema }

  let predict fitted ~feature_schema:_ ~x =
    Array.make (Matrix.rows x) fitted.value
    |> Vector.of_array |> Target.regression |> data_result

  let fitted_params fitted = fitted.value
  let feature_schema fitted = fitted.schema
end

module Invalid_estimator = struct
  include External_estimator

  let predict fitted ~feature_schema:_ ~x =
    Array.make (max 0 (Matrix.rows x - 1)) fitted.value
    |> Vector.of_array |> Target.regression |> data_result
end

type transformer_fitted = { transformer_schema : Feature_schema.t }

module External_transformer = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type fitted = transformer_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    Ok { transformer_schema = feature_schema }

  let transform _ ~feature_schema:_ ~x = Ok x
  let fitted_params _ = ()
  let input_schema fitted = fitted.transformer_schema
  let output_schema fitted = fitted.transformer_schema
end

module Invalid_transformer = struct
  include External_transformer

  let transform _ ~feature_schema:_ ~x =
    Matrix.init
      ~rows:(max 0 (Matrix.rows x - 1))
      ~columns:(Matrix.columns x)
      (fun row column -> Matrix.get x row column)
    |> data_result
end

module External_scorer = struct
  type t = { scorer_name : string; result : float }
  type params = t
  type truth = Target.regression Target.t
  type prediction = Target.regression Target.t

  let clone specification = specification
  let params specification = specification
  let name specification = specification.scorer_name

  let score specification ?sample_weight:_ ~truth:_ ~prediction:_ () =
    Ok specification.result
end

let fixture () =
  let x = matrix [| [| 0. |]; [| 1. |]; [| 2. |]; [| 3. |] |] in
  let y = regression [| 1.; 1.; 1.; 1. |] in
  let feature_schema = Feature_schema.of_matrix x |> get_data in
  (x, y, feature_schema)

let test_estimator_checks () =
  let x, y, feature_schema = fixture () in
  let fixture : (_, _, _, _, _, _) Conformance.Estimator.fixture =
    {
      Conformance.Estimator.specification = 1.;
      rng = (fun () -> Rng.create (Seed.of_int 7));
      feature_schema;
      x;
      y;
      sample_weight = None;
      equal_params = Float.equal;
      prediction_length = Target.length;
      equal_prediction = vector_equal;
    }
  in
  let valid = Conformance.Estimator.check (module External_estimator) fixture in
  let invalid =
    Conformance.Estimator.check (module Invalid_estimator) fixture
  in
  Alcotest.(check bool) "valid estimator" true (Conformance.passed valid);
  Alcotest.(check bool)
    "invalid estimator detected" false
    (Conformance.passed invalid);
  Alcotest.(check int)
    "one estimator violation" 1
    (Array.length (Conformance.failures invalid))

let test_transformer_checks () =
  let x, y, feature_schema = fixture () in
  let fixture : (_, _, _, _, _) Conformance.Transformer.fixture =
    {
      Conformance.Transformer.specification = ();
      rng = (fun () -> Rng.create (Seed.of_int 7));
      feature_schema;
      x;
      y = Some y;
      sample_weight = None;
      equal_params = (fun () () -> true);
    }
  in
  let valid =
    Conformance.Transformer.check (module External_transformer) fixture
  in
  let invalid =
    Conformance.Transformer.check (module Invalid_transformer) fixture
  in
  Alcotest.(check bool) "valid transformer" true (Conformance.passed valid);
  Alcotest.(check bool)
    "invalid transformer detected" false
    (Conformance.passed invalid);
  Alcotest.(check int)
    "row violation" 1
    (Array.length (Conformance.failures invalid))

let scorer_capabilities =
  Capability.scorer ~sample_weight:Capability.Supported
    ~prediction:Capability.Direct ()

let scorer_fixture specification =
  let _, y, _ = fixture () in
  ({
     Conformance.Scorer.specification;
     capabilities = scorer_capabilities;
     truth = y;
     prediction = y;
     sample_weight = None;
     equal_params = ( = );
   }
    : (_, _, _, _) Conformance.Scorer.fixture)

let test_scorer_checks () =
  let valid =
    Conformance.Scorer.check
      (module External_scorer)
      (scorer_fixture
         { External_scorer.scorer_name = "constant"; result = 0.5 })
  in
  let invalid =
    Conformance.Scorer.check
      (module External_scorer)
      (scorer_fixture { External_scorer.scorer_name = " "; result = Float.nan })
  in
  Alcotest.(check bool) "valid scorer" true (Conformance.passed valid);
  Alcotest.(check bool)
    "invalid scorer detected" false
    (Conformance.passed invalid);
  Alcotest.(check int)
    "blank and non-finite violations" 3
    (Array.length (Conformance.failures invalid))

let evaluation_dataset () =
  let x = Array.init 9 (fun row -> [| Float.of_int row |]) |> matrix in
  let y =
    Array.init 9 (fun row -> (2. *. Float.of_int row) +. 1.) |> regression
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data

let splitter () =
  K_fold.create ~folds:3 () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let pipeline () =
  let estimator =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.set_estimator Pipeline.empty estimator |> get

let custom_scorer () =
  Scorer.of_module ~capabilities:scorer_capabilities
    (module External_scorer)
    { External_scorer.scorer_name = "external_constant"; result = 0.25 }

let grid () =
  Grid_search.create ~base:() ~build:(fun () -> Ok (pipeline ())) [||] |> get

let space () =
  Randomized_search.create ~base:() ~build:(fun () -> Ok (pipeline ())) [||]
  |> get

let test_custom_scorer_selection () =
  let dataset = evaluation_dataset () in
  let custom_scorers = [| custom_scorer () |] in
  let cv =
    Cross_validation.Regression.cross_validate ~splitter:(splitter ())
      ~scorers:[||] ~custom_scorers ~seed:(Seed.of_int 11) (pipeline ()) dataset
    |> get
  in
  let cv_score = (Cross_validation.folds cv).(0).Cross_validation.scores.(0) in
  Alcotest.(check string)
    "CV custom field" "external_constant" cv_score.Cross_validation.name;
  let grid_report =
    Grid_search.Regression.search ~grid:(grid ()) ~splitter:(splitter ())
      ~scorers:[||] ~custom_scorers ~refit:"external_constant"
      ~seed:(Seed.of_int 11) dataset
    |> get
  in
  Alcotest.(check int)
    "grid custom candidate" 1
    (Array.length (Grid_search.candidates grid_report));
  let random_report =
    Randomized_search.Regression.search ~space:(space ())
      ~splitter:(splitter ()) ~scorers:[||] ~custom_scorers
      ~refit:"external_constant" ~seed:(Seed.of_int 11) dataset
    |> get
  in
  Alcotest.(check int)
    "random custom candidate" 1
    (Array.length (Randomized_search.candidates random_report))

let test_capability_preflight () =
  let base = evaluation_dataset () in
  let sample_weight =
    Sample_weight.of_array ~expected_length:9 (Array.make 9 1.) |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x:(Dataset.features base)
      ~y:(Dataset.target base) ~sample_weight ()
    |> get_data
  in
  let unsupported =
    Scorer.of_module
      ~capabilities:(Capability.scorer ~prediction:Capability.Direct ())
      (module External_scorer)
      { External_scorer.scorer_name = "unweighted"; result = 0. }
  in
  let result =
    Cross_validation.Regression.cross_validate ~splitter:(splitter ())
      ~scorers:[||] ~custom_scorers:[| unsupported |] ~seed:(Seed.of_int 11)
      (pipeline ()) dataset
  in
  Alcotest.(check bool)
    "unsupported weights rejected" true (Result.is_error result);
  let estimator =
    Capability.estimator ~sample_weight:Capability.Supported
      ~predict_proba:Capability.Supported ()
  in
  Alcotest.(check bool)
    "estimator capability description" true
    (estimator.Capability.estimator_sample_weight = Capability.Supported
    && estimator.Capability.estimator_predict_proba = Capability.Supported
    && estimator.Capability.estimator_decision_function = Capability.Unsupported
    )

let () =
  Alcotest.run "Public conformance"
    [
      ( "contracts",
        [
          Alcotest.test_case "estimator" `Quick test_estimator_checks;
          Alcotest.test_case "transformer" `Quick test_transformer_checks;
          Alcotest.test_case "scorer" `Quick test_scorer_checks;
          Alcotest.test_case "custom scorer CV and search" `Quick
            test_custom_scorer_selection;
          Alcotest.test_case "capability preflight" `Quick
            test_capability_preflight;
        ] );
    ]
