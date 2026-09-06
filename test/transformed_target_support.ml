open Modelkit
open Evaluation_metadata_support
module Ttr = Transformed_target_regressor
module Callback = Modelkit.Callback

let target_map f y =
  Target.regression_values y |> Vector.to_array |> Array.map f |> regression

module Center = struct
  type t = unit
  type params = unit
  type fitted = float

  let clone () = ()
  let params () = ()

  let fit_request () =
    Metadata.Request.create ~sample_weight:Metadata.Request.Required
      ~groups:Metadata.Request.Required ~callback:Metadata.Request.Optional ()

  let fit () ~metadata ~rng:_ ~y =
    let weights = Option.get (Metadata.sample_weight metadata) in
    let groups = Option.get (Metadata.groups metadata) in
    let values = Target.regression_values y in
    let sum = ref 0. and total = ref 0. in
    for i = 0 to Target.length y - 1 do
      let row = (Vector.get values i -. 5.) /. 3. |> int_of_float in
      if Sample_weight.get weights i <> float_of_int (row + 1) then
        failwith "misaligned target fit weights";
      if Groups.get groups i <> 100 + (row / 2) then
        failwith "misaligned target fit groups";
      let weight = Sample_weight.get weights i in
      sum := !sum +. (weight *. Vector.get values i);
      total := !total +. weight
    done;
    let* () = report metadata in
    Ok (!sum /. !total)

  let transform mean y = Ok (target_map (fun value -> value -. mean) y)
  let inverse_transform mean y = Ok (target_map (( +. ) mean) y)
end

module Zero = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = target
  type fitted = Feature_schema.t
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit_request () =
    Metadata.Request.create ~sample_weight:Metadata.Request.Required ()

  let fit () ~metadata ~rng:_ ~feature_schema ~x:_ ~y () =
    let weights = Option.get (Metadata.sample_weight metadata) in
    let sum = ref 0. in
    for i = 0 to Target.length y - 1 do
      sum :=
        !sum
        +. Sample_weight.get weights i
           *. Vector.get (Target.regression_values y) i
    done;
    if abs_float !sum > 1e-9 then
      failwith "regressor received uncentered targets";
    Ok feature_schema

  let predict _ ~feature_schema:_ ~x =
    Ok (regression (Array.make (Matrix.rows x) 0.))

  let fitted_params _ = ()
  let feature_schema schema = schema
end

let wrap transformer regressor =
  Ttr.create ~name:"response" ~transformer ~regressor ()
  |> get
  |> Pipeline.set_estimator Pipeline.empty
  |> get

let pipeline () =
  wrap
    (Ttr.transformer (module Center) ())
    (Pipeline.metadata_estimator ~name:"zero" (module Zero) () |> get)

let check_execution execution =
  let source = dataset () in
  let evaluation = run ~execution (pipeline ()) source |> get in
  Array.iter
    (fun fold ->
      let train = Option.get fold.Cross_validation.train_indices in
      let test = Option.get fold.Cross_validation.test_indices in
      let expected_mean = mean source train in
      let fitted = Option.get fold.Cross_validation.model in
      let predicted =
        Pipeline.predict fitted
          ~feature_schema:(Dataset.feature_schema source)
          ~x:(Dataset.features source)
        |> get |> Target.regression_values |> Vector.to_array
      in
      Alcotest.(check (array (Alcotest.float 1e-12)))
        "fold-local learned inverse, metadata-free inference"
        (Array.make 16 expected_mean)
        predicted;
      let weights = Option.get (Dataset.sample_weight source) in
      let target = Target.regression_values (Dataset.target source) in
      let sum, total =
        Array.fold_left
          (fun (sum, total) row ->
            let w = Sample_weight.get weights row in
            let residual = Vector.get target row -. expected_mean in
            (sum +. (w *. residual *. residual), total +. w))
          (0., 0.) test
      in
      Alcotest.check (Alcotest.float 1e-10) "original-space weighted scoring"
        (-.sum /. total) (score fold))
    (Cross_validation.folds evaluation);
  let grid =
    Grid_search.create ~base:() ~build:(fun () -> Ok (pipeline ())) [||] |> get
  in
  let selected =
    Grid_search.Regression.search ~execution ~grid ~splitter:(splitter ())
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 42) source
    |> get |> Grid_search.selection |> get
  in
  let predicted =
    Pipeline.predict selected.Grid_search.selected_model
      ~feature_schema:(Dataset.feature_schema source)
      ~x:(Dataset.features source)
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "full-data refit relearns target inverse"
    (Array.make 16 (mean source (Array.init 16 Fun.id)))
    predicted

let fit ?(metadata = Metadata.empty) specification source =
  Pipeline.fit_with_metadata specification ~metadata
    ~rng:(Rng.create (Seed.of_int 19))
    ~feature_schema:(Dataset.feature_schema source)
    ~x:(Dataset.features source) ~y:(Dataset.target source) ()

let ordinary () =
  Pipeline.estimator ~name:"linear"
    (module Linear_regression)
    (Linear_regression.create ())
  |> get

let expect_error label = function Ok _ -> Alcotest.fail label | Error _ -> ()

let test_functions () =
  let x =
    Matrix.of_arrays [| [| 0. |]; [| 1. |]; [| 2. |]; [| 3. |] |] |> data
  in
  let y =
    regression (Array.init 4 (fun i -> expm1 (0.5 +. (0.2 *. float_of_int i))))
  in
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> data
  in
  let specification =
    wrap (Ttr.functions ~transform:log1p ~inverse_transform:expm1) (ordinary ())
  in
  let fitted = fit specification source |> get in
  let prediction =
    Pipeline.predict fitted ~feature_schema:(Dataset.feature_schema source) ~x
    |> get
  in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "nonlinear inverse prediction"
    (Target.regression_values y |> Vector.to_array)
    (Target.regression_values prediction |> Vector.to_array);
  expect_error "wrapper has no artifact codec"
    (Artifact.encode_regression fitted);
  let bad = Matrix.of_arrays [| [| 1.; 2. |] |] |> data in
  expect_error "prediction schema remains checked"
    (Pipeline.predict fitted
       ~feature_schema:(Dataset.feature_schema source)
       ~x:bad)

let test_validation () =
  let source = dataset () in
  let identity = Ttr.functions ~transform:Fun.id ~inverse_transform:Fun.id in
  List.iter
    (fun tolerance ->
      expect_error "invalid tolerance"
        (Ttr.create ~rtol:tolerance ~name:"bad" ~transformer:identity
           ~regressor:(ordinary ()) ()))
    [ nan; infinity; -1. ];
  expect_error "blank name"
    (Ttr.create ~name:" " ~transformer:identity ~regressor:(ordinary ()) ());
  List.iter
    (fun transformer ->
      expect_error "invalid forward/inverse"
        (fit (wrap transformer (ordinary ())) source))
    [
      Ttr.functions ~transform:log ~inverse_transform:Fun.id;
      Ttr.functions ~transform:(fun _ -> nan) ~inverse_transform:Fun.id;
      Ttr.functions ~transform:Fun.id ~inverse_transform:(fun _ -> infinity);
    ];
  expect_error "requests preflight before training" (fit (pipeline ()) source);
  let callback = Callback.create (fun _ -> Ok Callback.Cancel) |> get in
  let metadata = Metadata.of_dataset ~callback source in
  let result = fit ~metadata (pipeline ()) source in
  match result with
  | Error error ->
      Alcotest.(check bool)
        "cancellation is control error" true
        (Callback.is_control_error error)
  | Ok _ -> Alcotest.fail "callback must cancel"

module Short = struct
  include Center

  let fit_request () = Metadata.Request.none
  let fit () ~metadata:_ ~rng:_ ~y:_ = Ok 0.
  let transform _ _ = Ok (regression [||])
end

module Short_inverse = struct
  include Short

  let transform _ y = Ok y
  let inverse_transform _ _ = Ok (regression [||])
end

module Late_inverse = struct
  include Short

  let transform _ y = Ok y

  let inverse_transform _ y =
    if Target.length y = 16 then Ok y else Ok (regression [||])
end

let test_lengths () =
  let source = dataset () in
  List.iter
    (fun transformer ->
      expect_error "target transform length"
        (fit (wrap transformer (ordinary ())) source))
    [
      Ttr.transformer (module Short) ();
      Ttr.transformer (module Short_inverse) ();
    ];
  let fitted =
    fit (wrap (Ttr.transformer (module Late_inverse) ()) (ordinary ())) source
    |> get
  in
  let x = Matrix.of_arrays [| [| 0.; 100. |] |] |> data in
  expect_error "inverse inference length"
    (Pipeline.predict fitted ~feature_schema:(Dataset.feature_schema source) ~x)

let test_parity () =
  let channel = open_in (Sys.getenv "MODELKIT_TARGET_FIXTURE") in
  let rows =
    Fun.protect
      ~finally:(fun () -> close_in channel)
      (fun () ->
        let rec read acc =
          match input_line channel with
          | line -> read (line :: acc)
          | exception End_of_file -> List.rev acc
        in
        read [])
  in
  let fields =
    List.filter_map
      (fun line ->
        match String.split_on_char '\t' line with
        | [ name; values ] ->
            Some
              ( name,
                String.split_on_char ',' values
                |> List.map float_of_string |> Array.of_list )
        | _ -> None)
      rows
  in
  let values name = List.assoc name fields in
  let matrix name =
    values name
    |> Array.map (fun value -> [| value |])
    |> Matrix.of_arrays |> data
  in
  let x = matrix "train" and y = regression (values "target") in
  let weights =
    Sample_weight.of_array ~expected_length:(Target.length y) (values "weights")
    |> data
  in
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights ~x
      ~y ()
    |> data
  in
  let fitted =
    fit
      ~metadata:(Metadata.of_dataset source)
      (wrap
         (Ttr.functions ~transform:log1p ~inverse_transform:expm1)
         (ordinary ()))
      source
    |> get
  in
  let predicted =
    Pipeline.predict fitted
      ~feature_schema:(Dataset.feature_schema source)
      ~x:(matrix "test")
    |> get |> Target.regression_values |> Vector.to_array
  in
  Alcotest.(check (array (Alcotest.float 1e-12)))
    "pinned sklearn weighted nonlinear regression" (values "prediction")
    predicted;
  let extreme = Matrix.of_arrays [| [| 1e6 |] |] |> data in
  expect_error "inverse overflow at prediction"
    (Pipeline.predict fitted
       ~feature_schema:(Dataset.feature_schema source)
       ~x:extreme)

let test_seeds_and_clone () =
  let clones = ref 0 and target_seeds = ref [] and regressor_seeds = ref [] in
  let module Identity = struct
    type t = unit
    type params = unit
    type fitted = unit

    let clone () = incr clones
    let params () = ()
    let fit_request () = Metadata.Request.none

    let fit () ~metadata:_ ~rng ~y:_ =
      target_seeds := Rng.to_seed rng :: !target_seeds;
      Ok ()

    let transform () y = Ok y
    let inverse_transform () y = Ok y
  end in
  let module Regressor = struct
    include Linear_regression

    let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
      regressor_seeds := Rng.to_seed rng :: !regressor_seeds;
      Linear_regression.fit specification ?sample_weight ~rng ~feature_schema ~x
        ~y ()
  end in
  let source = dataset () in
  let specification =
    wrap
      (Ttr.transformer (module Identity) ())
      (Pipeline.estimator ~name:"linear"
         (module Regressor)
         (Regressor.create ())
      |> get)
  in
  ignore (fit specification source |> get);
  ignore (fit (Pipeline.clone specification) source |> get);
  Alcotest.(check int) "fresh target specification per fit" 2 !clones;
  let same = function [ a; b ] -> Seed.equal a b | _ -> false in
  Alcotest.(check bool) "target seed reproducibility" true (same !target_seeds);
  Alcotest.(check bool)
    "regressor seed reproducibility" true (same !regressor_seeds);
  Alcotest.(check bool)
    "independent child streams" false
    (Seed.equal (List.hd !target_seeds) (List.hd !regressor_seeds))

let test_extreme_tolerances () =
  let x = Matrix.of_arrays [| [| 0. |]; [| 1. |] |] |> data in
  let y = regression [| 1e308; 1e308 |] in
  let source =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> data
  in
  let transformer =
    Ttr.functions ~transform:Fun.id ~inverse_transform:Float.neg
  in
  let regressor =
    Pipeline.metadata_estimator ~name:"terminal" (module Terminal) () |> get
  in
  let specification =
    Ttr.create ~rtol:1.5 ~atol:0. ~name:"response" ~transformer ~regressor ()
    |> get
    |> Pipeline.set_estimator Pipeline.empty
    |> get
  in
  expect_error "overflow cannot hide an inverse mismatch"
    (fit specification source);
  let specification =
    Ttr.create ~rtol:2. ~atol:0. ~name:"response" ~transformer ~regressor ()
    |> get
    |> Pipeline.set_estimator Pipeline.empty
    |> get
  in
  ignore (fit specification source |> get)

let tests =
  [
    ( "clones and independent deterministic streams",
      `Quick,
      test_seeds_and_clone );
    ("inverse tolerance near float limits", `Quick, test_extreme_tolerances);
    ("pinned sklearn parity and inverse overflow", `Quick, test_parity);
    ( "fold-local target fitting, scoring and refit",
      `Quick,
      fun () -> check_execution Execution.sequential );
    ("nonlinear functions and schema", `Quick, test_functions);
    ("validation and callbacks", `Quick, test_validation);
    ("row-preserving transform and inverse", `Quick, test_lengths);
  ]
