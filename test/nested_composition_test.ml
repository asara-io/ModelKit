open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let rng () = Rng.create (Seed.of_int 42)

let names schema =
  Feature_schema.names schema |> Option.get |> Feature_names.to_array

let expect_error = function
  | Ok _ -> Alcotest.fail "expected typed error"
  | Error _ -> ()

let impute name =
  Pipeline.transformer ~name (module Simple_imputer) (Simple_imputer.mean ())
  |> get

let scale ?route_sample_weight ?(with_std = true) name =
  Pipeline.transformer ?route_sample_weight ~name
    (module Standard_scaler)
    (Standard_scaler.create ~with_std ())
  |> get

let chain name stages =
  Transformer_pipeline.create stages
  |> get
  |> Transformer_pipeline.stage ~name
  |> get

let union name branches =
  Feature_union.create branches |> get |> Feature_union.stage ~name |> get

let columns values = Column_selector.indices values |> get

let terminal () =
  Pipeline.estimator ~name:"model"
    (module Ridge_regression)
    (Ridge_regression.create () |> get)
  |> get

let pipeline stage =
  Pipeline.add_transformer Pipeline.empty stage |> get |> fun builder ->
  Pipeline.set_estimator builder (terminal ()) |> get

let check_float name expected actual =
  Alcotest.(check bool)
    name true
    (Float.abs (expected -. actual)
    <= 1e-12 *. Float.max 1. (Float.abs expected))

let check_matrix name expected actual =
  Alcotest.(check int)
    (name ^ " rows") (Array.length expected) (Matrix.rows actual);
  Array.iteri
    (fun row values ->
      Alcotest.(check int)
        (name ^ " columns") (Array.length values) (Matrix.columns actual);
      Array.iteri
        (fun column value ->
          check_float name value (Matrix.get actual row column))
        values)
    expected

let read_fixture () =
  let matrices = Hashtbl.create 8 and names = Hashtbl.create 2 in
  In_channel.with_open_text (Sys.getenv "MODELKIT_NESTED_FIXTURE") (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if line <> "" && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ name; row; values ] ->
                let rows =
                  match Hashtbl.find_opt matrices name with
                  | Some rows -> rows
                  | None ->
                      let rows = Hashtbl.create 4 in
                      Hashtbl.add matrices name rows;
                      rows
                in
                Hashtbl.add rows (int_of_string row)
                  (String.split_on_char ',' values
                  |> List.map float_of_string |> Array.of_list)
            | [ name; values ] ->
                Hashtbl.add names name
                  (String.split_on_char ',' values |> Array.of_list)
            | _ -> Alcotest.fail "invalid nested fixture"));
  ( (fun name ->
      let rows = Hashtbl.find matrices name in
      Array.init (Hashtbl.length rows) (Hashtbl.find rows)),
    fun name -> Hashtbl.find names name )

let union_spec () =
  Feature_union.create
    [|
      Feature_union.transformer
        (chain "scaled" [| impute "impute"; scale "scale" |]);
      Feature_union.transformer (impute "imputed");
      Feature_union.drop ~name:"unused" |> get;
    |]
  |> get

let nested_spec () =
  let views =
    union "views"
      [|
        Feature_union.transformer (scale "scaled");
        Feature_union.passthrough ~name:"raw" |> get;
      |]
  in
  let numeric = chain "numeric" [| impute "impute"; views |] in
  let column_spec =
    Column_transformer.create
      [|
        Column_transformer.transformer ~columns:(columns [| 0 |]) numeric;
        Column_transformer.transformer ~columns:(columns [| 1 |])
          (impute "other");
      |]
    |> get
  in
  Transformer_pipeline.create
    [|
      Column_transformer.stage ~name:"columns" column_spec |> get; scale "scale";
    |]
  |> get

let test_parity () =
  let data, expected_names = read_fixture () in
  let train = matrix (data "train") and test = matrix (data "test") in
  let feature_schema = schema train in
  let fitted, output, allocation =
    Feature_union.fit_transform (union_spec ()) ~rng:(rng ()) ~feature_schema
      ~x:train ~y:None ()
    |> get
  in
  check_matrix "union training" (data "union_train") output;
  check_matrix "union inference" (data "union_test")
    (Feature_union.transform fitted ~feature_schema ~x:test |> get);
  Alcotest.(check (array string))
    "union names"
    (expected_names "union_names")
    (names (Feature_union.output_schema fitted));
  Alcotest.(check int64)
    "only final union payload reported" 128L
    allocation.Feature_union.output_bytes;
  let _, allocation =
    Feature_union.transform_with_report fitted ~feature_schema ~x:test |> get
  in
  Alcotest.(check int64)
    "inference allocation follows row count" 64L
    allocation.Feature_union.output_bytes;
  let fitted, output =
    Transformer_pipeline.fit_transform (nested_spec ()) ~rng:(rng ())
      ~feature_schema ~x:train ~y:None ()
    |> get
  in
  check_matrix "nested training" (data "nested_train") output;
  check_matrix "nested inference" (data "nested_test")
    (Transformer_pipeline.transform fitted ~feature_schema ~x:test |> get);
  Alcotest.(check (array string))
    "nested feature provenance"
    (expected_names "nested_names")
    (names (Transformer_pipeline.output_schema fitted))

module Probe = struct
  type t = {
    fits : int ref;
    transforms : int ref;
    expected_input : Matrix.t option;
    drop_on : int option;
    draws : int;
  }

  type params = t
  type target = unit
  type rng = Rng.t
  type fitted = Feature_schema.t * t * float

  let clone t = t
  let params t = t

  let fit spec ?sample_weight:_ ~rng ~feature_schema ~x ~y () =
    incr spec.fits;
    Alcotest.(check bool) "unsupervised child gets no targets" true (y = None);
    Option.iter
      (fun expected ->
        Alcotest.(check bool) "union input is shared" true (expected == x))
      spec.expected_input;
    let rec draw count rng =
      let value, rng = Rng.next_float rng in
      if count = 0 then value else draw (count - 1) rng
    in
    Ok (feature_schema, spec, draw spec.draws rng)

  let transform (_, spec, value) ~feature_schema:_ ~x =
    incr spec.transforms;
    let rows = Matrix.rows x in
    let rows = if spec.drop_on = Some rows then rows - 1 else rows in
    Matrix.init ~rows ~columns:(Matrix.columns x) (fun _ _ -> value)
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"preserve row count" error)

  let fitted_params (_, spec, _) = spec
  let input_schema (schema, _, _) = schema
  let output_schema (schema, _, _) = schema
end

let probe ?expected_input ?drop_on ?(draws = 0) name =
  let fits = ref 0 and transforms = ref 0 in
  let stage =
    Pipeline.transformer ~name
      (module Probe)
      Probe.{ fits; transforms; expected_input; drop_on; draws }
    |> get
  in
  (stage, fits, transforms)

let test_single_pass_and_rng () =
  let x = matrix [| [| 1. |]; [| 2. |]; [| 3. |] |] in
  let run draws =
    let first, fits1, transforms1 = probe ~expected_input:x ~draws "first" in
    let second, fits2, transforms2 = probe ~expected_input:x "second" in
    let last, fits3, transforms3 = probe "last" in
    let right = chain "right" [| second; last |] in
    let stage =
      union "features"
        [| Feature_union.transformer first; Feature_union.transformer right |]
    in
    let fitted =
      Pipeline.fit (pipeline stage) ~rng:(rng ()) ~feature_schema:(schema x) ~x
        ~y:(regression [| 1.; 2.; 3. |])
        ()
      |> get
    in
    List.iter
      (fun calls ->
        Alcotest.(check int)
          "one fit or transform per leaf during training" 1 !calls)
      [ fits1; transforms1; fits2; transforms2; fits3; transforms3 ];
    let output =
      Pipeline.transform fitted ~feature_schema:(schema x) ~x |> get
    in
    Artifact.encode_regression fitted |> expect_error;
    Matrix.get output 0 1
  in
  check_float "nested sibling random consumption is isolated" (run 0) (run 100)

let test_empty_and_admission () =
  let x = matrix [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let feature_schema = schema x in
  let empty = Transformer_pipeline.create [||] |> get in
  let fitted, output =
    Transformer_pipeline.fit_transform empty ~rng:(rng ()) ~feature_schema ~x
      ~y:None ()
    |> get
  in
  Alcotest.(check bool) "empty chain shares input" true (output == x);
  Alcotest.(check bool)
    "empty chain preserves schema identity" true
    (Feature_schema.equal feature_schema
       (Transformer_pipeline.output_schema fitted));
  let stage = scale "scale" in
  Transformer_pipeline.create [| stage; stage |] |> expect_error;
  Transformer_pipeline.stage ~name:" " empty |> expect_error;
  let stages = [| stage |] in
  let config = Transformer_pipeline.create stages |> get in
  stages.(0) <- impute "changed";
  let stage_names = Transformer_pipeline.stage_names config in
  stage_names.(0) <- "changed";
  Alcotest.(check (array string))
    "chain admission and inspection are defensive" [| "scale" |]
    (Transformer_pipeline.stage_names config);
  let config_array = [| Feature_union.passthrough ~name:"raw" |> get |] in
  let config = Feature_union.create config_array |> get in
  config_array.(0) <- Feature_union.drop ~name:"gone" |> get;
  let _, output, _ =
    Feature_union.fit_transform
      (Feature_union.clone config)
      ~rng:(rng ()) ~feature_schema ~x ~y:None ()
    |> get
  in
  check_matrix "union branches copied at admission" (Matrix.to_arrays x) output;
  List.iter
    (fun branches ->
      let empty = Feature_union.create ~max_output_features:0 branches |> get in
      let fitted, output, allocation =
        Feature_union.fit_transform empty ~rng:(rng ()) ~feature_schema ~x
          ~y:None ()
        |> get
      in
      Alcotest.(check (pair int int))
        "empty union keeps rows" (2, 0) (Matrix.shape output);
      Alcotest.(check int64)
        "zero output payload" 0L allocation.Feature_union.output_bytes;
      let zero_rows = Matrix.create ~rows:0 ~columns:2 0. |> get_data in
      let output =
        Feature_union.transform fitted ~feature_schema ~x:zero_rows |> get
      in
      Alcotest.(check (pair int int))
        "zero-row inference" (0, 0) (Matrix.shape output))
    [ [||]; [| Feature_union.drop ~name:"drop" |> get |] ];
  let active, fits, _ = probe "active" in
  let zero_columns = Matrix.create ~rows:2 ~columns:0 0. |> get_data in
  Feature_union.fit
    (Feature_union.create [| Feature_union.transformer active |] |> get)
    ~rng:(rng ()) ~feature_schema:(schema zero_columns) ~x:zero_columns ~y:None
    ()
  |> get |> ignore;
  Alcotest.(check int)
    "active empty-input branches decide their own policy" 1 !fits;
  Feature_union.create ~max_output_features:(-1) [||] |> expect_error;
  Feature_union.create ~max_output_features:max_int [||] |> expect_error;
  let duplicate = Feature_union.passthrough ~name:"same" |> get in
  Feature_union.create [| duplicate; duplicate |] |> expect_error;
  Feature_union.passthrough ~name:" " |> expect_error;
  let bounded =
    Feature_union.create ~max_output_features:1
      [| Feature_union.passthrough ~name:"wide" |> get |]
    |> get
  in
  Feature_union.fit bounded ~rng:(rng ()) ~feature_schema ~x ~y:None ()
  |> expect_error

let[@warning "-4"] test_nested_errors () =
  let x = matrix [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let child, fits, _ = probe ~drop_on:1 "bad" in
  let chain_stage = chain "branch" [| child |] in
  let union_stage =
    union "features" [| Feature_union.transformer chain_stage |]
  in
  let spec = pipeline union_stage in
  let fitted =
    Pipeline.fit spec ~rng:(rng ()) ~feature_schema:(schema x) ~x
      ~y:(regression [| 1.; 2. |])
      ()
    |> get
  in
  (match
     Pipeline.transform fitted ~feature_schema:(schema x)
       ~x:(matrix [| [| 1.; 2. |] |])
   with
  | Ok _ -> Alcotest.fail "row-changing child output was accepted"
  | Error error ->
      Alcotest.(check bool)
        "complete nested error path" true
        (Error.context error
        = [ Error.Stage "features"; Error.Stage "branch"; Error.Stage "bad" ]);
      Alcotest.(check bool)
        "typed shape failure" true
        (match Error.kind error with
        | Error.Shape_mismatch _ -> true
        | _ -> false));
  let short = Sample_weight.of_array ~expected_length:1 [| 1. |] |> get_data in
  let before = !fits in
  let union_config =
    Feature_union.create [| Feature_union.transformer chain_stage |] |> get
  in
  Feature_union.fit union_config ~sample_weight:short ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:None ()
  |> expect_error;
  Transformer_pipeline.fit
    (Transformer_pipeline.create [| union_stage |] |> get)
    ~sample_weight:short ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y:None ()
  |> expect_error;
  Alcotest.(check int) "bad weights rejected before nested fits" before !fits;
  let named =
    Feature_names.create ~expected_count:2 [| "a"; "b" |]
    |> get_data |> Feature_schema.named
  in
  let other =
    Feature_names.create ~expected_count:2 [| "b"; "a" |]
    |> get_data |> Feature_schema.named
  in
  let fitted =
    Feature_union.fit union_config ~rng:(rng ()) ~feature_schema:named ~x
      ~y:None ()
    |> get
  in
  Feature_union.transform fitted ~feature_schema:other ~x |> expect_error

let test_fold_local_weights () =
  let x = Array.init 12 (fun row -> [| Float.of_int row |]) |> matrix in
  let y = Array.init 12 Float.of_int |> regression in
  let weights =
    Sample_weight.of_array ~expected_length:12
      (Array.init 12 (fun row -> Float.of_int (row + 1)))
    |> get_data
  in
  let weighted =
    chain "weighted"
      [|
        impute "impute"; scale ~with_std:false ~route_sample_weight:true "scale";
      |]
  in
  let plain =
    chain "plain" [| impute "impute"; scale ~with_std:false "scale" |]
  in
  let stage =
    union "features"
      [| Feature_union.transformer weighted; Feature_union.transformer plain |]
  in
  let specification = pipeline stage in
  let dataset x =
    Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight:weights ~x
      ~y ()
    |> get_data
  in
  let splitter =
    K_fold.create ~folds:3 ~shuffle:true ()
    |> get
    |> Cross_validation.target_independent_splitter (module K_fold)
  in
  let run x =
    Cross_validation.Regression.cross_validate ~return_models:true
      ~return_indices:true ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~seed:(Seed.of_int 7) specification (dataset x)
    |> get
  in
  let report = run x in
  Alcotest.(check int)
    "all nested folds fit" 3
    (Cross_validation.successful_fold_count report);
  let transformed fold =
    Pipeline.transform
      (Option.get fold.Cross_validation.model)
      ~feature_schema:(schema x) ~x
    |> get
  in
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let rows = Option.get fold.Cross_validation.train_indices in
      let sum, numerator, denominator =
        Array.fold_left
          (fun (sum, n, d) row ->
            let value = Matrix.get x row 0
            and weight = Sample_weight.get weights row in
            (sum +. value, n +. (value *. weight), d +. weight))
          (0., 0., 0.) rows
      in
      let output = transformed fold in
      check_float "nested weighted fold mean"
        (-.numerator /. denominator)
        (Matrix.get output 0 0);
      check_float "nested opt-out fold mean"
        (-.sum /. Float.of_int (Array.length rows))
        (Matrix.get output 0 1))
    (Cross_validation.folds report);
  let first = (Cross_validation.folds report).(0) in
  let changed = Matrix.to_arrays x in
  Array.iter
    (fun row -> changed.(row).(0) <- 1e6)
    (Option.get first.Cross_validation.test_indices);
  check_matrix "nested fit cannot observe held-out features"
    (Matrix.to_arrays (transformed first))
    (transformed (Cross_validation.folds (run (matrix changed))).(0));
  let grid =
    Grid_search.create ~base:() ~build:(fun () -> Ok specification) [||] |> get
  in
  let search =
    Grid_search.Regression.search ~grid ~splitter
      ~scorers:[| Regression_scorer.neg_mean_squared_error |]
      ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 7) (dataset x)
    |> get
  in
  let selected = Grid_search.selection search |> get in
  let output =
    Pipeline.transform selected.Grid_search.selected_model
      ~feature_schema:(schema x) ~x
    |> get
  in
  check_float "nested grid refit sees all rows" (-5.5) (Matrix.get output 0 1)

let () =
  Alcotest.run "nested composition"
    [
      ( "contracts",
        [
          Alcotest.test_case "sklearn union and nested pipeline parity" `Quick
            test_parity;
          Alcotest.test_case "single-pass shared inputs and random isolation"
            `Quick test_single_pass_and_rng;
          Alcotest.test_case "empty compositions and immutable admission" `Quick
            test_empty_and_admission;
          Alcotest.test_case "nested error paths and preflight checks" `Quick
            test_nested_errors;
          Alcotest.test_case "fold-local weights, leakage, and grid refit"
            `Quick test_fold_local_weights;
        ] );
    ]
