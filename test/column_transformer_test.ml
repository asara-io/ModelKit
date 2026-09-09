open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix values = Matrix.of_arrays values |> get_data
let regression values = Target.regression (Vector.of_array values) |> get_data
let schema x = Feature_schema.of_matrix x |> get_data
let rng () = Rng.create (Seed.of_int 42)
let indices values = Column_selector.indices values |> get

let named_schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> get_data |> Feature_schema.named

let names schema =
  Feature_schema.names schema |> Option.get |> Feature_names.to_array

let pass name columns =
  Column_transformer.passthrough ~name ~columns:(indices columns) |> get

let drop name columns =
  Column_transformer.drop ~name ~columns:(indices columns) |> get

let scale ?route_sample_weight ?(with_std = true) name columns =
  Pipeline.transformer ?route_sample_weight ~name
    (module Standard_scaler)
    (Standard_scaler.create ~with_std ())
  |> get
  |> Column_transformer.transformer ~columns:(indices columns)

let expect_error = function
  | Ok _ -> Alcotest.fail "expected a typed error"
  | Error _ -> ()

let check_float label expected observed =
  Alcotest.(check bool)
    label true
    (if Float.is_nan expected then Float.is_nan observed
     else
       Float.abs (expected -. observed)
       <= 1e-12 *. Float.max 1. (Float.abs expected))

let check_matrix label expected actual =
  Alcotest.(check int)
    (label ^ " rows") (Array.length expected) (Matrix.rows actual);
  Array.iteri
    (fun row values ->
      Alcotest.(check int)
        (label ^ " columns") (Array.length values) (Matrix.columns actual);
      Array.iteri
        (fun column value ->
          check_float label value (Matrix.get actual row column))
        values)
    expected

let test_selectors () =
  let schema = named_schema [| "height"; "age"; "weight" |] in
  let source = [| 2; 0 |] in
  let selected = indices source in
  source.(0) <- 1;
  let result = Column_selector.resolve selected schema |> get in
  Alcotest.(check (array int))
    "index order and defensive admission" [| 2; 0 |] result;
  result.(0) <- 1;
  Alcotest.(check (array int))
    "defensive resolution" [| 2; 0 |]
    (Column_selector.resolve selected schema |> get);
  let source = [| "weight"; "height" |] in
  let selected = Column_selector.names source |> get in
  source.(0) <- "age";
  Alcotest.(check (array int))
    "name resolution order" [| 2; 0 |]
    (Column_selector.resolve selected schema |> get);
  List.iter expect_error
    [ Column_selector.indices [| -1 |]; Column_selector.indices [| 1; 1 |] ];
  List.iter expect_error
    [ Column_selector.names [| "age"; "age" |]; Column_selector.names [| "" |] ];
  Column_selector.resolve (indices [| 3 |]) schema |> expect_error;
  Column_selector.resolve Column_selector.all
    (Feature_schema.anonymous ~feature_count:max_int |> get_data)
  |> expect_error;
  Column_selector.resolve (Column_selector.names [| "absent" |] |> get) schema
  |> expect_error;
  Column_selector.resolve selected
    (Feature_schema.anonymous ~feature_count:3 |> get_data)
  |> expect_error;
  Alcotest.(check (array int))
    "all columns" [| 0; 1; 2 |]
    (Column_selector.resolve Column_selector.all schema |> get)

let read_fixture () =
  let matrices = Hashtbl.create 8 and names = Hashtbl.create 2 in
  In_channel.with_open_text (Sys.getenv "MODELKIT_COLUMN_FIXTURE") (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if line <> "" && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ name; row; values ] ->
                let values =
                  String.split_on_char ',' values
                  |> List.map float_of_string |> Array.of_list
                in
                let rows =
                  match Hashtbl.find_opt matrices name with
                  | Some rows -> rows
                  | None ->
                      let rows = Hashtbl.create 4 in
                      Hashtbl.add matrices name rows;
                      rows
                in
                Hashtbl.add rows (int_of_string row) values
            | [ name; values ] ->
                Hashtbl.add names name
                  (String.split_on_char ',' values |> Array.of_list)
            | _ -> Alcotest.fail "invalid column fixture row"));
  let matrix name =
    let rows = Hashtbl.find matrices name in
    Array.init (Hashtbl.length rows) (Hashtbl.find rows)
  in
  (matrix, fun name -> Hashtbl.find names name)

let fixture_specs () =
  let impute =
    Pipeline.transformer ~name:"impute"
      (module Simple_imputer)
      (Simple_imputer.mean ())
    |> get
    |> Column_transformer.transformer ~columns:(indices [| 3 |])
  in
  let encode =
    Pipeline.transformer ~name:"encode"
      (module One_hot_encoder)
      (One_hot_encoder.create ~unknown_category:One_hot_encoder.Ignore () |> get)
    |> get
    |> Column_transformer.transformer ~columns:(indices [| 1 |])
  in
  [
    ( "mixed",
      [|
        scale "scale" [| 2; 0 |];
        impute;
        pass "category" [| 1 |];
        drop "discard" [| 4 |];
      |],
      Column_transformer.Passthrough );
    ( "overlap",
      [| scale "scale" [| 0 |]; pass "raw" [| 0; 2 |]; drop "discard" [| 3 |] |],
      Column_transformer.Passthrough );
    ( "encoding",
      [| scale "scale" [| 0 |]; encode; scale "unused" [||] |],
      Column_transformer.Drop );
  ]

let test_parity () =
  let data, expected_names = read_fixture () in
  let train = matrix (data "train") and test = matrix (data "test") in
  List.iter
    (fun (name, branches, remainder) ->
      let spec = Column_transformer.create ~remainder branches |> get in
      let fitted, output, allocation =
        Column_transformer.fit_transform spec ~rng:(rng ())
          ~feature_schema:(schema train) ~x:train ~y:None ()
        |> get
      in
      check_matrix (name ^ " training") (data (name ^ "_train")) output;
      let inference, inference_allocation =
        Column_transformer.transform_with_report fitted
          ~feature_schema:(schema train) ~x:test
        |> get
      in
      check_matrix (name ^ " inference") (data (name ^ "_test")) inference;
      if name <> "encoding" then
        Alcotest.(check (array string))
          "sklearn names"
          (expected_names (name ^ "_names"))
          (names (Column_transformer.output_schema fitted));
      let copied_columns =
        if name = "mixed" then 3 else if name = "overlap" then 1 else 2
      in
      Alcotest.(check int64)
        "selection payload bytes"
        (Int64.of_int (4 * copied_columns * 8))
        allocation.Column_transformer.selected_input_bytes;
      Alcotest.(check int64)
        "final output bytes"
        (Int64.of_int (4 * Matrix.columns output * 8))
        allocation.Column_transformer.output_bytes;
      Alcotest.(check int64)
        "inference uses its own row count"
        (Int64.of_int (2 * copied_columns * 8))
        inference_allocation.Column_transformer.selected_input_bytes;
      let infos = Column_transformer.branches fitted in
      infos.(0).Column_transformer.input_indices.(0) <- 4;
      check_matrix "fitted selectors are immutable"
        (data (name ^ "_test"))
        (Column_transformer.transform fitted ~feature_schema:(schema train)
           ~x:test
        |> get))
    (fixture_specs ())

module Probe = struct
  type t = {
    calls : int ref;
    transforms : int ref;
    drop_at : int option;
    draws : int;
  }

  type params = t
  type target = unit
  type rng = Rng.t
  type fitted = Feature_schema.t * t * float

  let clone t = t
  let params t = t

  let fit specification ?sample_weight:_ ~rng ~feature_schema ~x:_ ~y:_ () =
    incr specification.calls;
    let rec draw count rng =
      let value, rng = Rng.next_float rng in
      if count = 0 then value else draw (count - 1) rng
    in
    Ok (feature_schema, specification, draw specification.draws rng)

  let transform (_, specification, value) ~feature_schema:_ ~x =
    incr specification.transforms;
    let rows = Matrix.rows x in
    let rows = if specification.drop_at = Some rows then rows - 1 else rows in
    Matrix.init ~rows ~columns:(Matrix.columns x) (fun _ _ -> value)
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"preserve rows" error)

  let fitted_params (_, specification, _) = specification
  let input_schema (schema, _, _) = schema
  let output_schema (schema, _, _) = schema
end

let probe ?(drop_at = None) ?(draws = 0) name columns =
  let calls = ref 0 and transforms = ref 0 in
  let stage =
    Pipeline.transformer ~name
      (module Probe)
      Probe.{ calls; transforms; drop_at; draws }
    |> get
  in
  ( Column_transformer.transformer ~columns:(indices columns) stage,
    calls,
    transforms )

let test_empty_and_limits () =
  let x = matrix [| [| Float.nan; Float.infinity |]; [| 2.; 3. |] |] in
  let empty_branch, calls, transforms = probe "empty" [||] in
  let specification =
    Column_transformer.create ~max_output_features:0
      [| empty_branch; drop "discard" [| 0; 1 |] |]
    |> get
  in
  let fitted, output, allocation =
    Column_transformer.fit_transform specification ~rng:(rng ())
      ~feature_schema:(schema x) ~x ~y:None ()
    |> get
  in
  Alcotest.(check (pair int int))
    "all-dropped retains row count" (2, 0) (Matrix.shape output);
  Alcotest.(check int) "empty branch never fits" 0 !calls;
  Alcotest.(check int) "empty branch never transforms" 0 !transforms;
  Alcotest.(check int64)
    "empty output has no payload" 0L allocation.Column_transformer.output_bytes;
  let empty_x = Matrix.create ~rows:0 ~columns:2 0. |> get_data in
  let empty_output =
    Column_transformer.transform fitted ~feature_schema:(schema x) ~x:empty_x
    |> get
  in
  Alcotest.(check (pair int int))
    "zero rows and zero output" (0, 0)
    (Matrix.shape empty_output);
  let all =
    Column_transformer.create ~remainder:Column_transformer.Passthrough [||]
    |> get
  in
  let _, output, allocation =
    Column_transformer.fit_transform all ~rng:(rng ())
      ~feature_schema:(schema x) ~x ~y:None ()
    |> get
  in
  Alcotest.(check bool)
    "passthrough preserves non-finite values" true
    (Float.is_nan (Matrix.get output 0 0)
    && Matrix.get output 0 1 = Float.infinity);
  Alcotest.(check int64)
    "passthrough has no selected intermediate" 0L
    allocation.Column_transformer.selected_input_bytes;
  let too_wide =
    Column_transformer.create ~max_output_features:1 [| pass "all" [| 0; 1 |] |]
    |> get
  in
  Column_transformer.fit too_wide ~rng:(rng ()) ~feature_schema:(schema x) ~x
    ~y:None ()
  |> expect_error;
  Column_transformer.create ~max_output_features:(-1) [||] |> expect_error;
  Column_transformer.create ~max_output_features:max_int [||] |> expect_error;
  Column_transformer.create [| pass "same" [| 0 |]; pass "same" [| 1 |] |]
  |> expect_error;
  Column_transformer.passthrough ~name:"remainder" ~columns:Column_selector.all
  |> expect_error

let[@warning "-4"] test_early_validation_and_context () =
  let x = matrix [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let branch, calls, _ = probe "first" [| 0 |] in
  let bad =
    Column_transformer.create [| branch; pass "invalid" [| 2 |] |] |> get
  in
  (match
     Column_transformer.fit bad ~rng:(rng ()) ~feature_schema:(schema x) ~x
       ~y:None ()
   with
  | Ok _ -> Alcotest.fail "invalid selector was accepted"
  | Error error ->
      Alcotest.(check bool)
        "selector branch context" true
        (Error.context error = [ Error.Stage "invalid" ]));
  Alcotest.(check int) "all selectors checked before any fitting" 0 !calls;
  let valid = Column_transformer.create [| branch |] |> get in
  let short_weights =
    Sample_weight.of_array ~expected_length:1 [| 1. |] |> get_data
  in
  Column_transformer.fit valid ~sample_weight:short_weights ~rng:(rng ())
    ~feature_schema:(schema x) ~x ~y:None ()
  |> expect_error;
  Alcotest.(check int) "weights checked before any fitting" 0 !calls;
  let stage, _, _ = probe ~drop_at:(Some 1) "bad-rows" [| 0 |] in
  let spec = Column_transformer.create [| stage |] |> get in
  let fitted =
    Column_transformer.fit spec ~rng:(rng ()) ~feature_schema:(schema x) ~x
      ~y:None ()
    |> get
  in
  (match
     Column_transformer.transform fitted ~feature_schema:(schema x)
       ~x:(matrix [| [| 1.; 2. |] |])
   with
  | Ok _ -> Alcotest.fail "row-changing output was accepted"
  | Error error ->
      Alcotest.(check bool)
        "shape error" true
        (match Error.kind error with
        | Error.Shape_mismatch _ -> true
        | _ -> false);
      Alcotest.(check bool)
        "inference branch context" true
        (Error.context error = [ Error.Stage "bad-rows" ]));
  let schema_a = named_schema [| "a"; "b" |] in
  let fitted =
    Column_transformer.fit spec ~rng:(rng ()) ~feature_schema:schema_a ~x
      ~y:None ()
    |> get
  in
  Column_transformer.transform fitted
    ~feature_schema:(named_schema [| "b"; "a" |])
    ~x
  |> expect_error;
  let colliding =
    Column_transformer.create [| pass "a" [| 0 |]; pass "a__b" [| 1 |] |] |> get
  in
  Column_transformer.fit colliding ~rng:(rng ())
    ~feature_schema:(named_schema [| "b__c"; "c" |])
    ~x ~y:None ()
  |> expect_error

let test_pipeline_and_random_streams () =
  let x = matrix [| [| 1.; 2. |]; [| 3.; 4. |]; [| 5.; 6. |] |] in
  let y = regression [| 1.; 2.; 3. |] in
  let run draws =
    let first, _, _ = probe ~draws "first" [| 0 |] in
    let second, calls, transforms = probe "second" [| 1 |] in
    let config = [| first; second |] in
    let spec = Column_transformer.create config |> get in
    config.(0) <- drop "replacement" [||];
    let stage =
      Column_transformer.stage ~name:"columns" (Column_transformer.clone spec)
      |> get
    in
    let pipeline =
      Pipeline.add_transformer Pipeline.empty stage |> get |> fun builder ->
      Pipeline.set_estimator builder
        (Pipeline.estimator ~name:"model"
           (module Ridge_regression)
           (Ridge_regression.create () |> get)
        |> get)
      |> get
    in
    let fitted =
      Pipeline.fit pipeline ~rng:(rng ()) ~feature_schema:(schema x) ~x ~y ()
      |> get
    in
    Alcotest.(check int) "one fit per selected branch" 1 !calls;
    Alcotest.(check int) "training output is reused" 1 !transforms;
    let output =
      Pipeline.transform fitted ~feature_schema:(schema x) ~x |> get
    in
    Alcotest.(check int)
      "both admitted branches remain" 2 (Matrix.columns output);
    Artifact.encode_regression fitted |> expect_error;
    Matrix.get output 0 1
  in
  check_float "sibling RNG consumption is isolated" (run 0) (run 100)

let test_weighted_cross_validation () =
  let x = Array.init 12 (fun row -> [| Float.of_int row |]) |> matrix in
  let y = Array.init 12 Float.of_int |> regression in
  let weights =
    Sample_weight.of_array ~expected_length:12
      (Array.init 12 (fun row -> Float.of_int (row + 1)))
    |> get_data
  in
  let spec =
    Column_transformer.create
      [|
        scale ~with_std:false ~route_sample_weight:true "weighted" [| 0 |];
        scale ~with_std:false "plain" [| 0 |];
      |]
    |> get
  in
  let pipeline =
    Pipeline.add_transformer Pipeline.empty
      (Column_transformer.stage ~name:"columns" spec |> get)
    |> get
    |> fun builder ->
    Pipeline.set_estimator builder
      (Pipeline.estimator ~name:"ridge"
         (module Ridge_regression)
         (Ridge_regression.create () |> get)
      |> get)
    |> get
  in
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
      ~seed:(Seed.of_int 9) pipeline (dataset x)
    |> get
  in
  let report = run x in
  Alcotest.(check int)
    "all folds succeed" 3
    (Cross_validation.successful_fold_count report);
  Array.iter
    (fun (fold : _ Cross_validation.fold) ->
      let rows = Option.get fold.Cross_validation.train_indices in
      let sum, numerator, denominator =
        Array.fold_left
          (fun (s, n, d) row ->
            let value = Matrix.get x row 0
            and weight = Sample_weight.get weights row in
            (s +. value, n +. (value *. weight), d +. weight))
          (0., 0., 0.) rows
      in
      let transformed =
        Pipeline.transform
          (Option.get fold.Cross_validation.model)
          ~feature_schema:(schema x) ~x
        |> get
      in
      check_float "fold-local weighted mean"
        (-.numerator /. denominator)
        (Matrix.get transformed 0 0);
      check_float "fold-local unweighted mean"
        (-.sum /. Float.of_int (Array.length rows))
        (Matrix.get transformed 0 1))
    (Cross_validation.folds report);
  let first = (Cross_validation.folds report).(0) in
  let altered = Matrix.to_arrays x in
  Array.iter
    (fun row -> altered.(row).(0) <- 1e6)
    (Option.get first.Cross_validation.test_indices);
  let changed = (Cross_validation.folds (run (matrix altered))).(0) in
  let transformed fold =
    Pipeline.transform
      (Option.get fold.Cross_validation.model)
      ~feature_schema:(schema x) ~x
    |> get
  in
  check_matrix "held-out values cannot affect fitted column branches"
    (Matrix.to_arrays (transformed first))
    (transformed changed)

let () =
  Alcotest.run "column transformation"
    [
      ( "contracts",
        [
          Alcotest.test_case "checked immutable selectors" `Quick test_selectors;
          Alcotest.test_case "sklearn parity and allocation reports" `Quick
            test_parity;
          Alcotest.test_case "empty selections, passthrough, and limits" `Quick
            test_empty_and_limits;
          Alcotest.test_case "preflight, schemas, and child error context"
            `Quick test_early_validation_and_context;
          Alcotest.test_case "single training pass and independent RNGs" `Quick
            test_pipeline_and_random_streams;
          Alcotest.test_case "weighted fold fitting and leakage" `Quick
            test_weighted_cross_validation;
        ] );
    ]
