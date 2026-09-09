open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let ( let* ) = Result.bind
let matrix values = Matrix.of_arrays values |> data
let target values = Target.regression (Vector.of_array values) |> data
let rng () = Rng.create (Seed.of_int 42)

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values |> data

let groups values =
  Groups.create ~expected_length:(Array.length values) values |> data

let x = matrix [| [| 1.; 11. |]; [| 2.; 12. |]; [| 3.; 13. |] |]
let schema = Feature_schema.of_matrix x |> data
let y = target [| 10.; 20.; 30. |]

let metadata =
  Metadata.create
    ~sample_weight:(weights [| 2.; 3.; 4. |])
    ~groups:(groups [| 101; 102; 103 |])
    ()

let both =
  Metadata.Request.create ~sample_weight:Metadata.Request.Required
    ~groups:Metadata.Request.Required ()

let neither = Metadata.Request.none

let[@warning "-4"] error_path expected = function
  | Ok _ -> Alcotest.fail "expected metadata error"
  | Error error ->
      Alcotest.(check bool)
        "typed validation" true
        (match Error.kind error with
        | Error.Validation _ | Error.Data (Data_error.Length_mismatch _) -> true
        | _ -> false);
      Alcotest.(check bool)
        "nested context" true
        (Error.context error = List.map (fun name -> Error.Stage name) expected)

module Consumer = struct
  type t = {
    fit_policy : Metadata.Request.t;
    transform_policy : Metadata.Request.t;
    fit_observer :
      Metadata.t -> target:Target.regression Target.t option -> Matrix.t -> unit;
    transform_observer : Metadata.t -> Matrix.t -> unit;
  }

  type params = t
  type target = Target.regression Target.t
  type rng = Rng.t
  type fitted = { specification : t; schema : Feature_schema.t }

  let clone t = t
  let params t = t
  let fit_request t = t.fit_policy
  let transform_request t = t.transform_policy

  let fit specification ~metadata ~rng:_ ~feature_schema ~x ~y () =
    specification.fit_observer metadata ~target:y x;
    Ok { specification; schema = feature_schema }

  let transform fitted ~metadata ~feature_schema:_ ~x =
    fitted.specification.transform_observer metadata x;
    let groups = Metadata.groups metadata in
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
      (fun row column ->
        Matrix.get x row column
        +.
        match groups with
        | None -> 0.
        | Some groups -> Float.of_int (Groups.get groups row))
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"preserve valid dimensions" error)

  let fitted_params fitted = fitted.specification
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
end

module Unsupervised = struct
  include Consumer

  type target = unit

  let fit specification ~metadata ~rng ~feature_schema ~x ~y () =
    Alcotest.(check bool)
      "unsupervised metadata stage receives no target" true (Option.is_none y);
    Consumer.fit specification ~metadata ~rng ~feature_schema ~x ~y:None ()
end

module Terminal = struct
  type t = Metadata.Request.t * (Metadata.t -> unit)
  type params = t
  type target = Target.regression Target.t
  type prediction = target
  type fitted = Feature_schema.t * t
  type rng = Rng.t

  let clone t = t
  let params t = t
  let fit_request (request, _) = request

  let fit ((_, observe) as specification) ~metadata ~rng:_ ~feature_schema ~x:_
      ~y:_ () =
    observe metadata;
    Ok (feature_schema, specification)

  let predict _ ~feature_schema:_ ~x =
    Ok (target (Array.init (Matrix.rows x) (fun row -> Matrix.get x row 0)))

  let feature_schema (schema, _) = schema
  let fitted_params (_, specification) = specification

  let decision_function fitted ~feature_schema ~x =
    let* result = predict fitted ~feature_schema ~x in
    Ok (Target.regression_values result)

  let predict_proba _ ~feature_schema:_ ~x = Ok x
end

let specification ?(fit = neither) ?(transform = neither)
    ?(on_fit = fun _ ~target:_ _ -> ()) ?(on_transform = fun _ _ -> ()) () =
  {
    Consumer.fit_policy = fit;
    transform_policy = transform;
    fit_observer = on_fit;
    transform_observer = on_transform;
  }

let stage name specification =
  Pipeline.Supervised.metadata_transformer ~name (module Consumer) specification
  |> get

let unsupervised name specification =
  Pipeline.metadata_transformer ~name (module Unsupervised) specification |> get

let terminal ?(request = neither) ?(observe = fun _ -> ()) () =
  Pipeline.metadata_estimator ~name:"terminal"
    (module Terminal)
    ~decision_function:Terminal.decision_function
    ~predict_proba:Terminal.predict_proba (request, observe)
  |> get

let pipeline ?terminal:(last = terminal ()) stages =
  let builder =
    Array.fold_left
      (fun builder stage ->
        Pipeline.Supervised.add_transformer builder stage |> get)
      Pipeline.Supervised.empty stages
  in
  Pipeline.Supervised.set_estimator builder last |> get

let nested stage =
  let chain =
    Transformer_pipeline.Supervised.create [| stage |]
    |> get
    |> Transformer_pipeline.Supervised.stage ~name:"chain"
    |> get
  in
  let union =
    Feature_union.Supervised.create
      [| Feature_union.Supervised.transformer chain |]
    |> get
    |> Feature_union.Supervised.stage ~name:"union"
    |> get
  in
  Column_transformer.Supervised.create
    [|
      Column_transformer.Supervised.transformer
        ~columns:(Column_selector.indices [| 1; 0 |] |> get)
        union;
    |]
  |> get
  |> Column_transformer.Supervised.stage ~name:"columns"
  |> get

let fit ?(metadata = metadata) specification =
  Pipeline.fit_with_metadata specification ~metadata ~rng:(rng ())
    ~feature_schema:schema ~x ~y ()

let check_present weights groups metadata =
  Alcotest.(check bool)
    "weights presence" weights
    (Option.is_some (Metadata.sample_weight metadata));
  Alcotest.(check bool)
    "groups presence" groups
    (Option.is_some (Metadata.groups metadata))

let test_requests () =
  List.iter
    (fun policy ->
      let request =
        Metadata.Request.create ~sample_weight:policy ~groups:policy ()
      in
      List.iter
        (fun supplied ->
          let result =
            Metadata.route request
              (if supplied then metadata else Metadata.empty)
          in
          match (policy, supplied) with
          | Metadata.Request.Required, false | Metadata.Request.Reject, true ->
              error_path [] result
          | Metadata.Request.Ignore, _ | Metadata.Request.Reject, false ->
              check_present false false (get result)
          | Metadata.Request.Optional, _ | Metadata.Request.Required, true ->
              check_present supplied supplied (get result))
        [ false; true ])
    [
      Metadata.Request.Ignore;
      Metadata.Request.Optional;
      Metadata.Request.Required;
      Metadata.Request.Reject;
    ];
  let delivered = Metadata.route both metadata |> get in
  Alcotest.(check bool)
    "weights are shared" true
    (Option.get (Metadata.sample_weight delivered)
    == Option.get (Metadata.sample_weight metadata));
  Alcotest.(check bool)
    "groups are shared" true
    (Option.get (Metadata.groups delivered)
    == Option.get (Metadata.groups metadata))

let test_nested_delivery () =
  let calls = ref [] in
  let consumer =
    specification ~fit:both
      ~transform:(Metadata.Request.create ~groups:Metadata.Request.Required ())
      ~on_fit:(fun metadata ~target:y selected ->
        calls := "fit" :: !calls;
        check_present true true metadata;
        let labels = Target.regression_values (Option.get y) in
        for row = 0 to Matrix.rows selected - 1 do
          let id = Matrix.get selected row 1 in
          Alcotest.check (Alcotest.float 0.) "target alignment" (id *. 10.)
            (Vector.get labels row);
          Alcotest.check (Alcotest.float 0.) "weight alignment" (id +. 1.)
            (Sample_weight.get
               (Option.get (Metadata.sample_weight metadata))
               row);
          Alcotest.(check int)
            "group alignment"
            (100 + int_of_float id)
            (Groups.get (Option.get (Metadata.groups metadata)) row)
        done)
      ~on_transform:(fun metadata _ ->
        calls := "transform" :: !calls;
        check_present false true metadata)
      ()
  in
  let terminal_calls = ref 0 in
  let last =
    terminal
      ~request:
        (Metadata.Request.create ~sample_weight:Metadata.Request.Required ())
      ~observe:(fun delivered ->
        incr terminal_calls;
        check_present true false delivered)
      ()
  in
  let fitted =
    fit (pipeline ~terminal:last [| nested (stage "consumer" consumer) |])
    |> get
  in
  Alcotest.(check (list string))
    "single training transform" [ "transform"; "fit" ] !calls;
  Alcotest.(check int) "terminal fit once" 1 !terminal_calls;
  let inference_x = matrix [| [| 7.; 17. |]; [| 8.; 18. |] |] in
  let inference_metadata = Metadata.create ~groups:(groups [| 200; 300 |]) () in
  let prediction =
    Pipeline.predict_with_metadata fitted ~metadata:inference_metadata
      ~feature_schema:schema ~x:inference_x
    |> get
  in
  Alcotest.(check (array (Alcotest.float 0.)))
    "fresh inference groups" [| 217.; 318. |]
    (Target.regression_values prediction |> Vector.to_array);
  let decision =
    Pipeline.decision_function_with_metadata fitted ~metadata:inference_metadata
      ~feature_schema:schema ~x:inference_x
    |> get
  in
  Alcotest.(check (array (Alcotest.float 0.)))
    "decision dispatch uses metadata" [| 217.; 318. |]
    (Vector.to_array decision);
  let proba =
    Pipeline.predict_proba_with_metadata fitted ~metadata:inference_metadata
      ~feature_schema:schema ~x:inference_x
    |> get
  in
  Alcotest.check (Alcotest.float 0.) "probability dispatch uses metadata" 217.
    (Matrix.get proba 0 0);
  calls := [];
  Pipeline.predict fitted ~feature_schema:schema ~x:inference_x
  |> error_path [ "columns"; "union"; "chain"; "consumer" ];
  Alcotest.(check (list string))
    "inference cannot reuse training metadata" [] !calls

let test_preflight () =
  let calls = ref 0 in
  let observe =
    specification
      ~on_fit:(fun _ ~target:_ _ -> incr calls)
      ~on_transform:(fun _ _ -> incr calls)
      ()
  in
  let required = specification ~transform:both () in
  let workflow =
    pipeline [| stage "first" observe; nested (stage "required" required) |]
  in
  fit ~metadata:Metadata.empty workflow
  |> error_path [ "columns"; "union"; "chain"; "required" ];
  Alcotest.(check int)
    "training-transform requests preflight before fit" 0 !calls;
  let malformed = Metadata.create ~groups:(groups [| 1 |]) () in
  fit ~metadata:malformed (pipeline [| stage "ignored" observe |])
  |> error_path [];
  Alcotest.(check int)
    "ignored but misaligned fields rejected before fit" 0 !calls;
  let fitted = fit workflow |> get in
  calls := 0;
  Pipeline.transform_with_metadata fitted ~metadata:Metadata.empty
    ~feature_schema:schema ~x
  |> error_path [ "columns"; "union"; "chain"; "required" ];
  Alcotest.(check int) "inference preflight before any transforms" 0 !calls;
  Pipeline.transform_with_metadata fitted ~metadata:malformed
    ~feature_schema:schema ~x
  |> error_path [];
  Alcotest.(check int) "inference alignment before any transforms" 0 !calls;
  fit ~metadata:Metadata.empty
    (pipeline ~terminal:(terminal ~request:both ()) [| stage "first" observe |])
  |> error_path [ "terminal" ];
  Alcotest.(check int)
    "terminal request preflight before preprocessing" 0 !calls;
  let reject = Metadata.Request.create ~groups:Metadata.Request.Reject () in
  fit
    (pipeline
       [|
         stage "first" observe;
         nested (stage "reject" (specification ~fit:reject ()));
       |])
  |> error_path [ "columns"; "union"; "chain"; "reject" ];
  Alcotest.(check int) "rejected metadata preflight" 0 !calls

let test_direct_composition () =
  let first =
    unsupervised "uses-groups"
      (specification ~fit:both
         ~transform:
           (Metadata.Request.create ~groups:Metadata.Request.Required ())
         ())
  in
  let second =
    unsupervised "ignores-groups"
      (specification
         ~on_fit:(fun metadata ~target:_ _ ->
           check_present false false metadata)
         ~on_transform:(fun metadata _ -> check_present false false metadata)
         ())
  in
  let union =
    Feature_union.create
      [| Feature_union.transformer first; Feature_union.transformer second |]
    |> get
  in
  let columns =
    Column_transformer.create
      [|
        Column_transformer.transformer ~columns:Column_selector.all
          (Feature_union.stage ~name:"union" union |> get);
      |]
    |> get
  in
  let chain =
    Transformer_pipeline.create
      [| Column_transformer.stage ~name:"columns" columns |> get |]
    |> get
  in
  let fitted, output =
    Transformer_pipeline.fit_transform_with_metadata chain ~metadata
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  Alcotest.check (Alcotest.float 0.) "requesting branch" 102.
    (Matrix.get output 0 0);
  Alcotest.check (Alcotest.float 0.) "ignored field leaves sibling unchanged" 1.
    (Matrix.get output 0 2);
  Transformer_pipeline.transform fitted ~feature_schema:schema ~x
  |> error_path [ "columns"; "union"; "uses-groups" ];
  let transformed =
    Transformer_pipeline.transform_with_metadata fitted ~metadata
      ~feature_schema:schema ~x
    |> get
  in
  Alcotest.(check bool)
    "direct inference receives metadata" true
    (Matrix.to_arrays output = Matrix.to_arrays transformed)

let test_retention_and_legacy () =
  let weak = Weak.create 1 in
  let fit_ephemeral () =
    let transient = groups [| 4; 5; 6 |] in
    Weak.set weak 0 (Some transient);
    let metadata = Metadata.create ~groups:transient () in
    let request =
      Metadata.Request.create ~groups:Metadata.Request.Required ()
    in
    fit ~metadata
      (pipeline [| nested (stage "fit-only" (specification ~fit:request ())) |])
    |> get
  in
  let fitted = fit_ephemeral () in
  Gc.full_major ();
  Alcotest.(check bool)
    "fitted composition does not retain training groups" false
    (Weak.check weak 0);
  ignore (Pipeline.predict fitted ~feature_schema:schema ~x |> get);
  let scaler =
    Artifact.standard_scaler_stage ~route_sample_weight:true ~name:"scale"
      (Standard_scaler.create ())
    |> get
  in
  let last =
    Artifact.linear_regression_estimator ~name:"linear"
      (Linear_regression.create ())
    |> get
  in
  let builder = Pipeline.add_transformer Pipeline.empty scaler |> get in
  let specification = Pipeline.set_estimator builder last |> get in
  let legacy =
    Pipeline.fit specification
      ?sample_weight:(Metadata.sample_weight metadata)
      ~rng:(rng ()) ~feature_schema:schema ~x ~y ()
    |> get
  in
  let routed = fit specification |> get in
  Alcotest.(check bytes)
    "ignored groups preserve legacy artifact bytes"
    (Artifact.encode_regression legacy |> get)
    (Artifact.encode_regression routed |> get);
  let loaded =
    Artifact.decode_regression (Artifact.encode_regression routed |> get) |> get
  in
  let prediction =
    Pipeline.predict_with_metadata (Artifact.model loaded) ~metadata
      ~feature_schema:schema ~x
    |> get
  in
  let expected = Pipeline.predict legacy ~feature_schema:schema ~x |> get in
  Alcotest.(check (array (Alcotest.float 0.)))
    "decoded legacy transformer accepts explicit metadata"
    (Target.regression_values expected |> Vector.to_array)
    (Target.regression_values prediction |> Vector.to_array)

let test_empty_and_alignment () =
  Metadata.validate ~rows:(-1) Metadata.empty |> error_path [];
  let wrong = Metadata.create ~sample_weight:(weights [| 1. |]) () in
  fit ~metadata:wrong (pipeline [||]) |> error_path [];
  let required = unsupervised "required" (specification ~transform:both ()) in
  let columns =
    Column_transformer.create
      [|
        Column_transformer.transformer
          ~columns:(Column_selector.indices [||] |> get)
          required;
      |]
    |> get
  in
  Column_transformer.fit_transform columns ~rng:(rng ()) ~feature_schema:schema
    ~x ~y:None ()
  |> error_path [ "required" ];
  let fitted, output, allocation =
    Column_transformer.fit_transform_with_metadata columns ~metadata
      ~rng:(rng ()) ~feature_schema:schema ~x ~y:None ()
    |> get
  in
  Alcotest.(check int) "empty selected output" 0 (Matrix.columns output);
  Alcotest.(check int64)
    "no selection allocation" 0L
    allocation.Column_transformer.selected_input_bytes;
  Column_transformer.transform fitted ~feature_schema:schema ~x
  |> error_path [ "required" ];
  let _, report =
    Column_transformer.transform_with_report_with_metadata fitted ~metadata
      ~feature_schema:schema ~x
    |> get
  in
  Alcotest.(check int64)
    "no inference allocation" 0L report.Column_transformer.output_bytes

let () =
  Alcotest.run "metadata routing"
    [
      ( "contracts",
        [
          Alcotest.test_case "training metadata lifetime and legacy codecs"
            `Quick test_retention_and_legacy;
          Alcotest.test_case "empty selections and metadata shape validation"
            `Quick test_empty_and_alignment;
          Alcotest.test_case "request policies" `Quick test_requests;
          Alcotest.test_case "nested alignment and fresh inference delivery"
            `Quick test_nested_delivery;
          Alcotest.test_case "preflight fit, transform, and terminal requests"
            `Quick test_preflight;
          Alcotest.test_case "direct composition and sibling isolation" `Quick
            test_direct_composition;
        ] );
    ]
