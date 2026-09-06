open Modelkit
module Callback = Modelkit.Callback

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let ( let* ) = Result.bind
let regression values = Target.regression (Vector.of_array values) |> data
let fits = Atomic.make 0
let transforms = Atomic.make 0

let invalid reason =
  Error
    (Error.make ~remediation:"supply aligned metadata"
       (Error.Validation { name = "test consumer"; reason }))

let report metadata =
  match Metadata.callback metadata with
  | None -> Ok ()
  | Some callback -> Callback.progress callback ~completed:1 ~total:1 ()

let dataset () =
  let x =
    Matrix.of_arrays
      (Array.init 16 (fun row ->
           [| Float.of_int row; Float.of_int (row + 100) |]))
    |> data
  in
  let y =
    regression (Array.init 16 (fun row -> Float.of_int ((3 * row) + 5)))
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:16
      (Array.init 16 (fun row -> Float.of_int (row + 1)))
    |> data
  in
  let groups =
    Groups.create ~expected_length:16
      (Array.init 16 (fun row -> 100 + (row / 2)))
    |> data
  in
  Dataset.create ~finiteness:Dataset.Require_finite ~sample_weight ~groups ~x ~y
    ()
  |> data

module Consumer = struct
  type t = { align : bool; offset : float; callbacks : Metadata.Request.policy }
  type params = t
  type target = Target.regression Target.t

  type fitted = {
    specification : t;
    input : Feature_schema.t;
    output : Feature_schema.t;
    mean : float;
  }

  type rng = Rng.t

  let clone t = t
  let params t = t

  let fit_request t =
    Metadata.Request.create ~sample_weight:Metadata.Request.Required
      ~groups:Metadata.Request.Required ~callback:t.callbacks ()

  let transform_request = fit_request

  let[@warning "-4"] fit specification ~metadata ~rng:_ ~feature_schema ~x ~y ()
      =
    Atomic.incr fits;
    let* () = report metadata in
    match (Metadata.sample_weight metadata, Metadata.groups metadata, y) with
    | Some weights, Some groups, Some y ->
        let labels = Target.regression_values y in
        let numerator = ref 0.
        and denominator = ref 0.
        and aligned = ref true in
        for row = 0 to Matrix.rows x - 1 do
          let id = Matrix.get x row 1 in
          let weight = Sample_weight.get weights row in
          if specification.align then
            aligned :=
              !aligned
              && weight = id +. 1.
              && Groups.get groups row = 100 + (int_of_float id / 2)
              && Vector.get labels row = (3. *. id) +. 5.;
          numerator := !numerator +. (weight *. Vector.get labels row);
          denominator := !denominator +. weight
        done;
        if not !aligned then
          invalid "training metadata was not selected with rows"
        else
          Ok
            {
              specification;
              input = feature_schema;
              output = Feature_schema.anonymous ~feature_count:1 |> data;
              mean = (!numerator /. !denominator) +. specification.offset;
            }
    | _ -> invalid "missing fit metadata or targets"

  let transform fitted ~metadata ~feature_schema:_ ~x =
    Atomic.incr transforms;
    let* () = report metadata in
    match (Metadata.sample_weight metadata, Metadata.groups metadata) with
    | Some weights, Some groups ->
        Matrix.init ~rows:(Matrix.rows x) ~columns:1 (fun row _ ->
            fitted.mean
            +. (0.01 *. Float.of_int (Groups.get groups row))
            +. (0.001 *. Sample_weight.get weights row))
        |> Result.map_error (fun error ->
            Error.of_data_error ~remediation:"preserve row count" error)
    | _ -> invalid "missing inference metadata"

  let fitted_params fitted = fitted.specification
  let input_schema fitted = fitted.input
  let output_schema fitted = fitted.output
end

module Terminal = struct
  type t = unit
  type params = unit
  type target = Target.regression Target.t
  type prediction = target
  type fitted = Feature_schema.t
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit_request () =
    Metadata.Request.create ~callback:Metadata.Request.Optional ()

  let fit () ~metadata ~rng:_ ~feature_schema ~x:_ ~y:_ () =
    let* () = report metadata in
    Ok feature_schema

  let predict _ ~feature_schema:_ ~x =
    Ok (regression (Array.init (Matrix.rows x) (fun row -> Matrix.get x row 0)))

  let fitted_params _ = ()
  let feature_schema fitted = fitted
end

let nest stage =
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

let pipeline ?(align = true) ?(offset = 0.)
    ?(callbacks = Metadata.Request.Optional) () =
  let stage =
    Pipeline.Supervised.metadata_transformer ~name:"consumer"
      (module Consumer)
      Consumer.{ align; offset; callbacks }
    |> get |> nest
  in
  let builder =
    Pipeline.Supervised.add_transformer Pipeline.Supervised.empty stage |> get
  in
  let terminal =
    Pipeline.metadata_estimator ~name:"terminal" (module Terminal) () |> get
  in
  Pipeline.Supervised.set_estimator builder terminal |> get

let splitter () =
  Group_k_fold.create ~folds:4 ()
  |> get
  |> Cross_validation.target_independent_splitter (module Group_k_fold)

let run ?metadata ?(execution = Execution.sequential)
    ?(failure_policy = Cross_validation.Record) specification source =
  Cross_validation.Regression.cross_validate ?metadata ~execution
    ~failure_policy ~return_models:true ~return_indices:true
    ~return_train_score:true ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~seed:(Seed.of_int 42) specification source

let predictions metadata source model =
  Pipeline.predict_with_metadata model ~metadata
    ~feature_schema:(Dataset.feature_schema source)
    ~x:(Dataset.features source)
  |> get |> Target.regression_values |> Vector.to_array

let mean source rows =
  let weights = Dataset.sample_weight source |> Option.get in
  let targets = Target.regression_values (Dataset.target source) in
  let numerator, denominator =
    Array.fold_left
      (fun (n, d) row ->
        let w = Sample_weight.get weights row in
        (n +. (w *. Vector.get targets row), d +. w))
      (0., 0.) rows
  in
  numerator /. denominator

let expected source metadata rows =
  let mean = mean source rows in
  Array.init (Dataset.sample_count source) (fun row ->
      mean
      +. 0.01
         *. Float.of_int
              (Groups.get (Option.get (Metadata.groups metadata)) row)
      +. 0.001
         *. Sample_weight.get (Option.get (Metadata.sample_weight metadata)) row)

let grid () =
  Grid_search.create ~base:0.
    ~build:(fun offset -> Ok (pipeline ~offset ()))
    [|
      Grid_search.axis ~name:"offset" ~values:[| 0.; 2. |]
        ~encode:(fun value -> Grid_search.Float value)
        ~set:(fun _ value -> Ok value)
      |> get;
    |]
  |> get

let search ?metadata ?(execution = Execution.sequential) source =
  Grid_search.Regression.search ?metadata ~execution ~grid:(grid ())
    ~splitter:(splitter ())
    ~scorers:[| Regression_scorer.neg_mean_squared_error |]
    ~refit:"neg_mean_squared_error" ~seed:(Seed.of_int 42) source

let score fold =
  match fold.Cross_validation.scores.(0).Cross_validation.test_score with
  | Some (Ok score) -> score
  | Some (Error error) -> Alcotest.fail (Error.to_string error)
  | None -> Alcotest.fail "missing score"
