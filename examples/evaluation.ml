open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> failwith (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> failwith (Error.to_string error)

let () =
  let x =
    Matrix.of_arrays
      [|
        [| -3.0; 1.0 |];
        [| -2.0; 2.0 |];
        [| -1.0; 1.5 |];
        [| 1.0; 3.0 |];
        [| 2.0; 2.5 |];
        [| 3.0; 4.0 |];
      |]
    |> get_data
  in
  let y = Target.classification [| 0; 0; 0; 1; 1; 1 |] in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let scale =
    Artifact.standard_scaler_stage ~name:"scale" (Standard_scaler.create ())
    |> get
  in
  let logistic = Logistic_regression.create ~c:2.0 () |> get in
  let terminal =
    Artifact.logistic_regression_estimator ~name:"logistic" logistic |> get
  in
  let pipeline =
    Pipeline.add_transformer Pipeline.empty scale |> fun result ->
    Result.bind result (fun builder -> Pipeline.set_estimator builder terminal)
    |> get
  in
  let fitted =
    Pipeline.fit pipeline
      ~rng:(Rng.create (Seed.of_int 42))
      ~feature_schema:(Dataset.feature_schema dataset)
      ~x:(Dataset.features dataset) ~y:(Dataset.target dataset) ()
    |> get
  in
  let artifact = Artifact.encode_binary_classification fitted |> get in
  let restored =
    Artifact.decode_binary_classification artifact |> get |> Artifact.model
  in
  let predictions =
    Pipeline.predict restored
      ~feature_schema:(Dataset.feature_schema dataset)
      ~x:(Dataset.features dataset)
    |> get
  in
  let accuracy =
    Binary_classification_metrics.accuracy ~truth:y ~prediction:predictions ()
    |> get
  in
  Printf.printf "training accuracy: %.3f\n" accuracy
