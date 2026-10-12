open Modelkit

let ( let* ) = Result.bind

let data result =
  Result.map_error
    (Error.of_data_error ~context:[]
       ~remediation:"supply valid codec fixture data")
    result

let failure reason =
  Error
    (Error.make ~remediation:"use a fixture-produced payload"
       (Error.Artifact { operation = "fixture"; reason }))

module Mean = struct
  type t = float
  type params = float
  type fitted = { offset : float; mean : float; schema : Feature_schema.t }
  type target = Target.regression Target.t
  type prediction = target
  type rng = Rng.t

  let clone value = value
  let params value = value
  let fitted_params fitted = fitted.offset
  let feature_schema fitted = fitted.schema

  let fit offset ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let* () = data (Feature_schema.validate_matrix feature_schema x) in
    let values = Target.regression_values y in
    let sum = ref 0. and weight_sum = ref 0. in
    for row = 0 to Vector.length values - 1 do
      let weight =
        match sample_weight with
        | None -> 1.
        | Some weights -> Sample_weight.get weights row
      in
      sum := !sum +. (weight *. Vector.get values row);
      weight_sum := !weight_sum +. weight
    done;
    Ok { offset; mean = !sum /. !weight_sum; schema = feature_schema }

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      failure "schema differs"
    else
      let* values =
        Vector.init ~length:(Matrix.rows x) (fun _ ->
            fitted.mean +. fitted.offset)
        |> data
      in
      Target.regression values |> data
end

module Classifier = struct
  type t = int
  type params = int

  type fitted = {
    labels : int array;
    probabilities : float array;
    schema : Feature_schema.t;
  }

  type target = Target.classification Target.t
  type prediction = target
  type rng = Rng.t

  let clone value = value
  let params value = value
  let fitted_params fitted = Array.length fitted.labels
  let feature_schema fitted = fitted.schema
  let classes fitted = Array.copy fitted.labels
  let labels count = Array.sub [| 19; -7; 61 |] 0 count

  let fit count ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let* () = data (Feature_schema.validate_matrix feature_schema x) in
    let labels = labels count in
    let mass = Array.make count 0. in
    Array.iteri
      (fun row label ->
        let index = ref 0 in
        while labels.(!index) <> label do
          incr index
        done;
        let weight =
          match sample_weight with
          | None -> 1.
          | Some weights -> Sample_weight.get weights row
        in
        mass.(!index) <- mass.(!index) +. weight)
      (Target.classification_values y);
    let total = Array.fold_left ( +. ) 0. mass in
    Ok
      {
        labels;
        probabilities = Array.map (fun value -> value /. total) mass;
        schema = feature_schema;
      }

  let predict fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      failure "schema differs"
    else
      let best = ref 0 in
      Array.iteri
        (fun index value ->
          if value > fitted.probabilities.(!best) then best := index)
        fitted.probabilities;
      Ok
        (Target.classification
           (Array.make (Matrix.rows x) fitted.labels.(!best)))

  let predict_proba fitted ~feature_schema ~x =
    if not (Feature_schema.equal fitted.schema feature_schema) then
      failure "schema differs"
    else
      Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length fitted.labels)
        (fun _ column -> fitted.probabilities.(column))
      |> data
end

let package = "example-modelkit-estimators"

let provenance implementation =
  Pipeline.provenance ~package ~version:"1.2.3" ~implementation

let encode_floats values =
  let payload = Bytes.create (8 * Array.length values) in
  Array.iteri
    (fun index value ->
      Bytes.set_int64_be payload (8 * index) (Int64.bits_of_float value))
    values;
  Ok payload

let decode_floats count payload =
  if Bytes.length payload <> count * 8 then
    failure "invalid numeric payload length"
  else
    let values =
      Array.init count (fun index ->
          Int64.float_of_bits (Bytes.get_int64_be payload (8 * index)))
    in
    if Array.for_all Float.is_finite values then Ok values
    else failure "nonfinite payload"

let mean_codec ?(calls = ref 0) ?(bad_schema = false) ?(version = 1)
    ?(name = "Mean") () =
  Artifact.Codec.regression ~package ~name ~version
    (module Mean)
    ~encode:(fun fitted ->
      encode_floats [| fitted.Mean.offset; fitted.Mean.mean |])
    ~decode:(fun ~provenance:_ ~feature_schema payload ->
      incr calls;
      let* values = decode_floats 2 payload in
      let* schema =
        if bad_schema then Feature_schema.anonymous ~feature_count:99 |> data
        else Ok feature_schema
      in
      Ok { Mean.offset = values.(0); mean = values.(1); schema })

let classifier_codec ?(calls = ref 0) ?(bad_classes = false) () =
  Artifact.Codec.classification ~package ~name:"Classifier" ~version:1
    (module Classifier)
    ~classes:Classifier.classes ~predict_proba:Classifier.predict_proba
    ~encode:(fun fitted -> encode_floats fitted.Classifier.probabilities)
    ~decode:(fun ~provenance:_ ~feature_schema ~classes payload ->
      incr calls;
      let* probabilities = decode_floats (Array.length classes) payload in
      if bad_classes then (
        let first = classes.(0) in
        classes.(0) <- classes.(1);
        classes.(1) <- first);
      Ok { Classifier.labels = classes; probabilities; schema = feature_schema })
    ()
