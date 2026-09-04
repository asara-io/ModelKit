open Modelkit

type 'a conversion = { value : 'a; report : Conversion_report.t }

type features = {
  matrix : Matrix.t;
  schema : Feature_schema.t;
  null_mask : Null_mask.t option;
  feature_reports : Conversion_report.t list;
}

type 'kind admitted_dataset = {
  dataset : 'kind Dataset.t;
  feature_null_mask : Null_mask.t option;
  dataset_reports : Conversion_report.t list;
}

let ( let* ) = Result.bind

let validation_error ~name ~reason ~remediation =
  Error (Error.make ~remediation (Error.Validation { name; reason }))

let shape_error ~name ~expected ~observed =
  Error
    (Error.make
       ~remediation:
         (Format.sprintf "Pass %s with shape (%s)." name
            (String.concat ", " (List.map string_of_int expected)))
       (Error.Shape_mismatch
          { name; expected; observed = Array.to_list observed }))

let expect_rank ~name rank tensor =
  let shape = Nx.shape tensor in
  if Array.length shape = rank then Ok shape
  else
    validation_error ~name
      ~reason:
        (Format.sprintf "expected rank %d but observed rank %d with shape %s"
           rank (Array.length shape) (Nx.shape_to_string shape))
      ~remediation:(Format.sprintf "Pass a rank-%d %s tensor." rank name)

let report ~source ~source_dtype ~shape ~source_contiguous
    ~temporary_payload_bytes ~retained_payload_bytes =
  Conversion_report.create ~source ~source_dtype ~source_shape:shape
    ~source_contiguous:(Some source_contiguous) ~temporary_payload_bytes
    ~retained_payload_bytes

let float_bytes count = Int64.mul 8L (Int64.of_int count)

let int_bytes count =
  Int64.mul (Int64.of_int (Sys.word_size / 8)) (Int64.of_int count)

let map_data_error ~remediation = function
  | Ok value -> Ok value
  | Error error -> Error (Error.of_data_error ~remediation error)

let feature_names names columns =
  match names with
  | None -> Ok None
  | Some names ->
      let* names =
        Feature_names.create ~expected_count:columns names
        |> map_data_error
             ~remediation:
               "Provide one non-empty, unique feature name per tensor column."
      in
      Ok (Some names)

let feature_null_mask tensor expected_shape =
  let* shape = expect_rank ~name:"feature null mask" 2 tensor in
  if shape <> expected_shape then
    shape_error ~name:"feature null mask"
      ~expected:(Array.to_list expected_shape)
      ~observed:shape
  else
    let rows = shape.(0) in
    let columns = shape.(1) in
    let* mask =
      Null_mask.init ~rows ~columns (fun row column ->
          Nx.item [ row; column ] tensor)
      |> map_data_error ~remediation:"Pass a valid rank-two Boolean null mask."
    in
    let* report =
      report ~source:"feature null mask" ~source_dtype:"bool" ~shape
        ~source_contiguous:(Nx.is_c_contiguous tensor)
        ~temporary_payload_bytes:0L
        ~retained_payload_bytes:(int_bytes (Nx.size tensor))
    in
    Ok (mask, report)

let validate_feature_values matrix rows columns =
  let rec loop row column =
    if row = rows then Ok ()
    else if column = columns then loop (row + 1) 0
    else
      let value = Matrix.get matrix row column in
      if Float.is_finite value || Float.is_nan value then loop row (column + 1)
      else
        Error
          (Error.of_data_error
             ~remediation:
               "Replace infinity, mark the position null, or reject the source \
                row before admission."
             (Data_error.Non_finite
                {
                  name =
                    Format.sprintf "Nx features at row %d, column %d" row column;
                  index = (row * columns) + column;
                  value;
                }))
  in
  loop 0 0

let features ?names ?null_mask tensor =
  let* shape = expect_rank ~name:"features" 2 tensor in
  let rows = shape.(0) in
  let columns = shape.(1) in
  let* names = feature_names names columns in
  let* null_mask, mask_reports =
    match null_mask with
    | None -> Ok (None, [])
    | Some tensor ->
        let* mask, report = feature_null_mask tensor shape in
        Ok (Some mask, [ report ])
  in
  let* matrix =
    Matrix.init ~rows ~columns (fun row column ->
        match null_mask with
        | Some mask when Null_mask.get mask row column -> Float.nan
        | None | Some _ -> Nx.item [ row; column ] tensor)
    |> map_data_error
         ~remediation:"Pass a valid rank-two float64 feature tensor."
  in
  let* () = validate_feature_values matrix rows columns in
  let* schema =
    Feature_schema.of_matrix ?names matrix
    |> map_data_error
         ~remediation:"Align feature names with the tensor column order."
  in
  let* value_report =
    report ~source:"features" ~source_dtype:"float64" ~shape
      ~source_contiguous:(Nx.is_c_contiguous tensor)
      ~temporary_payload_bytes:0L
      ~retained_payload_bytes:(float_bytes (Nx.size tensor))
  in
  Ok
    ({
       matrix;
       schema;
       null_mask;
       feature_reports = value_report :: mask_reports;
     }
      : features)

let regression_target tensor =
  let* shape = expect_rank ~name:"regression target" 1 tensor in
  let length = shape.(0) in
  let* values =
    Vector.init ~length (fun index -> Nx.item [ index ] tensor)
    |> map_data_error
         ~remediation:"Pass a valid rank-one float64 regression target."
  in
  let* value =
    Target.regression values
    |> map_data_error
         ~remediation:"Remove NaN and infinity from the regression target."
  in
  let* report =
    report ~source:"regression target" ~source_dtype:"float64" ~shape
      ~source_contiguous:(Nx.is_c_contiguous tensor)
      ~temporary_payload_bytes:0L ~retained_payload_bytes:(float_bytes length)
  in
  Ok { value; report }

let int_bounds = (Int64.of_int Int.min_int, Int64.of_int Int.max_int)

let checked_int ~name index value =
  let minimum, maximum = int_bounds in
  if Int64.compare value minimum >= 0 && Int64.compare value maximum <= 0 then
    Ok (Int64.to_int value)
  else
    validation_error ~name
      ~reason:
        (Format.sprintf "value %Ld at index %d does not fit OCaml int" value
           index)
      ~remediation:
        "Remap labels to values representable by OCaml int on the target \
         platform."

let int_array ~name tensor =
  let* shape = expect_rank ~name 1 tensor in
  let length = shape.(0) in
  let values = Array.make length 0 in
  let rec fill index =
    if index = length then Ok (shape, values)
    else
      let* value = checked_int ~name index (Nx.item [ index ] tensor) in
      values.(index) <- value;
      fill (index + 1)
  in
  fill 0

let classification_target tensor =
  let* shape, values = int_array ~name:"classification target" tensor in
  let length = Array.length values in
  let value = Target.classification values in
  let payload_bytes = int_bytes length in
  let* report =
    report ~source:"classification target" ~source_dtype:"int64" ~shape
      ~source_contiguous:(Nx.is_c_contiguous tensor)
      ~temporary_payload_bytes:payload_bytes
      ~retained_payload_bytes:payload_bytes
  in
  Ok { value; report }

let sample_weight tensor =
  let* shape = expect_rank ~name:"sample weights" 1 tensor in
  let length = shape.(0) in
  let* vector =
    Vector.init ~length (fun index -> Nx.item [ index ] tensor)
    |> map_data_error
         ~remediation:"Pass a valid rank-one float64 weight tensor."
  in
  let* value =
    Sample_weight.create ~expected_length:length vector
    |> map_data_error
         ~remediation:
           "Use finite, non-negative weights with at least one positive value."
  in
  let* report =
    report ~source:"sample weights" ~source_dtype:"float64" ~shape
      ~source_contiguous:(Nx.is_c_contiguous tensor)
      ~temporary_payload_bytes:0L ~retained_payload_bytes:(float_bytes length)
  in
  Ok { value; report }

let groups tensor =
  let* shape, values = int_array ~name:"groups" tensor in
  let length = Array.length values in
  let* value =
    Groups.create ~expected_length:length values
    |> map_data_error
         ~remediation:
           "Provide one representable integer group label per sample."
  in
  let payload_bytes = int_bytes length in
  let* report =
    report ~source:"groups" ~source_dtype:"int64" ~shape
      ~source_contiguous:(Nx.is_c_contiguous tensor)
      ~temporary_payload_bytes:payload_bytes
      ~retained_payload_bytes:payload_bytes
  in
  Ok { value; report }

let optional_conversion convert = function
  | None -> Ok (None, [])
  | Some tensor ->
      let* converted = convert tensor in
      Ok (Some converted.value, [ converted.report ])

let admitted_dataset ~(features : features) ~target ?sample_weight ?groups
    ~metadata_reports () =
  let feature_names = Feature_schema.names features.schema in
  let* dataset =
    Dataset.create ~finiteness:Dataset.Allow_nan ?feature_names ?sample_weight
      ?groups ~x:features.matrix ~y:target ()
    |> map_data_error
         ~remediation:
           "Align every target and metadata tensor with the feature row count."
  in
  Ok
    ({
       dataset;
       feature_null_mask = features.null_mask;
       dataset_reports = features.feature_reports @ metadata_reports;
     }
      : _ admitted_dataset)

let regression_dataset ?names ?feature_null_mask ?sample_weight:weights
    ?groups:group_tensor ~x ~y () =
  let* features = features ?names ?null_mask:feature_null_mask x in
  let* target = regression_target y in
  let* sample_weight, weight_reports =
    optional_conversion sample_weight weights
  in
  let* groups, group_reports = optional_conversion groups group_tensor in
  admitted_dataset ~features ~target:target.value ?sample_weight ?groups
    ~metadata_reports:((target.report :: weight_reports) @ group_reports)
    ()

let classification_dataset ?names ?feature_null_mask ?sample_weight:weights
    ?groups:group_tensor ~x ~y () =
  let* features = features ?names ?null_mask:feature_null_mask x in
  let* target = classification_target y in
  let* sample_weight, weight_reports =
    optional_conversion sample_weight weights
  in
  let* groups, group_reports = optional_conversion groups group_tensor in
  admitted_dataset ~features ~target:target.value ?sample_weight ?groups
    ~metadata_reports:((target.report :: weight_reports) @ group_reports)
    ()
