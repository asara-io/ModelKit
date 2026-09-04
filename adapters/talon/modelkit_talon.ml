open Modelkit

type 'a conversion = 'a Admission.conversion = {
  value : 'a;
  report : Conversion_report.t;
}

type features = Admission.features = {
  matrix : Matrix.t;
  schema : Feature_schema.t;
  null_mask : Null_mask.t option;
  feature_reports : Conversion_report.t list;
}

type 'kind admitted_dataset = 'kind Admission.dataset = {
  dataset : 'kind Dataset.t;
  feature_null_mask : Null_mask.t option;
  dataset_reports : Conversion_report.t list;
}

let ( let* ) = Result.bind

let validation_error ~role ~reason ~remediation =
  Error (Error.make ~remediation (Error.Validation { name = role; reason }))

let map_data_error ~remediation = function
  | Ok value -> Ok value
  | Error error -> Error (Error.of_data_error ~remediation error)

let float_bytes count = Int64.mul 8L (Int64.of_int count)

let int_bytes count =
  Int64.mul (Int64.of_int (Sys.word_size / 8)) (Int64.of_int count)

let dtype_name = function
  | `Float32 -> "float32"
  | `Float64 -> "float64"
  | `Int32 -> "int32"
  | `Int64 -> "int64"
  | `Bool -> "bool"
  | `String -> "string"
  | `Other -> "an unsupported dtype"

let column ~role frame name =
  match Talon.get_column frame name with
  | Some column -> Ok column
  | None ->
      validation_error ~role
        ~reason:
          (Format.sprintf "column %S is not present in the dataframe" name)
        ~remediation:
          (Format.sprintf
             "Select a column that exists in the dataframe for the %s." role)

let typed_tensor ~role ~name ~dtype ~expected column =
  match Talon.Col.to_tensor dtype column with
  | Some tensor -> Ok tensor
  | None ->
      validation_error ~role
        ~reason:
          (Format.sprintf "column %S has %s but %s is required" name
             (dtype_name (Talon.Col.dtype column))
             expected)
        ~remediation:
          (Format.sprintf
             "Cast column %S to %s explicitly (for example with \
              Talon.cast_column) before admission; ModelKit does not coerce \
              column types."
             name expected)

let reject_nulls ~role ~name column =
  let nulls = Talon.Col.null_count column in
  if nulls = 0 then Ok ()
  else
    validation_error ~role
      ~reason:(Format.sprintf "column %S contains %d null values" name nulls)
      ~remediation:
        (Format.sprintf
           "Drop or fill the null rows of column %S in Talon before admission; \
            ModelKit never imputes a %s."
           name role)

let report ~source ~source_dtype ~shape ~source_contiguous
    ~temporary_payload_bytes ~retained_payload_bytes =
  Conversion_report.create ~source ~source_dtype ~source_shape:shape
    ~source_contiguous ~temporary_payload_bytes ~retained_payload_bytes

let contiguity tensor = Some (Nx.is_c_contiguous tensor)

(* Column tensors are read through their flat buffer with the tensor's offset
   and element stride, which avoids allocating an index list per element. *)
let reader tensor =
  let buffer = Nx.data tensor in
  let offset = Nx.offset tensor in
  let stride = (Nx.strides tensor).(0) / Nx.itemsize tensor in
  fun index -> Nx_buffer.get buffer (offset + (index * stride))

let rec unique_selection seen = function
  | [] -> Ok ()
  | name :: rest ->
      if List.mem name seen then
        validation_error ~role:"features"
          ~reason:(Format.sprintf "column %S is selected more than once" name)
          ~remediation:"Select each feature column at most once."
      else unique_selection (name :: seen) rest

let feature_columns frame names =
  let rec collect accumulated = function
    | [] -> Ok (Array.of_list (List.rev accumulated))
    | name :: rest ->
        let* column = column ~role:"features" frame name in
        let* tensor =
          typed_tensor ~role:"features" ~name ~dtype:Nx.float64
            ~expected:"float64" column
        in
        collect
          ((reader tensor, Talon.Col.null_mask column) :: accumulated)
          rest
  in
  collect [] names

let feature_null_mask ~rows ~columns masks =
  if Array.exists Option.is_some masks then
    let* mask =
      Null_mask.init ~rows ~columns (fun row column ->
          match masks.(column) with Some mask -> mask.(row) | None -> false)
      |> map_data_error
           ~remediation:"Select feature columns with well-formed null masks."
    in
    let* report =
      report ~source:"feature null mask" ~source_dtype:"bool"
        ~shape:[| rows; columns |] ~source_contiguous:None
        ~temporary_payload_bytes:0L
        ~retained_payload_bytes:(int_bytes (rows * columns))
    in
    Ok (Some mask, [ report ])
  else Ok (None, [])

let validate_feature_values ~names matrix rows columns =
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
               "Replace infinity, mark the position null, or drop the row in \
                Talon before admission."
             (Data_error.Non_finite
                {
                  name =
                    Format.sprintf "Talon feature column %S at row %d"
                      names.(column) row;
                  index = (row * columns) + column;
                  value;
                }))
  in
  loop 0 0

let features frame names =
  let* () =
    if names = [] then
      validation_error ~role:"features"
        ~reason:"no feature columns were selected"
        ~remediation:"Select at least one float64 feature column."
    else Ok ()
  in
  let* () = unique_selection [] names in
  let* columns = feature_columns frame names in
  let names = Array.of_list names in
  let rows = Talon.num_rows frame in
  let column_count = Array.length columns in
  let* null_mask, mask_reports =
    feature_null_mask ~rows ~columns:column_count (Array.map snd columns)
  in
  let* matrix =
    Matrix.init ~rows ~columns:column_count (fun row column ->
        match null_mask with
        | Some mask when Null_mask.get mask row column -> Float.nan
        | None | Some _ -> (fst columns.(column)) row)
    |> map_data_error
         ~remediation:"Select float64 feature columns with at least one row."
  in
  let* () = validate_feature_values ~names matrix rows column_count in
  let* feature_names =
    Feature_names.create ~expected_count:column_count names
    |> map_data_error
         ~remediation:
           "Rename the selected columns so that every feature name is \
            non-empty and unique."
  in
  let* schema =
    Feature_schema.of_matrix ~names:feature_names matrix
    |> map_data_error
         ~remediation:"Align feature names with the selected column order."
  in
  let* value_report =
    report ~source:"features" ~source_dtype:"float64"
      ~shape:[| rows; column_count |] ~source_contiguous:None
      ~temporary_payload_bytes:0L
      ~retained_payload_bytes:(float_bytes (rows * column_count))
  in
  Ok
    ({
       matrix;
       schema;
       null_mask;
       feature_reports = value_report :: mask_reports;
     }
      : features)

let float_column ~role frame name =
  let* column = column ~role frame name in
  let* tensor =
    typed_tensor ~role ~name ~dtype:Nx.float64 ~expected:"float64" column
  in
  let* () = reject_nulls ~role ~name column in
  let length = Talon.Col.length column in
  let* vector =
    Vector.init ~length (reader tensor)
    |> map_data_error
         ~remediation:
           (Format.sprintf "Select a float64 %s column with at least one row."
              role)
  in
  Ok (vector, tensor, length)

let regression_target frame name =
  let role = "regression target" in
  let* vector, tensor, length = float_column ~role frame name in
  let* value =
    Target.regression vector
    |> map_data_error
         ~remediation:
           (Format.sprintf
              "Remove NaN and infinity from column %S before admission." name)
  in
  let* report =
    report ~source:role ~source_dtype:"float64" ~shape:[| length |]
      ~source_contiguous:(contiguity tensor) ~temporary_payload_bytes:0L
      ~retained_payload_bytes:(float_bytes length)
  in
  Ok { value; report }

let sample_weight frame name =
  let role = "sample weights" in
  let* vector, tensor, length = float_column ~role frame name in
  let* value =
    Sample_weight.create ~expected_length:length vector
    |> map_data_error
         ~remediation:
           (Format.sprintf
              "Use finite, non-negative weights with at least one positive \
               value in column %S."
              name)
  in
  let* report =
    report ~source:role ~source_dtype:"float64" ~shape:[| length |]
      ~source_contiguous:(contiguity tensor) ~temporary_payload_bytes:0L
      ~retained_payload_bytes:(float_bytes length)
  in
  Ok { value; report }

let int_bounds = (Int64.of_int Int.min_int, Int64.of_int Int.max_int)

let checked_int ~role ~name index value =
  let minimum, maximum = int_bounds in
  if Int64.compare value minimum >= 0 && Int64.compare value maximum <= 0 then
    Ok (Int64.to_int value)
  else
    validation_error ~role
      ~reason:
        (Format.sprintf
           "value %Ld at row %d of column %S does not fit OCaml int" value index
           name)
      ~remediation:
        "Remap labels to values representable by OCaml int on the target \
         platform."

let int_column ~role frame name =
  let* column = column ~role frame name in
  let* tensor =
    typed_tensor ~role ~name ~dtype:Nx.int64 ~expected:"int64" column
  in
  let* () = reject_nulls ~role ~name column in
  let length = Talon.Col.length column in
  let read = reader tensor in
  let values = Array.make length 0 in
  let rec fill index =
    if index = length then Ok (values, tensor)
    else
      let* value = checked_int ~role ~name index (read index) in
      values.(index) <- value;
      fill (index + 1)
  in
  fill 0

let classification_target frame name =
  let role = "classification target" in
  let* values, tensor = int_column ~role frame name in
  let length = Array.length values in
  let value = Target.classification values in
  let payload_bytes = int_bytes length in
  let* report =
    report ~source:role ~source_dtype:"int64" ~shape:[| length |]
      ~source_contiguous:(contiguity tensor)
      ~temporary_payload_bytes:payload_bytes
      ~retained_payload_bytes:payload_bytes
  in
  Ok { value; report }

let groups frame name =
  let role = "groups" in
  let* values, tensor = int_column ~role frame name in
  let length = Array.length values in
  let* value =
    Groups.create ~expected_length:length values
    |> map_data_error
         ~remediation:
           (Format.sprintf
              "Provide one representable integer group label per row in column \
               %S."
              name)
  in
  let payload_bytes = int_bytes length in
  let* report =
    report ~source:role ~source_dtype:"int64" ~shape:[| length |]
      ~source_contiguous:(contiguity tensor)
      ~temporary_payload_bytes:payload_bytes
      ~retained_payload_bytes:payload_bytes
  in
  Ok { value; report }

let disjoint_roles ~features ~target ?sample_weight ?groups () =
  let assignments =
    List.map (fun name -> ("features", name)) features
    @ [ ("target", target) ]
    @ Option.fold ~none:[]
        ~some:(fun name -> [ ("sample weights", name) ])
        sample_weight
    @ Option.fold ~none:[] ~some:(fun name -> [ ("groups", name) ]) groups
  in
  let rec check seen = function
    | [] -> Ok ()
    | (role, name) :: rest -> (
        match List.assoc_opt name seen with
        | Some previous when previous <> role ->
            validation_error ~role:"dataset"
              ~reason:
                (Format.sprintf "column %S is selected as both %s and %s" name
                   previous role)
              ~remediation:
                "Select distinct columns for features, the target, sample \
                 weights, and groups."
        | Some _ | None -> check ((name, role) :: seen) rest)
  in
  check [] assignments

let optional_conversion convert frame = function
  | None -> Ok (None, [])
  | Some name ->
      let* converted = convert frame name in
      Ok (Some converted.value, [ converted.report ])

let admitted_dataset ~(features : features) ~target ?sample_weight ?groups
    ~metadata_reports () =
  let feature_names = Feature_schema.names features.schema in
  let* dataset =
    Dataset.create ~finiteness:Dataset.Allow_nan ?feature_names ?sample_weight
      ?groups ~x:features.matrix ~y:target ()
    |> map_data_error
         ~remediation:
           "Select feature, target, and metadata columns from the same \
            dataframe."
  in
  Ok
    ({
       dataset;
       feature_null_mask = features.null_mask;
       dataset_reports = features.feature_reports @ metadata_reports;
     }
      : _ admitted_dataset)

let regression_dataset ?sample_weight:weight_column ?groups:group_column
    ~features:feature_columns ~target:target_column frame =
  let* () =
    disjoint_roles ~features:feature_columns ~target:target_column
      ?sample_weight:weight_column ?groups:group_column ()
  in
  let* features = features frame feature_columns in
  let* target = regression_target frame target_column in
  let* sample_weight, weight_reports =
    optional_conversion sample_weight frame weight_column
  in
  let* groups, group_reports = optional_conversion groups frame group_column in
  admitted_dataset ~features ~target:target.value ?sample_weight ?groups
    ~metadata_reports:((target.report :: weight_reports) @ group_reports)
    ()

let classification_dataset ?sample_weight:weight_column ?groups:group_column
    ~features:feature_columns ~target:target_column frame =
  let* () =
    disjoint_roles ~features:feature_columns ~target:target_column
      ?sample_weight:weight_column ?groups:group_column ()
  in
  let* features = features frame feature_columns in
  let* target = classification_target frame target_column in
  let* sample_weight, weight_reports =
    optional_conversion sample_weight frame weight_column
  in
  let* groups, group_reports = optional_conversion groups frame group_column in
  admitted_dataset ~features ~target:target.value ?sample_weight ?groups
    ~metadata_reports:((target.report :: weight_reports) @ group_reports)
    ()
