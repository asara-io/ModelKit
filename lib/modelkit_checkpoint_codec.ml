open Modelkit_data

exception Invalid_checkpoint

let invalid () = raise Invalid_checkpoint
let limit = 64 * 1024 * 1024

let token emit value =
  emit (string_of_int (String.length value));
  emit ":";
  emit value

let int emit value = token emit (string_of_int value)
let float emit value = token emit (Int64.to_string (Int64.bits_of_float value))
let bool emit value = int emit (if value then 1 else 0)

let array write emit values =
  int emit (Array.length values);
  Array.iter (write emit) values

let option write emit = function
  | None -> int emit 0
  | Some value ->
      int emit 1;
      write emit value

let encode write value =
  let buffer = Buffer.create 1024 in
  let emit value =
    if String.length value > limit - Buffer.length buffer then invalid ();
    Buffer.add_string buffer value
  in
  write emit value;
  Buffer.contents buffer

let digest write value =
  let state = ref "modelkit-search-identity-v1" in
  let buffer = Buffer.create 4096 in
  let flush () =
    state := Digest.string (!state ^ Buffer.contents buffer);
    Buffer.clear buffer
  in
  let emit value =
    String.iter
      (fun byte ->
        Buffer.add_char buffer byte;
        if Buffer.length buffer = 4096 then flush ())
      value
  in
  write emit value;
  flush ();
  Digest.to_hex !state

type reader = { source : string; mutable position : int }

let reader source =
  if String.length source > limit then invalid ();
  { source; position = 0 }

let remaining reader = String.length reader.source - reader.position

let read_token reader =
  let start = reader.position in
  while
    reader.position < String.length reader.source
    && reader.source.[reader.position] <> ':'
  do
    let byte = reader.source.[reader.position] in
    if byte < '0' || byte > '9' || reader.position - start > 9 then invalid ();
    reader.position <- reader.position + 1
  done;
  if remaining reader = 0 then invalid ();
  let length =
    int_of_string (String.sub reader.source start (reader.position - start))
  in
  reader.position <- reader.position + 1;
  if length > remaining reader then invalid ();
  let value = String.sub reader.source reader.position length in
  reader.position <- reader.position + length;
  value

let read_int reader = int_of_string (read_token reader)

let read_float reader =
  Int64.float_of_bits (Int64.of_string (read_token reader))

let read_bool reader =
  match read_int reader with 0 -> false | 1 -> true | _ -> invalid ()

let read_array read reader =
  let count = read_int reader in
  if count < 0 || count > remaining reader / 2 || count > Sys.max_array_length
  then invalid ();
  Array.init count (fun _ -> read reader)

let read_option read reader =
  match read_int reader with
  | 0 -> None
  | 1 -> Some (read reader)
  | _ -> invalid ()

let finish reader = if remaining reader <> 0 then invalid ()

let protect f =
  try Ok (f ())
  with Invalid_checkpoint | Failure _ | Invalid_argument _ ->
    Error
      (Error.make
         (Error.Artifact
            {
              operation = "search checkpoint";
              reason = "invalid, oversized, or unsupported checkpoint";
            })
         ~remediation:
           "use an intact checkpoint produced by this checkpoint schema")

let schema emit value =
  int emit (Feature_schema.feature_count value);
  option (array token) emit
    (Option.map Feature_names.to_array (Feature_schema.names value))

let read_schema reader =
  let count = read_int reader in
  let names = read_option (read_array read_token) reader in
  let result =
    match names with
    | None -> Feature_schema.anonymous ~feature_count:count
    | Some names ->
        Result.map Feature_schema.named
          (Feature_names.create ~expected_count:count names)
  in
  match result with Ok value -> value | Error _ -> invalid ()

let data_error emit value =
  match value with
  | Data_error.Negative_dimension { name; value } ->
      token emit "Negative_dimension";
      token emit name;
      int emit value
  | Data_error.Ragged_matrix { row; expected_columns; observed_columns } ->
      token emit "Ragged_matrix";
      int emit row;
      int emit expected_columns;
      int emit observed_columns
  | Data_error.Length_mismatch { name; expected; observed } ->
      token emit "Length_mismatch";
      token emit name;
      int emit expected;
      int emit observed
  | Data_error.Index_out_of_bounds { name; index; upper_bound } ->
      token emit "Index_out_of_bounds";
      token emit name;
      int emit index;
      int emit upper_bound
  | Data_error.Non_finite { name; index; value } ->
      token emit "Non_finite";
      token emit name;
      int emit index;
      float emit value
  | Data_error.Negative_weight { index; value } ->
      token emit "Negative_weight";
      int emit index;
      float emit value
  | Data_error.All_zero_weights ->
      token emit "All_zero_weights";
      ()
  | Data_error.Empty_feature_name { index } ->
      token emit "Empty_feature_name";
      int emit index
  | Data_error.Duplicate_feature_name { name; first_index; duplicate_index } ->
      token emit "Duplicate_feature_name";
      token emit name;
      int emit first_index;
      int emit duplicate_index
  | Data_error.Csr_row_offset_mismatch { position; expected; observed } ->
      token emit "Csr_row_offset_mismatch";
      int emit position;
      int emit expected;
      int emit observed
  | Data_error.Invalid_csr_row_offset
      { position; previous; observed; nonzero_count } ->
      token emit "Invalid_csr_row_offset";
      int emit position;
      int emit previous;
      int emit observed;
      int emit nonzero_count
  | Data_error.Invalid_csr_column_order { row; previous; observed } ->
      token emit "Invalid_csr_column_order";
      int emit row;
      int emit previous;
      int emit observed

let read_data_error reader =
  match read_token reader with
  | "Negative_dimension" ->
      let name = read_token reader in
      let value = read_int reader in
      Data_error.Negative_dimension { name; value }
  | "Ragged_matrix" ->
      let row = read_int reader in
      let expected_columns = read_int reader in
      let observed_columns = read_int reader in
      Data_error.Ragged_matrix { row; expected_columns; observed_columns }
  | "Length_mismatch" ->
      let name = read_token reader in
      let expected = read_int reader in
      let observed = read_int reader in
      Data_error.Length_mismatch { name; expected; observed }
  | "Index_out_of_bounds" ->
      let name = read_token reader in
      let index = read_int reader in
      let upper_bound = read_int reader in
      Data_error.Index_out_of_bounds { name; index; upper_bound }
  | "Non_finite" ->
      let name = read_token reader in
      let index = read_int reader in
      let value = read_float reader in
      Data_error.Non_finite { name; index; value }
  | "Negative_weight" ->
      let index = read_int reader in
      let value = read_float reader in
      Data_error.Negative_weight { index; value }
  | "All_zero_weights" -> Data_error.All_zero_weights
  | "Empty_feature_name" ->
      let index = read_int reader in
      Data_error.Empty_feature_name { index }
  | "Duplicate_feature_name" ->
      let name = read_token reader in
      let first_index = read_int reader in
      let duplicate_index = read_int reader in
      Data_error.Duplicate_feature_name { name; first_index; duplicate_index }
  | "Csr_row_offset_mismatch" ->
      let position = read_int reader in
      let expected = read_int reader in
      let observed = read_int reader in
      Data_error.Csr_row_offset_mismatch { position; expected; observed }
  | "Invalid_csr_row_offset" ->
      let position = read_int reader in
      let previous = read_int reader in
      let observed = read_int reader in
      let nonzero_count = read_int reader in
      Data_error.Invalid_csr_row_offset
        { position; previous; observed; nonzero_count }
  | "Invalid_csr_column_order" ->
      let row = read_int reader in
      let previous = read_int reader in
      let observed = read_int reader in
      Data_error.Invalid_csr_column_order { row; previous; observed }
  | _ -> invalid ()

let int_list emit values = array int emit (Array.of_list values)
let read_int_list reader = Array.to_list (read_array read_int reader)

let kind emit value =
  match value with
  | Error.Data value ->
      token emit "Data";
      data_error emit value
  | Error.Shape_mismatch { name; expected; observed } ->
      token emit "Shape_mismatch";
      token emit name;
      int_list emit expected;
      int_list emit observed
  | Error.Feature_schema_mismatch { expected; observed } ->
      token emit "Feature_schema_mismatch";
      schema emit expected;
      schema emit observed
  | Error.Validation { name; reason } ->
      token emit "Validation";
      token emit name;
      token emit reason
  | Error.Numerical { operation; reason } ->
      token emit "Numerical";
      token emit operation;
      token emit reason
  | Error.Convergence { algorithm; reason } ->
      token emit "Convergence";
      token emit algorithm;
      token emit reason
  | Error.Compatibility { component; reason } ->
      token emit "Compatibility";
      token emit component;
      token emit reason
  | Error.Artifact { operation; reason } ->
      token emit "Artifact";
      token emit operation;
      token emit reason
  | Error.Callback_failure { reason } ->
      token emit "Callback_failure";
      token emit reason
  | Error.Cancelled ->
      token emit "Cancelled";
      ()

let read_kind reader =
  match read_token reader with
  | "Data" -> Error.Data (read_data_error reader)
  | "Shape_mismatch" ->
      let name = read_token reader in
      let expected = read_int_list reader in
      let observed = read_int_list reader in
      Error.Shape_mismatch { name; expected; observed }
  | "Feature_schema_mismatch" ->
      let expected = read_schema reader in
      let observed = read_schema reader in
      Error.Feature_schema_mismatch { expected; observed }
  | "Validation" ->
      let name = read_token reader in
      let reason = read_token reader in
      Error.Validation { name; reason }
  | "Numerical" ->
      let operation = read_token reader in
      let reason = read_token reader in
      Error.Numerical { operation; reason }
  | "Convergence" ->
      let algorithm = read_token reader in
      let reason = read_token reader in
      Error.Convergence { algorithm; reason }
  | "Compatibility" ->
      let component = read_token reader in
      let reason = read_token reader in
      Error.Compatibility { component; reason }
  | "Artifact" ->
      let operation = read_token reader in
      let reason = read_token reader in
      Error.Artifact { operation; reason }
  | "Callback_failure" ->
      let reason = read_token reader in
      Error.Callback_failure { reason }
  | "Cancelled" -> Error.Cancelled
  | _ -> invalid ()

let context emit = function
  | Error.Stage name ->
      int emit 0;
      token emit name
  | Error.Fold index ->
      int emit 1;
      int emit index
  | Error.Candidate index ->
      int emit 2;
      int emit index
  | Error.Feature name ->
      int emit 3;
      token emit (Feature_name.to_string name)

let read_context reader =
  match read_int reader with
  | 0 -> Error.Stage (read_token reader)
  | 1 -> Error.Fold (read_int reader)
  | 2 -> Error.Candidate (read_int reader)
  | 3 -> (
      match Feature_name.create (read_token reader) with
      | Ok name -> Error.Feature name
      | Error _ -> invalid ())
  | _ -> invalid ()

let error emit value =
  kind emit (Error.kind value);
  array context emit (Array.of_list (Error.context value));
  token emit (Error.remediation value)

let read_error reader =
  let kind = read_kind reader in
  let context = read_array read_context reader |> Array.to_list in
  let remediation = read_token reader in
  Error.make ~context ~remediation kind

let result write emit = function
  | Ok value ->
      int emit 0;
      write emit value
  | Error value ->
      int emit 1;
      error emit value

let read_result read reader =
  match read_int reader with
  | 0 -> Ok (read reader)
  | 1 -> Error (read_error reader)
  | _ -> invalid ()
