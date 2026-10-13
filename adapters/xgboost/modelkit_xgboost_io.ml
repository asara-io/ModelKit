open Modelkit

let ( let* ) = Result.bind

let failure reason =
  Error
    (Error.make
       ~remediation:
         "Load a trusted model produced by a supported ModelKit XGBoost writer."
       (Error.Artifact { operation = "xgboost model"; reason }))

type limits = {
  max_bytes : int;
  max_features : int;
  max_classes : int;
  max_string_bytes : int;
}

let default_limits =
  {
    max_bytes = 64 * 1024 * 1024;
    max_features = 1_000_000;
    max_classes = 1_000_000;
    max_string_bytes = 1_048_576;
  }

let limits ?(max_bytes = default_limits.max_bytes)
    ?(max_features = default_limits.max_features)
    ?(max_classes = default_limits.max_classes)
    ?(max_string_bytes = default_limits.max_string_bytes) () =
  if
    max_bytes <= 0 || max_features <= 0 || max_classes <= 0
    || max_string_bytes <= 0
  then failure "reader limits must be positive"
  else Ok { max_bytes; max_features; max_classes; max_string_bytes }

module Writer = struct
  type t = Buffer.t

  let create () = Buffer.create 256
  let byte writer value = Buffer.add_char writer (Char.chr value)

  let i64 writer value =
    for shift = 7 downto 0 do
      byte writer
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical value (shift * 8)) 255L))
    done

  let int writer value = i64 writer (Int64.of_int value)
  let float writer value = i64 writer (Int64.bits_of_float value)

  let string writer value =
    int writer (String.length value);
    Buffer.add_string writer value

  let bytes writer value =
    int writer (Bytes.length value);
    Buffer.add_bytes writer value

  let contents writer = Bytes.of_string (Buffer.contents writer)
end

module Reader = struct
  type t = { bytes : bytes; limits : limits; mutable position : int }

  let create ~limits bytes = { bytes; limits; position = 0 }
  let remaining reader = Bytes.length reader.bytes - reader.position

  let require reader length =
    if length < 0 || length > remaining reader then
      failure "model artifact is truncated"
    else Ok ()

  let byte reader =
    let* () = require reader 1 in
    let value = Char.code (Bytes.get reader.bytes reader.position) in
    reader.position <- reader.position + 1;
    Ok value

  let i64 reader =
    let* () = require reader 8 in
    let value = Bytes.get_int64_be reader.bytes reader.position in
    reader.position <- reader.position + 8;
    Ok value

  let int reader =
    let* value = i64 reader in
    if value < Int64.of_int min_int || value > Int64.of_int max_int then
      failure "integer cannot be represented on this platform"
    else Ok (Int64.to_int value)

  let length reader ~maximum =
    let* value = i64 reader in
    if value < 0L || value > Int64.of_int maximum then
      failure "declared length exceeds its reader limit"
    else Ok (Int64.to_int value)

  let float reader = Result.map Int64.float_of_bits (i64 reader)

  let string reader =
    let* length = length reader ~maximum:reader.limits.max_string_bytes in
    let* () = require reader length in
    let value = Bytes.sub_string reader.bytes reader.position length in
    reader.position <- reader.position + length;
    if String.is_valid_utf_8 value then Ok value
    else failure "model string is not valid UTF-8"

  let bytes reader =
    let* length = length reader ~maximum:reader.limits.max_bytes in
    let* () = require reader length in
    let value = Bytes.sub reader.bytes reader.position length in
    reader.position <- reader.position + length;
    Ok value

  let ints reader ~maximum =
    let* length = length reader ~maximum in
    let* () =
      if length > remaining reader / 8 then failure "numeric array is truncated"
      else require reader (length * 8)
    in
    let values = Array.make length 0 in
    let rec read index =
      if index = length then Ok values
      else
        let* value = int reader in
        values.(index) <- value;
        read (index + 1)
    in
    read 0

  let finish reader =
    if remaining reader = 0 then Ok ()
    else failure "model artifact contains trailing data"
end

let frame ~limits ~task payload =
  if Bytes.length payload > limits.max_bytes - 36 then
    failure "model artifact exceeds the byte limit"
  else
    let writer = Writer.create () in
    Buffer.add_string writer "MKXGB001";
    Writer.byte writer 1;
    Writer.byte writer 0;
    Writer.byte writer task;
    Writer.int writer (Bytes.length payload);
    Writer.byte writer 1;
    Buffer.add_string writer (Digest.bytes payload);
    Buffer.add_bytes writer payload;
    Ok (Writer.contents writer)

let unframe ~limits ~tasks bytes =
  if Bytes.length bytes > limits.max_bytes then
    failure "model artifact exceeds the byte limit"
  else if Bytes.length bytes < 36 then failure "model artifact is truncated"
  else if Bytes.sub_string bytes 0 8 <> "MKXGB001" then
    failure "model artifact magic is invalid"
  else if Bytes.get bytes 8 <> '\001' || Bytes.get bytes 9 <> '\000' then
    failure "unsupported model container version"
  else
    let task = Char.code (Bytes.get bytes 10) in
    if not (List.mem task tasks) then
      failure "model task differs from the requested reader"
    else if Bytes.get bytes 19 <> '\001' then
      failure "unsupported model checksum algorithm"
    else
      let length = Bytes.get_int64_be bytes 11 in
      if length <> Int64.of_int (Bytes.length bytes - 36) then
        failure "model payload framing is invalid"
      else
        let payload = Bytes.sub bytes 36 (Bytes.length bytes - 36) in
        if Bytes.sub_string bytes 20 16 <> Digest.bytes payload then
          failure "model checksum differs"
        else Ok (task, payload)

let save ~path bytes =
  try
    let channel = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () ->
        output_bytes channel bytes;
        close_out channel;
        Ok ())
  with Sys_error reason -> failure reason

let read ~limits ~path =
  try
    let channel = open_in_bin path in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        let length = in_channel_length channel in
        if length > limits.max_bytes then
          failure "model file exceeds the byte limit"
        else
          let bytes = Bytes.create length in
          really_input channel bytes 0 length;
          Ok bytes)
  with
  | Sys_error reason -> failure reason
  | End_of_file -> failure "model file is truncated"

module Json = struct
  type t = Yojson.Basic.t

  let parse ~max_bytes text =
    if String.length text > max_bytes then
      failure "native JSON exceeds the byte limit"
    else
      let depth = ref 0
      and maximum = ref 0
      and quoted = ref false
      and escaped = ref false in
      String.iter
        (fun char ->
          if !quoted then (
            if !escaped then escaped := false
            else if char = '\\' then escaped := true
            else if char = '"' then quoted := false)
          else
            match char with
            | '"' -> quoted := true
            | '{' | '[' ->
                incr depth;
                maximum := max !maximum !depth
            | '}' | ']' -> decr depth
            | _ -> ())
        text;
      if !maximum > 64 then
        failure "native JSON nesting exceeds the supported bound"
      else
        try Ok (Yojson.Basic.from_string text)
        with Yojson.Json_error reason ->
          failure ("invalid native JSON: " ^ reason)

  let[@warning "-4"] field name = function
    | `Assoc fields -> (
        match List.filter (fun (key, _) -> key = name) fields with
        | [ (_, value) ] -> Ok value
        | _ -> failure ("missing or duplicate native JSON field: " ^ name))
    | _ -> failure ("native JSON object required for " ^ name)

  let[@warning "-4"] string = function
    | `String value -> Ok value
    | _ -> failure "native JSON string required"

  let[@warning "-4"] array = function
    | `List values -> Ok values
    | _ -> failure "native JSON array required"

  let[@warning "-4"] int = function
    | `Int value -> Ok value
    | `String value -> (
        if
          String.length value = 0
          || not
               (String.for_all
                  (function '0' .. '9' -> true | _ -> false)
                  value)
        then failure "native JSON integer must be nonnegative decimal"
        else
          try Ok (int_of_string value)
          with Failure _ -> failure "native JSON integer is invalid")
    | _ -> failure "native JSON integer required"

  let get_string name object_ =
    let* value = field name object_ in
    string value

  let get_int name object_ =
    let* value = field name object_ in
    int value

  let get_array name object_ =
    let* value = field name object_ in
    array value

  let version json =
    let* values = get_array "version" json in
    match values with
    | [ major; minor; patch ] ->
        let* major = int major in
        let* minor = int minor in
        let* patch = int patch in
        Ok (major, minor, patch)
    | _ -> failure "native version must contain three integers"

  let summary json =
    let* learner = field "learner" json in
    let* params = field "learner_model_param" learner in
    let* features = get_int "num_feature" params in
    let* classes = get_int "num_class" params in
    let* targets = get_int "num_target" params in
    let* objective = field "objective" learner in
    let* name = get_string "name" objective in
    let* booster = field "gradient_booster" learner in
    let* booster_name = get_string "name" booster in
    Ok (learner, features, classes, targets, name, booster, booster_name)

  let check_summary ~columns ~classes ~objective json =
    let* learner, features, native_classes, targets, name, booster, booster_name
        =
      summary json
    in
    let expected_classes = if classes > 2 then classes else 0 in
    if
      features <> columns
      || native_classes <> expected_classes
      || targets <> 1 || name <> objective || booster_name <> "gbtree"
    then
      failure
        "native task/objective or feature/output dimensions differ from the \
         envelope"
    else if classes > 2 then
      let* objective = field "objective" learner in
      let* params = field "softmax_multiclass_param" objective in
      let* count = get_int "num_class" params in
      if count <> classes then
        failure "native objective class count differs from the envelope"
      else Ok (learner, booster)
    else Ok (learner, booster)

  let check_model ~columns ~classes ~rounds ~objective ~version:expected_version
      json =
    let* actual_version = version json in
    if actual_version <> expected_version then
      failure "native JSON version differs from the envelope"
    else
      let* learner, booster = check_summary ~columns ~classes ~objective json in
      let* names = get_array "feature_names" learner in
      let* types = get_array "feature_types" learner in
      if names <> [] || types <> [] then
        failure "native feature metadata is outside the dense adapter contract"
      else
        let* model = field "model" booster in
        let* params = field "gbtree_model_param" model in
        let* parallel = get_int "num_parallel_tree" params in
        let* count = get_int "num_trees" params in
        let groups = if classes > 2 then classes else 1 in
        if
          parallel <> 1 || rounds > max_int / groups || count <> rounds * groups
        then
          failure "native tree count differs from the declared prediction range"
        else
          let* trees = get_array "trees" model in
          let* info = get_array "tree_info" model in
          let* pointers = get_array "iteration_indptr" model in
          if
            List.length trees <> count
            || List.length info <> count
            || List.length pointers <> rounds + 1
          then failure "native tree/iteration dimensions are inconsistent"
          else
            let rec check_pointers index = function
              | [] -> Ok ()
              | value :: rest ->
                  let* value = int value in
                  if value <> index * groups then
                    failure "native iteration range is inconsistent"
                  else check_pointers (index + 1) rest
            in
            let* () = check_pointers 0 pointers in
            let rec check_groups index = function
              | [] -> Ok ()
              | value :: rest ->
                  let* value = int value in
                  if value <> index mod groups then
                    failure "native output group is inconsistent"
                  else check_groups (index + 1) rest
            in
            let* () = check_groups 0 info in
            let rec check_trees = function
              | [] -> Ok ()
              | tree :: rest ->
                  let* params = field "tree_param" tree in
                  let* features = get_int "num_feature" params in
                  let* leaf_vector = get_int "size_leaf_vector" params in
                  let* splits = get_array "split_type" tree in
                  let* categories = get_array "categories" tree in
                  if features <> columns || leaf_vector <> 1 || categories <> []
                  then failure "native tree representation is unsupported"
                  else
                    let rec numeric = function
                      | [] -> Ok ()
                      | value :: tail ->
                          let* value = int value in
                          if value <> 0 then
                            failure "categorical splits are unsupported"
                          else numeric tail
                    in
                    let* () = numeric splits in
                    check_trees rest
            in
            check_trees trees

  let check_config ~columns ~classes ~objective ~threads json =
    let* learner, _ = check_summary ~columns ~classes ~objective json in
    let* generic = field "generic_param" learner in
    let* actual_threads = get_int "nthread" generic in
    let* device = get_string "device" generic in
    if actual_threads <> threads || device <> "cpu" then
      failure "native inference configuration differs from load options"
    else Ok ()
end
