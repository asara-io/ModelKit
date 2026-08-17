module Data_error = struct
  type t =
    | Negative_dimension of { name : string; value : int }
    | Ragged_matrix of {
        row : int;
        expected_columns : int;
        observed_columns : int;
      }
    | Length_mismatch of { name : string; expected : int; observed : int }
    | Index_out_of_bounds of { name : string; index : int; upper_bound : int }
    | Non_finite of { name : string; index : int; value : float }
    | Negative_weight of { index : int; value : float }
    | All_zero_weights
    | Empty_feature_name of { index : int }
    | Duplicate_feature_name of {
        name : string;
        first_index : int;
        duplicate_index : int;
      }

  let pp formatter = function
    | Negative_dimension { name; value } ->
        Format.fprintf formatter "%s must be non-negative, observed %d" name
          value
    | Ragged_matrix { row; expected_columns; observed_columns } ->
        Format.fprintf formatter
          "matrix row %d has %d columns; expected %d columns" row
          observed_columns expected_columns
    | Length_mismatch { name; expected; observed } ->
        Format.fprintf formatter "%s has length %d; expected %d" name observed
          expected
    | Index_out_of_bounds { name; index; upper_bound } ->
        Format.fprintf formatter
          "%s index %d is outside the half-open range [0, %d)" name index
          upper_bound
    | Non_finite { name; index; value } ->
        Format.fprintf formatter "%s contains non-finite value %g at index %d"
          name value index
    | Negative_weight { index; value } ->
        Format.fprintf formatter "sample weight %g at index %d is negative"
          value index
    | All_zero_weights ->
        Format.pp_print_string formatter
          "sample weights must contain at least one positive value"
    | Empty_feature_name { index } ->
        Format.fprintf formatter "feature name at index %d is empty" index
    | Duplicate_feature_name { name; first_index; duplicate_index } ->
        Format.fprintf formatter
          "feature name %S at index %d duplicates index %d" name duplicate_index
          first_index

  let to_string error = Format.asprintf "%a" pp error
end

let check_index ~name ~length index =
  if index < 0 || index >= length then
    invalid_arg
      (Format.asprintf "%a" Data_error.pp
         (Data_error.Index_out_of_bounds { name; index; upper_bound = length }))

module Vector = struct
  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  type t = bigarray

  let unsafe_init length f =
    let values =
      Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout length
    in
    for index = 0 to length - 1 do
      Bigarray.Array1.set values index (f index)
    done;
    values

  let create ~length value =
    if length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "vector length"; value = length })
    else Ok (unsafe_init length (fun _ -> value))

  let init ~length f =
    if length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "vector length"; value = length })
    else Ok (unsafe_init length f)

  let of_array values = unsafe_init (Array.length values) (Array.get values)

  let of_bigarray values =
    unsafe_init (Bigarray.Array1.dim values) (Bigarray.Array1.get values)

  let length = Bigarray.Array1.dim

  let get values index =
    check_index ~name:"vector" ~length:(length values) index;
    Bigarray.Array1.get values index

  let to_array values = Array.init (length values) (get values)
  let to_bigarray values = of_bigarray values
end

module Matrix = struct
  type bigarray =
    (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array2.t

  type t = bigarray

  let dimensions_error ~rows ~columns =
    if rows < 0 then
      Some
        (Data_error.Negative_dimension { name = "matrix rows"; value = rows })
    else if columns < 0 then
      Some
        (Data_error.Negative_dimension
           { name = "matrix columns"; value = columns })
    else None

  let unsafe_init ~rows ~columns f =
    let values =
      Bigarray.Array2.create Bigarray.float64 Bigarray.c_layout rows columns
    in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        Bigarray.Array2.set values row column (f row column)
      done
    done;
    values

  let init ~rows ~columns f =
    match dimensions_error ~rows ~columns with
    | Some error -> Error error
    | None -> Ok (unsafe_init ~rows ~columns f)

  let create ~rows ~columns value = init ~rows ~columns (fun _ _ -> value)

  let of_arrays values =
    let rows = Array.length values in
    let columns = if rows = 0 then 0 else Array.length values.(0) in
    let rec validate row =
      if row = rows then Ok ()
      else
        let observed_columns = Array.length values.(row) in
        if observed_columns <> columns then
          Error
            (Data_error.Ragged_matrix
               { row; expected_columns = columns; observed_columns })
        else validate (row + 1)
    in
    match validate 0 with
    | Error _ as error -> error
    | Ok () ->
        Ok
          (unsafe_init ~rows ~columns (fun row column -> values.(row).(column)))

  let of_bigarray values =
    let rows = Bigarray.Array2.dim1 values in
    let columns = Bigarray.Array2.dim2 values in
    unsafe_init ~rows ~columns (Bigarray.Array2.get values)

  let rows = Bigarray.Array2.dim1
  let columns = Bigarray.Array2.dim2
  let shape values = (rows values, columns values)

  let get values row column =
    check_index ~name:"matrix row" ~length:(rows values) row;
    check_index ~name:"matrix column" ~length:(columns values) column;
    Bigarray.Array2.get values row column

  let row values index =
    check_index ~name:"matrix row" ~length:(rows values) index;
    Bigarray.Array2.slice_left values index

  let to_arrays values =
    Array.init (rows values) (fun row ->
        Array.init (columns values) (fun column -> get values row column))

  let to_bigarray values = of_bigarray values
end

module Row_view = struct
  type t = { source_size : int; indices : int array }

  let create ~source_size indices =
    if source_size < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "row-view source size"; value = source_size })
    else
      let rec validate position =
        if position = Array.length indices then Ok ()
        else
          let index = indices.(position) in
          if index < 0 || index >= source_size then
            Error
              (Data_error.Index_out_of_bounds
                 { name = "row view"; index; upper_bound = source_size })
          else validate (position + 1)
      in
      match validate 0 with
      | Error _ as error -> error
      | Ok () -> Ok { source_size; indices = Array.copy indices }

  let all ~source_size =
    if source_size < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "row-view source size"; value = source_size })
    else Ok { source_size; indices = Array.init source_size Fun.id }

  let source_size view = view.source_size
  let length view = Array.length view.indices

  let get view position =
    check_index ~name:"row-view position" ~length:(length view) position;
    view.indices.(position)

  let indices view = Array.copy view.indices
end

let check_view_alignment ~name ~length view =
  let observed = Row_view.source_size view in
  if observed = length then Ok ()
  else Error (Data_error.Length_mismatch { name; expected = length; observed })

module Target = struct
  (* Closed markers make the GADT indices provably disjoint to exhaustiveness checking. *)
  type regression = [ `Regression ]
  type classification = [ `Classification ]

  type _ t =
    | Regression_values : Vector.t -> regression t
    | Classification_values : int array -> classification t

  let regression values =
    let rec validate index =
      if index = Vector.length values then Ok (Regression_values values)
      else
        let value = Vector.get values index in
        if Float.is_finite value then validate (index + 1)
        else
          Error
            (Data_error.Non_finite { name = "regression target"; index; value })
    in
    validate 0

  let classification values = Classification_values (Array.copy values)

  let length : type kind. kind t -> int = function
    | Regression_values values -> Vector.length values
    | Classification_values values -> Array.length values

  let regression_values : regression t -> Vector.t = function
    | Regression_values values -> values

  let classification_values : classification t -> int array = function
    | Classification_values values -> Array.copy values

  let classification_get : classification t -> int -> int = function
    | Classification_values values -> Array.get values

  let select : type kind. kind t -> Row_view.t -> (kind t, Data_error.t) result
      =
   fun target view ->
    match
      check_view_alignment ~name:"target row view" ~length:(length target) view
    with
    | Error _ as error -> error
    | Ok () -> (
        match target with
        | Regression_values values ->
            let selected =
              Vector.unsafe_init (Row_view.length view) (fun position ->
                  Vector.get values (Row_view.get view position))
            in
            Ok (Regression_values selected)
        | Classification_values values ->
            let selected =
              Array.init (Row_view.length view) (fun position ->
                  values.(Row_view.get view position))
            in
            Ok (Classification_values selected))
end

module Feature_name = struct
  type t = string

  let create name =
    if String.length name = 0 then
      Error (Data_error.Empty_feature_name { index = 0 })
    else Ok name

  let to_string name = name
end

module Feature_names = struct
  type t = Feature_name.t array

  let create ~expected_count names =
    if expected_count < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected feature count"; value = expected_count })
    else
      let observed = Array.length names in
      if observed <> expected_count then
        Error
          (Data_error.Length_mismatch
             { name = "feature names"; expected = expected_count; observed })
      else
        let seen = Hashtbl.create observed in
        let rec validate index =
          if index = observed then Ok (Array.copy names)
          else
            let name = names.(index) in
            if String.length name = 0 then
              Error (Data_error.Empty_feature_name { index })
            else
              match Hashtbl.find_opt seen name with
              | Some first_index ->
                  Error
                    (Data_error.Duplicate_feature_name
                       { name; first_index; duplicate_index = index })
              | None ->
                  Hashtbl.add seen name index;
                  validate (index + 1)
        in
        validate 0

  let length = Array.length

  let get names index =
    check_index ~name:"feature name" ~length:(length names) index;
    names.(index)

  let to_array names = Array.map Feature_name.to_string names
end

module Sample_weight = struct
  type t = Vector.t

  let create ~expected_length values =
    if expected_length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected sample-weight length"; value = expected_length })
    else
      let observed = Vector.length values in
      if observed <> expected_length then
        Error
          (Data_error.Length_mismatch
             { name = "sample weights"; expected = expected_length; observed })
      else
        let rec validate index has_positive =
          if index = observed then
            if has_positive then Ok values
            else Error Data_error.All_zero_weights
          else
            let value = Vector.get values index in
            if not (Float.is_finite value) then
              Error
                (Data_error.Non_finite { name = "sample weights"; index; value })
            else if value < 0.0 then
              Error (Data_error.Negative_weight { index; value })
            else validate (index + 1) (has_positive || value > 0.0)
        in
        validate 0 false

  let of_array ~expected_length values =
    create ~expected_length (Vector.of_array values)

  let length = Vector.length
  let get = Vector.get
  let to_vector weights = weights

  let select weights view =
    match
      check_view_alignment ~name:"sample-weight row view"
        ~length:(length weights) view
    with
    | Error _ as error -> error
    | Ok () ->
        let selected =
          Vector.unsafe_init (Row_view.length view) (fun position ->
              get weights (Row_view.get view position))
        in
        create ~expected_length:(Row_view.length view) selected
end

module Groups = struct
  type t = int array

  let create ~expected_length groups =
    if expected_length < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "expected group length"; value = expected_length })
    else
      let observed = Array.length groups in
      if observed = expected_length then Ok (Array.copy groups)
      else
        Error
          (Data_error.Length_mismatch
             { name = "groups"; expected = expected_length; observed })

  let length = Array.length

  let get groups index =
    check_index ~name:"group" ~length:(length groups) index;
    groups.(index)

  let to_array = Array.copy

  let distinct_count groups =
    let distinct = Hashtbl.create (length groups) in
    Array.iter (fun group -> Hashtbl.replace distinct group ()) groups;
    Hashtbl.length distinct

  let select groups view =
    match
      check_view_alignment ~name:"group row view" ~length:(length groups) view
    with
    | Error _ as error -> error
    | Ok () ->
        Ok
          (Array.init (Row_view.length view) (fun position ->
               get groups (Row_view.get view position)))
end

module Schema_fingerprint = struct
  type t = string

  let equal = String.equal
  let pp = Format.pp_print_string
  let to_string fingerprint = fingerprint
end

module Feature_schema = struct
  type t = { feature_count : int; names : Feature_names.t option }

  let anonymous ~feature_count =
    if feature_count < 0 then
      Error
        (Data_error.Negative_dimension
           { name = "feature-schema width"; value = feature_count })
    else Ok { feature_count; names = None }

  let named names =
    { feature_count = Feature_names.length names; names = Some names }

  let of_matrix ?names matrix =
    let feature_count = Matrix.columns matrix in
    match names with
    | None -> Ok { feature_count; names = None }
    | Some names ->
        let observed = Feature_names.length names in
        if observed = feature_count then
          Ok { feature_count; names = Some names }
        else
          Error
            (Data_error.Length_mismatch
               { name = "feature names"; expected = feature_count; observed })

  let feature_count schema = schema.feature_count
  let names schema = schema.names

  let equal_names left right =
    let length = Feature_names.length left in
    if length <> Feature_names.length right then false
    else
      let rec loop index =
        index = length
        || String.equal
             (Feature_name.to_string (Feature_names.get left index))
             (Feature_name.to_string (Feature_names.get right index))
           && loop (index + 1)
      in
      loop 0

  let equal left right =
    left.feature_count = right.feature_count
    &&
    match (left.names, right.names) with
    | None, None -> true
    | Some left, Some right -> equal_names left right
    | None, Some _ | Some _, None -> false

  let add_int64 buffer value =
    for shift = 7 downto 0 do
      Buffer.add_char buffer
        (Char.chr
           (Int64.to_int
              (Int64.logand (Int64.shift_right_logical value (shift * 8)) 0xffL)))
    done

  let fingerprint schema =
    let canonical = Buffer.create 64 in
    Buffer.add_string canonical "modelkit-feature-schema-v1";
    add_int64 canonical (Int64.of_int schema.feature_count);
    (match schema.names with
    | None -> Buffer.add_char canonical '\000'
    | Some names ->
        Buffer.add_char canonical '\001';
        for index = 0 to Feature_names.length names - 1 do
          let name = Feature_name.to_string (Feature_names.get names index) in
          add_int64 canonical (Int64.of_int (String.length name));
          Buffer.add_string canonical name
        done);
    "modelkit-schema-v1:"
    ^ Digest.to_hex (Digest.string (Buffer.contents canonical))

  let validate_matrix schema matrix =
    let observed = Matrix.columns matrix in
    if observed = schema.feature_count then Ok ()
    else
      Error
        (Data_error.Length_mismatch
           {
             name = "matrix feature count";
             expected = schema.feature_count;
             observed;
           })

  let pp formatter schema =
    match schema.names with
    | None ->
        Format.fprintf formatter "%d anonymous features" schema.feature_count
    | Some names ->
        Format.fprintf formatter "@[<hov 1>[";
        for index = 0 to Feature_names.length names - 1 do
          if index > 0 then Format.fprintf formatter ";@ ";
          Format.fprintf formatter "%S"
            (Feature_name.to_string (Feature_names.get names index))
        done;
        Format.fprintf formatter "]@]"

  let to_string schema = Format.asprintf "%a" pp schema
end

module Dataset = struct
  type finiteness = Require_finite | Allow_nan
  type data_access = Copy | View

  type access_report = {
    feature_access : data_access;
    target_access : data_access;
    sample_weight_access : data_access option;
    group_access : data_access option;
  }

  type 'kind t = {
    x : Matrix.t;
    y : 'kind Target.t;
    sample_weight : Sample_weight.t option;
    groups : Groups.t option;
    feature_schema : Feature_schema.t;
    schema_fingerprint : Schema_fingerprint.t;
    finiteness : finiteness;
    access_report : access_report;
  }

  type 'kind view = { dataset : 'kind t; rows : Row_view.t }

  let aligned_length ~name ~expected observed =
    if observed = expected then Ok ()
    else Error (Data_error.Length_mismatch { name; expected; observed })

  let validate_features finiteness x =
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let rec validate row column =
      if row = rows then Ok ()
      else if column = columns then validate (row + 1) 0
      else
        let value = Matrix.get x row column in
        let accepted =
          Float.is_finite value || (finiteness = Allow_nan && Float.is_nan value)
        in
        if accepted then validate row (column + 1)
        else
          Error
            (Data_error.Non_finite
               {
                 name =
                   Format.sprintf "features at row %d, column %d" row column;
                 index = (row * columns) + column;
                 value;
               })
    in
    validate 0 0

  let report access sample_weight groups : access_report =
    {
      feature_access = access;
      target_access = access;
      sample_weight_access = Option.map (fun _ -> access) sample_weight;
      group_access = Option.map (fun _ -> access) groups;
    }

  let ( let* ) = Result.bind

  let create_with_report ~access ~finiteness ?feature_names ?sample_weight
      ?groups ~x ~y () =
    let sample_count = Matrix.rows x in
    let* () = validate_features finiteness x in
    let* () =
      aligned_length ~name:"target" ~expected:sample_count (Target.length y)
    in
    let* () =
      match sample_weight with
      | Some weights ->
          aligned_length ~name:"sample weights" ~expected:sample_count
            (Sample_weight.length weights)
      | None -> Ok ()
    in
    let* () =
      match groups with
      | Some groups ->
          aligned_length ~name:"groups" ~expected:sample_count
            (Groups.length groups)
      | None -> Ok ()
    in
    let* feature_schema = Feature_schema.of_matrix ?names:feature_names x in
    Ok
      {
        x;
        y;
        sample_weight;
        groups;
        schema_fingerprint = Feature_schema.fingerprint feature_schema;
        feature_schema;
        finiteness;
        access_report = report access sample_weight groups;
      }

  let create ~finiteness ?feature_names ?sample_weight ?groups ~x ~y () =
    create_with_report ~access:View ~finiteness ?feature_names ?sample_weight
      ?groups ~x ~y ()

  let sample_count dataset = Matrix.rows dataset.x
  let feature_count dataset = Matrix.columns dataset.x
  let features dataset = dataset.x
  let target dataset = dataset.y
  let sample_weight dataset = dataset.sample_weight
  let groups dataset = dataset.groups
  let feature_schema dataset = dataset.feature_schema
  let schema_fingerprint dataset = dataset.schema_fingerprint
  let finiteness dataset = dataset.finiteness
  let access_report dataset = dataset.access_report

  let all dataset =
    let rows =
      Result.get_ok (Row_view.all ~source_size:(sample_count dataset))
    in
    { dataset; rows }

  let view dataset rows =
    match
      aligned_length ~name:"dataset row view" ~expected:(sample_count dataset)
        (Row_view.source_size rows)
    with
    | Error _ as error -> error
    | Ok () -> Ok { dataset; rows }

  let view_sample_count view = Row_view.length view.rows
  let row_view view = view.rows

  let view_access_report view =
    report View view.dataset.sample_weight view.dataset.groups

  let source_row view position = Row_view.get view.rows position

  let feature view ~row ~column =
    check_index ~name:"dataset feature column"
      ~length:(feature_count view.dataset)
      column;
    Matrix.get view.dataset.x (source_row view row) column

  let regression_target view position =
    Vector.get
      (Target.regression_values view.dataset.y)
      (source_row view position)

  let classification_target view position =
    Target.classification_get view.dataset.y (source_row view position)

  let sample_weight_value view position =
    let source = source_row view position in
    Option.map
      (fun weights -> Sample_weight.get weights source)
      view.dataset.sample_weight

  let group view position =
    let source = source_row view position in
    Option.map (fun groups -> Groups.get groups source) view.dataset.groups

  let materialize : type kind. kind view -> (kind t, Data_error.t) result =
   fun view ->
    let dataset = view.dataset in
    let rows = view.rows in
    let x =
      Matrix.unsafe_init ~rows:(Row_view.length rows)
        ~columns:(feature_count dataset) (fun row column ->
          Matrix.get dataset.x (Row_view.get rows row) column)
    in
    let* y = Target.select dataset.y rows in
    let* sample_weight =
      match dataset.sample_weight with
      | None -> Ok None
      | Some weights ->
          Result.map Option.some (Sample_weight.select weights rows)
    in
    let* groups =
      match dataset.groups with
      | None -> Ok None
      | Some groups -> Result.map Option.some (Groups.select groups rows)
    in
    create_with_report ~access:Copy ~finiteness:dataset.finiteness
      ?feature_names:(Feature_schema.names dataset.feature_schema)
      ?sample_weight ?groups ~x ~y ()
end

module Error = struct
  type context =
    | Stage of string
    | Fold of int
    | Candidate of int
    | Feature of Feature_name.t

  type kind =
    | Data of Data_error.t
    | Shape_mismatch of {
        name : string;
        expected : int list;
        observed : int list;
      }
    | Feature_schema_mismatch of {
        expected : Feature_schema.t;
        observed : Feature_schema.t;
      }
    | Validation of { name : string; reason : string }
    | Numerical of { operation : string; reason : string }
    | Convergence of { algorithm : string; reason : string }
    | Compatibility of { component : string; reason : string }
    | Artifact of { operation : string; reason : string }
    | Cancelled

  type t = { kind : kind; context : context list; remediation : string }

  let make ?(context = []) ~remediation kind = { kind; context; remediation }

  let of_data_error ?context ~remediation error =
    make ?context ~remediation (Data error)

  let kind error = error.kind
  let context error = error.context
  let remediation error = error.remediation
  let with_context outer error = { error with context = outer :: error.context }

  let pp_dimensions formatter dimensions =
    Format.fprintf formatter "(";
    List.iteri
      (fun index dimension ->
        if index > 0 then Format.fprintf formatter ", ";
        Format.pp_print_int formatter dimension)
      dimensions;
    Format.fprintf formatter ")"

  let pp_kind formatter = function
    | Data error -> Data_error.pp formatter error
    | Shape_mismatch { name; expected; observed } ->
        Format.fprintf formatter "%s has shape %a; expected %a" name
          pp_dimensions observed pp_dimensions expected
    | Feature_schema_mismatch { expected; observed } ->
        Format.fprintf formatter
          "feature schema %a does not match fitted schema %a" Feature_schema.pp
          observed Feature_schema.pp expected
    | Validation { name; reason } ->
        Format.fprintf formatter "%s is invalid: %s" name reason
    | Numerical { operation; reason } ->
        Format.fprintf formatter "%s failed numerically: %s" operation reason
    | Convergence { algorithm; reason } ->
        Format.fprintf formatter "%s did not converge: %s" algorithm reason
    | Compatibility { component; reason } ->
        Format.fprintf formatter "%s is incompatible: %s" component reason
    | Artifact { operation; reason } ->
        Format.fprintf formatter "artifact %s failed: %s" operation reason
    | Cancelled -> Format.pp_print_string formatter "operation was cancelled"

  let pp_context formatter = function
    | Stage stage -> Format.fprintf formatter "stage %S" stage
    | Fold fold -> Format.fprintf formatter "fold %d" fold
    | Candidate candidate -> Format.fprintf formatter "candidate %d" candidate
    | Feature feature ->
        Format.fprintf formatter "feature %S" (Feature_name.to_string feature)

  let pp formatter error =
    Format.fprintf formatter "%a" pp_kind error.kind;
    (match error.context with
    | [] -> ()
    | context ->
        Format.fprintf formatter " (@[<hov>";
        List.iteri
          (fun index item ->
            if index > 0 then Format.fprintf formatter ",@ ";
            pp_context formatter item)
          context;
        Format.fprintf formatter "@])");
    Format.fprintf formatter ". Remediation: %s" error.remediation

  let to_string error = Format.asprintf "%a" pp error
end
