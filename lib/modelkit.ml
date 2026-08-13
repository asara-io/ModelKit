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

module type SPECIFICATION = sig
  type t
  type params

  val clone : t -> t
  val params : t -> params
end

module type ESTIMATOR = sig
  include SPECIFICATION

  type target
  type prediction
  type fitted
  type rng

  val fit :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:rng ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:target ->
    unit ->
    (fitted, Error.t) result

  val predict :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (prediction, Error.t) result

  val fitted_params : fitted -> params
  val feature_schema : fitted -> Feature_schema.t
end

module type CLASSIFIER = sig
  include
    ESTIMATOR
      with type target = Target.classification Target.t
       and type prediction = Target.classification Target.t
end

module type REGRESSOR = sig
  include
    ESTIMATOR
      with type target = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

module type TRANSFORMER = sig
  include SPECIFICATION

  type target
  type fitted
  type rng

  val fit :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:rng ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:target option ->
    unit ->
    (fitted, Error.t) result

  val transform :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t, Error.t) result

  val fitted_params : fitted -> params
  val input_schema : fitted -> Feature_schema.t
  val output_schema : fitted -> Feature_schema.t
end

module type SCORER = sig
  include SPECIFICATION

  type truth
  type prediction

  val name : t -> string

  val score :
    t ->
    ?sample_weight:Sample_weight.t ->
    truth:truth ->
    prediction:prediction ->
    unit ->
    (float, Error.t) result
end

module type SPLITTER = sig
  include SPECIFICATION

  type target
  type rng

  val split :
    t ->
    rng:rng ->
    ?groups:Groups.t ->
    x:Matrix.t ->
    y:target option ->
    unit ->
    ((Row_view.t * Row_view.t) array, Error.t) result
end

module type EXECUTION = sig
  type t

  val concurrency : t -> int

  val map :
    t ->
    f:(index:int -> 'input -> ('output, 'error) result) ->
    'input array ->
    ('output array, 'error) result
end

module type RNG = sig
  type seed
  type t

  val create : seed -> t
  val derive : seed -> operation:string -> index:int -> seed
  val next_int64 : t -> int64 * t
  val next_float : t -> float * t
end

module type NUMERICAL_BACKEND = sig
  val name : string
  val sum : Vector.t -> float
  val dot : Vector.t -> Vector.t -> (float, Error.t) result
  val matrix_vector_product : Matrix.t -> Vector.t -> (Vector.t, Error.t) result

  val transposed_matrix_vector_product :
    Matrix.t -> Vector.t -> (Vector.t, Error.t) result
end

module Seed = struct
  type t = int64

  let of_int = Int64.of_int
  let of_int64 seed = seed
  let to_int64 seed = seed
  let equal = Int64.equal

  (* FNV-1a. *)
  let mix value =
    let value =
      Int64.mul
        (Int64.logxor value (Int64.shift_right_logical value 30))
        (-4658895280553007687L)
    in
    let value =
      Int64.mul
        (Int64.logxor value (Int64.shift_right_logical value 27))
        (-7723592293110705685L)
    in
    Int64.logxor value (Int64.shift_right_logical value 31)

  let hash_operation operation =
    let hash = ref (-3750763034362895579L) in
    for index = 0 to String.length operation - 1 do
      hash :=
        Int64.mul
          (Int64.logxor !hash (Int64.of_int (Char.code operation.[index])))
          1099511628211L
    done;
    mix !hash

  let derive seed ~operation ~index =
    let operation_hash = hash_operation operation in
    let index_hash = mix (Int64.of_int index) in
    mix (Int64.logxor seed (Int64.logxor operation_hash index_hash))

  let pp formatter seed = Format.fprintf formatter "0x%016Lx" seed
  let to_string seed = Format.asprintf "%a" pp seed
end

module Rng = struct
  type seed = Seed.t
  type t = int64

  let create = Seed.to_int64
  let derive = Seed.derive
  let to_seed = Seed.of_int64

  let next_int64 state =
    let successor = Int64.add state (-7046029254386353131L) in
    (Seed.mix successor, successor)

  let next_float state =
    let bits, successor = next_int64 state in
    let significand = Int64.shift_right_logical bits 11 in
    (Int64.to_float significand /. 9_007_199_254_740_992.0, successor)
end

module Sequential_execution = struct
  type t = unit

  let default = ()
  let concurrency () = 1

  let map () ~f inputs =
    let rec loop index outputs =
      if index = Array.length inputs then Ok (Array.of_list (List.rev outputs))
      else
        match f ~index inputs.(index) with
        | Error _ as error -> error
        | Ok output -> loop (index + 1) (output :: outputs)
    in
    loop 0 []
end

module Reference_backend = struct
  let name = "reference"

  module Accumulator = struct
    type t = {
      mutable total : float;
      mutable correction : float;
      mutable positive_infinity : bool;
      mutable negative_infinity : bool;
      mutable nan : bool;
    }

    let create () =
      {
        total = 0.0;
        correction = 0.0;
        positive_infinity = false;
        negative_infinity = false;
        nan = false;
      }

    let record_non_finite accumulator value =
      if Float.is_nan value then accumulator.nan <- true
      else if value > 0.0 then accumulator.positive_infinity <- true
      else accumulator.negative_infinity <- true

    let add accumulator value =
      if not (Float.is_finite value) then record_non_finite accumulator value
      else if
        not
          (accumulator.nan || accumulator.positive_infinity
         || accumulator.negative_infinity)
      then (
        let next = accumulator.total +. value in
        if not (Float.is_finite next) then record_non_finite accumulator next
        else
          let correction =
            if Float.abs accumulator.total >= Float.abs value then
              accumulator.total -. next +. value
            else value -. next +. accumulator.total
          in
          let corrected = accumulator.correction +. correction in
          if Float.is_finite corrected then accumulator.correction <- corrected
          else record_non_finite accumulator corrected;
          accumulator.total <- next)

    let value accumulator =
      if
        accumulator.nan
        || (accumulator.positive_infinity && accumulator.negative_infinity)
      then Float.nan
      else if accumulator.positive_infinity then Float.infinity
      else if accumulator.negative_infinity then Float.neg_infinity
      else accumulator.total +. accumulator.correction
  end

  let length_error ~name ~expected ~observed =
    Error.of_data_error ~remediation:"provide operands with aligned dimensions"
      (Data_error.Length_mismatch { name; expected; observed })

  let vector_result = function
    | Ok vector -> Ok vector
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"provide a representable output dimension" error)

  let sum vector =
    let accumulator = Accumulator.create () in
    for index = 0 to Vector.length vector - 1 do
      Accumulator.add accumulator (Vector.get vector index)
    done;
    Accumulator.value accumulator

  let dot left right =
    let expected = Vector.length left in
    let observed = Vector.length right in
    if expected <> observed then
      Error (length_error ~name:"dot-product right operand" ~expected ~observed)
    else
      let accumulator = Accumulator.create () in
      for index = 0 to expected - 1 do
        Accumulator.add accumulator
          (Vector.get left index *. Vector.get right index)
      done;
      Ok (Accumulator.value accumulator)

  let matrix_vector_product matrix vector =
    let expected = Matrix.columns matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error (length_error ~name:"matrix-vector operand" ~expected ~observed)
    else
      vector_result
        (Vector.init ~length:(Matrix.rows matrix) (fun row ->
             let accumulator = Accumulator.create () in
             for column = 0 to expected - 1 do
               Accumulator.add accumulator
                 (Matrix.get matrix row column *. Vector.get vector column)
             done;
             Accumulator.value accumulator))

  let transposed_matrix_vector_product matrix vector =
    let expected = Matrix.rows matrix in
    let observed = Vector.length vector in
    if expected <> observed then
      Error
        (length_error ~name:"transposed-matrix-vector operand" ~expected
           ~observed)
    else
      vector_result
        (Vector.init ~length:(Matrix.columns matrix) (fun column ->
             let accumulator = Accumulator.create () in
             for row = 0 to expected - 1 do
               Accumulator.add accumulator
                 (Matrix.get matrix row column *. Vector.get vector row)
             done;
             Accumulator.value accumulator))
end

module Preprocessing_internal = struct
  let ( let* ) = Result.bind
  let data_error ~remediation error = Error.of_data_error ~remediation error

  let validate_width feature_schema x =
    match Feature_schema.validate_matrix feature_schema x with
    | Ok () -> Ok ()
    | Error error ->
        Error
          (data_error ~remediation:"provide features matching the schema width"
             error)

  let validate_schema ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide features with the fitted schema"
           (Error.Feature_schema_mismatch { expected; observed }))

  let reject_sample_weight transformer = function
    | None -> Ok ()
    | Some _ ->
        Error
          (Error.make
             ~remediation:
               "route sample weights only to components that declare support"
             (Error.Validation
                {
                  name = transformer ^ " sample_weight";
                  reason = "sample weights are not supported";
                }))

  let non_finite_error ~operation ~columns row column value =
    data_error
      ~remediation:("impute or remove non-finite values before " ^ operation)
      (Data_error.Non_finite
         {
           name =
             Format.sprintf "%s input at row %d, column %d" operation row column;
           index = (row * columns) + column;
           value;
         })

  let validate_values ~operation ~allow_nan x =
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let rec loop row column =
      if row = rows then Ok ()
      else if column = columns then loop (row + 1) 0
      else
        let value = Matrix.get x row column in
        if Float.is_finite value || (allow_nan && Float.is_nan value) then
          loop row (column + 1)
        else Error (non_finite_error ~operation ~columns row column value)
    in
    loop 0 0

  let validate_fit_input ~operation ~allow_nan feature_schema x =
    let* () = validate_width feature_schema x in
    validate_values ~operation ~allow_nan x

  let validate_transform_input ~operation ~allow_nan ~expected_schema
      feature_schema x =
    let* () = validate_schema ~expected:expected_schema feature_schema in
    validate_fit_input ~operation ~allow_nan feature_schema x

  let feature_context schema column =
    match Feature_schema.names schema with
    | None -> []
    | Some names -> [ Error.Feature (Feature_names.get names column) ]

  let no_observations ~operation schema column =
    Error.make
      ~context:(feature_context schema column)
      ~remediation:
        "provide at least one observed training value for the feature"
      (Error.Validation
         {
           name = operation ^ " training feature";
           reason = "contains no observed values";
         })

  let numerical_error ~operation schema column reason =
    Error.make
      ~context:(feature_context schema column)
      ~remediation:"rescale the feature or remove numerically extreme values"
      (Error.Numerical { operation; reason })

  let column_mean ~operation ~skip_nan schema x column =
    let rows = Matrix.rows x in
    let count = ref 0 in
    let maximum = ref 0.0 in
    for row = 0 to rows - 1 do
      let value = Matrix.get x row column in
      if not (skip_nan && Float.is_nan value) then (
        incr count;
        maximum := Float.max !maximum (Float.abs value))
    done;
    if !count = 0 then Error (no_observations ~operation schema column)
    else if !maximum = 0.0 then Ok 0.0
    else
      let mean = ref 0.0 in
      let seen = ref 0 in
      for row = 0 to rows - 1 do
        let value = Matrix.get x row column in
        if not (skip_nan && Float.is_nan value) then (
          incr seen;
          let count = Float.of_int !seen in
          let normalized = value /. !maximum in
          mean := ((!mean *. (count -. 1.0)) +. normalized) /. count)
      done;
      let mean = !mean *. !maximum in
      if Float.is_finite mean then Ok mean
      else
        Error
          (numerical_error ~operation schema column
             "the fitted mean is not finite")

  let column_moments ~operation schema x column =
    let rows = Matrix.rows x in
    if rows = 0 then Error (no_observations ~operation schema column)
    else
      let maximum = ref 0.0 in
      for row = 0 to rows - 1 do
        maximum := Float.max !maximum (Float.abs (Matrix.get x row column))
      done;
      if !maximum = 0.0 then Ok (0.0, 0.0)
      else
        let mean = ref 0.0 in
        let m2 = ref 0.0 in
        for row = 0 to rows - 1 do
          let count = Float.of_int (row + 1) in
          let value = Matrix.get x row column /. !maximum in
          let delta = value -. !mean in
          mean := !mean +. (delta /. count);
          let delta_after_update = value -. !mean in
          m2 := !m2 +. (delta *. delta_after_update)
        done;
        let mean = !mean *. !maximum in
        let normalized_variance = Float.max 0.0 (!m2 /. Float.of_int rows) in
        let standard_deviation = Float.sqrt normalized_variance *. !maximum in
        let variance = standard_deviation *. standard_deviation in
        if Float.is_finite mean && Float.is_finite variance then
          Ok (mean, variance)
        else
          Error
            (numerical_error ~operation schema column
               "the fitted mean or variance is not finite")

  let matrix ~rows ~columns f =
    match Matrix.init ~rows ~columns f with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (data_error ~remediation:"provide representable matrix dimensions"
             error)

  let subset_schema schema selected =
    match Feature_schema.names schema with
    | None -> (
        match
          Feature_schema.anonymous ~feature_count:(Array.length selected)
        with
        | Ok schema -> Ok schema
        | Error error ->
            Error
              (data_error
                 ~remediation:"provide a representable selected feature count"
                 error))
    | Some names -> (
        let selected_names =
          Array.map
            (fun column ->
              Feature_name.to_string (Feature_names.get names column))
            selected
        in
        match
          Feature_names.create ~expected_count:(Array.length selected)
            selected_names
        with
        | Ok names -> Ok (Feature_schema.named names)
        | Error error ->
            Error
              (data_error ~remediation:"provide valid selected feature names"
                 error))
end

module Simple_imputer = struct
  type strategy = Mean | Median | Constant of float
  type params = { strategy : strategy }
  type t = params

  type fitted = {
    params : params;
    statistics : Vector.t;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let mean () = { strategy = Mean }
  let median () = { strategy = Median }

  let constant value =
    if Float.is_finite value then Ok { strategy = Constant value }
    else
      Error
        (Error.make ~remediation:"choose a finite imputation constant"
           (Error.Validation
              { name = "simple imputer constant"; reason = "must be finite" }))

  let clone specification = specification
  let params specification = specification

  let median_statistic schema x column =
    let observed = ref [] in
    for row = 0 to Matrix.rows x - 1 do
      let value = Matrix.get x row column in
      if not (Float.is_nan value) then observed := value :: !observed
    done;
    match !observed with
    | [] ->
        Error
          (Preprocessing_internal.no_observations ~operation:"median imputation"
             schema column)
    | observed ->
        let values = Array.of_list observed in
        Array.sort Float.compare values;
        let length = Array.length values in
        if length mod 2 = 1 then Ok values.(length / 2)
        else
          let left = values.((length / 2) - 1) in
          let right = values.(length / 2) in
          if left < 0.0 = (right < 0.0) then
            Ok (left +. ((right -. left) /. 2.0))
          else Ok ((left /. 2.0) +. (right /. 2.0))

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Preprocessing_internal in
    let* () = reject_sample_weight "simple imputer" sample_weight in
    let* () =
      validate_fit_input ~operation:"simple imputer" ~allow_nan:true
        feature_schema x
    in
    let columns = Matrix.columns x in
    let statistics = Array.make columns 0.0 in
    let rec fit_column column =
      if column = columns then Ok ()
      else
        let statistic =
          match specification.strategy with
          | Constant value -> Ok value
          | Mean ->
              column_mean ~operation:"mean imputation" ~skip_nan:true
                feature_schema x column
          | Median -> median_statistic feature_schema x column
        in
        let* statistic = statistic in
        statistics.(column) <- statistic;
        fit_column (column + 1)
    in
    let* () = fit_column 0 in
    Ok
      {
        params = specification;
        statistics = Vector.of_array statistics;
        schema = feature_schema;
      }

  let transform fitted ~feature_schema ~x =
    let open Preprocessing_internal in
    let* () =
      validate_transform_input ~operation:"simple imputer" ~allow_nan:true
        ~expected_schema:fitted.schema feature_schema x
    in
    matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x) (fun row column ->
        let value = Matrix.get x row column in
        if Float.is_nan value then Vector.get fitted.statistics column
        else value)

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let statistics fitted = fitted.statistics
end

module Standard_scaler = struct
  type params = { with_mean : bool; with_std : bool }
  type t = params

  type fitted = {
    params : params;
    mean : Vector.t;
    variance : Vector.t;
    scale : Vector.t;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(with_mean = true) ?(with_std = true) () = { with_mean; with_std }
  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Preprocessing_internal in
    let* () = reject_sample_weight "standard scaler" sample_weight in
    let* () =
      validate_fit_input ~operation:"standard scaler" ~allow_nan:false
        feature_schema x
    in
    let columns = Matrix.columns x in
    let means = Array.make columns 0.0 in
    let variances = Array.make columns 0.0 in
    let scales = Array.make columns 1.0 in
    let rec fit_column column =
      if column = columns then Ok ()
      else
        let* mean, variance =
          column_moments ~operation:"standard scaler fit" feature_schema x
            column
        in
        means.(column) <- mean;
        variances.(column) <- variance;
        if variance > 0.0 then scales.(column) <- Float.sqrt variance;
        fit_column (column + 1)
    in
    let* () = fit_column 0 in
    Ok
      {
        params = specification;
        mean = Vector.of_array means;
        variance = Vector.of_array variances;
        scale = Vector.of_array scales;
        schema = feature_schema;
      }

  let transform fitted ~feature_schema ~x =
    let open Preprocessing_internal in
    let* () =
      validate_transform_input ~operation:"standard scaler" ~allow_nan:false
        ~expected_schema:fitted.schema feature_schema x
    in
    let* transformed =
      matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
        (fun row column ->
          let value = Matrix.get x row column in
          let centered =
            if fitted.params.with_mean then
              value -. Vector.get fitted.mean column
            else value
          in
          if fitted.params.with_std then
            centered /. Vector.get fitted.scale column
          else centered)
    in
    let* () =
      validate_values ~operation:"standard scaler output" ~allow_nan:false
        transformed
    in
    Ok transformed

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let mean fitted = fitted.mean
  let variance fitted = fitted.variance
  let scale fitted = fitted.scale
end

module Variance_threshold = struct
  type params = { threshold : float }
  type t = params

  type fitted = {
    params : params;
    variances : Vector.t;
    selected : int array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(threshold = 0.0) () =
    if Float.is_finite threshold && threshold >= 0.0 then Ok { threshold }
    else
      Error
        (Error.make
           ~remediation:"choose a finite, non-negative variance threshold"
           (Error.Validation
              {
                name = "variance threshold";
                reason = "must be finite and non-negative";
              }))

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Preprocessing_internal in
    let* () = reject_sample_weight "variance threshold" sample_weight in
    let* () =
      validate_fit_input ~operation:"variance threshold" ~allow_nan:false
        feature_schema x
    in
    let columns = Matrix.columns x in
    let variances = Array.make columns 0.0 in
    let selected = ref [] in
    let rec fit_column column =
      if column = columns then Ok ()
      else
        let* _, variance =
          column_moments ~operation:"variance threshold fit" feature_schema x
            column
        in
        variances.(column) <- variance;
        if variance > specification.threshold then
          selected := column :: !selected;
        fit_column (column + 1)
    in
    let* () = fit_column 0 in
    let selected = Array.of_list (List.rev !selected) in
    if Array.length selected = 0 then
      Error
        (Error.make
           ~remediation:"lower the threshold or provide varying features"
           (Error.Validation
              {
                name = "variance threshold selection";
                reason = "no feature exceeds the threshold";
              }))
    else
      let* output_schema = subset_schema feature_schema selected in
      Ok
        {
          params = specification;
          variances = Vector.of_array variances;
          selected;
          input_schema = feature_schema;
          output_schema;
        }

  let transform fitted ~feature_schema ~x =
    let open Preprocessing_internal in
    let* () =
      validate_transform_input ~operation:"variance threshold" ~allow_nan:false
        ~expected_schema:fitted.input_schema feature_schema x
    in
    matrix ~rows:(Matrix.rows x) ~columns:(Array.length fitted.selected)
      (fun row output_column ->
        Matrix.get x row fitted.selected.(output_column))

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema
  let variances fitted = fitted.variances
  let selected_indices fitted = Array.copy fitted.selected
end

module Pipeline = struct
  type capabilities = { decision_function : bool; predict_proba : bool }

  type fitted_transformer = {
    stage_name : string;
    transform_input_schema : Feature_schema.t;
    transform_output_schema : Feature_schema.t;
    apply_transform :
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result;
  }

  type transformer = {
    transformer_name : string;
    fit_transform :
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (fitted_transformer * Matrix.t * Feature_schema.t, Error.t) result;
  }

  type builder = {
    reversed_transformers : transformer list;
    names : string list;
  }

  type 'prediction fitted_estimator = {
    terminal_name : string;
    terminal_predict :
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      ('prediction, Error.t) result;
    terminal_decision_function :
      (feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Vector.t, Error.t) result)
      option;
    terminal_predict_proba :
      (feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      (Matrix.t, Error.t) result)
      option;
  }

  type ('target, 'prediction) estimator = {
    estimator_name : string;
    estimator_capabilities : capabilities;
    fit_estimator :
      ?sample_weight:Sample_weight.t ->
      rng:Rng.t ->
      feature_schema:Feature_schema.t ->
      x:Matrix.t ->
      y:'target ->
      unit ->
      ('prediction fitted_estimator, Error.t) result;
  }

  type ('target, 'prediction) t = {
    transformers : transformer array;
    estimator : ('target, 'prediction) estimator;
  }

  type ('target, 'prediction) fitted = {
    fitted_transformers : fitted_transformer array;
    fitted_estimator : 'prediction fitted_estimator;
    pipeline_input_schema : Feature_schema.t;
    pipeline_output_schema : Feature_schema.t;
  }

  let ( let* ) = Result.bind

  let validation_error ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let validate_name name =
    if String.length (String.trim name) = 0 then
      Error
        (validation_error ~name:"pipeline stage name"
           ~reason:"must not be blank"
           ~remediation:"choose a non-empty stage name")
    else Ok ()

  let duplicate_name name =
    validation_error ~name:"pipeline stage names"
      ~reason:(Format.sprintf "stage name %S is duplicated" name)
      ~remediation:"choose a unique name for every pipeline stage"

  let validate_schema ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide features with the fitted schema"
           (Error.Feature_schema_mismatch { expected; observed }))

  let validate_matrix schema matrix =
    match Feature_schema.validate_matrix schema matrix with
    | Ok () -> Ok ()
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"provide a matrix matching the current stage schema"
             error)

  let validate_transform_output ~input ~output_schema output =
    let expected_rows = Matrix.rows input in
    let observed_rows = Matrix.rows output in
    if expected_rows <> observed_rows then
      Error
        (Error.make
           ~remediation:"preserve sample count in every transformer stage"
           (Error.Shape_mismatch
              {
                name = "transformer output rows";
                expected = [ expected_rows ];
                observed = [ observed_rows ];
              }))
    else validate_matrix output_schema output

  let validate_sample_weight x = function
    | None -> Ok ()
    | Some weights ->
        let expected = Matrix.rows x in
        let observed = Sample_weight.length weights in
        if expected = observed then Ok ()
        else
          Error
            (Error.of_data_error
               ~remediation:"provide one sample weight per training row"
               (Data_error.Length_mismatch
                  { name = "pipeline sample weights"; expected; observed }))

  let validate_fitted_schema ~stage ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make
           ~remediation:
             "fix the component so its fitted schema matches its fit input"
           (Error.Compatibility
              {
                component = stage;
                reason = "component returned an inconsistent fitted schema";
              }))

  let with_stage name result =
    Result.map_error (Error.with_context (Error.Stage name)) result

  let transformer (type specification fitted) ~name
      (module Transformer : TRANSFORMER
        with type t = specification
         and type target = unit
         and type fitted = fitted
         and type rng = Rng.t) (specification : specification) =
    let* () = validate_name name in
    let fit_transform ~rng ~feature_schema ~x =
      let* fitted =
        Transformer.fit specification ~rng ~feature_schema ~x ~y:None ()
      in
      let* () =
        validate_fitted_schema ~stage:name ~expected:feature_schema
          (Transformer.input_schema fitted)
      in
      let output_schema = Transformer.output_schema fitted in
      let* transformed = Transformer.transform fitted ~feature_schema ~x in
      let* () = validate_transform_output ~input:x ~output_schema transformed in
      let fitted_transformer : fitted_transformer =
        {
          stage_name = name;
          transform_input_schema = feature_schema;
          transform_output_schema = output_schema;
          apply_transform = Transformer.transform fitted;
        }
      in
      Ok (fitted_transformer, transformed, output_schema)
    in
    Ok { transformer_name = name; fit_transform }

  let estimator (type specification target prediction fitted) ~name
      (module Estimator : ESTIMATOR
        with type t = specification
         and type target = target
         and type prediction = prediction
         and type fitted = fitted
         and type rng = Rng.t) ?decision_function ?predict_proba
      (specification : specification) =
    let* () = validate_name name in
    let capabilities : capabilities =
      {
        decision_function = Option.is_some decision_function;
        predict_proba = Option.is_some predict_proba;
      }
    in
    let fit ?sample_weight ~rng ~feature_schema ~x ~y () =
      let* fitted =
        Estimator.fit specification ?sample_weight ~rng ~feature_schema ~x ~y ()
      in
      let fitted_schema = Estimator.feature_schema fitted in
      let* () =
        validate_fitted_schema ~stage:name ~expected:feature_schema
          fitted_schema
      in
      let fitted_estimator : prediction fitted_estimator =
        {
          terminal_name = name;
          terminal_predict = Estimator.predict fitted;
          terminal_decision_function =
            Option.map (fun dispatch -> dispatch fitted) decision_function;
          terminal_predict_proba =
            Option.map (fun dispatch -> dispatch fitted) predict_proba;
        }
      in
      Ok fitted_estimator
    in
    Ok
      {
        estimator_name = name;
        estimator_capabilities = capabilities;
        fit_estimator = fit;
      }

  let empty = { reversed_transformers = []; names = [] }

  let add_transformer (builder : builder) (transformer : transformer) =
    if List.exists (String.equal transformer.transformer_name) builder.names
    then Error (duplicate_name transformer.transformer_name)
    else
      Ok
        {
          reversed_transformers = transformer :: builder.reversed_transformers;
          names = transformer.transformer_name :: builder.names;
        }

  let set_estimator (builder : builder)
      (estimator : ('target, 'prediction) estimator) =
    if List.exists (String.equal estimator.estimator_name) builder.names then
      Error (duplicate_name estimator.estimator_name)
    else
      Ok
        {
          transformers = Array.of_list (List.rev builder.reversed_transformers);
          estimator;
        }

  let clone specification = specification

  let transformer_names specification =
    Array.map
      (fun transformer -> transformer.transformer_name)
      specification.transformers

  let estimator_name specification = specification.estimator.estimator_name

  let capabilities specification =
    specification.estimator.estimator_capabilities

  let child_rng root ~kind ~name ~index =
    let seed =
      Seed.derive (Rng.to_seed root) ~operation:(kind ^ ":" ^ name) ~index
    in
    Rng.create seed

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = validate_matrix feature_schema x in
    let* () = validate_sample_weight x sample_weight in
    let rec fit_transformers index current_schema current_x reversed_fitted =
      if index = Array.length specification.transformers then
        Ok (Array.of_list (List.rev reversed_fitted), current_schema, current_x)
      else
        let transformer = specification.transformers.(index) in
        let stage_rng =
          child_rng rng ~kind:"pipeline-transformer"
            ~name:transformer.transformer_name ~index
        in
        let* fitted, transformed, output_schema =
          with_stage transformer.transformer_name
            (transformer.fit_transform ~rng:stage_rng
               ~feature_schema:current_schema ~x:current_x)
        in
        fit_transformers (index + 1) output_schema transformed
          (fitted :: reversed_fitted)
    in
    let* fitted_transformers, output_schema, transformed_x =
      fit_transformers 0 feature_schema x []
    in
    let estimator_index = Array.length specification.transformers in
    let estimator_rng =
      child_rng rng ~kind:"pipeline-estimator"
        ~name:specification.estimator.estimator_name ~index:estimator_index
    in
    let* fitted_estimator =
      with_stage specification.estimator.estimator_name
        (specification.estimator.fit_estimator ?sample_weight ~rng:estimator_rng
           ~feature_schema:output_schema ~x:transformed_x ~y ())
    in
    Ok
      {
        fitted_transformers;
        fitted_estimator;
        pipeline_input_schema = feature_schema;
        pipeline_output_schema = output_schema;
      }

  let transform fitted ~feature_schema ~x =
    let* () =
      validate_schema ~expected:fitted.pipeline_input_schema feature_schema
    in
    let* () = validate_matrix feature_schema x in
    let rec apply index current_schema current_x =
      if index = Array.length fitted.fitted_transformers then Ok current_x
      else
        let transformer = fitted.fitted_transformers.(index) in
        let* () =
          with_stage transformer.stage_name
            (validate_schema ~expected:transformer.transform_input_schema
               current_schema)
        in
        let* transformed =
          with_stage transformer.stage_name
            (transformer.apply_transform ~feature_schema:current_schema
               ~x:current_x)
        in
        let* () =
          with_stage transformer.stage_name
            (validate_transform_output ~input:current_x
               ~output_schema:transformer.transform_output_schema transformed)
        in
        apply (index + 1) transformer.transform_output_schema transformed
    in
    apply 0 feature_schema x

  let predict fitted ~feature_schema ~x =
    let* transformed = transform fitted ~feature_schema ~x in
    with_stage fitted.fitted_estimator.terminal_name
      (fitted.fitted_estimator.terminal_predict
         ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let unsupported fitted capability =
    Error.make
      ~context:[ Error.Stage fitted.fitted_estimator.terminal_name ]
      ~remediation:("configure a terminal estimator that supports " ^ capability)
      (Error.Compatibility
         {
           component = "pipeline terminal estimator";
           reason = capability ^ " is unavailable";
         })

  let decision_function fitted ~feature_schema ~x =
    match fitted.fitted_estimator.terminal_decision_function with
    | None -> Error (unsupported fitted "decision_function")
    | Some dispatch ->
        let* transformed = transform fitted ~feature_schema ~x in
        with_stage fitted.fitted_estimator.terminal_name
          (dispatch ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let predict_proba fitted ~feature_schema ~x =
    match fitted.fitted_estimator.terminal_predict_proba with
    | None -> Error (unsupported fitted "predict_proba")
    | Some dispatch ->
        let* transformed = transform fitted ~feature_schema ~x in
        with_stage fitted.fitted_estimator.terminal_name
          (dispatch ~feature_schema:fitted.pipeline_output_schema ~x:transformed)

  let input_schema fitted = fitted.pipeline_input_schema
  let output_schema fitted = fitted.pipeline_output_schema
end
