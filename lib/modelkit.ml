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

module Solver_report = struct
  type stopping_reason = Direct_solution | Gradient_tolerance | Step_tolerance

  type t = {
    converged : bool;
    iterations : int;
    objective : float;
    stopping_reason : stopping_reason;
    rank : int option;
  }

  let converged report = report.converged
  let iterations report = report.iterations
  let objective report = report.objective
  let stopping_reason report = report.stopping_reason
  let rank report = report.rank

  let create ~iterations ~objective ~stopping_reason ~rank =
    { converged = true; iterations; objective; stopping_reason; rank }
end

module Linear_model_internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let numerical ~operation ~reason ~remediation =
    Error.make ~remediation (Error.Numerical { operation; reason })

  let validate_schema ~expected observed =
    if Feature_schema.equal expected observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide features with the fitted schema"
           (Error.Feature_schema_mismatch { expected; observed }))

  let validate_matrix ?(require_samples = true) feature_schema x =
    let* () =
      match Feature_schema.validate_matrix feature_schema x with
      | Ok () -> Ok ()
      | Error error ->
          Error
            (Error.of_data_error
               ~remediation:"provide features matching the declared schema"
               error)
    in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let rec loop row column =
      if row = rows then Ok ()
      else if column = columns then loop (row + 1) 0
      else
        let value = Matrix.get x row column in
        if Float.is_finite value then loop row (column + 1)
        else
          Error
            (Error.of_data_error
               ~remediation:"impute or remove non-finite estimator inputs"
               (Data_error.Non_finite
                  {
                    name = "linear-model features";
                    index = (row * columns) + column;
                    value;
                  }))
    in
    let* () = loop 0 0 in
    if require_samples && rows = 0 then
      Error
        (validation ~name:"linear-model samples"
           ~reason:"at least one training sample is required"
           ~remediation:"provide a non-empty training matrix")
    else Ok ()

  let validate_prediction_input ~schema feature_schema x =
    let* () = validate_schema ~expected:schema feature_schema in
    validate_matrix ~require_samples:false feature_schema x

  let validate_target_length x target_length =
    let expected = Matrix.rows x in
    if target_length = expected then Ok ()
    else
      Error
        (Error.make ~remediation:"provide one target per training sample"
           (Error.Shape_mismatch
              {
                name = "linear-model target";
                expected = [ expected ];
                observed = [ target_length ];
              }))

  let validate_sample_weight x = function
    | None -> Ok ()
    | Some weights ->
        let expected = Matrix.rows x in
        let observed = Sample_weight.length weights in
        if expected = observed then Ok ()
        else
          Error
            (Error.of_data_error
               ~remediation:"provide one sample weight per training sample"
               (Data_error.Length_mismatch
                  { name = "linear-model sample weights"; expected; observed }))

  let weight sample_weight row =
    match sample_weight with
    | None -> 1.0
    | Some weights -> Sample_weight.get weights row

  let maximum_weight sample_weight =
    match sample_weight with
    | None -> 1.0
    | Some weights ->
        let maximum = ref 0.0 in
        for row = 0 to Sample_weight.length weights - 1 do
          maximum := Float.max !maximum (Sample_weight.get weights row)
        done;
        !maximum

  let stable_norm matrix ~column ~first_row =
    let scale = ref 0.0 in
    let sum_squares = ref 1.0 in
    for row = first_row to Array.length matrix - 1 do
      let value = Float.abs matrix.(row).(column) in
      if value <> 0.0 then
        if !scale < value then (
          let ratio = !scale /. value in
          sum_squares := 1.0 +. (!sum_squares *. ratio *. ratio);
          scale := value)
        else
          let ratio = value /. !scale in
          sum_squares := !sum_squares +. (ratio *. ratio)
    done;
    if !scale = 0.0 then 0.0 else !scale *. Float.sqrt !sum_squares

  type least_squares_solution = {
    least_squares_coefficients : float array;
    least_squares_rank : int;
  }

  let solve_least_squares ~operation matrix target =
    let rows = Array.length matrix in
    let columns = if rows = 0 then 0 else Array.length matrix.(0) in
    let matrix = Array.map Array.copy matrix in
    let target = Array.copy target in
    let permutation = Array.init columns Fun.id in
    let limit = Int.min rows columns in
    let first_pivot = ref 0.0 in
    let rank = ref 0 in
    let rec factor column =
      if column = limit then Ok ()
      else
        let pivot = ref column in
        let pivot_norm = ref (stable_norm matrix ~column ~first_row:column) in
        for candidate = column + 1 to columns - 1 do
          let norm = stable_norm matrix ~column:candidate ~first_row:column in
          if norm > !pivot_norm then (
            pivot := candidate;
            pivot_norm := norm)
        done;
        if column = 0 then first_pivot := !pivot_norm;
        let tolerance =
          Float.epsilon *. Float.of_int (Int.max rows columns) *. !first_pivot
        in
        if !pivot_norm <= tolerance then Ok ()
        else (
          if !pivot <> column then (
            for row = 0 to rows - 1 do
              let value = matrix.(row).(column) in
              matrix.(row).(column) <- matrix.(row).(!pivot);
              matrix.(row).(!pivot) <- value
            done;
            let original = permutation.(column) in
            permutation.(column) <- permutation.(!pivot);
            permutation.(!pivot) <- original);
          let norm = stable_norm matrix ~column ~first_row:column in
          let leading = matrix.(column).(column) in
          let reflected = if leading >= 0.0 then -.norm else norm in
          let reflector = Array.make (rows - column) 0.0 in
          for offset = 0 to Array.length reflector - 1 do
            reflector.(offset) <- matrix.(column + offset).(column)
          done;
          reflector.(0) <- reflector.(0) -. reflected;
          let reflector_norm =
            let scale = ref 0.0 in
            let squares = ref 1.0 in
            Array.iter
              (fun raw ->
                let value = Float.abs raw in
                if value <> 0.0 then
                  if !scale < value then (
                    let ratio = !scale /. value in
                    squares := 1.0 +. (!squares *. ratio *. ratio);
                    scale := value)
                  else
                    let ratio = value /. !scale in
                    squares := !squares +. (ratio *. ratio))
              reflector;
            if !scale = 0.0 then 0.0 else !scale *. Float.sqrt !squares
          in
          if reflector_norm = 0.0 || not (Float.is_finite reflector_norm) then
            Error
              (numerical ~operation
                 ~reason:"Householder reflector is not finite"
                 ~remediation:"rescale the features and target")
          else (
            Array.iteri
              (fun index value -> reflector.(index) <- value /. reflector_norm)
              reflector;
            for affected = column to columns - 1 do
              let projection = ref 0.0 in
              for offset = 0 to Array.length reflector - 1 do
                projection :=
                  !projection
                  +. (reflector.(offset) *. matrix.(column + offset).(affected))
              done;
              for offset = 0 to Array.length reflector - 1 do
                matrix.(column + offset).(affected) <-
                  matrix.(column + offset).(affected)
                  -. (2.0 *. reflector.(offset) *. !projection)
              done
            done;
            let projection = ref 0.0 in
            for offset = 0 to Array.length reflector - 1 do
              projection :=
                !projection +. (reflector.(offset) *. target.(column + offset))
            done;
            for offset = 0 to Array.length reflector - 1 do
              target.(column + offset) <-
                target.(column + offset)
                -. (2.0 *. reflector.(offset) *. !projection)
            done;
            matrix.(column).(column) <- reflected;
            for row = column + 1 to rows - 1 do
              matrix.(row).(column) <- 0.0
            done;
            incr rank;
            factor (column + 1)))
    in
    let* () = factor 0 in
    let pivoted = Array.make columns 0.0 in
    for row = !rank - 1 downto 0 do
      let residual = ref target.(row) in
      for column = row + 1 to !rank - 1 do
        residual := !residual -. (matrix.(row).(column) *. pivoted.(column))
      done;
      pivoted.(row) <- !residual /. matrix.(row).(row)
    done;
    let solution = Array.make columns 0.0 in
    for column = 0 to columns - 1 do
      solution.(permutation.(column)) <- pivoted.(column)
    done;
    if Array.for_all Float.is_finite solution then
      Ok { least_squares_coefficients = solution; least_squares_rank = !rank }
    else
      Error
        (numerical ~operation ~reason:"solver produced non-finite coefficients"
           ~remediation:"rescale the features and target")

  let weighted_means x target sample_weight =
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let maximum = maximum_weight sample_weight in
    let total = ref 0.0 in
    let feature_means = Array.make columns 0.0 in
    let target_mean = ref 0.0 in
    for row = 0 to rows - 1 do
      let normalized_weight = weight sample_weight row /. maximum in
      if normalized_weight > 0.0 then (
        let next_total = !total +. normalized_weight in
        let fraction = normalized_weight /. next_total in
        for column = 0 to columns - 1 do
          let value = Matrix.get x row column in
          feature_means.(column) <-
            feature_means.(column)
            +. (fraction *. (value -. feature_means.(column)))
        done;
        let value = Vector.get target row in
        target_mean := !target_mean +. (fraction *. (value -. !target_mean));
        total := next_total)
    done;
    (feature_means, !target_mean)

  type regression_solution = {
    regression_coefficients : float array;
    regression_intercept : float;
    regression_rank : int;
    regression_objective : float;
  }

  let fit_least_squares ~operation ~alpha ~fit_intercept ?sample_weight x target
      =
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let maximum_weight = maximum_weight sample_weight in
    let objective_scale = Float.max maximum_weight alpha in
    let feature_means, target_mean =
      if fit_intercept then weighted_means x target sample_weight
      else (Array.make columns 0.0, 0.0)
    in
    let penalty_rows = if alpha = 0.0 then 0 else columns in
    let design = Array.make_matrix (rows + penalty_rows) columns 0.0 in
    let response = Array.make (rows + penalty_rows) 0.0 in
    for row = 0 to rows - 1 do
      let factor = Float.sqrt (weight sample_weight row /. objective_scale) in
      for column = 0 to columns - 1 do
        design.(row).(column) <-
          factor *. (Matrix.get x row column -. feature_means.(column))
      done;
      response.(row) <- factor *. (Vector.get target row -. target_mean)
    done;
    (if penalty_rows > 0 then
       let penalty = Float.sqrt (alpha /. objective_scale) in
       for column = 0 to columns - 1 do
         design.(rows + column).(column) <- penalty
       done);
    let* solved = solve_least_squares ~operation design response in
    let coefficients = solved.least_squares_coefficients in
    let intercept =
      if fit_intercept then (
        let value = ref target_mean in
        for column = 0 to columns - 1 do
          value := !value -. (feature_means.(column) *. coefficients.(column))
        done;
        !value)
      else 0.0
    in
    let objective = ref 0.0 in
    for row = 0 to rows - 1 do
      let prediction = ref intercept in
      for column = 0 to columns - 1 do
        prediction :=
          !prediction +. (Matrix.get x row column *. coefficients.(column))
      done;
      let residual = !prediction -. Vector.get target row in
      objective :=
        !objective
        +. 0.5
           *. (weight sample_weight row /. objective_scale)
           *. residual *. residual
    done;
    for column = 0 to columns - 1 do
      objective :=
        !objective
        +. 0.5 *. (alpha /. objective_scale) *. coefficients.(column)
           *. coefficients.(column)
    done;
    if Float.is_finite intercept && Float.is_finite !objective then
      Ok
        {
          regression_coefficients = coefficients;
          regression_intercept = intercept;
          regression_rank = solved.least_squares_rank;
          regression_objective = !objective;
        }
    else
      Error
        (numerical ~operation
           ~reason:"fitted intercept or objective is not finite"
           ~remediation:"rescale the features and target")

  let regression_prediction ~operation ~schema ~coefficients ~intercept
      feature_schema x =
    let* () = validate_prediction_input ~schema feature_schema x in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let values = Array.make rows 0.0 in
    let rec predict row =
      if row = rows then Ok ()
      else
        let value = ref intercept in
        for column = 0 to columns - 1 do
          value := !value +. (Matrix.get x row column *. coefficients.(column))
        done;
        if Float.is_finite !value then (
          values.(row) <- !value;
          predict (row + 1))
        else
          Error
            (numerical ~operation ~reason:"prediction is not finite"
               ~remediation:"rescale the prediction features")
    in
    let* () = predict 0 in
    match Target.regression (Vector.of_array values) with
    | Ok target -> Ok target
    | Error error ->
        Error
          (Error.of_data_error ~remediation:"rescale the prediction features"
             error)

  let stable_sigmoid value =
    if value >= 0.0 then 1.0 /. (1.0 +. Float.exp (-.value))
    else
      let exponential = Float.exp value in
      exponential /. (1.0 +. exponential)

  let softplus value =
    if value > 0.0 then value +. Float.log1p (Float.exp (-.value))
    else Float.log1p (Float.exp value)
end

module Linear_regression = struct
  type params = { fit_intercept : bool }
  type t = params

  type fitted = {
    linear_params : params;
    linear_coefficients : float array;
    linear_intercept : float;
    linear_schema : Feature_schema.t;
    linear_report : Solver_report.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let create ?(fit_intercept = true) () = { fit_intercept }
  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let open Linear_model_internal in
    let* () = validate_matrix feature_schema x in
    let* () = validate_target_length x (Target.length y) in
    let* () = validate_sample_weight x sample_weight in
    let target = Target.regression_values y in
    let* solution =
      fit_least_squares ~operation:"ordinary least squares" ~alpha:0.0
        ~fit_intercept:specification.fit_intercept ?sample_weight x target
    in
    Ok
      {
        linear_params = specification;
        linear_coefficients = solution.regression_coefficients;
        linear_intercept = solution.regression_intercept;
        linear_schema = feature_schema;
        linear_report =
          Solver_report.create ~iterations:1
            ~objective:solution.regression_objective
            ~stopping_reason:Solver_report.Direct_solution
            ~rank:(Some solution.regression_rank);
      }

  let predict fitted ~feature_schema ~x =
    Linear_model_internal.regression_prediction
      ~operation:"linear regression prediction" ~schema:fitted.linear_schema
      ~coefficients:fitted.linear_coefficients
      ~intercept:fitted.linear_intercept feature_schema x

  let fitted_params fitted = fitted.linear_params
  let feature_schema fitted = fitted.linear_schema
  let coefficients fitted = Vector.of_array fitted.linear_coefficients
  let intercept fitted = fitted.linear_intercept
  let report fitted = fitted.linear_report
end

module Ridge_regression = struct
  type params = { alpha : float; fit_intercept : bool }
  type t = params

  type fitted = {
    ridge_params : params;
    ridge_coefficients : float array;
    ridge_intercept : float;
    ridge_schema : Feature_schema.t;
    ridge_report : Solver_report.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let create ?(alpha = 1.0) ?(fit_intercept = true) () =
    if Float.is_finite alpha && alpha >= 0.0 then Ok { alpha; fit_intercept }
    else
      Error
        (Linear_model_internal.validation ~name:"ridge alpha"
           ~reason:"must be finite and non-negative"
           ~remediation:"choose a finite alpha greater than or equal to zero")

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let open Linear_model_internal in
    let* () = validate_matrix feature_schema x in
    let* () = validate_target_length x (Target.length y) in
    let* () = validate_sample_weight x sample_weight in
    let target = Target.regression_values y in
    let* solution =
      fit_least_squares ~operation:"ridge regression" ~alpha:specification.alpha
        ~fit_intercept:specification.fit_intercept ?sample_weight x target
    in
    Ok
      {
        ridge_params = specification;
        ridge_coefficients = solution.regression_coefficients;
        ridge_intercept = solution.regression_intercept;
        ridge_schema = feature_schema;
        ridge_report =
          Solver_report.create ~iterations:1
            ~objective:solution.regression_objective
            ~stopping_reason:Solver_report.Direct_solution
            ~rank:(Some solution.regression_rank);
      }

  let predict fitted ~feature_schema ~x =
    Linear_model_internal.regression_prediction
      ~operation:"ridge regression prediction" ~schema:fitted.ridge_schema
      ~coefficients:fitted.ridge_coefficients ~intercept:fitted.ridge_intercept
      feature_schema x

  let fitted_params fitted = fitted.ridge_params
  let feature_schema fitted = fitted.ridge_schema
  let coefficients fitted = Vector.of_array fitted.ridge_coefficients
  let intercept fitted = fitted.ridge_intercept
  let report fitted = fitted.ridge_report
end

module Logistic_regression = struct
  type params = {
    c : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    logistic_params : params;
    logistic_coefficients : float array;
    logistic_intercept : float;
    logistic_classes : int * int;
    logistic_schema : Feature_schema.t;
    logistic_report : Solver_report.t;
  }

  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  let create ?(c = 1.0) ?(fit_intercept = true) ?(tolerance = 1e-8)
      ?(max_iterations = 100) () =
    if (not (Float.is_finite c)) || c <= 0.0 then
      Error
        (Linear_model_internal.validation ~name:"logistic regression c"
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite c greater than zero")
    else if (not (Float.is_finite tolerance)) || tolerance <= 0.0 then
      Error
        (Linear_model_internal.validation ~name:"logistic regression tolerance"
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite tolerance greater than zero")
    else if max_iterations <= 0 then
      Error
        (Linear_model_internal.validation
           ~name:"logistic regression max_iterations" ~reason:"must be positive"
           ~remediation:"choose at least one iteration")
    else Ok { c; fit_intercept; tolerance; max_iterations }

  let clone specification = specification
  let params specification = specification

  let classes target sample_weight =
    let labels = Target.classification_values target in
    let first = ref None in
    let second = ref None in
    let invalid = ref false in
    for row = 0 to Array.length labels - 1 do
      if Linear_model_internal.weight sample_weight row > 0.0 then
        let label = labels.(row) in
        match (!first, !second) with
        | None, _ -> first := Some label
        | Some existing, None when label <> existing -> second := Some label
        | Some _, None -> ()
        | Some left, Some right ->
            if label <> left && label <> right then invalid := true
    done;
    match (!first, !second) with
    | Some left, Some right when not !invalid ->
        let low = Int.min left right in
        let high = Int.max left right in
        Ok (low, high)
    | Some _, Some _ ->
        Error
          (Linear_model_internal.validation ~name:"logistic regression classes"
             ~reason:"more than two positively weighted classes were found"
             ~remediation:"provide exactly two effective classes")
    | None, None | Some _, None | None, Some _ ->
        Error
          (Linear_model_internal.validation ~name:"logistic regression classes"
             ~reason:"exactly two positively weighted classes are required"
             ~remediation:"provide training rows from two classes")

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let open Linear_model_internal in
    let* () = validate_matrix feature_schema x in
    let* () = validate_target_length x (Target.length y) in
    let* () = validate_sample_weight x sample_weight in
    let* low_class, high_class = classes y sample_weight in
    let labels = Target.classification_values y in
    let rows = Matrix.rows x in
    let features = Matrix.columns x in
    let dimensions = features + if specification.fit_intercept then 1 else 0 in
    let regularization = 1.0 /. specification.c in
    let objective_scale =
      Float.max (maximum_weight sample_weight) regularization
    in
    let scaled_regularization = regularization /. objective_scale in
    let parameters = Array.make dimensions 0.0 in
    let linear row values =
      let result = ref 0.0 in
      for column = 0 to features - 1 do
        result := !result +. (Matrix.get x row column *. values.(column))
      done;
      if specification.fit_intercept then !result +. values.(features)
      else !result
    in
    let evaluate values ~with_hessian =
      let objective = ref 0.0 in
      let gradient = Array.make dimensions 0.0 in
      let hessian =
        if with_hessian then Some (Array.make_matrix dimensions dimensions 0.0)
        else None
      in
      for row = 0 to rows - 1 do
        let row_weight = weight sample_weight row /. objective_scale in
        if row_weight > 0.0 then (
          let expected = if labels.(row) = high_class then 1.0 else 0.0 in
          let decision = linear row values in
          let probability = stable_sigmoid decision in
          objective :=
            !objective
            +. (row_weight *. (softplus decision -. (expected *. decision)));
          let residual = row_weight *. (probability -. expected) in
          let curvature = row_weight *. probability *. (1.0 -. probability) in
          for left = 0 to dimensions - 1 do
            let left_value =
              if left = features then 1.0 else Matrix.get x row left
            in
            gradient.(left) <- gradient.(left) +. (residual *. left_value);
            match hessian with
            | None -> ()
            | Some matrix ->
                for right = 0 to left do
                  let right_value =
                    if right = features then 1.0 else Matrix.get x row right
                  in
                  matrix.(left).(right) <-
                    matrix.(left).(right)
                    +. (curvature *. left_value *. right_value)
                done
          done)
      done;
      for column = 0 to features - 1 do
        objective :=
          !objective
          +. (0.5 *. scaled_regularization *. values.(column) *. values.(column));
        gradient.(column) <-
          gradient.(column) +. (scaled_regularization *. values.(column));
        match hessian with
        | None -> ()
        | Some matrix ->
            matrix.(column).(column) <-
              matrix.(column).(column) +. scaled_regularization
      done;
      (match hessian with
      | None -> ()
      | Some matrix ->
          for left = 0 to dimensions - 1 do
            for right = 0 to left - 1 do
              matrix.(right).(left) <- matrix.(left).(right)
            done
          done);
      (!objective, gradient, hessian)
    in
    let infinity_norm values =
      Array.fold_left
        (fun maximum value -> Float.max maximum (Float.abs value))
        0.0 values
    in
    let rec iterate iteration =
      let objective, gradient, hessian =
        evaluate parameters ~with_hessian:true
      in
      if
        (not (Float.is_finite objective))
        || not (Array.for_all Float.is_finite gradient)
      then
        Error
          (numerical ~operation:"logistic regression"
             ~reason:"objective or gradient is not finite"
             ~remediation:"rescale the features or strengthen regularization")
      else if infinity_norm gradient <= specification.tolerance then
        Ok (Solver_report.Gradient_tolerance, iteration, objective)
      else if iteration = specification.max_iterations then
        Error
          (Error.make
             ~remediation:
               "increase max_iterations, rescale features, or strengthen \
                regularization"
             (Error.Convergence
                {
                  algorithm = "logistic regression";
                  reason =
                    Format.sprintf
                      "gradient tolerance was not reached after %d iterations"
                      specification.max_iterations;
                }))
      else
        let hessian = Option.get hessian in
        let* solved =
          solve_least_squares ~operation:"logistic regression Newton step"
            hessian gradient
        in
        if solved.least_squares_rank < dimensions then
          Error
            (numerical ~operation:"logistic regression Newton step"
               ~reason:"the Hessian is numerically rank deficient"
               ~remediation:
                 "rescale features, remove redundant columns, or strengthen \
                  regularization")
        else
          let directional = ref 0.0 in
          for index = 0 to dimensions - 1 do
            directional :=
              !directional
              +. (gradient.(index) *. solved.least_squares_coefficients.(index))
          done;
          let rec line_search attempts step =
            if attempts = 30 then None
            else
              let candidate =
                Array.mapi
                  (fun index value ->
                    value -. (step *. solved.least_squares_coefficients.(index)))
                  parameters
              in
              let candidate_objective, _, _ =
                evaluate candidate ~with_hessian:false
              in
              if
                Float.is_finite candidate_objective
                && candidate_objective
                   <= objective -. (1e-4 *. step *. !directional)
              then Some (step, candidate, candidate_objective)
              else line_search (attempts + 1) (step /. 2.0)
          in
          match line_search 0 1.0 with
          | None ->
              Error
                (Error.make
                   ~remediation:
                     "rescale features or choose stronger regularization"
                   (Error.Convergence
                      {
                        algorithm = "logistic regression";
                        reason = "damped Newton line search made no progress";
                      }))
          | Some (step, candidate, candidate_objective) ->
              let step_norm =
                step *. infinity_norm solved.least_squares_coefficients
              in
              let parameter_norm = infinity_norm parameters in
              Array.blit candidate 0 parameters 0 dimensions;
              if step_norm <= specification.tolerance *. (1.0 +. parameter_norm)
              then
                Ok
                  ( Solver_report.Step_tolerance,
                    iteration + 1,
                    candidate_objective )
              else iterate (iteration + 1)
    in
    let* stopping_reason, iterations, objective = iterate 0 in
    let coefficients = Array.sub parameters 0 features in
    let intercept =
      if specification.fit_intercept then parameters.(features) else 0.0
    in
    Ok
      {
        logistic_params = specification;
        logistic_coefficients = coefficients;
        logistic_intercept = intercept;
        logistic_classes = (low_class, high_class);
        logistic_schema = feature_schema;
        logistic_report =
          Solver_report.create ~iterations ~objective ~stopping_reason
            ~rank:None;
      }

  let decision_function fitted ~feature_schema ~x =
    let open Linear_model_internal in
    let* () =
      validate_prediction_input ~schema:fitted.logistic_schema feature_schema x
    in
    let rows = Matrix.rows x in
    let features = Matrix.columns x in
    let decisions = Array.make rows 0.0 in
    let rec predict row =
      if row = rows then Ok ()
      else
        let value = ref fitted.logistic_intercept in
        for column = 0 to features - 1 do
          value :=
            !value
            +. (Matrix.get x row column *. fitted.logistic_coefficients.(column))
        done;
        if Float.is_finite !value then (
          decisions.(row) <- !value;
          predict (row + 1))
        else
          Error
            (numerical ~operation:"logistic regression prediction"
               ~reason:"decision value is not finite"
               ~remediation:"rescale the prediction features")
    in
    let* () = predict 0 in
    Ok (Vector.of_array decisions)

  let predict_proba fitted ~feature_schema ~x =
    let open Linear_model_internal in
    let* decisions = decision_function fitted ~feature_schema ~x in
    match
      Matrix.init ~rows:(Vector.length decisions) ~columns:2 (fun row column ->
          let positive = stable_sigmoid (Vector.get decisions row) in
          if column = 0 then 1.0 -. positive else positive)
    with
    | Ok probabilities -> Ok probabilities
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"provide representable prediction dimensions" error)

  let predict fitted ~feature_schema ~x =
    let open Linear_model_internal in
    let* decisions = decision_function fitted ~feature_schema ~x in
    let low, high = fitted.logistic_classes in
    Ok
      (Target.classification
         (Array.init (Vector.length decisions) (fun row ->
              if Vector.get decisions row > 0.0 then high else low)))

  let fitted_params fitted = fitted.logistic_params
  let feature_schema fitted = fitted.logistic_schema
  let coefficients fitted = Vector.of_array fitted.logistic_coefficients
  let intercept fitted = fitted.logistic_intercept

  let classes fitted =
    let low, high = fitted.logistic_classes in
    [| low; high |]

  let report fitted = fitted.logistic_report
end

module Split = struct
  type t = {
    split_source_size : int;
    split_train : Row_view.t;
    split_test : Row_view.t;
  }

  let validation ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name = "split"; reason })

  let validate_partition ~source_size ~name ~seen view =
    if Row_view.source_size view <> source_size then
      Error
        (validation
           ~reason:
             (Format.sprintf "%s rows refer to source size %d instead of %d"
                name
                (Row_view.source_size view)
                source_size)
           ~remediation:"construct both partitions from the same aligned source")
    else if Row_view.length view = 0 then
      Error
        (validation ~reason:(name ^ " rows are empty")
           ~remediation:"provide at least one row in each split partition")
    else
      let rec loop position =
        if position = Row_view.length view then Ok ()
        else
          let row = Row_view.get view position in
          if seen.(row) then
            Error
              (validation
                 ~reason:
                   (Format.sprintf "%s row %d occurs more than once" name row)
                 ~remediation:"remove duplicate rows from each split partition")
          else (
            seen.(row) <- true;
            loop (position + 1))
      in
      loop 0

  let of_views ~train ~test =
    let source_size = Row_view.source_size train in
    let train_seen = Array.make source_size false in
    let test_seen = Array.make source_size false in
    let ( let* ) = Result.bind in
    let* () =
      validate_partition ~source_size ~name:"training" ~seen:train_seen train
    in
    let* () =
      validate_partition ~source_size ~name:"test" ~seen:test_seen test
    in
    let rec disjoint row =
      if row = source_size then Ok ()
      else if train_seen.(row) && test_seen.(row) then
        Error
          (validation
             ~reason:
               (Format.sprintf "row %d occurs in both training and test data"
                  row)
             ~remediation:"make the training and test partitions disjoint")
      else disjoint (row + 1)
    in
    let* () = disjoint 0 in
    Ok
      {
        split_source_size = source_size;
        split_train = train;
        split_test = test;
      }

  let create ~source_size ~train ~test =
    let ( let* ) = Result.bind in
    let* train =
      Row_view.create ~source_size train
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide valid training row indices"
            error)
    in
    let* test =
      Row_view.create ~source_size test
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide valid test row indices"
            error)
    in
    of_views ~train ~test

  let train split = split.split_train
  let test split = split.split_test
  let pair split = (split.split_train, split.split_test)

  let materialize dataset split =
    let expected = Dataset.sample_count dataset in
    if split.split_source_size <> expected then
      Error
        (validation
           ~reason:
             (Format.sprintf
                "split source size is %d but the dataset has %d rows"
                split.split_source_size expected)
           ~remediation:"use a split generated for this dataset")
    else
      let ( let* ) = Result.bind in
      let materialize_partition rows =
        let* view =
          Dataset.view dataset rows
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"use rows aligned with the source dataset" error)
        in
        Dataset.materialize view
        |> Result.map_error (fun error ->
            Error.of_data_error
              ~remediation:
                "ensure selected targets, weights, and groups remain valid"
              error)
      in
      let* train = materialize_partition split.split_train in
      let* test = materialize_partition split.split_test in
      Ok (train, test)
end

module Splitter_internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let validate_folds ~name folds =
    if folds >= 2 then Ok ()
    else
      Error
        (validation ~name ~reason:"fold count must be at least two"
           ~remediation:"choose two or more folds")

  let validate_sample_count ~name ~folds sample_count =
    if folds <= sample_count then Ok ()
    else
      Error
        (validation ~name
           ~reason:
             (Format.sprintf "%d folds exceed the %d available samples" folds
                sample_count)
           ~remediation:"reduce the fold count or provide more samples")

  let validate_aligned_length ~name ~expected observed =
    if expected = observed then Ok ()
    else
      Error
        (Error.make
           ~remediation:("provide " ^ name ^ " aligned to feature rows")
           (Error.Shape_mismatch
              { name; expected = [ expected ]; observed = [ observed ] }))

  let shuffle rng values =
    let state = ref rng in
    for upper = Array.length values - 1 downto 1 do
      let random, successor = Rng.next_int64 !state in
      state := successor;
      let nonnegative = Int64.shift_right_logical random 1 in
      let selected =
        Int64.rem nonnegative (Int64.of_int (upper + 1)) |> Int64.to_int
      in
      let value = values.(upper) in
      values.(upper) <- values.(selected);
      values.(selected) <- value
    done

  let split_from_assignments ~source_size ~folds assignments =
    Array.init folds (fun fold ->
        let train_count = ref 0 in
        let test_count = ref 0 in
        Array.iter
          (fun assigned ->
            if assigned = fold then incr test_count else incr train_count)
          assignments;
        let train = Array.make !train_count 0 in
        let test = Array.make !test_count 0 in
        let train_position = ref 0 in
        let test_position = ref 0 in
        Array.iteri
          (fun row assigned ->
            if assigned = fold then (
              test.(!test_position) <- row;
              incr test_position)
            else (
              train.(!train_position) <- row;
              incr train_position))
          assignments;
        Split.create ~source_size ~train ~test)
    |> Array.fold_left
         (fun accumulated split ->
           let* values = accumulated in
           let* value = split in
           Ok (value :: values))
         (Ok [])
    |> Result.map (fun reversed ->
        reversed |> List.rev_map Split.pair |> Array.of_list)

  let balanced_assignments ~sample_count ~folds ~shuffle:rng =
    let order = Array.init sample_count Fun.id in
    Option.iter (fun state -> shuffle state order) rng;
    let assignments = Array.make sample_count 0 in
    let base_size = sample_count / folds in
    let remainder = sample_count mod folds in
    let offset = ref 0 in
    for fold = 0 to folds - 1 do
      let size = base_size + if fold < remainder then 1 else 0 in
      for position = !offset to !offset + size - 1 do
        assignments.(order.(position)) <- fold
      done;
      offset := !offset + size
    done;
    assignments
end

module K_fold = struct
  type params = { folds : int; shuffle : bool }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) ?(shuffle = false) () =
    Result.map
      (fun () -> { folds; shuffle })
      (Splitter_internal.validate_folds ~name:"K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_sample_count ~name:"K-fold"
        ~folds:specification.folds sample_count
    in
    let shuffle = if specification.shuffle then Some rng else None in
    let assignments =
      Splitter_internal.balanced_assignments ~sample_count
        ~folds:specification.folds ~shuffle
    in
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Stratified_k_fold = struct
  type params = { folds : int; shuffle : bool }
  type t = params
  type target = Target.classification Target.t
  type rng = Rng.t

  let create ?(folds = 5) ?(shuffle = false) () =
    Result.map
      (fun () -> { folds; shuffle })
      (Splitter_internal.validate_folds ~name:"stratified K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng ?groups:_ ~x ~y () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_sample_count ~name:"stratified K-fold"
        ~folds:specification.folds sample_count
    in
    let* target =
      match y with
      | Some target -> Ok target
      | None ->
          Error
            (Splitter_internal.validation ~name:"stratified K-fold target"
               ~reason:"a classification target is required"
               ~remediation:"provide aligned integer class labels")
    in
    let labels = Target.classification_values target in
    let* () =
      Splitter_internal.validate_aligned_length ~name:"stratified K-fold target"
        ~expected:sample_count (Array.length labels)
    in
    let class_by_label = Hashtbl.create sample_count in
    let encoded = Array.make sample_count 0 in
    let counts = Array.make sample_count 0 in
    let class_count = ref 0 in
    for row = 0 to sample_count - 1 do
      let class_index =
        match Hashtbl.find_opt class_by_label labels.(row) with
        | Some index -> index
        | None ->
            let index = !class_count in
            incr class_count;
            Hashtbl.add class_by_label labels.(row) index;
            index
      in
      encoded.(row) <- class_index;
      counts.(class_index) <- counts.(class_index) + 1
    done;
    let assignments = Array.make sample_count 0 in
    let class_offset = ref 0 in
    for class_index = 0 to !class_count - 1 do
      let allocation = Array.make specification.folds 0 in
      for position = 0 to counts.(class_index) - 1 do
        let fold = (!class_offset + position) mod specification.folds in
        allocation.(fold) <- allocation.(fold) + 1
      done;
      let class_assignments = Array.make counts.(class_index) 0 in
      let position = ref 0 in
      for fold = 0 to specification.folds - 1 do
        for _ = 1 to allocation.(fold) do
          class_assignments.(!position) <- fold;
          incr position
        done
      done;
      (if specification.shuffle then
         let class_rng =
           Seed.derive (Rng.to_seed rng) ~operation:"stratified-k-fold-class"
             ~index:class_index
           |> Rng.create
         in
         Splitter_internal.shuffle class_rng class_assignments);
      let position = ref 0 in
      for row = 0 to sample_count - 1 do
        if encoded.(row) = class_index then (
          assignments.(row) <- class_assignments.(!position);
          incr position)
      done;
      class_offset := !class_offset + counts.(class_index)
    done;
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Group_k_fold = struct
  type params = { folds : int }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) () =
    Result.map
      (fun () -> { folds })
      (Splitter_internal.validate_folds ~name:"group K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng:_ ?groups ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* groups =
      match groups with
      | Some groups -> Ok groups
      | None ->
          Error
            (Splitter_internal.validation ~name:"group K-fold groups"
               ~reason:"group labels are required"
               ~remediation:"provide one integer group per feature row")
    in
    let* () =
      Splitter_internal.validate_aligned_length ~name:"group K-fold groups"
        ~expected:sample_count (Groups.length groups)
    in
    let counts_by_group = Hashtbl.create sample_count in
    for row = 0 to sample_count - 1 do
      let group = Groups.get groups row in
      let count =
        Option.value (Hashtbl.find_opt counts_by_group group) ~default:0
      in
      Hashtbl.replace counts_by_group group (count + 1)
    done;
    let group_ids = Hashtbl.to_seq_keys counts_by_group |> Array.of_seq in
    let* () =
      Splitter_internal.validate_sample_count
        ~name:"group K-fold distinct groups" ~folds:specification.folds
        (Array.length group_ids)
    in
    Array.sort
      (fun left right ->
        let by_count =
          Int.compare
            (Hashtbl.find counts_by_group right)
            (Hashtbl.find counts_by_group left)
        in
        if by_count <> 0 then by_count else Int.compare right left)
      group_ids;
    let fold_sizes = Array.make specification.folds 0 in
    let fold_by_group = Hashtbl.create (Array.length group_ids) in
    Array.iter
      (fun group ->
        let selected = ref 0 in
        for fold = 1 to specification.folds - 1 do
          if fold_sizes.(fold) < fold_sizes.(!selected) then selected := fold
        done;
        Hashtbl.add fold_by_group group !selected;
        fold_sizes.(!selected) <-
          fold_sizes.(!selected) + Hashtbl.find counts_by_group group)
      group_ids;
    let assignments =
      Array.init sample_count (fun row ->
          Hashtbl.find fold_by_group (Groups.get groups row))
    in
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Time_series_split = struct
  type params = { folds : int; test_size : int option; gap : int }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) ?test_size ?(gap = 0) () =
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_folds ~name:"time-series split" folds
    in
    let* () =
      match test_size with
      | None -> Ok ()
      | Some size when size > 0 -> Ok ()
      | Some _ ->
          Error
            (Splitter_internal.validation ~name:"time-series test size"
               ~reason:"test size must be positive"
               ~remediation:"choose at least one test row")
    in
    if gap >= 0 then Ok { folds; test_size; gap }
    else
      Error
        (Splitter_internal.validation ~name:"time-series gap"
           ~reason:"gap must be non-negative"
           ~remediation:"choose zero or more excluded rows")

  let clone specification = specification
  let params specification = specification

  let split specification ~rng:_ ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let test_size =
      Option.value specification.test_size
        ~default:(sample_count / (specification.folds + 1))
    in
    if test_size <= 0 then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:"the default test window is empty"
           ~remediation:"provide more samples or an explicit positive test size")
    else if specification.gap >= sample_count then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:"the gap leaves no initial training rows"
           ~remediation:"reduce the gap or provide more samples")
    else if
      test_size > (sample_count - specification.gap - 1) / specification.folds
    then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:
             "fold count, test size, and gap leave no initial training rows"
           ~remediation:
             "reduce folds, test size, or gap, or provide more samples")
    else
      Array.init specification.folds (fun fold ->
          let test_start =
            sample_count - (specification.folds * test_size) + (fold * test_size)
          in
          let train_size = test_start - specification.gap in
          let train = Array.init train_size Fun.id in
          let test = Array.init test_size (fun offset -> test_start + offset) in
          Split.create ~source_size:sample_count ~train ~test)
      |> Array.fold_left
           (fun accumulated split ->
             let ( let* ) = Result.bind in
             let* values = accumulated in
             let* value = split in
             Ok (value :: values))
           (Ok [])
      |> Result.map (fun reversed ->
          reversed |> List.rev_map Split.pair |> Array.of_list)
end

module Undefined_metric_policy = struct
  type t = Error | Return_nan | Use_fallback
end

module Metric_internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let numerical ~operation ~reason =
    Error.make ~remediation:"rescale inputs or inspect the metric inputs"
      (Error.Numerical { operation; reason })

  let validate_nonempty ~name length =
    if length > 0 then Ok ()
    else
      Error
        (validation ~name ~reason:"metric inputs are empty"
           ~remediation:"provide at least one observed and predicted value")

  let validate_length ~name ~expected ~observed =
    if expected = observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide aligned truth and prediction values"
           (Error.Shape_mismatch
              { name; expected = [ expected ]; observed = [ observed ] }))

  let validate_weights ~expected = function
    | None -> Ok ()
    | Some weights ->
        validate_length ~name:"metric sample weights" ~expected
          ~observed:(Sample_weight.length weights)

  let weight sample_weight index =
    match sample_weight with
    | None -> 1.0
    | Some weights -> Sample_weight.get weights index

  let sum length f =
    let accumulator = Reference_backend.Accumulator.create () in
    for index = 0 to length - 1 do
      Reference_backend.Accumulator.add accumulator (f index)
    done;
    Reference_backend.Accumulator.value accumulator

  let finite ~operation value =
    if Float.is_finite value then Ok value
    else
      Error
        (numerical ~operation
           ~reason:"the result is not representable as a finite float64 value")

  let total_weight sample_weight length =
    sum length (weight sample_weight) |> finite ~operation:"metric weighting"

  let average ~operation ~sample_weight length f =
    let numerator =
      sum length (fun index -> weight sample_weight index *. f index)
    in
    let* numerator = finite ~operation numerator in
    let* denominator = total_weight sample_weight length in
    finite ~operation (numerator /. denominator)

  let undefined policy ~name ~reason ~fallback =
    match policy with
    | Undefined_metric_policy.Error ->
        Error
          (validation ~name ~reason
             ~remediation:
               "choose Return_nan or Use_fallback, or provide metric inputs \
                with the required variation")
    | Undefined_metric_policy.Return_nan -> Ok Float.nan
    | Undefined_metric_policy.Use_fallback -> Ok fallback

  let validate_probabilities ~expected probabilities =
    let observed = Vector.length probabilities in
    let* () =
      validate_length ~name:"positive-class probabilities" ~expected ~observed
    in
    let rec loop index =
      if index = observed then Ok ()
      else
        let probability = Vector.get probabilities index in
        if not (Float.is_finite probability) then
          Error
            (validation ~name:"positive-class probabilities"
               ~reason:
                 (Format.sprintf "value %g at index %d is not finite"
                    probability index)
               ~remediation:
                 "provide finite probabilities in the interval [0, 1]")
        else if probability < 0.0 || probability > 1.0 then
          Error
            (validation ~name:"positive-class probabilities"
               ~reason:
                 (Format.sprintf "value %g at index %d lies outside [0, 1]"
                    probability index)
               ~remediation:"provide probabilities in the interval [0, 1]")
        else loop (index + 1)
    in
    loop 0

  let validate_at_most_two_labels arrays =
    let labels = Hashtbl.create 3 in
    Array.iter
      (Array.iter (fun label -> Hashtbl.replace labels label ()))
      arrays;
    if Hashtbl.length labels <= 2 then Ok ()
    else
      Error
        (validation ~name:"binary classification labels"
           ~reason:"more than two distinct labels were observed"
           ~remediation:"provide binary truth and prediction labels")

  let validate_positive_label ~positive_label arrays =
    let negative = ref None in
    let rec validate_array collection index =
      if index = Array.length collection then Ok ()
      else
        let label = collection.(index) in
        if label = positive_label then validate_array collection (index + 1)
        else
          match !negative with
          | None ->
              negative := Some label;
              validate_array collection (index + 1)
          | Some expected when label = expected ->
              validate_array collection (index + 1)
          | Some expected ->
              Error
                (validation ~name:"binary classification labels"
                   ~reason:
                     (Format.sprintf
                        "label %d is neither positive label %d nor negative \
                         label %d"
                        label positive_label expected)
                   ~remediation:
                     "choose the intended positive label and provide binary \
                      labels")
    in
    let rec loop index =
      if index = Array.length arrays then Ok ()
      else
        let* () = validate_array arrays.(index) 0 in
        loop (index + 1)
    in
    loop 0
end

module Regression_metrics = struct
  type residual_curve = { predictions : Vector.t; residuals : Vector.t }

  let prepare ?sample_weight ~truth ~prediction () =
    let truth = Target.regression_values truth in
    let prediction = Target.regression_values prediction in
    let length = Vector.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"regression metric" length
    in
    let* () =
      Metric_internal.validate_length ~name:"regression prediction"
        ~expected:length ~observed:(Vector.length prediction)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    Ok (truth, prediction, length)

  let mean_absolute_error ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    Metric_internal.average ~operation:"mean absolute error" ~sample_weight
      length (fun index ->
        Float.abs (Vector.get truth index -. Vector.get prediction index))

  let mean_squared_error ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    Metric_internal.average ~operation:"mean squared error" ~sample_weight
      length (fun index ->
        let residual = Vector.get truth index -. Vector.get prediction index in
        residual *. residual)

  let root_mean_squared_error ?sample_weight ~truth ~prediction () =
    mean_squared_error ?sample_weight ~truth ~prediction ()
    |> Result.map Float.sqrt

  let r2 ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    let* mean =
      Metric_internal.average ~operation:"R-squared truth mean" ~sample_weight
        length (Vector.get truth)
    in
    let residual_sum =
      Metric_internal.sum length (fun index ->
          let residual =
            Vector.get truth index -. Vector.get prediction index
          in
          Metric_internal.weight sample_weight index *. residual *. residual)
    in
    let total_sum =
      Metric_internal.sum length (fun index ->
          let centered = Vector.get truth index -. mean in
          Metric_internal.weight sample_weight index *. centered *. centered)
    in
    let* residual_sum =
      Metric_internal.finite ~operation:"R-squared residual sum" residual_sum
    in
    let* total_sum =
      Metric_internal.finite ~operation:"R-squared total sum" total_sum
    in
    if total_sum = 0.0 then
      Metric_internal.undefined undefined ~name:"R-squared"
        ~reason:"the weighted truth values are constant"
        ~fallback:(if residual_sum = 0.0 then 1.0 else 0.0)
    else
      Metric_internal.finite ~operation:"R-squared"
        (1.0 -. (residual_sum /. total_sum))

  let residual_curve ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length = prepare ~truth ~prediction () in
    let residuals = Array.make length 0.0 in
    let rec fill index =
      if index = length then Ok ()
      else
        let residual = Vector.get truth index -. Vector.get prediction index in
        if Float.is_finite residual then (
          residuals.(index) <- residual;
          fill (index + 1))
        else
          Error
            (Metric_internal.numerical ~operation:"residual curve"
               ~reason:
                 (Format.sprintf
                    "the residual at index %d is not representable as float64"
                    index))
    in
    let* () = fill 0 in
    Ok { predictions = prediction; residuals = Vector.of_array residuals }
end

module Binary_prediction = struct
  type t = {
    binary_labels : Target.classification Target.t option;
    binary_positive_probabilities : Vector.t option;
    binary_length : int;
  }

  let create ?labels ?positive_probabilities () =
    let ( let* ) = Result.bind in
    match (labels, positive_probabilities) with
    | None, None ->
        Error
          (Metric_internal.validation ~name:"binary prediction"
             ~reason:
               "neither labels nor positive-class probabilities were provided"
             ~remediation:"provide at least one classifier response")
    | Some labels, None ->
        Ok
          {
            binary_labels = Some labels;
            binary_positive_probabilities = None;
            binary_length = Target.length labels;
          }
    | None, Some probabilities ->
        let length = Vector.length probabilities in
        let* () =
          Metric_internal.validate_probabilities ~expected:length probabilities
        in
        Ok
          {
            binary_labels = None;
            binary_positive_probabilities = Some probabilities;
            binary_length = length;
          }
    | Some labels, Some probabilities ->
        let length = Target.length labels in
        let* () =
          Metric_internal.validate_probabilities ~expected:length probabilities
        in
        Ok
          {
            binary_labels = Some labels;
            binary_positive_probabilities = Some probabilities;
            binary_length = length;
          }

  let length prediction = prediction.binary_length
  let labels prediction = prediction.binary_labels

  let positive_probabilities prediction =
    prediction.binary_positive_probabilities
end

module Binary_classification_metrics = struct
  type roc_curve = {
    thresholds : Vector.t;
    false_positive_rates : Vector.t;
    true_positive_rates : Vector.t;
  }

  type precision_recall_curve = {
    decision_thresholds : Vector.t;
    precisions : Vector.t;
    recalls : Vector.t;
  }

  type confusion = { tp : float; tn : float; fp : float; fn : float }

  let prepare_labels ?sample_weight ~positive_label ~truth ~prediction () =
    let truth = Target.classification_values truth in
    let prediction = Target.classification_values prediction in
    let length = Array.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"binary classification metric"
        length
    in
    let* () =
      Metric_internal.validate_length ~name:"classification prediction"
        ~expected:length ~observed:(Array.length prediction)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_positive_label ~positive_label
        [| truth; prediction |]
    in
    Ok (truth, prediction, length)

  let prepare_probabilities ?sample_weight ~positive_label ~truth
      ~positive_probabilities () =
    let truth = Target.classification_values truth in
    let length = Array.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"binary classification metric"
        length
    in
    let* () =
      Metric_internal.validate_probabilities ~expected:length
        positive_probabilities
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_positive_label ~positive_label [| truth |]
    in
    Ok (truth, length)

  let accuracy ?sample_weight ~truth ~prediction () =
    let truth_values = Target.classification_values truth in
    let prediction_values = Target.classification_values prediction in
    let length = Array.length truth_values in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"classification accuracy" length
    in
    let* () =
      Metric_internal.validate_length ~name:"classification prediction"
        ~expected:length
        ~observed:(Array.length prediction_values)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_at_most_two_labels
        [| truth_values; prediction_values |]
    in
    Metric_internal.average ~operation:"classification accuracy" ~sample_weight
      length (fun index ->
        if truth_values.(index) = prediction_values.(index) then 1.0 else 0.0)

  let confusion ?sample_weight ~positive_label ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare_labels ?sample_weight ~positive_label ~truth ~prediction ()
    in
    let tp = Reference_backend.Accumulator.create () in
    let tn = Reference_backend.Accumulator.create () in
    let fp = Reference_backend.Accumulator.create () in
    let fn = Reference_backend.Accumulator.create () in
    for index = 0 to length - 1 do
      let weight = Metric_internal.weight sample_weight index in
      match
        (truth.(index) = positive_label, prediction.(index) = positive_label)
      with
      | true, true -> Reference_backend.Accumulator.add tp weight
      | false, false -> Reference_backend.Accumulator.add tn weight
      | false, true -> Reference_backend.Accumulator.add fp weight
      | true, false -> Reference_backend.Accumulator.add fn weight
    done;
    let* tp =
      Reference_backend.Accumulator.value tp
      |> Metric_internal.finite ~operation:"binary true-positive weight"
    in
    let* tn =
      Reference_backend.Accumulator.value tn
      |> Metric_internal.finite ~operation:"binary true-negative weight"
    in
    let* fp =
      Reference_backend.Accumulator.value fp
      |> Metric_internal.finite ~operation:"binary false-positive weight"
    in
    let* fn =
      Reference_backend.Accumulator.value fn
      |> Metric_internal.finite ~operation:"binary false-negative weight"
    in
    Ok { tp; tn; fp; fn }

  let ratio ~undefined ~name ~reason ~fallback numerator denominator =
    if denominator = 0.0 then
      Metric_internal.undefined undefined ~name ~reason ~fallback
    else Ok (numerator /. denominator)

  let balanced_accuracy ?(positive_label = 1)
      ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    let positive_support = values.tp +. values.fn in
    let negative_support = values.tn +. values.fp in
    if positive_support = 0.0 || negative_support = 0.0 then
      Metric_internal.undefined undefined ~name:"balanced accuracy"
        ~reason:"positive and negative weighted support are both required"
        ~fallback:0.0
    else
      Ok
        (0.5
        *. ((values.tp /. positive_support) +. (values.tn /. negative_support))
        )

  let precision ?(positive_label = 1)
      ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"precision"
      ~reason:"no positive prediction has positive sample weight" ~fallback:0.0
      values.tp (values.tp +. values.fp)

  let recall ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"recall"
      ~reason:"no positive truth label has positive sample weight" ~fallback:0.0
      values.tp (values.tp +. values.fn)

  let f1 ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"F1"
      ~reason:
        "true-positive, false-positive, and false-negative weights are zero"
      ~fallback:0.0 (2.0 *. values.tp)
      ((2.0 *. values.tp) +. values.fp +. values.fn)

  let log_loss ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* truth, length =
      prepare_probabilities ?sample_weight ~positive_label ~truth
        ~positive_probabilities ()
    in
    let epsilon = Float.epsilon in
    Metric_internal.average ~operation:"binary log loss" ~sample_weight length
      (fun index ->
        let probability = Vector.get positive_probabilities index in
        let probability =
          Float.max epsilon (Float.min (1.0 -. epsilon) probability)
        in
        if truth.(index) = positive_label then -.Float.log probability
        else -.Float.log1p (-.probability))

  type curve_counts = {
    curve_thresholds : float array;
    true_positives : float array;
    false_positives : float array;
    positive_weight : float;
    negative_weight : float;
  }

  type scored = {
    score : float;
    positive : bool;
    weight : float;
    source_index : int;
  }

  let curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
      () =
    let ( let* ) = Result.bind in
    let* truth, length =
      prepare_probabilities ?sample_weight ~positive_label ~truth
        ~positive_probabilities ()
    in
    let included_count = ref 0 in
    for index = 0 to length - 1 do
      if Metric_internal.weight sample_weight index > 0.0 then
        incr included_count
    done;
    let included =
      Array.make !included_count
        { score = 0.0; positive = false; weight = 0.0; source_index = 0 }
    in
    let position = ref 0 in
    for index = 0 to length - 1 do
      let weight = Metric_internal.weight sample_weight index in
      if weight > 0.0 then (
        included.(!position) <-
          {
            score = Vector.get positive_probabilities index;
            positive = truth.(index) = positive_label;
            weight;
            source_index = index;
          };
        incr position)
    done;
    Array.stable_sort
      (fun left right ->
        let by_score = Float.compare right.score left.score in
        if by_score <> 0 then by_score
        else Int.compare left.source_index right.source_index)
      included;
    let maximum = Array.length included in
    let thresholds = Array.make maximum 0.0 in
    let true_positives = Array.make maximum 0.0 in
    let false_positives = Array.make maximum 0.0 in
    let tp = Reference_backend.Accumulator.create () in
    let fp = Reference_backend.Accumulator.create () in
    let group_count = ref 0 in
    let index = ref 0 in
    while !index < maximum do
      let threshold = included.(!index).score in
      while !index < maximum && included.(!index).score = threshold do
        let item = included.(!index) in
        Reference_backend.Accumulator.add
          (if item.positive then tp else fp)
          item.weight;
        incr index
      done;
      thresholds.(!group_count) <- threshold;
      true_positives.(!group_count) <- Reference_backend.Accumulator.value tp;
      false_positives.(!group_count) <- Reference_backend.Accumulator.value fp;
      incr group_count
    done;
    let thresholds = Array.sub thresholds 0 !group_count in
    let true_positives = Array.sub true_positives 0 !group_count in
    let false_positives = Array.sub false_positives 0 !group_count in
    let positive_weight = true_positives.(!group_count - 1) in
    let negative_weight = false_positives.(!group_count - 1) in
    let* positive_weight =
      Metric_internal.finite ~operation:"ROC positive support" positive_weight
    in
    let* negative_weight =
      Metric_internal.finite ~operation:"ROC negative support" negative_weight
    in
    if positive_weight = 0.0 || negative_weight = 0.0 then
      Error
        (Metric_internal.validation ~name:"binary ranking curve"
           ~reason:"positive and negative weighted support are both required"
           ~remediation:"provide a fold containing both binary classes")
    else
      Ok
        {
          curve_thresholds = thresholds;
          true_positives;
          false_positives;
          positive_weight;
          negative_weight;
        }

  let roc_curve ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* counts =
      curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
        ()
    in
    let points = Array.length counts.curve_thresholds + 1 in
    Ok
      ({
         thresholds =
           Vector.unsafe_init points (fun index ->
               if index = 0 then Float.infinity
               else counts.curve_thresholds.(index - 1));
         false_positive_rates =
           Vector.unsafe_init points (fun index ->
               if index = 0 then 0.0
               else counts.false_positives.(index - 1) /. counts.negative_weight);
         true_positive_rates =
           Vector.unsafe_init points (fun index ->
               if index = 0 then 0.0
               else counts.true_positives.(index - 1) /. counts.positive_weight);
       }
        : roc_curve)

  let precision_recall_curve ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* counts =
      curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
        ()
    in
    let threshold_count = Array.length counts.curve_thresholds in
    Ok
      ({
         decision_thresholds =
           Vector.unsafe_init threshold_count (fun index ->
               counts.curve_thresholds.(threshold_count - index - 1));
         precisions =
           Vector.unsafe_init (threshold_count + 1) (fun index ->
               if index = threshold_count then 1.0
               else
                 let source = threshold_count - index - 1 in
                 let positives = counts.true_positives.(source) in
                 positives /. (positives +. counts.false_positives.(source)));
         recalls =
           Vector.unsafe_init (threshold_count + 1) (fun index ->
               if index = threshold_count then 0.0
               else
                 counts.true_positives.(threshold_count - index - 1)
                 /. counts.positive_weight);
       }
        : precision_recall_curve)

  let roc_auc ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~positive_probabilities () =
    match
      roc_curve ~positive_label ?sample_weight ~truth ~positive_probabilities ()
    with
    | Error error -> (
        match Error.kind error with
        | Error.Validation { name = "binary ranking curve"; reason } ->
            Metric_internal.undefined undefined ~name:"ROC AUC" ~reason
              ~fallback:0.5
        | Error.Data _ | Error.Shape_mismatch _
        | Error.Feature_schema_mismatch _ | Error.Validation _
        | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
        | Error.Artifact _ | Error.Cancelled ->
            Error error)
    | Ok curve ->
        let length = Vector.length curve.false_positive_rates in
        let area =
          Metric_internal.sum (length - 1) (fun index ->
              let left_x = Vector.get curve.false_positive_rates index in
              let right_x = Vector.get curve.false_positive_rates (index + 1) in
              let left_y = Vector.get curve.true_positive_rates index in
              let right_y = Vector.get curve.true_positive_rates (index + 1) in
              (right_x -. left_x) *. (left_y +. right_y) *. 0.5)
        in
        Metric_internal.finite ~operation:"ROC AUC" area
end

module Regression_scorer = struct
  type metric =
    | Mean_absolute_error
    | Mean_squared_error
    | Root_mean_squared_error
    | R2

  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t = params
  type truth = Target.regression Target.t
  type prediction = Target.regression Target.t

  let create ?(undefined = Undefined_metric_policy.Error) metric =
    { metric; undefined }

  let neg_mean_absolute_error = create Mean_absolute_error
  let neg_mean_squared_error = create Mean_squared_error
  let neg_root_mean_squared_error = create Root_mean_squared_error
  let r2 ?undefined () = create ?undefined R2
  let clone specification = specification
  let params specification = specification

  let name specification =
    match specification.metric with
    | Mean_absolute_error -> "neg_mean_absolute_error"
    | Mean_squared_error -> "neg_mean_squared_error"
    | Root_mean_squared_error -> "neg_root_mean_squared_error"
    | R2 -> "r2"

  let score specification ?sample_weight ~truth ~prediction () =
    match specification.metric with
    | Mean_absolute_error ->
        Regression_metrics.mean_absolute_error ?sample_weight ~truth ~prediction
          ()
        |> Result.map Float.neg
    | Mean_squared_error ->
        Regression_metrics.mean_squared_error ?sample_weight ~truth ~prediction
          ()
        |> Result.map Float.neg
    | Root_mean_squared_error ->
        Regression_metrics.root_mean_squared_error ?sample_weight ~truth
          ~prediction ()
        |> Result.map Float.neg
    | R2 ->
        Regression_metrics.r2 ~undefined:specification.undefined ?sample_weight
          ~truth ~prediction ()
end

module Binary_classification_scorer = struct
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision
    | Recall
    | F1
    | Log_loss
    | Roc_auc

  type response = Labels | Positive_probabilities

  type params = {
    metric : metric;
    positive_label : int;
    undefined : Undefined_metric_policy.t;
  }

  type t = params
  type truth = Target.classification Target.t
  type prediction = Binary_prediction.t

  let create ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      metric =
    { metric; positive_label; undefined }

  let accuracy = create Accuracy

  let balanced_accuracy ?positive_label ?undefined () =
    create ?positive_label ?undefined Balanced_accuracy

  let precision ?positive_label ?undefined () =
    create ?positive_label ?undefined Precision

  let recall ?positive_label ?undefined () =
    create ?positive_label ?undefined Recall

  let f1 ?positive_label ?undefined () = create ?positive_label ?undefined F1
  let neg_log_loss ?positive_label () = create ?positive_label Log_loss

  let roc_auc ?positive_label ?undefined () =
    create ?positive_label ?undefined Roc_auc

  let clone specification = specification
  let params specification = specification

  let response specification =
    match specification.metric with
    | Accuracy | Balanced_accuracy | Precision | Recall | F1 -> Labels
    | Log_loss | Roc_auc -> Positive_probabilities

  let name specification =
    match specification.metric with
    | Accuracy -> "accuracy"
    | Balanced_accuracy -> "balanced_accuracy"
    | Precision -> "precision"
    | Recall -> "recall"
    | F1 -> "f1"
    | Log_loss -> "neg_log_loss"
    | Roc_auc -> "roc_auc"

  let missing_response name =
    Error
      (Metric_internal.validation ~name:"binary scorer prediction"
         ~reason:(name ^ " are required by this scorer")
         ~remediation:"provide the classifier response requested by the scorer")

  let score specification ?sample_weight ~truth ~prediction () =
    let score_labels metric =
      match Binary_prediction.labels prediction with
      | None -> missing_response "predicted labels"
      | Some prediction -> metric prediction
    in
    let score_probabilities metric =
      match Binary_prediction.positive_probabilities prediction with
      | None -> missing_response "positive-class probabilities"
      | Some prediction -> metric prediction
    in
    match specification.metric with
    | Accuracy ->
        score_labels (fun prediction ->
            Binary_classification_metrics.accuracy ?sample_weight ~truth
              ~prediction ())
    | Balanced_accuracy ->
        score_labels (fun prediction ->
            Binary_classification_metrics.balanced_accuracy
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Precision ->
        score_labels (fun prediction ->
            Binary_classification_metrics.precision
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Recall ->
        score_labels (fun prediction ->
            Binary_classification_metrics.recall
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | F1 ->
        score_labels (fun prediction ->
            Binary_classification_metrics.f1
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Log_loss ->
        score_probabilities (fun positive_probabilities ->
            Binary_classification_metrics.log_loss
              ~positive_label:specification.positive_label ?sample_weight ~truth
              ~positive_probabilities ()
            |> Result.map Float.neg)
    | Roc_auc ->
        score_probabilities (fun positive_probabilities ->
            Binary_classification_metrics.roc_auc
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~positive_probabilities ())
end

module Score_aggregation = struct
  type t = {
    count : int;
    mean : float;
    standard_deviation : float;
    minimum : float;
    maximum : float;
  }

  let summarize ?(undefined = Undefined_metric_policy.Error) values =
    let count = Array.length values in
    if count = 0 then
      Error
        (Metric_internal.validation ~name:"score aggregation"
           ~reason:"no scores were provided"
           ~remediation:"provide at least one fold score")
    else
      let has_nan = ref false in
      let invalid_infinity = ref None in
      Array.iteri
        (fun index value ->
          if Float.is_nan value then has_nan := true
          else if not (Float.is_finite value) then
            invalid_infinity := Some index)
        values;
      match !invalid_infinity with
      | Some index ->
          Error
            (Metric_internal.validation ~name:"score aggregation"
               ~reason:(Format.sprintf "score at index %d is infinite" index)
               ~remediation:"provide finite scores or resolve the failed fold")
      | None when !has_nan && undefined = Undefined_metric_policy.Error ->
          Error
            (Metric_internal.validation ~name:"score aggregation"
               ~reason:"at least one score is undefined (NaN)"
               ~remediation:
                 "resolve undefined fold metrics or choose an explicit \
                  aggregation policy")
      | None when !has_nan && undefined = Undefined_metric_policy.Return_nan ->
          Ok
            {
              count;
              mean = Float.nan;
              standard_deviation = Float.nan;
              minimum = Float.nan;
              maximum = Float.nan;
            }
      | None ->
          let value index =
            let observed = values.(index) in
            if Float.is_nan observed then 0.0 else observed
          in
          let mean = Metric_internal.sum count value /. Float.of_int count in
          let variance =
            Metric_internal.sum count (fun index ->
                let difference = value index -. mean in
                difference *. difference)
            /. Float.of_int count
          in
          let minimum = ref (value 0) in
          let maximum = ref (value 0) in
          for index = 1 to count - 1 do
            minimum := Float.min !minimum (value index);
            maximum := Float.max !maximum (value index)
          done;
          if Float.is_finite mean && Float.is_finite variance then
            Ok
              {
                count;
                mean;
                standard_deviation = Float.sqrt (Float.max 0.0 variance);
                minimum = !minimum;
                maximum = !maximum;
              }
          else
            Error
              (Metric_internal.numerical ~operation:"score aggregation"
                 ~reason:"the aggregate is not representable as float64")
end
