open Modelkit_data
open Modelkit_protocols
module P = Modelkit_preprocessing.Preprocessing_internal

module Internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let data_error ~remediation error = Error.of_data_error ~remediation error

  let output_schema input_schema names =
    match Feature_schema.names input_schema with
    | None -> (
        match Feature_schema.anonymous ~feature_count:(Array.length names) with
        | Ok schema -> Ok schema
        | Error error ->
            Error
              (data_error ~remediation:"reduce the number of output features"
                 error))
    | Some _ -> (
        match
          Feature_names.create ~expected_count:(Array.length names) names
        with
        | Ok names -> Ok (Feature_schema.named names)
        | Error error ->
            Error
              (data_error
                 ~remediation:"use valid unique generated feature names" error))

  let input_name schema column =
    match Feature_schema.names schema with
    | None -> string_of_int column
    | Some names -> Feature_name.to_string (Feature_names.get names column)

  let generated_name ~kind ~output source =
    Format.sprintf "%s[%d]:%s" kind output source

  let finite_output ~operation matrix =
    P.validate_values ~operation ~allow_nan:false matrix

  let sorted_unique_float values =
    Array.sort Float.compare values;
    if Array.length values = 0 then [||]
    else
      let count = ref 1 in
      for index = 1 to Array.length values - 1 do
        if Float.compare values.(index) values.(!count - 1) <> 0 then (
          values.(!count) <- values.(index);
          incr count)
      done;
      Array.sub values 0 !count

  let learn_categories schema x column operation =
    let rows = Matrix.rows x in
    if rows = 0 then Error (P.no_observations ~operation schema column)
    else
      Ok
        (Array.init rows (fun row -> Matrix.get x row column)
        |> sorted_unique_float |> Vector.of_array)

  let find_float vector value =
    let rec search lower upper =
      if lower >= upper then None
      else
        let middle = lower + ((upper - lower) / 2) in
        let comparison = Float.compare (Vector.get vector middle) value in
        if comparison = 0 then Some middle
        else if comparison < 0 then search (middle + 1) upper
        else search lower middle
    in
    search 0 (Vector.length vector)

  let unknown_category schema column value =
    Error.make
      ~context:
        (match Feature_schema.names schema with
        | None -> []
        | Some names -> [ Error.Feature (Feature_names.get names column) ])
      ~remediation:
        "fit on all expected categories or select a permissive unknown policy"
      (Error.Validation
         {
           name = "categorical transform";
           reason = Format.sprintf "encountered unknown category %.17g" value;
         })

  let validate_categories ~reject schema categories x =
    if not reject then Ok ()
    else
      let rows = Matrix.rows x in
      let columns = Matrix.columns x in
      let rec loop row column =
        if row = rows then Ok ()
        else if column = columns then loop (row + 1) 0
        else
          let value = Matrix.get x row column in
          match find_float categories.(column) value with
          | Some _ -> loop row (column + 1)
          | None -> Error (unknown_category schema column value)
      in
      loop 0 0

  let copy_vectors vectors = Array.copy vectors
end

module Min_max_scaler = struct
  type params = { feature_range : float * float; clip : bool }
  type t = params

  type fitted = {
    params : params;
    data_min : Vector.t;
    data_max : Vector.t;
    data_range : Vector.t;
    scale : Vector.t;
    offset : Vector.t;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(feature_range = (0.0, 1.0)) ?(clip = false) () =
    let lower, upper = feature_range in
    let width = upper -. lower in
    if
      Float.is_finite lower && Float.is_finite upper && lower < upper
      && Float.is_finite width
    then Ok { feature_range; clip }
    else
      Error
        (Internal.validation ~name:"min-max feature range"
           ~reason:"bounds must be finite, ordered, and have a finite width"
           ~remediation:
             "choose finite lower and upper bounds with lower < upper")

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "min-max scaler" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"min-max scaler" ~allow_nan:false
        feature_schema x
    in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    if rows = 0 && columns > 0 then
      Error (P.no_observations ~operation:"min-max scaler" feature_schema 0)
    else
      let lower, upper = specification.feature_range in
      let target_range = upper -. lower in
      let minima = Array.make columns 0.0 in
      let maxima = Array.make columns 0.0 in
      let ranges = Array.make columns 0.0 in
      let scales = Array.make columns 1.0 in
      let offsets = Array.make columns 0.0 in
      let rec fit_column column =
        if column = columns then Ok ()
        else
          let minimum = ref (Matrix.get x 0 column) in
          let maximum = ref !minimum in
          for row = 1 to rows - 1 do
            let value = Matrix.get x row column in
            minimum := Float.min !minimum value;
            maximum := Float.max !maximum value
          done;
          let range = !maximum -. !minimum in
          if not (Float.is_finite range) then
            Error
              (P.numerical_error ~operation:"min-max scaler fit" feature_schema
                 column "the feature range is not finite")
          else
            let denominator = if range = 0.0 then 1.0 else range in
            let scale = target_range /. denominator in
            let offset = lower -. (!minimum *. scale) in
            if not (Float.is_finite scale && Float.is_finite offset) then
              Error
                (P.numerical_error ~operation:"min-max scaler fit"
                   feature_schema column
                   "the fitted scale or offset is not finite")
            else (
              minima.(column) <- !minimum;
              maxima.(column) <- !maximum;
              ranges.(column) <- range;
              scales.(column) <- scale;
              offsets.(column) <- offset;
              fit_column (column + 1))
      in
      let* () = fit_column 0 in
      Ok
        {
          params = specification;
          data_min = Vector.of_array minima;
          data_max = Vector.of_array maxima;
          data_range = Vector.of_array ranges;
          scale = Vector.of_array scales;
          offset = Vector.of_array offsets;
          schema = feature_schema;
        }

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"min-max scaler" ~allow_nan:false
        ~expected_schema:fitted.schema feature_schema x
    in
    let lower, upper = fitted.params.feature_range in
    let* transformed =
      P.matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
        (fun row column ->
          let value =
            (Matrix.get x row column *. Vector.get fitted.scale column)
            +. Vector.get fitted.offset column
          in
          if fitted.params.clip then Float.min upper (Float.max lower value)
          else value)
    in
    let* () = finite_output ~operation:"min-max scaler output" transformed in
    Ok transformed

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let data_min fitted = fitted.data_min
  let data_max fitted = fitted.data_max
  let data_range fitted = fitted.data_range
  let scale fitted = fitted.scale
  let offset fitted = fitted.offset
end

module Max_abs_scaler = struct
  type params = unit
  type t = params

  type fitted = {
    max_abs : Vector.t;
    scale : Vector.t;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create () = ()
  let clone () = ()
  let params () = ()

  let fit () ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "max-absolute scaler" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"max-absolute scaler" ~allow_nan:false
        feature_schema x
    in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    if rows = 0 && columns > 0 then
      Error
        (P.no_observations ~operation:"max-absolute scaler" feature_schema 0)
    else
      let maxima = Array.make columns 0.0 in
      for column = 0 to columns - 1 do
        for row = 0 to rows - 1 do
          maxima.(column) <-
            Float.max maxima.(column) (Float.abs (Matrix.get x row column))
        done
      done;
      let scales =
        Array.map (fun value -> if value = 0.0 then 1.0 else value) maxima
      in
      Ok
        {
          max_abs = Vector.of_array maxima;
          scale = Vector.of_array scales;
          schema = feature_schema;
        }

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"max-absolute scaler"
        ~allow_nan:false ~expected_schema:fitted.schema feature_schema x
    in
    let* transformed =
      P.matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
        (fun row column ->
          Matrix.get x row column /. Vector.get fitted.scale column)
    in
    let* () =
      finite_output ~operation:"max-absolute scaler output" transformed
    in
    Ok transformed

  let fitted_params _ = ()
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let max_abs fitted = fitted.max_abs
  let scale fitted = fitted.scale
end

module Robust_scaler = struct
  type params = {
    with_centering : bool;
    with_scaling : bool;
    quantile_range : float * float;
  }

  type t = params

  type fitted = {
    params : params;
    center : Vector.t;
    scale : Vector.t;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(with_centering = true) ?(with_scaling = true)
      ?(quantile_range = (25.0, 75.0)) () =
    let lower, upper = quantile_range in
    if
      Float.is_finite lower && Float.is_finite upper && lower >= 0.0
      && lower < upper && upper <= 100.0
    then Ok { with_centering; with_scaling; quantile_range }
    else
      Error
        (Internal.validation ~name:"robust scaler quantile range"
           ~reason:"percentiles must satisfy 0 <= lower < upper <= 100"
           ~remediation:"choose two ordered finite percentiles from 0 to 100")

  let clone specification = specification
  let params specification = specification

  let quantile sorted percentile =
    let position =
      percentile /. 100.0 *. Float.of_int (Array.length sorted - 1)
    in
    let lower = int_of_float (Float.floor position) in
    let upper = int_of_float (Float.ceil position) in
    if lower = upper then sorted.(lower)
    else
      let fraction = position -. Float.of_int lower in
      (sorted.(lower) *. (1.0 -. fraction)) +. (sorted.(upper) *. fraction)

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "robust scaler" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"robust scaler" ~allow_nan:false
        feature_schema x
    in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    if
      rows = 0 && columns > 0
      && (specification.with_centering || specification.with_scaling)
    then Error (P.no_observations ~operation:"robust scaler" feature_schema 0)
    else
      let centers = Array.make columns 0.0 in
      let scales = Array.make columns 1.0 in
      let lower, upper = specification.quantile_range in
      let rec fit_column column =
        if column = columns then Ok ()
        else if not (specification.with_centering || specification.with_scaling)
        then fit_column (column + 1)
        else
          let values = Array.init rows (fun row -> Matrix.get x row column) in
          Array.sort Float.compare values;
          let center =
            if specification.with_centering then quantile values 50.0 else 0.0
          in
          let scale =
            if specification.with_scaling then
              let range = quantile values upper -. quantile values lower in
              if range = 0.0 then 1.0 else range
            else 1.0
          in
          if not (Float.is_finite center && Float.is_finite scale) then
            Error
              (P.numerical_error ~operation:"robust scaler fit" feature_schema
                 column "the fitted center or quantile range is not finite")
          else (
            centers.(column) <- center;
            scales.(column) <- scale;
            fit_column (column + 1))
      in
      let* () = fit_column 0 in
      Ok
        {
          params = specification;
          center = Vector.of_array centers;
          scale = Vector.of_array scales;
          schema = feature_schema;
        }

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"robust scaler" ~allow_nan:false
        ~expected_schema:fitted.schema feature_schema x
    in
    let* transformed =
      P.matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
        (fun row column ->
          let centered =
            if fitted.params.with_centering then
              Matrix.get x row column -. Vector.get fitted.center column
            else Matrix.get x row column
          in
          if fitted.params.with_scaling then
            centered /. Vector.get fitted.scale column
          else centered)
    in
    let* () = finite_output ~operation:"robust scaler output" transformed in
    Ok transformed

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let center fitted = fitted.center
  let scale fitted = fitted.scale
end

module Normalizer = struct
  type norm = L1 | L2 | Max
  type params = { norm : norm }
  type t = params
  type fitted = { params : params; schema : Feature_schema.t }
  type target = unit
  type rng = Rng.t

  let create ?(norm = L2) () = { norm }
  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "normalizer" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"normalizer" ~allow_nan:false
        feature_schema x
    in
    Ok { params = specification; schema = feature_schema }

  let l2_norm x row =
    let scale = ref 0.0 in
    let sum_squares = ref 1.0 in
    for column = 0 to Matrix.columns x - 1 do
      let absolute = Float.abs (Matrix.get x row column) in
      if absolute <> 0.0 then
        if !scale < absolute then (
          let ratio = !scale /. absolute in
          sum_squares := 1.0 +. (!sum_squares *. ratio *. ratio);
          scale := absolute)
        else
          let ratio = absolute /. !scale in
          sum_squares := !sum_squares +. (ratio *. ratio)
    done;
    if !scale = 0.0 then 0.0 else !scale *. Float.sqrt !sum_squares

  let row_norm norm x row =
    match norm with
    | L1 ->
        let accumulator = Reference_backend.Accumulator.create () in
        for column = 0 to Matrix.columns x - 1 do
          Reference_backend.Accumulator.add accumulator
            (Float.abs (Matrix.get x row column))
        done;
        Reference_backend.Accumulator.value accumulator
    | L2 -> l2_norm x row
    | Max ->
        let maximum = ref 0.0 in
        for column = 0 to Matrix.columns x - 1 do
          maximum := Float.max !maximum (Float.abs (Matrix.get x row column))
        done;
        !maximum

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"normalizer" ~allow_nan:false
        ~expected_schema:fitted.schema feature_schema x
    in
    let norms = Array.init (Matrix.rows x) (row_norm fitted.params.norm x) in
    let rec validate row =
      if row = Array.length norms then Ok ()
      else if Float.is_finite norms.(row) then validate (row + 1)
      else
        Error
          (Error.make ~remediation:"rescale numerically extreme input rows"
             (Error.Numerical
                {
                  operation = "normalizer";
                  reason = Format.sprintf "row %d norm is not finite" row;
                }))
    in
    let* () = validate 0 in
    P.matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
      (fun row column ->
        if norms.(row) = 0.0 then Matrix.get x row column
        else Matrix.get x row column /. norms.(row))

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
end

module One_hot_encoder = struct
  type unknown_category = Reject | Ignore

  type params = {
    unknown_category : unknown_category;
    max_output_features : int;
  }

  type t = params

  type fitted = {
    params : params;
    categories : Vector.t array;
    offsets : int array;
    source_columns : int array;
    category_values : float array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(unknown_category = Reject) ?(max_output_features = 100_000) () =
    if max_output_features <= 0 then
      Error
        (Internal.validation ~name:"one-hot maximum output features"
           ~reason:"must be positive"
           ~remediation:"choose a positive output-feature limit")
    else Ok { unknown_category; max_output_features }

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "one-hot encoder" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"one-hot encoder" ~allow_nan:false
        feature_schema x
    in
    let columns = Matrix.columns x in
    let categories = Array.make columns (Vector.of_array [||]) in
    let offsets = Array.make (columns + 1) 0 in
    let rec fit_column column =
      if column = columns then Ok ()
      else
        let* values =
          learn_categories feature_schema x column "one-hot encoder"
        in
        let next = offsets.(column) + Vector.length values in
        if next < offsets.(column) || next > specification.max_output_features
        then
          Error
            (validation ~name:"one-hot output width"
               ~reason:
                 (Format.sprintf "exceeds the configured limit of %d"
                    specification.max_output_features)
               ~remediation:
                 "reduce category cardinality or raise max_output_features")
        else (
          categories.(column) <- values;
          offsets.(column + 1) <- next;
          fit_column (column + 1))
    in
    let* () = fit_column 0 in
    let output_columns = offsets.(columns) in
    let source_columns = Array.make output_columns 0 in
    let category_values = Array.make output_columns 0.0 in
    let names = Array.make output_columns "" in
    for column = 0 to columns - 1 do
      for category = 0 to Vector.length categories.(column) - 1 do
        let output = offsets.(column) + category in
        let value = Vector.get categories.(column) category in
        source_columns.(output) <- column;
        category_values.(output) <- value;
        names.(output) <-
          generated_name ~kind:"one_hot" ~output
            (Format.sprintf "%s=%.17g" (input_name feature_schema column) value)
      done
    done;
    let* output_schema = output_schema feature_schema names in
    Ok
      {
        params = specification;
        categories;
        offsets;
        source_columns;
        category_values;
        input_schema = feature_schema;
        output_schema;
      }

  let validate_input fitted feature_schema x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"one-hot encoder" ~allow_nan:false
        ~expected_schema:fitted.input_schema feature_schema x
    in
    validate_categories
      ~reject:(fitted.params.unknown_category = Reject)
      feature_schema fitted.categories x

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () = validate_input fitted feature_schema x in
    P.matrix ~rows:(Matrix.rows x) ~columns:(Array.length fitted.source_columns)
      (fun row output ->
        if
          Float.compare
            (Matrix.get x row fitted.source_columns.(output))
            fitted.category_values.(output)
          = 0
        then 1.0
        else 0.0)

  let transform_csr fitted ~feature_schema ~x =
    let open Internal in
    let* () = validate_input fitted feature_schema x in
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let row_counts = Array.make rows 0 in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        match
          find_float fitted.categories.(column) (Matrix.get x row column)
        with
        | Some _ -> row_counts.(row) <- row_counts.(row) + 1
        | None -> ()
      done
    done;
    let row_offsets = Array.make (rows + 1) 0 in
    for row = 0 to rows - 1 do
      row_offsets.(row + 1) <- row_offsets.(row) + row_counts.(row)
    done;
    let stored = row_offsets.(rows) in
    let column_indices = Array.make stored 0 in
    let values = Array.make stored 1.0 in
    let entry = ref 0 in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        match
          find_float fitted.categories.(column) (Matrix.get x row column)
        with
        | None -> ()
        | Some category ->
            column_indices.(!entry) <- fitted.offsets.(column) + category;
            incr entry
      done
    done;
    match
      Csr_matrix.of_arrays ~rows
        ~columns:(Array.length fitted.source_columns)
        ~row_offsets ~column_indices ~values
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (data_error ~remediation:"report the invalid generated CSR structure"
             error)

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema
  let categories fitted = Internal.copy_vectors fitted.categories
end

module Ordinal_encoder = struct
  type unknown_category = Reject | Use_encoded_value of float
  type params = { unknown_category : unknown_category }
  type t = params

  type fitted = {
    params : params;
    categories : Vector.t array;
    schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(unknown_category = Reject) () =
    match unknown_category with
    | Reject -> Ok { unknown_category }
    | Use_encoded_value value ->
        if Float.is_finite value then Ok { unknown_category }
        else
          Error
            (Internal.validation ~name:"ordinal unknown value"
               ~reason:"must be finite"
               ~remediation:"choose a finite value such as -1")

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "ordinal encoder" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"ordinal encoder" ~allow_nan:false
        feature_schema x
    in
    let categories = Array.make (Matrix.columns x) (Vector.of_array [||]) in
    let rec fit_column column =
      if column = Matrix.columns x then Ok ()
      else
        let* values =
          learn_categories feature_schema x column "ordinal encoder"
        in
        categories.(column) <- values;
        fit_column (column + 1)
    in
    let* () = fit_column 0 in
    let collision =
      match specification.unknown_category with
      | Reject -> None
      | Use_encoded_value value ->
          let rec find column =
            if column = Array.length categories then None
            else if
              value >= 0.0
              && value < Float.of_int (Vector.length categories.(column))
              && Float.floor value = value
            then Some column
            else find (column + 1)
          in
          find 0
    in
    match collision with
    | Some column ->
        Error
          (validation ~name:"ordinal unknown value"
             ~reason:
               (Format.sprintf "collides with a learned code in column %d"
                  column)
             ~remediation:"choose a value outside every learned code range")
    | None -> Ok { params = specification; categories; schema = feature_schema }

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"ordinal encoder" ~allow_nan:false
        ~expected_schema:fitted.schema feature_schema x
    in
    let reject = fitted.params.unknown_category = Reject in
    let* () = validate_categories ~reject feature_schema fitted.categories x in
    P.matrix ~rows:(Matrix.rows x) ~columns:(Matrix.columns x)
      (fun row column ->
        match
          find_float fitted.categories.(column) (Matrix.get x row column)
        with
        | Some category -> Float.of_int category
        | None -> (
            match fitted.params.unknown_category with
            | Reject -> assert false
            | Use_encoded_value value -> value))

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.schema
  let output_schema fitted = fitted.schema
  let categories fitted = Internal.copy_vectors fitted.categories
end

module Label_encoder = struct
  type t = unit
  type fitted = { classes : int array }

  let create () = ()

  let sorted_unique values =
    Array.sort Int.compare values;
    if Array.length values = 0 then values
    else
      let count = ref 1 in
      for index = 1 to Array.length values - 1 do
        if values.(index) <> values.(!count - 1) then (
          values.(!count) <- values.(index);
          incr count)
      done;
      Array.sub values 0 !count

  let fit () ~y =
    let classes = Target.classification_values y |> sorted_unique in
    Ok { classes }

  let find classes value =
    let rec search lower upper =
      if lower >= upper then None
      else
        let middle = lower + ((upper - lower) / 2) in
        if classes.(middle) = value then Some middle
        else if classes.(middle) < value then search (middle + 1) upper
        else search lower middle
    in
    search 0 (Array.length classes)

  let transform fitted target =
    let values = Target.classification_values target in
    let encoded = Array.make (Array.length values) 0 in
    let rec loop index =
      if index = Array.length values then Ok (Target.classification encoded)
      else
        match find fitted.classes values.(index) with
        | Some code ->
            encoded.(index) <- code;
            loop (index + 1)
        | None ->
            Error
              (Internal.validation ~name:"label encoder"
                 ~reason:
                   (Format.sprintf "encountered unknown label %d" values.(index))
                 ~remediation:"fit the encoder on all expected labels")
    in
    loop 0

  let inverse_transform fitted target =
    let codes = Target.classification_values target in
    let decoded = Array.make (Array.length codes) 0 in
    let rec loop index =
      if index = Array.length codes then Ok (Target.classification decoded)
      else
        let code = codes.(index) in
        if code < 0 || code >= Array.length fitted.classes then
          Error
            (Internal.data_error
               ~remediation:"provide codes returned by this fitted encoder"
               (Data_error.Index_out_of_bounds
                  {
                    name = "label code";
                    index = code;
                    upper_bound = Array.length fitted.classes;
                  }))
        else (
          decoded.(index) <- fitted.classes.(code);
          loop (index + 1))
    in
    loop 0

  let classes fitted = Array.copy fitted.classes
end

module Polynomial_features = struct
  type params = {
    degree : int;
    include_bias : bool;
    interaction_only : bool;
    max_output_features : int;
  }

  type t = params

  type fitted = {
    params : params;
    terms : int array array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(degree = 2) ?(include_bias = true) ?(interaction_only = false)
      ?(max_output_features = 100_000) () =
    if degree < 1 then
      Error
        (Internal.validation ~name:"polynomial degree"
           ~reason:"must be at least one" ~remediation:"choose degree >= 1")
    else if max_output_features <= 0 then
      Error
        (Internal.validation ~name:"polynomial maximum output features"
           ~reason:"must be positive"
           ~remediation:"choose a positive output-feature limit")
    else Ok { degree; include_bias; interaction_only; max_output_features }

  let clone specification = specification
  let params specification = specification

  let generate_terms specification columns =
    let terms = ref (if specification.include_bias then [ [||] ] else []) in
    let count = ref (List.length !terms) in
    let exceeded = ref false in
    let add term =
      if !count = specification.max_output_features then exceeded := true
      else (
        terms := Array.copy term :: !terms;
        incr count)
    in
    let rec combinations degree position start term =
      if !exceeded then ()
      else if position = degree then add term
      else
        let remaining = degree - position - 1 in
        let last =
          if specification.interaction_only then columns - remaining - 1
          else columns - 1
        in
        for column = start to last do
          term.(position) <- column;
          combinations degree (position + 1)
            (if specification.interaction_only then column + 1 else column)
            term
        done
    in
    for degree = 1 to specification.degree do
      combinations degree 0 0 (Array.make degree 0)
    done;
    if !exceeded then
      Error
        (Internal.validation ~name:"polynomial output width"
           ~reason:
             (Format.sprintf "exceeds the configured limit of %d"
                specification.max_output_features)
           ~remediation:"reduce degree or raise max_output_features")
    else Ok (Array.of_list (List.rev !terms))

  let term_name schema output term =
    let source =
      if Array.length term = 0 then "1"
      else
        Array.to_list term
        |> List.map (Internal.input_name schema)
        |> String.concat " * "
    in
    Internal.generated_name ~kind:"polynomial" ~output source

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "polynomial features" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"polynomial features" ~allow_nan:false
        feature_schema x
    in
    let* terms = generate_terms specification (Matrix.columns x) in
    let names = Array.mapi (term_name feature_schema) terms in
    let* output_schema = output_schema feature_schema names in
    Ok
      {
        params = specification;
        terms;
        input_schema = feature_schema;
        output_schema;
      }

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"polynomial features"
        ~allow_nan:false ~expected_schema:fitted.input_schema feature_schema x
    in
    let* transformed =
      P.matrix ~rows:(Matrix.rows x) ~columns:(Array.length fitted.terms)
        (fun row output ->
          Array.fold_left
            (fun product column -> product *. Matrix.get x row column)
            1.0 fitted.terms.(output))
    in
    let* () =
      finite_output ~operation:"polynomial features output" transformed
    in
    Ok transformed

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema
  let terms fitted = Array.map Array.copy fitted.terms
end

module Missing_indicator = struct
  type features = Missing_only | All
  type params = { features : features; error_on_new : bool }
  type t = params

  type fitted = {
    params : params;
    selected : int array;
    selected_mask : bool array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  type target = unit
  type rng = Rng.t

  let create ?(features = Missing_only) ?(error_on_new = true) () =
    { features; error_on_new }

  let clone specification = specification
  let params specification = specification

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y:_ () =
    let open Internal in
    let* () = P.reject_sample_weight "missing indicator" sample_weight in
    let* () =
      P.validate_fit_input ~operation:"missing indicator" ~allow_nan:true
        feature_schema x
    in
    let columns = Matrix.columns x in
    let selected = ref [] in
    for column = columns - 1 downto 0 do
      let missing = ref false in
      for row = 0 to Matrix.rows x - 1 do
        if Float.is_nan (Matrix.get x row column) then missing := true
      done;
      if specification.features = All || !missing then
        selected := column :: !selected
    done;
    let selected = Array.of_list !selected in
    let selected_mask = Array.make columns false in
    Array.iter (fun column -> selected_mask.(column) <- true) selected;
    let names =
      Array.mapi
        (fun output column ->
          generated_name ~kind:"missing" ~output
            (input_name feature_schema column))
        selected
    in
    let* output_schema = output_schema feature_schema names in
    Ok
      {
        params = specification;
        selected;
        selected_mask;
        input_schema = feature_schema;
        output_schema;
      }

  let validate_new_missing fitted x =
    if fitted.params.features = All || not fitted.params.error_on_new then Ok ()
    else
      let rows = Matrix.rows x in
      let columns = Matrix.columns x in
      let rec loop row column =
        if row = rows then Ok ()
        else if column = columns then loop (row + 1) 0
        else if
          (not fitted.selected_mask.(column))
          && Float.is_nan (Matrix.get x row column)
        then
          Error
            (Internal.validation ~name:"missing indicator"
               ~reason:
                 (Format.sprintf
                    "feature %d has missing values absent during fitting" column)
               ~remediation:
                 "fit with representative missingness, select All, or disable \
                  error_on_new")
        else loop row (column + 1)
      in
      loop 0 0

  let transform fitted ~feature_schema ~x =
    let open Internal in
    let* () =
      P.validate_transform_input ~operation:"missing indicator" ~allow_nan:true
        ~expected_schema:fitted.input_schema feature_schema x
    in
    let* () = validate_new_missing fitted x in
    P.matrix ~rows:(Matrix.rows x) ~columns:(Array.length fitted.selected)
      (fun row output ->
        if Float.is_nan (Matrix.get x row fitted.selected.(output)) then 1.0
        else 0.0)

  let fitted_params fitted = fitted.params
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema
  let selected_features fitted = Array.copy fitted.selected
end
