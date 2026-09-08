open Modelkit_data
open Modelkit_protocols
module Transform_cache = Modelkit_transform_cache.Transform_cache

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

  let validate_sample_weight transformer x = function
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
                  { name = transformer ^ " sample weights"; expected; observed }))

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

  (* Weighted West/Welford moments over positively weighted rows; the
     unweighted path keeps the historical arithmetic so fitted values remain
     byte-stable. *)
  let weighted_column_moments ~operation schema weights x column =
    let rows = Matrix.rows x in
    let maximum = ref 0.0 in
    let total = ref 0.0 in
    for row = 0 to rows - 1 do
      if Sample_weight.get weights row > 0.0 then (
        total := !total +. Sample_weight.get weights row;
        maximum := Float.max !maximum (Float.abs (Matrix.get x row column)))
    done;
    if !total <= 0.0 then Error (no_observations ~operation schema column)
    else if !maximum = 0.0 then Ok (0.0, 0.0)
    else
      let mean = ref 0.0 in
      let m2 = ref 0.0 in
      let accumulated = ref 0.0 in
      for row = 0 to rows - 1 do
        let weight = Sample_weight.get weights row in
        if weight > 0.0 then (
          accumulated := !accumulated +. weight;
          let value = Matrix.get x row column /. !maximum in
          let delta = value -. !mean in
          mean := !mean +. (weight /. !accumulated *. delta);
          m2 := !m2 +. (weight *. delta *. (value -. !mean)))
      done;
      let mean = !mean *. !maximum in
      let normalized_variance = Float.max 0.0 (!m2 /. !accumulated) in
      let standard_deviation = Float.sqrt normalized_variance *. !maximum in
      let variance = standard_deviation *. standard_deviation in
      if Float.is_finite mean && Float.is_finite variance then
        Ok (mean, variance)
      else
        Error
          (numerical_error ~operation schema column
             "weighted moments overflowed")

  let column_moments ?sample_weight ~operation schema x column =
    let rows = Matrix.rows x in
    match sample_weight with
    | Some weights -> weighted_column_moments ~operation schema weights x column
    | None ->
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
            let normalized_variance =
              Float.max 0.0 (!m2 /. Float.of_int rows)
            in
            let standard_deviation =
              Float.sqrt normalized_variance *. !maximum
            in
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

module Cache_wire = struct
  let ( let* ) = Result.bind
  let max_features = 1_000_000
  let max_string_bytes = 1_048_576

  let failure component reason =
    Error
      (Error.make
         ~remediation:"discard the cache entry and refit the transformer"
         (Error.Compatibility { component; reason }))

  module Writer = struct
    let create magic =
      let writer = Buffer.create 256 in
      Buffer.add_string writer magic;
      writer

    let u8 writer value = Buffer.add_char writer (Char.chr value)

    let i64 writer value =
      for shift = 7 downto 0 do
        u8 writer
          (Int64.to_int
             (Int64.logand (Int64.shift_right_logical value (shift * 8)) 0xffL))
      done

    let length writer value = i64 writer (Int64.of_int value)
    let bool writer value = u8 writer (if value then 1 else 0)
    let float writer value = i64 writer (Int64.bits_of_float value)

    let string writer value =
      length writer (String.length value);
      Buffer.add_string writer value

    let vector writer vector =
      length writer (Vector.length vector);
      for index = 0 to Vector.length vector - 1 do
        float writer (Vector.get vector index)
      done

    let schema writer schema =
      length writer (Feature_schema.feature_count schema);
      match Feature_schema.names schema with
      | None -> bool writer false
      | Some names ->
          bool writer true;
          for index = 0 to Feature_names.length names - 1 do
            string writer
              (Feature_name.to_string (Feature_names.get names index))
          done

    let contents writer = Bytes.of_string (Buffer.contents writer)
  end

  module Reader = struct
    type t = { bytes : bytes; component : string; mutable position : int }

    let create ~component bytes = { bytes; component; position = 0 }
    let remaining reader = Bytes.length reader.bytes - reader.position

    let require reader count =
      if count < 0 || count > remaining reader then
        failure reader.component "cache payload is truncated"
      else Ok ()

    let literal reader expected =
      let length = String.length expected in
      let* () = require reader length in
      let observed = Bytes.sub_string reader.bytes reader.position length in
      reader.position <- reader.position + length;
      if String.equal observed expected then Ok ()
      else failure reader.component "cache payload has an invalid header"

    let u8 reader =
      let* () = require reader 1 in
      let value = Char.code (Bytes.get reader.bytes reader.position) in
      reader.position <- reader.position + 1;
      Ok value

    let i64 reader =
      let* () = require reader 8 in
      let value = ref 0L in
      for _ = 0 to 7 do
        value :=
          Int64.logor
            (Int64.shift_left !value 8)
            (Int64.of_int (Char.code (Bytes.get reader.bytes reader.position)));
        reader.position <- reader.position + 1
      done;
      Ok !value

    let bounded_length reader ~name ~maximum =
      let* value = i64 reader in
      if value < 0L || value > Int64.of_int maximum then
        failure reader.component (name ^ " exceeds the decoder limit")
      else Ok (Int64.to_int value)

    let bool reader =
      let* value = u8 reader in
      match value with
      | 0 -> Ok false
      | 1 -> Ok true
      | _ -> failure reader.component "cache payload contains an invalid bool"

    let float reader = Result.map Int64.float_of_bits (i64 reader)

    let string reader =
      let* length =
        bounded_length reader ~name:"feature-name length"
          ~maximum:max_string_bytes
      in
      let* () = require reader length in
      let value = Bytes.sub_string reader.bytes reader.position length in
      reader.position <- reader.position + length;
      Ok value

    let vector reader =
      let* length =
        bounded_length reader ~name:"vector length" ~maximum:max_features
      in
      let* () =
        if length > remaining reader / 8 then
          failure reader.component "cache vector is truncated"
        else Ok ()
      in
      let values = Array.make length 0.0 in
      let rec loop index =
        if index = length then Ok (Vector.of_array values)
        else
          let* value = float reader in
          values.(index) <- value;
          loop (index + 1)
      in
      loop 0

    let schema reader =
      let* feature_count =
        bounded_length reader ~name:"feature count" ~maximum:max_features
      in
      let* named = bool reader in
      if not named then
        Feature_schema.anonymous ~feature_count
        |> Result.map_error (fun error ->
            Error.of_data_error
              ~remediation:"discard the cache entry and refit the transformer"
              error)
      else
        let names = Array.make feature_count "" in
        let rec loop index =
          if index = feature_count then Ok ()
          else
            let* name = string reader in
            names.(index) <- name;
            loop (index + 1)
        in
        let* () = loop 0 in
        let* names =
          Feature_names.create ~expected_count:feature_count names
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"discard the cache entry and refit the transformer"
                error)
        in
        Ok (Feature_schema.named names)

    let finish reader =
      if remaining reader = 0 then Ok ()
      else failure reader.component "cache payload contains trailing data"
  end

  let component name =
    match
      Transform_cache.Component.create ~package:"modelkit" ~name ~version:1
    with
    | Ok component -> component
    | Error _ -> invalid_arg "invalid built-in cache component identity"

  let configuration writer =
    Transform_cache.Content_id.of_bytes (Writer.contents writer)

  let finite_vector vector =
    let rec loop index =
      index = Vector.length vector
      || (Float.is_finite (Vector.get vector index) && loop (index + 1))
    in
    loop 0
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
  let cache_component = Cache_wire.component "simple-imputer"

  let cache_configuration specification =
    let writer = Cache_wire.Writer.create "MKIC01" in
    (match specification.strategy with
    | Mean -> Cache_wire.Writer.u8 writer 0
    | Median -> Cache_wire.Writer.u8 writer 1
    | Constant value ->
        Cache_wire.Writer.u8 writer 2;
        Cache_wire.Writer.float writer value);
    Cache_wire.configuration writer

  let encode_fitted fitted =
    let writer = Cache_wire.Writer.create "MKIF01" in
    (match fitted.params.strategy with
    | Mean -> Cache_wire.Writer.u8 writer 0
    | Median -> Cache_wire.Writer.u8 writer 1
    | Constant value ->
        Cache_wire.Writer.u8 writer 2;
        Cache_wire.Writer.float writer value);
    Cache_wire.Writer.vector writer fitted.statistics;
    Cache_wire.Writer.schema writer fitted.schema;
    Ok (Cache_wire.Writer.contents writer)

  let decode_fitted payload =
    let component_name = "simple imputer" in
    let reader = Cache_wire.Reader.create ~component:component_name payload in
    let open Cache_wire in
    let* () = Reader.literal reader "MKIF01" in
    let* strategy_tag = Reader.u8 reader in
    let* strategy =
      match strategy_tag with
      | 0 -> Ok Mean
      | 1 -> Ok Median
      | 2 -> Result.map (fun value -> Constant value) (Reader.float reader)
      | _ -> failure component_name "cache payload has an unknown strategy"
    in
    let* statistics = Reader.vector reader in
    let* schema = Reader.schema reader in
    let* () = Reader.finish reader in
    let valid_strategy =
      match strategy with
      | Constant value -> Float.is_finite value
      | Mean | Median -> true
    in
    if
      (not valid_strategy)
      || Vector.length statistics <> Feature_schema.feature_count schema
      || not (finite_vector statistics)
    then failure component_name "cache payload contains invalid fitted state"
    else Ok { params = { strategy }; statistics; schema }
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
    let* () =
      validate_fit_input ~operation:"standard scaler" ~allow_nan:false
        feature_schema x
    in
    let* () = validate_sample_weight "standard scaler" x sample_weight in
    let columns = Matrix.columns x in
    let means = Array.make columns 0.0 in
    let variances = Array.make columns 0.0 in
    let scales = Array.make columns 1.0 in
    let rec fit_column column =
      if column = columns then Ok ()
      else
        let* mean, variance =
          column_moments ?sample_weight ~operation:"standard scaler fit"
            feature_schema x column
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
  let cache_component = Cache_wire.component "standard-scaler"

  let cache_configuration specification =
    let writer = Cache_wire.Writer.create "MKSC01" in
    Cache_wire.Writer.bool writer specification.with_mean;
    Cache_wire.Writer.bool writer specification.with_std;
    Cache_wire.configuration writer

  let encode_fitted fitted =
    let writer = Cache_wire.Writer.create "MKSF01" in
    Cache_wire.Writer.bool writer fitted.params.with_mean;
    Cache_wire.Writer.bool writer fitted.params.with_std;
    Cache_wire.Writer.vector writer fitted.mean;
    Cache_wire.Writer.vector writer fitted.variance;
    Cache_wire.Writer.vector writer fitted.scale;
    Cache_wire.Writer.schema writer fitted.schema;
    Ok (Cache_wire.Writer.contents writer)

  let decode_fitted payload =
    let component_name = "standard scaler" in
    let reader = Cache_wire.Reader.create ~component:component_name payload in
    let open Cache_wire in
    let* () = Reader.literal reader "MKSF01" in
    let* with_mean = Reader.bool reader in
    let* with_std = Reader.bool reader in
    let* mean = Reader.vector reader in
    let* variance = Reader.vector reader in
    let* scale = Reader.vector reader in
    let* schema = Reader.schema reader in
    let* () = Reader.finish reader in
    let width = Feature_schema.feature_count schema in
    let valid_variance value = Float.is_finite value && value >= 0.0 in
    let valid_scale value = Float.is_finite value && value > 0.0 in
    if
      Vector.length mean <> width
      || Vector.length variance <> width
      || Vector.length scale <> width
      || (not (finite_vector mean))
      || (not (Array.for_all valid_variance (Vector.to_array variance)))
      || not (Array.for_all valid_scale (Vector.to_array scale))
    then failure component_name "cache payload contains invalid fitted state"
    else Ok { params = { with_mean; with_std }; mean; variance; scale; schema }
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
