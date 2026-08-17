open Modelkit_data
open Modelkit_protocols

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
