open Modelkit_data
open Modelkit_protocols

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
