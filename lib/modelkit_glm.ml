open Modelkit_data
open Modelkit_protocols
module Solver_report = Modelkit_linear_models.Solver_report
module Internal = Modelkit_linear_models.Linear_model_internal

module Glm_internal = struct
  let ( let* ) = Result.bind

  type link = Identity | Log

  type specification = {
    name : string;
    power : float;
    alpha : float;
    fit_intercept : bool;
    link : link;
    tolerance : float;
    max_iterations : int;
  }

  type fitted = {
    coefficients : float array;
    intercept : float;
    schema : Feature_schema.t;
    report : Solver_report.t;
  }

  let validate_hyperparameters ~name ~power ~alpha ~tolerance ~max_iterations =
    if not (Float.is_finite power) then
      Error
        (Internal.validation ~name:(name ^ " power") ~reason:"must be finite"
           ~remediation:"choose a finite Tweedie power")
    else if (not (Float.is_finite alpha)) || alpha < 0.0 then
      Error
        (Internal.validation ~name:(name ^ " alpha")
           ~reason:"must be finite and non-negative"
           ~remediation:"choose a finite alpha greater than or equal to zero")
    else if (not (Float.is_finite tolerance)) || tolerance <= 0.0 then
      Error
        (Internal.validation ~name:(name ^ " tolerance")
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite tolerance greater than zero")
    else if max_iterations <= 0 then
      Error
        (Internal.validation ~name:(name ^ " max_iterations")
           ~reason:"must be positive"
           ~remediation:"choose at least one iteration")
    else Ok ()

  let target_domain power value =
    if power <= 0.0 then true
    else if power < 2.0 then value >= 0.0
    else value > 0.0

  let validate_target specification target =
    let values = Target.regression_values target in
    let invalid = ref None in
    for row = 0 to Vector.length values - 1 do
      let value = Vector.get values row in
      if
        Option.is_none !invalid && not (target_domain specification.power value)
      then invalid := Some (row, value)
    done;
    match !invalid with
    | None -> Ok ()
    | Some (row, value) ->
        let domain =
          if specification.power <= 0.0 then "finite real values"
          else if specification.power < 2.0 then "non-negative values"
          else "strictly positive values"
        in
        Error
          (Internal.validation
             ~name:(specification.name ^ " target")
             ~reason:
               (Format.sprintf
                  "effective target at row %d is %.17g; power %.17g requires %s"
                  row value specification.power domain)
             ~remediation:"provide targets in the Tweedie power's domain")

  let normalized_weights rows sample_weight =
    let maximum = ref 0.0 in
    for row = 0 to rows - 1 do
      maximum := Float.max !maximum (Internal.weight sample_weight row)
    done;
    let scaled_total = ref 0.0 in
    let scaled =
      Array.init rows (fun row ->
          let value = Internal.weight sample_weight row /. !maximum in
          scaled_total := !scaled_total +. value;
          value)
    in
    Array.map (fun value -> value /. !scaled_total) scaled

  let inverse_link link power raw =
    match link with
    | Identity ->
        if Float.is_finite raw && (power = 0.0 || raw > 0.0) then Some raw
        else None
    | Log ->
        let mean = Float.exp raw in
        if Float.is_finite mean && mean > 0.0 then Some mean else None

  let loss link power target raw mean =
    if power = 0.0 && link = Identity then
      let residual = mean -. target in
      0.5 *. residual *. residual
    else if power = 1.0 then
      if link = Log then mean -. (target *. raw)
      else mean -. (target *. Float.log mean)
    else if power = 2.0 then
      if link = Log then raw +. (target /. mean)
      else Float.log mean +. (target /. mean)
    else
      (Float.pow mean (2.0 -. power) /. (2.0 -. power))
      -. (target *. Float.pow mean (1.0 -. power) /. (1.0 -. power))

  let gradient_and_curvature link power target mean =
    match link with
    | Identity ->
        let curvature = Float.pow mean (-.power) in
        ((mean -. target) *. curvature, curvature)
    | Log ->
        let scale = Float.pow mean (1.0 -. power) in
        ((mean -. target) *. scale, mean *. scale)

  let infinity_norm values =
    Array.fold_left
      (fun maximum value -> Float.max maximum (Float.abs value))
      0.0 values

  let fit specification ?sample_weight ~feature_schema ~x ~y () =
    let* () = Internal.validate_matrix feature_schema x in
    let* () = Internal.validate_target_length x (Target.length y) in
    let* () = Internal.validate_sample_weight x sample_weight in
    let* () = validate_target specification y in
    let rows = Matrix.rows x in
    let features = Matrix.columns x in
    let dimensions = features + if specification.fit_intercept then 1 else 0 in
    let target = Target.regression_values y in
    let weights = normalized_weights rows sample_weight in
    let parameters = Array.make dimensions 0.0 in
    let weighted_mean =
      let value = ref 0.0 in
      for row = 0 to rows - 1 do
        value := !value +. (weights.(row) *. Vector.get target row)
      done;
      !value
    in
    let* () =
      if not specification.fit_intercept then Ok ()
      else
        match specification.link with
        | Identity when specification.power = 0.0 ->
            parameters.(features) <- weighted_mean;
            Ok ()
        | Identity ->
            if weighted_mean > 0.0 && Float.is_finite weighted_mean then (
              parameters.(features) <- weighted_mean;
              Ok ())
            else
              Error
                (Internal.validation
                   ~name:(specification.name ^ " target mean")
                   ~reason:
                     "the identity link requires a positive effective target \
                      mean"
                   ~remediation:
                     "use a log link, rescale the target, or provide a \
                      positive weighted mean")
        | Log ->
            if weighted_mean > 0.0 && Float.is_finite weighted_mean then (
              parameters.(features) <- Float.log weighted_mean;
              Ok ())
            else
              Error
                (Internal.validation
                   ~name:(specification.name ^ " target mean")
                   ~reason:
                     "the log link requires a positive effective target mean"
                   ~remediation:
                     "provide at least one positive, positively weighted target")
    in
    let linear row values =
      let raw = ref 0.0 in
      for column = 0 to features - 1 do
        raw := !raw +. (Matrix.get x row column *. values.(column))
      done;
      if specification.fit_intercept then !raw +. values.(features) else !raw
    in
    let evaluate values ~with_hessian =
      let objective = ref 0.0 in
      let gradient = Array.make dimensions 0.0 in
      let hessian =
        if with_hessian then Some (Array.make_matrix dimensions dimensions 0.0)
        else None
      in
      let valid = ref true in
      for row = 0 to rows - 1 do
        if !valid && weights.(row) > 0.0 then
          let raw = linear row values in
          match inverse_link specification.link specification.power raw with
          | None -> valid := false
          | Some mean ->
              let row_loss =
                loss specification.link specification.power
                  (Vector.get target row) raw mean
              in
              let derivative, curvature =
                gradient_and_curvature specification.link specification.power
                  (Vector.get target row) mean
              in
              if
                (not (Float.is_finite row_loss))
                || (not (Float.is_finite derivative))
                || (not (Float.is_finite curvature))
                || curvature <= 0.0
              then valid := false
              else (
                objective := !objective +. (weights.(row) *. row_loss);
                let residual = weights.(row) *. derivative in
                let weighted_curvature = weights.(row) *. curvature in
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
                          if right = features then 1.0
                          else Matrix.get x row right
                        in
                        matrix.(left).(right) <-
                          matrix.(left).(right)
                          +. (weighted_curvature *. left_value *. right_value)
                      done
                done)
      done;
      for column = 0 to features - 1 do
        objective :=
          !objective
          +. (0.5 *. specification.alpha *. values.(column) *. values.(column));
        gradient.(column) <-
          gradient.(column) +. (specification.alpha *. values.(column));
        match hessian with
        | None -> ()
        | Some matrix ->
            matrix.(column).(column) <-
              matrix.(column).(column) +. specification.alpha
      done;
      (match hessian with
      | None -> ()
      | Some matrix ->
          for left = 0 to dimensions - 1 do
            for right = 0 to left - 1 do
              matrix.(right).(left) <- matrix.(left).(right)
            done
          done);
      if
        !valid && Float.is_finite !objective
        && Array.for_all Float.is_finite gradient
      then Some (!objective, gradient, hessian)
      else None
    in
    let rec iterate iteration =
      match evaluate parameters ~with_hessian:true with
      | None ->
          Error
            (Internal.numerical ~operation:specification.name
               ~reason:
                 "objective, gradient, or inverse-link value is not finite"
               ~remediation:
                 "rescale features or target, or strengthen regularization")
      | Some (objective, gradient, hessian) -> (
          if infinity_norm gradient <= specification.tolerance then
            Ok (Solver_report.Gradient_tolerance, iteration, objective)
          else if iteration = specification.max_iterations then
            Error
              (Error.make
                 ~remediation:
                   "increase max_iterations, rescale features, or strengthen \
                    regularization"
                 (Error.Convergence
                    {
                      algorithm = specification.name;
                      reason =
                        Format.sprintf
                          "gradient tolerance was not reached after %d \
                           iterations"
                          specification.max_iterations;
                    }))
          else
            let* solved =
              Internal.solve_least_squares
                ~operation:(specification.name ^ " IRLS step")
                (Option.get hessian) gradient
            in
            let {
              Internal.least_squares_rank;
              least_squares_coefficients = direction;
            } =
              solved
            in
            if least_squares_rank < dimensions then
              Error
                (Internal.numerical
                   ~operation:(specification.name ^ " IRLS step")
                   ~reason:"the weighted system is numerically rank deficient"
                   ~remediation:
                     "rescale features, remove redundant columns, or \
                      strengthen regularization")
            else
              let directional = ref 0.0 in
              for index = 0 to dimensions - 1 do
                directional :=
                  !directional +. (gradient.(index) *. direction.(index))
              done;
              let rec line_search attempts step =
                if attempts = 40 then None
                else
                  let candidate =
                    Array.mapi
                      (fun index value -> value -. (step *. direction.(index)))
                      parameters
                  in
                  match evaluate candidate ~with_hessian:false with
                  | Some (candidate_objective, _, _)
                    when candidate_objective
                         <= objective -. (1e-4 *. step *. !directional) ->
                      Some (step, candidate, candidate_objective)
                  | Some _ | None -> line_search (attempts + 1) (step /. 2.0)
              in
              match line_search 0 1.0 with
              | None ->
                  Error
                    (Error.make
                       ~remediation:
                         "rescale features or choose stronger regularization"
                       (Error.Convergence
                          {
                            algorithm = specification.name;
                            reason = "damped IRLS line search made no progress";
                          }))
              | Some (step, candidate, candidate_objective) ->
                  let step_norm = step *. infinity_norm direction in
                  let parameter_norm = infinity_norm parameters in
                  Array.blit candidate 0 parameters 0 dimensions;
                  if
                    step_norm
                    <= specification.tolerance *. (1.0 +. parameter_norm)
                  then
                    Ok
                      ( Solver_report.Step_tolerance,
                        iteration + 1,
                        candidate_objective )
                  else iterate (iteration + 1))
    in
    let* stopping_reason, iterations, objective = iterate 0 in
    Ok
      {
        coefficients = Array.sub parameters 0 features;
        intercept =
          (if specification.fit_intercept then parameters.(features) else 0.0);
        schema = feature_schema;
        report =
          Solver_report.create ~iterations ~objective ~stopping_reason
            ~rank:None;
      }

  let predict ~name ~power ~link fitted ~feature_schema ~x =
    let* () =
      Internal.validate_prediction_input ~schema:fitted.schema feature_schema x
    in
    let rows = Matrix.rows x in
    let features = Matrix.columns x in
    let predictions = Array.make rows 0.0 in
    let rec loop row =
      if row = rows then Ok ()
      else
        let raw = ref fitted.intercept in
        for column = 0 to features - 1 do
          raw :=
            !raw +. (Matrix.get x row column *. fitted.coefficients.(column))
        done;
        match inverse_link link power !raw with
        | Some mean ->
            predictions.(row) <- mean;
            loop (row + 1)
        | None ->
            Error
              (Internal.numerical ~operation:(name ^ " prediction")
                 ~reason:
                   "the inverse link produced a non-finite or out-of-domain \
                    mean"
                 ~remediation:"rescale the prediction features")
    in
    let* () = loop 0 in
    match Target.regression (Vector.of_array predictions) with
    | Ok target -> Ok target
    | Error error ->
        Error
          (Error.of_data_error ~remediation:"rescale the prediction features"
             error)
end

module Tweedie_regression = struct
  let ( let* ) = Result.bind

  type link = Auto | Identity | Log

  type params = {
    power : float;
    alpha : float;
    fit_intercept : bool;
    link : link;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    tweedie_params : params;
    tweedie_coefficients : float array;
    tweedie_intercept : float;
    tweedie_link : link;
    tweedie_schema : Feature_schema.t;
    tweedie_report : Solver_report.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let create ?(power = 0.0) ?(alpha = 1.0) ?(fit_intercept = true)
      ?(link = Auto) ?(tolerance = 1e-8) ?(max_iterations = 100) () =
    let* () =
      Glm_internal.validate_hyperparameters ~name:"Tweedie regression" ~power
        ~alpha ~tolerance ~max_iterations
    in
    Ok ({ power; alpha; fit_intercept; link; tolerance; max_iterations } : t)

  let clone specification = specification
  let params specification = specification

  let resolve_link (specification : t) =
    match specification.link with
    | Auto -> if specification.power <= 0.0 then Identity else Log
    | Identity -> Identity
    | Log -> Log

  let internal_link = function
    | Auto -> assert false
    | Identity -> Glm_internal.Identity
    | Log -> Glm_internal.Log

  let fit (specification : t) ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let resolved = resolve_link specification in
    let* fitted =
      Glm_internal.fit
        {
          Glm_internal.name = "Tweedie regression";
          power = specification.power;
          alpha = specification.alpha;
          fit_intercept = specification.fit_intercept;
          link = internal_link resolved;
          tolerance = specification.tolerance;
          max_iterations = specification.max_iterations;
        }
        ?sample_weight ~feature_schema ~x ~y ()
    in
    Ok
      {
        tweedie_params = specification;
        tweedie_coefficients = fitted.Glm_internal.coefficients;
        tweedie_intercept = fitted.Glm_internal.intercept;
        tweedie_link = resolved;
        tweedie_schema = fitted.Glm_internal.schema;
        tweedie_report = fitted.Glm_internal.report;
      }

  let predict fitted ~feature_schema ~x =
    Glm_internal.predict ~name:"Tweedie regression"
      ~power:fitted.tweedie_params.power
      ~link:(internal_link fitted.tweedie_link)
      {
        Glm_internal.coefficients = fitted.tweedie_coefficients;
        intercept = fitted.tweedie_intercept;
        schema = fitted.tweedie_schema;
        report = fitted.tweedie_report;
      }
      ~feature_schema ~x

  let fitted_params fitted = fitted.tweedie_params
  let feature_schema fitted = fitted.tweedie_schema
  let coefficients fitted = Vector.of_array fitted.tweedie_coefficients
  let intercept fitted = fitted.tweedie_intercept
  let resolved_link fitted = fitted.tweedie_link
  let report fitted = fitted.tweedie_report
end

module Poisson_regression = struct
  let ( let* ) = Result.bind

  type params = {
    alpha : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    poisson_params : params;
    poisson_coefficients : float array;
    poisson_intercept : float;
    poisson_schema : Feature_schema.t;
    poisson_report : Solver_report.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let create ?(alpha = 1.0) ?(fit_intercept = true) ?(tolerance = 1e-8)
      ?(max_iterations = 100) () =
    let* () =
      Glm_internal.validate_hyperparameters ~name:"Poisson regression"
        ~power:1.0 ~alpha ~tolerance ~max_iterations
    in
    Ok ({ alpha; fit_intercept; tolerance; max_iterations } : t)

  let clone specification = specification
  let params specification = specification

  let fit (specification : t) ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let* fitted =
      Glm_internal.fit
        {
          Glm_internal.name = "Poisson regression";
          power = 1.0;
          alpha = specification.alpha;
          fit_intercept = specification.fit_intercept;
          link = Glm_internal.Log;
          tolerance = specification.tolerance;
          max_iterations = specification.max_iterations;
        }
        ?sample_weight ~feature_schema ~x ~y ()
    in
    Ok
      {
        poisson_params = specification;
        poisson_coefficients = fitted.Glm_internal.coefficients;
        poisson_intercept = fitted.Glm_internal.intercept;
        poisson_schema = fitted.Glm_internal.schema;
        poisson_report = fitted.Glm_internal.report;
      }

  let predict fitted ~feature_schema ~x =
    Glm_internal.predict ~name:"Poisson regression" ~power:1.0
      ~link:Glm_internal.Log
      {
        Glm_internal.coefficients = fitted.poisson_coefficients;
        intercept = fitted.poisson_intercept;
        schema = fitted.poisson_schema;
        report = fitted.poisson_report;
      }
      ~feature_schema ~x

  let fitted_params fitted = fitted.poisson_params
  let feature_schema fitted = fitted.poisson_schema
  let coefficients fitted = Vector.of_array fitted.poisson_coefficients
  let intercept fitted = fitted.poisson_intercept
  let report fitted = fitted.poisson_report
end
