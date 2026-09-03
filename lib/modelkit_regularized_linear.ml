open Modelkit_data
open Modelkit_protocols
module Solver_report = Modelkit_linear_models.Solver_report
module Linear = Modelkit_linear_models.Linear_model_internal

module Internal = struct
  let ( let* ) = Result.bind

  type solver_params = {
    alpha : float;
    l1_ratio : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type prepared = {
    centered : float array array;
    target : float array;
    weights : float array;
    feature_means : float array;
    target_mean : float;
    feature_norms : float array;
    gradient_scale : float;
  }

  type solution = {
    coefficients : float array;
    intercept : float;
    report : Solver_report.t;
  }

  let validate_solver_params ~name params =
    if (not (Float.is_finite params.alpha)) || params.alpha < 0.0 then
      Error
        (Linear.validation ~name:(name ^ " alpha")
           ~reason:"must be finite and non-negative"
           ~remediation:"choose a finite alpha greater than or equal to zero")
    else if
      (not (Float.is_finite params.l1_ratio))
      || params.l1_ratio < 0.0 || params.l1_ratio > 1.0
    then
      Error
        (Linear.validation ~name:(name ^ " l1_ratio")
           ~reason:"must be finite and between zero and one"
           ~remediation:"choose an l1_ratio in the closed interval [0, 1]")
    else if (not (Float.is_finite params.tolerance)) || params.tolerance <= 0.0
    then
      Error
        (Linear.validation ~name:(name ^ " tolerance")
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite tolerance greater than zero")
    else if params.max_iterations <= 0 then
      Error
        (Linear.validation ~name:(name ^ " max_iterations")
           ~reason:"must be positive"
           ~remediation:"choose at least one coordinate-descent iteration")
    else Ok ()

  let checked_accumulator ~operation ~reason accumulator =
    let value = Reference_backend.Accumulator.value accumulator in
    if Float.is_finite value then Ok value
    else
      Error
        (Linear.numerical ~operation ~reason
           ~remediation:"rescale the features, target, or sample weights")

  let prepare ~operation ~fit_intercept ?sample_weight x target =
    let rows = Matrix.rows x in
    let columns = Matrix.columns x in
    let maximum_weight =
      match sample_weight with
      | None -> 1.0
      | Some weights ->
          let maximum = ref 0.0 in
          for row = 0 to Sample_weight.length weights - 1 do
            maximum := Float.max !maximum (Sample_weight.get weights row)
          done;
          !maximum
    in
    let scaled_weights =
      Array.init rows (fun row ->
          Linear.weight sample_weight row /. maximum_weight)
    in
    let total_weight = Array.fold_left ( +. ) 0.0 scaled_weights in
    if not (Float.is_finite total_weight && total_weight > 0.0) then
      Error
        (Linear.numerical ~operation
           ~reason:"the normalized sample-weight sum is not finite and positive"
           ~remediation:"rescale the sample weights")
    else
      let weights =
        Array.map (fun weight -> weight /. total_weight) scaled_weights
      in
      let feature_means, target_mean =
        if fit_intercept then Linear.weighted_means x target sample_weight
        else (Array.make columns 0.0, 0.0)
      in
      let centered = Array.make_matrix rows columns 0.0 in
      let centered_target = Array.make rows 0.0 in
      let invalid = ref None in
      for row = 0 to rows - 1 do
        let value = Vector.get target row -. target_mean in
        if Float.is_finite value then centered_target.(row) <- value
        else invalid := Some "centering produced a non-finite target";
        for column = 0 to columns - 1 do
          let value = Matrix.get x row column -. feature_means.(column) in
          if Float.is_finite value then centered.(row).(column) <- value
          else invalid := Some "centering produced a non-finite feature"
        done
      done;
      match !invalid with
      | Some reason ->
          Error
            (Linear.numerical ~operation ~reason
               ~remediation:"rescale the features and target")
      | None ->
          let feature_norms = Array.make columns 0.0 in
          let covariances = Array.make columns 0.0 in
          let rec summarize column =
            if column = columns then Ok ()
            else
              let norm = Reference_backend.Accumulator.create () in
              let covariance = Reference_backend.Accumulator.create () in
              for row = 0 to rows - 1 do
                let feature = centered.(row).(column) in
                Reference_backend.Accumulator.add norm
                  (weights.(row) *. feature *. feature);
                Reference_backend.Accumulator.add covariance
                  (weights.(row) *. feature *. centered_target.(row))
              done;
              let* norm =
                checked_accumulator ~operation
                  ~reason:"a weighted feature norm is not finite" norm
              in
              let* covariance =
                checked_accumulator ~operation
                  ~reason:"a weighted feature-target covariance is not finite"
                  covariance
              in
              feature_norms.(column) <- norm;
              covariances.(column) <- covariance;
              summarize (column + 1)
          in
          let* () = summarize 0 in
          let gradient_scale =
            Array.fold_left
              (fun maximum value -> Float.max maximum (Float.abs value))
              1.0 covariances
          in
          Ok
            {
              centered;
              target = centered_target;
              weights;
              feature_means;
              target_mean;
              feature_norms;
              gradient_scale;
            }

  let soft_threshold value threshold =
    if value > threshold then value -. threshold
    else if value < -.threshold then value +. threshold
    else 0.0

  let objective ~operation prepared params coefficients residual =
    let loss = Reference_backend.Accumulator.create () in
    for row = 0 to Array.length residual - 1 do
      Reference_backend.Accumulator.add loss
        (0.5 *. prepared.weights.(row) *. residual.(row) *. residual.(row))
    done;
    let l1 = Reference_backend.Accumulator.create () in
    let l2 = Reference_backend.Accumulator.create () in
    Array.iter
      (fun coefficient ->
        Reference_backend.Accumulator.add l1 (Float.abs coefficient);
        Reference_backend.Accumulator.add l2 (coefficient *. coefficient))
      coefficients;
    let value =
      Reference_backend.Accumulator.value loss
      +. params.alpha *. params.l1_ratio
         *. Reference_backend.Accumulator.value l1
      +. 0.5 *. params.alpha *. (1.0 -. params.l1_ratio)
         *. Reference_backend.Accumulator.value l2
    in
    if Float.is_finite value then Ok value
    else
      Error
        (Linear.numerical ~operation ~reason:"the objective is not finite"
           ~remediation:"rescale the features and target")

  let intercept ~operation prepared fit_intercept coefficients =
    if not fit_intercept then Ok 0.0
    else
      let value = Reference_backend.Accumulator.create () in
      Reference_backend.Accumulator.add value prepared.target_mean;
      for column = 0 to Array.length coefficients - 1 do
        Reference_backend.Accumulator.add value
          (-.(prepared.feature_means.(column) *. coefficients.(column)))
      done;
      checked_accumulator ~operation
        ~reason:"the fitted intercept is not finite" value

  let kkt_violation ~operation prepared params coefficients residual =
    let l1_penalty = params.alpha *. params.l1_ratio in
    let l2_penalty = params.alpha *. (1.0 -. params.l1_ratio) in
    let maximum = ref 0.0 in
    let invalid = ref false in
    for column = 0 to Array.length coefficients - 1 do
      let gradient = Reference_backend.Accumulator.create () in
      for row = 0 to Array.length residual - 1 do
        Reference_backend.Accumulator.add gradient
          (-.(prepared.weights.(row)
             *. prepared.centered.(row).(column)
             *. residual.(row)))
      done;
      let gradient =
        Reference_backend.Accumulator.value gradient
        +. (l2_penalty *. coefficients.(column))
      in
      if not (Float.is_finite gradient) then invalid := true
      else
        let violation =
          if coefficients.(column) > 0.0 then Float.abs (gradient +. l1_penalty)
          else if coefficients.(column) < 0.0 then
            Float.abs (gradient -. l1_penalty)
          else Float.max 0.0 (Float.abs gradient -. l1_penalty)
        in
        maximum := Float.max !maximum violation
    done;
    if !invalid then
      Error
        (Linear.numerical ~operation
           ~reason:"the optimality residual is not finite"
           ~remediation:"rescale the features and target")
    else Ok !maximum

  let solve ~operation prepared params ~initial =
    let rows = Array.length prepared.centered in
    let columns = Array.length prepared.feature_norms in
    let coefficients = Array.copy initial in
    let residual = Array.copy prepared.target in
    let invalid = ref false in
    for row = 0 to rows - 1 do
      for column = 0 to columns - 1 do
        residual.(row) <-
          residual.(row)
          -. (prepared.centered.(row).(column) *. coefficients.(column))
      done;
      if not (Float.is_finite residual.(row)) then invalid := true
    done;
    if !invalid then
      Error
        (Linear.numerical ~operation
           ~reason:"the warm-start residual is not finite"
           ~remediation:"rescale the features and target")
    else
      let l1_penalty = params.alpha *. params.l1_ratio in
      let l2_penalty = params.alpha *. (1.0 -. params.l1_ratio) in
      let rec iterate iteration =
        if iteration > params.max_iterations then
          Error
            (Error.make
               ~remediation:
                 "increase max_iterations, loosen tolerance, or rescale the \
                  data"
               (Error.Convergence
                  {
                    algorithm = "cyclic coordinate descent";
                    reason =
                      Format.sprintf
                        "did not converge for alpha %.17g within %d iterations"
                        params.alpha params.max_iterations;
                  }))
        else
          let maximum_update = ref 0.0 in
          let maximum_coefficient = ref 0.0 in
          let invalid = ref false in
          for column = 0 to columns - 1 do
            let old = coefficients.(column) in
            let covariance = Reference_backend.Accumulator.create () in
            for row = 0 to rows - 1 do
              Reference_backend.Accumulator.add covariance
                (prepared.weights.(row)
                *. prepared.centered.(row).(column)
                *. (residual.(row) +. (prepared.centered.(row).(column) *. old))
                )
            done;
            let covariance = Reference_backend.Accumulator.value covariance in
            let denominator = prepared.feature_norms.(column) +. l2_penalty in
            let updated =
              if
                (not (Float.is_finite covariance))
                || not (Float.is_finite denominator)
              then (
                invalid := true;
                old)
              else if denominator = 0.0 then 0.0
              else soft_threshold covariance l1_penalty /. denominator
            in
            if not (Float.is_finite updated) then invalid := true
            else
              let delta = updated -. old in
              coefficients.(column) <- updated;
              maximum_update := Float.max !maximum_update (Float.abs delta);
              maximum_coefficient :=
                Float.max !maximum_coefficient (Float.abs updated);
              if delta <> 0.0 then
                for row = 0 to rows - 1 do
                  residual.(row) <-
                    residual.(row) -. (prepared.centered.(row).(column) *. delta);
                  if not (Float.is_finite residual.(row)) then invalid := true
                done
          done;
          if !invalid then
            Error
              (Linear.numerical ~operation
                 ~reason:"a coordinate update is not finite"
                 ~remediation:"rescale the features and target")
          else
            let update_threshold =
              params.tolerance *. Float.max 1.0 !maximum_coefficient
            in
            if !maximum_update > update_threshold then iterate (iteration + 1)
            else
              let* violation =
                kkt_violation ~operation prepared params coefficients residual
              in
              if violation > params.tolerance *. prepared.gradient_scale then
                iterate (iteration + 1)
              else
                let* objective =
                  objective ~operation prepared params coefficients residual
                in
                let* intercept =
                  intercept ~operation prepared params.fit_intercept
                    coefficients
                in
                Ok
                  {
                    coefficients;
                    intercept;
                    report =
                      Solver_report.create ~iterations:iteration ~objective
                        ~stopping_reason:Solver_report.Gradient_tolerance
                        ~rank:None;
                  }
      in
      iterate 1

  let fit ~name params ?sample_weight ~feature_schema ~x ~y ~initial () =
    let* () = Linear.validate_matrix feature_schema x in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let target = Target.regression_values y in
    let* prepared =
      prepare ~operation:name ~fit_intercept:params.fit_intercept ?sample_weight
        x target
    in
    let* solution = solve ~operation:name prepared params ~initial in
    Ok (prepared, solution)

  let default_initial x = Array.make (Matrix.columns x) 0.0

  let prediction ~operation ~schema solution feature_schema x =
    Linear.regression_prediction ~operation ~schema
      ~coefficients:solution.coefficients ~intercept:solution.intercept
      feature_schema x

  let validate_path_params ~name ~epsilon ~count ~tolerance ~max_iterations =
    if (not (Float.is_finite epsilon)) || epsilon <= 0.0 || epsilon > 1.0 then
      Error
        (Linear.validation ~name:(name ^ " epsilon")
           ~reason:"must be finite and in the interval (0, 1]"
           ~remediation:
             "choose the desired positive alpha_min / alpha_max ratio")
    else if count <= 0 then
      Error
        (Linear.validation ~name:(name ^ " count") ~reason:"must be positive"
           ~remediation:"request at least one path point")
    else
      validate_solver_params ~name
        {
          alpha = 1.0;
          l1_ratio = 1.0;
          fit_intercept = true;
          tolerance;
          max_iterations;
        }

  let explicit_alphas ~name alphas =
    let values = Vector.to_array alphas in
    if Array.length values = 0 then
      Error
        (Linear.validation ~name:(name ^ " alphas") ~reason:"must not be empty"
           ~remediation:"provide at least one finite non-negative alpha")
    else
      let rec validate index =
        if index = Array.length values then Ok ()
        else if Float.is_finite values.(index) && values.(index) >= 0.0 then
          validate (index + 1)
        else
          Error
            (Linear.validation ~name:(name ^ " alphas")
               ~reason:
                 (Format.sprintf "alpha %d is not finite and non-negative" index)
               ~remediation:"provide only finite non-negative alpha values")
      in
      let* () = validate 0 in
      Array.sort (fun left right -> Float.compare right left) values;
      Ok values

  let automatic_alphas ~name prepared ~l1_ratio ~epsilon ~count =
    if l1_ratio = 0.0 then
      Error
        (Linear.validation ~name:(name ^ " l1_ratio")
           ~reason:"automatic alpha generation is undefined for pure L2 penalty"
           ~remediation:"choose l1_ratio > 0 or supply explicit alphas")
    else
      let maximum = ref 0.0 in
      for column = 0 to Array.length prepared.feature_norms - 1 do
        let covariance = Reference_backend.Accumulator.create () in
        for row = 0 to Array.length prepared.target - 1 do
          Reference_backend.Accumulator.add covariance
            (prepared.weights.(row)
            *. prepared.centered.(row).(column)
            *. prepared.target.(row))
        done;
        maximum :=
          Float.max !maximum
            (Float.abs (Reference_backend.Accumulator.value covariance))
      done;
      let alpha_max = Float.max Float.epsilon (!maximum /. l1_ratio) in
      if count = 1 then Ok [| alpha_max |]
      else
        let alpha_min = alpha_max *. epsilon in
        if not (Float.is_finite alpha_max && alpha_min > 0.0) then
          Error
            (Linear.numerical ~operation:name
               ~reason:"the automatic alpha range is not finite and positive"
               ~remediation:"rescale the data or supply explicit alphas")
        else
          let start = Float.log alpha_max in
          let finish = Float.log alpha_min in
          Ok
            (Array.init count (fun index ->
                 let fraction =
                   Float.of_int index /. Float.of_int (count - 1)
                 in
                 Float.exp (start +. (fraction *. (finish -. start)))))

  let fit_path ~name ~l1_ratio ~fit_intercept ~epsilon ~count ~tolerance
      ~max_iterations ?alphas ?sample_weight ~feature_schema ~x ~y () =
    let* () = Linear.validate_matrix feature_schema x in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let target = Target.regression_values y in
    let* prepared =
      prepare ~operation:name ~fit_intercept ?sample_weight x target
    in
    let* alphas =
      match alphas with
      | Some alphas -> explicit_alphas ~name alphas
      | None -> automatic_alphas ~name prepared ~l1_ratio ~epsilon ~count
    in
    let solutions = Array.make (Array.length alphas) None in
    let rec solve_path index initial =
      if index = Array.length alphas then
        Ok (alphas, Array.map Option.get solutions)
      else
        let params =
          {
            alpha = alphas.(index);
            l1_ratio;
            fit_intercept;
            tolerance;
            max_iterations;
          }
        in
        let* solution = solve ~operation:name prepared params ~initial in
        solutions.(index) <- Some solution;
        solve_path (index + 1) solution.coefficients
    in
    solve_path 0 (default_initial x)

  let path_matrix ~name solutions =
    match
      Matrix.init ~rows:(Array.length solutions)
        ~columns:
          (if Array.length solutions = 0 then 0
           else Array.length solutions.(0).coefficients)
        (fun row column -> solutions.(row).coefficients.(column))
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:("report invalid " ^ name ^ " path dimensions")
             error)

  let path_index ~name length index =
    if index >= 0 && index < length then Ok ()
    else
      Error
        (Error.of_data_error
           ~remediation:"select a path index within the fitted alpha sequence"
           (Data_error.Index_out_of_bounds { name; index; upper_bound = length }))
end

module Lasso_regression = struct
  type params = {
    alpha : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    lasso_params : params;
    lasso_solution : Internal.solution;
    lasso_schema : Feature_schema.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let solver_params params =
    {
      Internal.alpha = params.alpha;
      l1_ratio = 1.0;
      fit_intercept = params.fit_intercept;
      tolerance = params.tolerance;
      max_iterations = params.max_iterations;
    }

  let create ?(alpha = 1.0) ?(fit_intercept = true) ?(tolerance = 1e-4)
      ?(max_iterations = 1000) () =
    let params = { alpha; fit_intercept; tolerance; max_iterations } in
    match
      Internal.validate_solver_params ~name:"lasso regression"
        (solver_params params)
    with
    | Ok () -> Ok params
    | Error _ as error -> error

  let clone specification = specification
  let params specification = specification

  let of_solution params schema solution =
    { lasso_params = params; lasso_solution = solution; lasso_schema = schema }

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let initial = Internal.default_initial x in
    match
      Internal.fit ~name:"lasso regression"
        (solver_params specification)
        ?sample_weight ~feature_schema ~x ~y ~initial ()
    with
    | Ok (_, solution) -> Ok (of_solution specification feature_schema solution)
    | Error _ as error -> error

  let predict fitted ~feature_schema ~x =
    Internal.prediction ~operation:"lasso regression prediction"
      ~schema:fitted.lasso_schema fitted.lasso_solution feature_schema x

  let fitted_params fitted = fitted.lasso_params
  let feature_schema fitted = fitted.lasso_schema

  let coefficients fitted =
    Vector.of_array fitted.lasso_solution.Internal.coefficients

  let intercept fitted = fitted.lasso_solution.Internal.intercept
  let report fitted = fitted.lasso_solution.Internal.report
end

module Elastic_net_regression = struct
  type params = {
    alpha : float;
    l1_ratio : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    elastic_net_params : params;
    elastic_net_solution : Internal.solution;
    elastic_net_schema : Feature_schema.t;
  }

  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  let solver_params params =
    {
      Internal.alpha = params.alpha;
      l1_ratio = params.l1_ratio;
      fit_intercept = params.fit_intercept;
      tolerance = params.tolerance;
      max_iterations = params.max_iterations;
    }

  let create ?(alpha = 1.0) ?(l1_ratio = 0.5) ?(fit_intercept = true)
      ?(tolerance = 1e-4) ?(max_iterations = 1000) () =
    let params =
      { alpha; l1_ratio; fit_intercept; tolerance; max_iterations }
    in
    match
      Internal.validate_solver_params ~name:"elastic-net regression"
        (solver_params params)
    with
    | Ok () -> Ok params
    | Error _ as error -> error

  let clone specification = specification
  let params specification = specification

  let of_solution params schema solution =
    {
      elastic_net_params = params;
      elastic_net_solution = solution;
      elastic_net_schema = schema;
    }

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let initial = Internal.default_initial x in
    match
      Internal.fit ~name:"elastic-net regression"
        (solver_params specification)
        ?sample_weight ~feature_schema ~x ~y ~initial ()
    with
    | Ok (_, solution) -> Ok (of_solution specification feature_schema solution)
    | Error _ as error -> error

  let predict fitted ~feature_schema ~x =
    Internal.prediction ~operation:"elastic-net regression prediction"
      ~schema:fitted.elastic_net_schema fitted.elastic_net_solution
      feature_schema x

  let fitted_params fitted = fitted.elastic_net_params
  let feature_schema fitted = fitted.elastic_net_schema

  let coefficients fitted =
    Vector.of_array fitted.elastic_net_solution.Internal.coefficients

  let intercept fitted = fitted.elastic_net_solution.Internal.intercept
  let report fitted = fitted.elastic_net_solution.Internal.report
end

module Lasso_path = struct
  let ( let* ) = Result.bind

  type params = {
    fit_intercept : bool;
    epsilon : float;
    count : int;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    path_params : params;
    path_alphas : float array;
    path_solutions : Internal.solution array;
    path_coefficients : Matrix.t;
    path_schema : Feature_schema.t;
  }

  let create ?(fit_intercept = true) ?(epsilon = 1e-3) ?(count = 100)
      ?(tolerance = 1e-4) ?(max_iterations = 1000) () =
    match
      Internal.validate_path_params ~name:"lasso path" ~epsilon ~count
        ~tolerance ~max_iterations
    with
    | Ok () -> Ok { fit_intercept; epsilon; count; tolerance; max_iterations }
    | Error _ as error -> error

  let fit (specification : t) ?alphas ?sample_weight ~rng:_ ~feature_schema ~x
      ~y () =
    let* alphas, solutions =
      Internal.fit_path ~name:"lasso path" ~l1_ratio:1.0
        ~fit_intercept:specification.fit_intercept
        ~epsilon:specification.epsilon ~count:specification.count
        ~tolerance:specification.tolerance
        ~max_iterations:specification.max_iterations ?alphas ?sample_weight
        ~feature_schema ~x ~y ()
    in
    let* coefficients = Internal.path_matrix ~name:"lasso" solutions in
    Ok
      {
        path_params = specification;
        path_alphas = alphas;
        path_solutions = solutions;
        path_coefficients = coefficients;
        path_schema = feature_schema;
      }

  let params specification = specification
  let alphas fitted = Vector.of_array fitted.path_alphas
  let coefficients fitted = fitted.path_coefficients

  let intercepts fitted =
    Vector.unsafe_init (Array.length fitted.path_solutions) (fun index ->
        fitted.path_solutions.(index).Internal.intercept)

  let reports fitted =
    Array.map (fun solution -> solution.Internal.report) fitted.path_solutions

  let model fitted ~index =
    let* () =
      Internal.path_index ~name:"lasso path index"
        (Array.length fitted.path_solutions)
        index
    in
    let params =
      {
        Lasso_regression.alpha = fitted.path_alphas.(index);
        fit_intercept = fitted.path_params.fit_intercept;
        tolerance = fitted.path_params.tolerance;
        max_iterations = fitted.path_params.max_iterations;
      }
    in
    Ok
      (Lasso_regression.of_solution params fitted.path_schema
         fitted.path_solutions.(index))
end

module Elastic_net_path = struct
  let ( let* ) = Result.bind

  type params = {
    l1_ratio : float;
    fit_intercept : bool;
    epsilon : float;
    count : int;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    path_params : params;
    path_alphas : float array;
    path_solutions : Internal.solution array;
    path_coefficients : Matrix.t;
    path_schema : Feature_schema.t;
  }

  let create ?(l1_ratio = 0.5) ?(fit_intercept = true) ?(epsilon = 1e-3)
      ?(count = 100) ?(tolerance = 1e-4) ?(max_iterations = 1000) () =
    let* () =
      Internal.validate_path_params ~name:"elastic-net path" ~epsilon ~count
        ~tolerance ~max_iterations
    in
    let solver : Internal.solver_params =
      {
        Internal.alpha = 1.0;
        l1_ratio;
        fit_intercept;
        tolerance;
        max_iterations;
      }
    in
    let* () = Internal.validate_solver_params ~name:"elastic-net path" solver in
    Ok
      ({ l1_ratio; fit_intercept; epsilon; count; tolerance; max_iterations }
        : params)

  let fit (specification : t) ?alphas ?sample_weight ~rng:_ ~feature_schema ~x
      ~y () =
    let* alphas, solutions =
      Internal.fit_path ~name:"elastic-net path"
        ~l1_ratio:specification.l1_ratio
        ~fit_intercept:specification.fit_intercept
        ~epsilon:specification.epsilon ~count:specification.count
        ~tolerance:specification.tolerance
        ~max_iterations:specification.max_iterations ?alphas ?sample_weight
        ~feature_schema ~x ~y ()
    in
    let* coefficients = Internal.path_matrix ~name:"elastic-net" solutions in
    Ok
      {
        path_params = specification;
        path_alphas = alphas;
        path_solutions = solutions;
        path_coefficients = coefficients;
        path_schema = feature_schema;
      }

  let params specification = specification
  let alphas fitted = Vector.of_array fitted.path_alphas
  let coefficients fitted = fitted.path_coefficients

  let intercepts fitted =
    Vector.unsafe_init (Array.length fitted.path_solutions) (fun index ->
        fitted.path_solutions.(index).Internal.intercept)

  let reports fitted =
    Array.map (fun solution -> solution.Internal.report) fitted.path_solutions

  let model fitted ~index =
    let* () =
      Internal.path_index ~name:"elastic-net path index"
        (Array.length fitted.path_solutions)
        index
    in
    let params =
      {
        Elastic_net_regression.alpha = fitted.path_alphas.(index);
        l1_ratio = fitted.path_params.l1_ratio;
        fit_intercept = fitted.path_params.fit_intercept;
        tolerance = fitted.path_params.tolerance;
        max_iterations = fitted.path_params.max_iterations;
      }
    in
    Ok
      (Elastic_net_regression.of_solution params fitted.path_schema
         fitted.path_solutions.(index))
end
