open Modelkit_data
open Modelkit_protocols
module Linear = Modelkit_linear_models.Linear_model_internal
module Ridge = Modelkit_linear_models.Ridge_regression
module Solver_report = Modelkit_linear_models.Solver_report

module Ridge_classifier = struct
  let ( let* ) = Result.bind

  type params = { alpha : float; fit_intercept : bool }
  type t = params

  type fitted = {
    ridge_classifier_params : params;
    ridge_classifier_coefficients : Matrix.t;
    ridge_classifier_intercepts : float array;
    ridge_classifier_classes : int array;
    ridge_classifier_schema : Feature_schema.t;
    ridge_classifier_reports : Solver_report.t array;
  }

  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  let create ?(alpha = 1.0) ?(fit_intercept = true) () =
    if Float.is_finite alpha && alpha >= 0.0 then Ok { alpha; fit_intercept }
    else
      Error
        (Linear.validation ~name:"ridge classifier alpha"
           ~reason:"must be finite and non-negative"
           ~remediation:"choose a finite alpha greater than or equal to zero")

  let clone specification = specification
  let params specification = specification

  let effective_classes target sample_weight =
    let values = Target.classification_values target in
    let distinct = Hashtbl.create (Array.length values) in
    Array.iteri
      (fun row value ->
        if Linear.weight sample_weight row > 0.0 then
          Hashtbl.replace distinct value ())
      values;
    let classes = Hashtbl.to_seq_keys distinct |> Array.of_seq in
    Array.sort Int.compare classes;
    if Array.length classes >= 2 then Ok classes
    else
      Error
        (Linear.validation ~name:"ridge classifier classes"
           ~reason:"at least two positively weighted classes are required"
           ~remediation:"provide training rows from at least two classes")

  let coefficient_matrix models =
    let rows = Array.length models in
    let columns =
      if rows = 0 then 0 else Vector.length (Ridge.coefficients models.(0))
    in
    let values =
      Array.map
        (fun model -> Ridge.coefficients model |> Vector.to_array)
        models
    in
    match Matrix.of_arrays values with
    | Ok matrix when Matrix.shape matrix = (rows, columns) -> Ok matrix
    | Ok _ -> assert false
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"report invalid fitted ridge-classifier dimensions"
             error)

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = Linear.validate_matrix feature_schema x in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let* classes = effective_classes y sample_weight in
    let labels = Target.classification_values y in
    let* ridge_specification =
      Ridge.create ~alpha:specification.alpha
        ~fit_intercept:specification.fit_intercept ()
    in
    let models = Array.make (Array.length classes) None in
    let rec fit_class index =
      if index = Array.length classes then Ok (Array.map Option.get models)
      else
        let encoded =
          Vector.unsafe_init (Array.length labels) (fun row ->
              if labels.(row) = classes.(index) then 1.0 else -1.0)
        in
        let* target =
          match Target.regression encoded with
          | Ok target -> Ok target
          | Error error ->
              Error
                (Error.of_data_error
                   ~remediation:"report invalid ridge-classifier target data"
                   error)
        in
        let class_rng =
          Rng.derive (Rng.to_seed rng) ~operation:"ridge classifier class"
            ~index
          |> Rng.create
        in
        let* model =
          Ridge.fit ridge_specification ?sample_weight ~rng:class_rng
            ~feature_schema ~x ~y:target ()
        in
        models.(index) <- Some model;
        fit_class (index + 1)
    in
    let* models = fit_class 0 in
    let* coefficients = coefficient_matrix models in
    Ok
      {
        ridge_classifier_params = specification;
        ridge_classifier_coefficients = coefficients;
        ridge_classifier_intercepts = Array.map Ridge.intercept models;
        ridge_classifier_classes = classes;
        ridge_classifier_schema = feature_schema;
        ridge_classifier_reports = Array.map Ridge.report models;
      }

  let decision_function fitted ~feature_schema ~x =
    let* () =
      Linear.validate_prediction_input ~schema:fitted.ridge_classifier_schema
        feature_schema x
    in
    let rows = Matrix.rows x in
    let class_count = Array.length fitted.ridge_classifier_classes in
    let feature_count = Matrix.columns x in
    let values = Array.make_matrix rows class_count 0.0 in
    let rec score row class_index =
      if row = rows then Ok ()
      else if class_index = class_count then score (row + 1) 0
      else
        let accumulator = Reference_backend.Accumulator.create () in
        Reference_backend.Accumulator.add accumulator
          fitted.ridge_classifier_intercepts.(class_index);
        for feature = 0 to feature_count - 1 do
          Reference_backend.Accumulator.add accumulator
            (Matrix.get x row feature
            *. Matrix.get fitted.ridge_classifier_coefficients class_index
                 feature)
        done;
        let value = Reference_backend.Accumulator.value accumulator in
        if Float.is_finite value then (
          values.(row).(class_index) <- value;
          score row (class_index + 1))
        else
          Error
            (Linear.numerical ~operation:"ridge classifier decision function"
               ~reason:
                 (Format.sprintf "score for row %d and class %d is not finite"
                    row
                    fitted.ridge_classifier_classes.(class_index))
               ~remediation:"rescale the prediction features")
    in
    let* () = score 0 0 in
    match
      Matrix.init ~rows ~columns:class_count (fun row class_index ->
          values.(row).(class_index))
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"report invalid ridge-classifier score dimensions"
             error)

  let predict fitted ~feature_schema ~x =
    let* decisions = decision_function fitted ~feature_schema ~x in
    let predictions =
      Array.init (Matrix.rows decisions) (fun row ->
          let best = ref 0 in
          for class_index = 1 to Matrix.columns decisions - 1 do
            if
              Matrix.get decisions row class_index
              > Matrix.get decisions row !best
            then best := class_index
          done;
          fitted.ridge_classifier_classes.(!best))
    in
    Ok (Target.classification predictions)

  let fitted_params fitted = fitted.ridge_classifier_params
  let feature_schema fitted = fitted.ridge_classifier_schema
  let coefficients fitted = fitted.ridge_classifier_coefficients
  let intercepts fitted = Vector.of_array fitted.ridge_classifier_intercepts
  let classes fitted = Array.copy fitted.ridge_classifier_classes
  let reports fitted = Array.copy fitted.ridge_classifier_reports
end

module Multinomial_logistic_regression = struct
  let ( let* ) = Result.bind

  type params = {
    c : float;
    fit_intercept : bool;
    tolerance : float;
    max_iterations : int;
  }

  type t = params

  type fitted = {
    multinomial_params : params;
    multinomial_coefficients : Matrix.t;
    multinomial_intercepts : float array;
    multinomial_classes : int array;
    multinomial_schema : Feature_schema.t;
    multinomial_report : Solver_report.t;
  }

  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  let create ?(c = 1.0) ?(fit_intercept = true) ?(tolerance = 1e-8)
      ?(max_iterations = 100) () =
    if (not (Float.is_finite c)) || c <= 0.0 then
      Error
        (Linear.validation ~name:"multinomial logistic regression c"
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite c greater than zero")
    else if (not (Float.is_finite tolerance)) || tolerance <= 0.0 then
      Error
        (Linear.validation ~name:"multinomial logistic regression tolerance"
           ~reason:"must be finite and positive"
           ~remediation:"choose a finite tolerance greater than zero")
    else if max_iterations <= 0 then
      Error
        (Linear.validation
           ~name:"multinomial logistic regression max_iterations"
           ~reason:"must be positive"
           ~remediation:"choose at least one iteration")
    else Ok { c; fit_intercept; tolerance; max_iterations }

  let clone specification = specification
  let params specification = specification

  let effective_classes target sample_weight =
    let values = Target.classification_values target in
    let distinct = Hashtbl.create (Array.length values) in
    Array.iteri
      (fun row value ->
        if Linear.weight sample_weight row > 0.0 then
          Hashtbl.replace distinct value ())
      values;
    let classes = Hashtbl.to_seq_keys distinct |> Array.of_seq in
    Array.sort Int.compare classes;
    if Array.length classes >= 3 then Ok classes
    else
      Error
        (Linear.validation ~name:"multinomial logistic regression classes"
           ~reason:"at least three positively weighted classes are required"
           ~remediation:
             "use binary logistic regression for two classes or provide at \
              least three effective classes")

  let maximum_weight sample_weight rows =
    let maximum = ref 0.0 in
    for row = 0 to rows - 1 do
      maximum := Float.max !maximum (Linear.weight sample_weight row)
    done;
    !maximum

  let infinity_norm values =
    Array.fold_left
      (fun maximum value -> Float.max maximum (Float.abs value))
      0.0 values

  let class_scores ~free_classes ~features ~width ~fit_intercept x row values =
    let scores = Array.make (free_classes + 1) 0.0 in
    let total = ref 0.0 in
    for class_index = 0 to free_classes - 1 do
      let offset = class_index * width in
      let accumulator = Reference_backend.Accumulator.create () in
      for feature = 0 to features - 1 do
        Reference_backend.Accumulator.add accumulator
          (Matrix.get x row feature *. values.(offset + feature))
      done;
      if fit_intercept then
        Reference_backend.Accumulator.add accumulator values.(offset + features);
      let score = Reference_backend.Accumulator.value accumulator in
      scores.(class_index) <- score;
      total := !total +. score
    done;
    scores.(free_classes) <- -. !total;
    scores

  let probabilities scores =
    let maximum = Array.fold_left Float.max Float.neg_infinity scores in
    let values = Array.map (fun score -> Float.exp (score -. maximum)) scores in
    let total = Array.fold_left ( +. ) 0.0 values in
    Array.map (fun value -> value /. total) values

  let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
    let* () = Linear.validate_matrix feature_schema x in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let* classes = effective_classes y sample_weight in
    let rows = Matrix.rows x in
    let features = Matrix.columns x in
    let class_count = Array.length classes in
    let free_classes = class_count - 1 in
    (* The final class is the negative sum of the free classes. This removes
       softmax's common-score null direction without changing its symmetric L2
       penalty. *)
    let width = features + if specification.fit_intercept then 1 else 0 in
    let dimensions = free_classes * width in
    let regularization = 1.0 /. specification.c in
    let objective_scale =
      Float.max (maximum_weight sample_weight rows) regularization
    in
    let scaled_regularization = regularization /. objective_scale in
    let labels = Target.classification_values y in
    let class_indices = Hashtbl.create class_count in
    Array.iteri
      (fun index label -> Hashtbl.add class_indices label index)
      classes;
    let expected =
      Array.map
        (fun label ->
          match Hashtbl.find_opt class_indices label with
          | Some index -> index
          | None -> -1)
        labels
    in
    let parameters = Array.make dimensions 0.0 in
    let evaluate values ~with_hessian =
      let objective = ref 0.0 in
      let gradient = Array.make dimensions 0.0 in
      let hessian =
        if with_hessian then Some (Array.make_matrix dimensions dimensions 0.0)
        else None
      in
      for row = 0 to rows - 1 do
        let row_weight = Linear.weight sample_weight row /. objective_scale in
        if row_weight > 0.0 then (
          let scores =
            class_scores ~free_classes ~features ~width
              ~fit_intercept:specification.fit_intercept x row values
          in
          let probability = probabilities scores in
          let maximum = Array.fold_left Float.max Float.neg_infinity scores in
          let exponential_sum =
            Array.fold_left
              (fun total score -> total +. Float.exp (score -. maximum))
              0.0 scores
          in
          objective :=
            !objective
            +. row_weight
               *. (maximum
                  -. scores.(expected.(row))
                  +. Float.log exponential_sum);
          let last_residual =
            probability.(free_classes)
            -. if expected.(row) = free_classes then 1.0 else 0.0
          in
          for left = 0 to dimensions - 1 do
            let left_class = left / width in
            let left_coordinate = left mod width in
            let left_value =
              if left_coordinate = features then 1.0
              else Matrix.get x row left_coordinate
            in
            let residual =
              probability.(left_class)
              -. (if expected.(row) = left_class then 1.0 else 0.0)
              -. last_residual
            in
            gradient.(left) <-
              gradient.(left) +. (row_weight *. residual *. left_value);
            match hessian with
            | None -> ()
            | Some matrix ->
                for right = 0 to left do
                  let right_class = right / width in
                  let right_coordinate = right mod width in
                  let right_value =
                    if right_coordinate = features then 1.0
                    else Matrix.get x row right_coordinate
                  in
                  let left_probability = probability.(left_class) in
                  let right_probability = probability.(right_class) in
                  let last_probability = probability.(free_classes) in
                  let curvature =
                    (if left_class = right_class then left_probability else 0.0)
                    -. (left_probability *. right_probability)
                    +. (left_probability *. last_probability)
                    +. (last_probability *. right_probability)
                    +. (last_probability *. (1.0 -. last_probability))
                  in
                  matrix.(left).(right) <-
                    matrix.(left).(right)
                    +. (row_weight *. curvature *. left_value *. right_value)
                done
          done)
      done;
      for feature = 0 to features - 1 do
        let sum = ref 0.0 in
        for class_index = 0 to free_classes - 1 do
          sum := !sum +. values.((class_index * width) + feature)
        done;
        objective := !objective +. (0.5 *. scaled_regularization *. !sum *. !sum);
        for class_index = 0 to free_classes - 1 do
          let index = (class_index * width) + feature in
          let value = values.(index) in
          objective :=
            !objective +. (0.5 *. scaled_regularization *. value *. value);
          gradient.(index) <-
            gradient.(index) +. (scaled_regularization *. (value +. !sum));
          match hessian with
          | None -> ()
          | Some matrix ->
              for other_class = 0 to free_classes - 1 do
                let other = (other_class * width) + feature in
                matrix.(index).(other) <-
                  (matrix.(index).(other)
                  +. scaled_regularization
                     *. if class_index = other_class then 2.0 else 1.0)
              done
        done
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
    let rec iterate iteration =
      let objective, gradient, hessian =
        evaluate parameters ~with_hessian:true
      in
      if
        (not (Float.is_finite objective))
        || not (Array.for_all Float.is_finite gradient)
      then
        Error
          (Linear.numerical ~operation:"multinomial logistic regression"
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
                  algorithm = "multinomial logistic regression";
                  reason =
                    Format.sprintf
                      "gradient tolerance was not reached after %d iterations"
                      specification.max_iterations;
                }))
      else
        let hessian = Option.get hessian in
        let* solved =
          Linear.solve_least_squares
            ~operation:"multinomial logistic regression Newton step" hessian
            gradient
        in
        if solved.Linear.least_squares_rank < dimensions then
          Error
            (Linear.numerical
               ~operation:"multinomial logistic regression Newton step"
               ~reason:"the Hessian is numerically rank deficient"
               ~remediation:"rescale features or strengthen regularization")
        else
          let directional = ref 0.0 in
          for index = 0 to dimensions - 1 do
            directional :=
              !directional
              +. gradient.(index)
                 *. solved.Linear.least_squares_coefficients.(index)
          done;
          let rec line_search attempts step =
            if attempts = 30 then None
            else
              let candidate =
                Array.mapi
                  (fun index value ->
                    value
                    -. (step *. solved.Linear.least_squares_coefficients.(index)))
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
                        algorithm = "multinomial logistic regression";
                        reason = "damped Newton line search made no progress";
                      }))
          | Some (step, candidate, candidate_objective) ->
              let step_norm =
                step *. infinity_norm solved.Linear.least_squares_coefficients
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
    let coefficients = Array.make_matrix class_count features 0.0 in
    let intercepts = Array.make class_count 0.0 in
    for class_index = 0 to free_classes - 1 do
      let offset = class_index * width in
      for feature = 0 to features - 1 do
        coefficients.(class_index).(feature) <- parameters.(offset + feature);
        coefficients.(free_classes).(feature) <-
          coefficients.(free_classes).(feature) -. parameters.(offset + feature)
      done;
      if specification.fit_intercept then (
        intercepts.(class_index) <- parameters.(offset + features);
        intercepts.(free_classes) <-
          intercepts.(free_classes) -. parameters.(offset + features))
    done;
    let* coefficient_matrix =
      match Matrix.of_arrays coefficients with
      | Ok matrix -> Ok matrix
      | Error error ->
          Error
            (Error.of_data_error
               ~remediation:
                 "report invalid multinomial-logistic coefficient dimensions"
               error)
    in
    Ok
      {
        multinomial_params = specification;
        multinomial_coefficients = coefficient_matrix;
        multinomial_intercepts = intercepts;
        multinomial_classes = classes;
        multinomial_schema = feature_schema;
        multinomial_report =
          Solver_report.create ~iterations ~objective ~stopping_reason
            ~rank:None;
      }

  let decision_function fitted ~feature_schema ~x =
    let* () =
      Linear.validate_prediction_input ~schema:fitted.multinomial_schema
        feature_schema x
    in
    let rows = Matrix.rows x in
    let class_count = Array.length fitted.multinomial_classes in
    let features = Matrix.columns x in
    let values = Array.make_matrix rows class_count 0.0 in
    let rec score row class_index =
      if row = rows then Ok ()
      else if class_index = class_count then score (row + 1) 0
      else
        let accumulator = Reference_backend.Accumulator.create () in
        Reference_backend.Accumulator.add accumulator
          fitted.multinomial_intercepts.(class_index);
        for feature = 0 to features - 1 do
          Reference_backend.Accumulator.add accumulator
            (Matrix.get x row feature
            *. Matrix.get fitted.multinomial_coefficients class_index feature)
        done;
        let value = Reference_backend.Accumulator.value accumulator in
        if Float.is_finite value then (
          values.(row).(class_index) <- value;
          score row (class_index + 1))
        else
          Error
            (Linear.numerical
               ~operation:"multinomial logistic regression decision function"
               ~reason:
                 (Format.sprintf "score for row %d and class %d is not finite"
                    row
                    fitted.multinomial_classes.(class_index))
               ~remediation:"rescale the prediction features")
    in
    let* () = score 0 0 in
    match
      Matrix.init ~rows ~columns:class_count (fun row class_index ->
          values.(row).(class_index))
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:
               "report invalid multinomial-logistic decision dimensions"
             error)

  let predict_proba fitted ~feature_schema ~x =
    let* decisions = decision_function fitted ~feature_schema ~x in
    let rows = Matrix.rows decisions in
    let class_count = Matrix.columns decisions in
    let values = Array.make_matrix rows class_count 0.0 in
    for row = 0 to rows - 1 do
      let scores = Array.init class_count (Matrix.get decisions row) in
      let row_probabilities = probabilities scores in
      Array.blit row_probabilities 0 values.(row) 0 class_count
    done;
    match
      Matrix.init ~rows ~columns:class_count (fun row class_index ->
          values.(row).(class_index))
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:
               "report invalid multinomial-logistic probability dimensions"
             error)

  let predict fitted ~feature_schema ~x =
    let* decisions = decision_function fitted ~feature_schema ~x in
    let predictions =
      Array.init (Matrix.rows decisions) (fun row ->
          let best = ref 0 in
          for class_index = 1 to Matrix.columns decisions - 1 do
            if
              Matrix.get decisions row class_index
              > Matrix.get decisions row !best
            then best := class_index
          done;
          fitted.multinomial_classes.(!best))
    in
    Ok (Target.classification predictions)

  let fitted_params fitted = fitted.multinomial_params
  let feature_schema fitted = fitted.multinomial_schema
  let coefficients fitted = fitted.multinomial_coefficients
  let intercepts fitted = Vector.of_array fitted.multinomial_intercepts
  let classes fitted = Array.copy fitted.multinomial_classes
  let report fitted = fitted.multinomial_report
end
