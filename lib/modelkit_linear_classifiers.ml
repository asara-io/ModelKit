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
