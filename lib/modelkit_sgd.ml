open Modelkit_data
open Modelkit_protocols
module Linear = Modelkit_linear_models.Linear_model_internal

module Sgd_regressor = struct
  let ( let* ) = Result.bind

  type penalty = No_penalty | L1 | L2 | Elastic_net
  type learning_rate = Constant | Inverse_scaling of { power_t : float }
  type stopping_reason = Epoch_limit | Step_tolerance | Partial_fit

  type report = {
    converged : bool;
    batches_processed : int;
    updates : int;
    objective : float;
    stopping_reason : stopping_reason;
  }

  type params = {
    penalty : penalty;
    alpha : float;
    l1_ratio : float;
    fit_intercept : bool;
    learning_rate : learning_rate;
    eta0 : float;
    max_epochs : int;
    tolerance : float option;
    shuffle : bool;
  }

  type t = params
  type target = Target.regression Target.t
  type prediction = Target.regression Target.t
  type rng = Rng.t

  type checkpoint = {
    checkpoint_params : params;
    checkpoint_coefficients : float array;
    checkpoint_intercept : float;
    checkpoint_schema : Feature_schema.t;
    checkpoint_rng : Rng.t;
    checkpoint_updates : int;
    checkpoint_batches : int;
    checkpoint_objective : float option;
  }

  type fitted = { fitted_checkpoint : checkpoint; fitted_report : report }

  let validation ~name ~reason ~remediation =
    Linear.validation ~name:("SGD regression " ^ name) ~reason ~remediation

  let create ?(penalty = L2) ?(alpha = 0.0001) ?(l1_ratio = 0.15)
      ?(fit_intercept = true)
      ?(learning_rate = Inverse_scaling { power_t = 0.25 }) ?(eta0 = 0.01)
      ?(max_epochs = 1000) ?tolerance ?(shuffle = true) () =
    if (not (Float.is_finite alpha)) || alpha < 0.0 then
      Error
        (validation ~name:"alpha" ~reason:"must be finite and non-negative"
           ~remediation:"choose a finite non-negative regularization strength")
    else if (not (Float.is_finite l1_ratio)) || l1_ratio < 0.0 || l1_ratio > 1.0
    then
      Error
        (validation ~name:"l1_ratio"
           ~reason:"must be finite and in the interval [0, 1]"
           ~remediation:"choose the L1 share of the elastic-net penalty")
    else if (not (Float.is_finite eta0)) || eta0 <= 0.0 then
      Error
        (validation ~name:"eta0" ~reason:"must be finite and positive"
           ~remediation:"choose a positive initial learning rate")
    else if max_epochs <= 0 then
      Error
        (validation ~name:"max_epochs" ~reason:"must be positive"
           ~remediation:"allow at least one complete pass over the data")
    else
      let* () =
        match learning_rate with
        | Constant -> Ok ()
        | Inverse_scaling { power_t }
          when Float.is_finite power_t && power_t >= 0.0 ->
            Ok ()
        | Inverse_scaling _ ->
            Error
              (validation ~name:"power_t"
                 ~reason:"must be finite and non-negative"
                 ~remediation:"choose a non-negative inverse-scaling exponent")
      in
      let* () =
        match tolerance with
        | None -> Ok ()
        | Some value when Float.is_finite value && value > 0.0 -> Ok ()
        | Some _ ->
            Error
              (validation ~name:"tolerance"
                 ~reason:"must be finite and positive when supplied"
                 ~remediation:
                   "omit tolerance for a fixed epoch budget or choose a \
                    positive threshold")
      in
      Ok
        {
          penalty;
          alpha;
          l1_ratio;
          fit_intercept;
          learning_rate;
          eta0;
          max_epochs;
          tolerance;
          shuffle;
        }

  let clone specification = specification
  let params specification = specification

  let start checkpoint_params ~rng ~feature_schema =
    {
      checkpoint_params;
      checkpoint_coefficients =
        Array.make (Feature_schema.feature_count feature_schema) 0.0;
      checkpoint_intercept = 0.0;
      checkpoint_schema = feature_schema;
      checkpoint_rng = rng;
      checkpoint_updates = 0;
      checkpoint_batches = 0;
      checkpoint_objective = None;
    }

  let soft_threshold value threshold =
    if value > threshold then value -. threshold
    else if value < -.threshold then value +. threshold
    else 0.0

  let penalty_shares params =
    match params.penalty with
    | No_penalty -> (0.0, 0.0)
    | L1 -> (1.0, 0.0)
    | L2 -> (0.0, 1.0)
    | Elastic_net -> (params.l1_ratio, 1.0 -. params.l1_ratio)

  let learning_rate params updates =
    match params.learning_rate with
    | Constant -> params.eta0
    | Inverse_scaling { power_t } ->
        params.eta0 /. ((Float.of_int updates +. 1.0) ** power_t)

  let shuffle rng order =
    let state = ref rng in
    for upper = Array.length order - 1 downto 1 do
      let bits, successor = Rng.next_int64 !state in
      state := successor;
      let non_negative = Int64.logand bits Int64.max_int in
      let selected =
        Int64.to_int (Int64.rem non_negative (Int64.of_int (upper + 1)))
      in
      let value = order.(upper) in
      order.(upper) <- order.(selected);
      order.(selected) <- value
    done;
    !state

  let objective params coefficients intercept sample_weight x target =
    let accumulated = Reference_backend.Accumulator.create () in
    let total_weight = Reference_backend.Accumulator.create () in
    for row = 0 to Matrix.rows x - 1 do
      let prediction = Reference_backend.Accumulator.create () in
      Reference_backend.Accumulator.add prediction intercept;
      for column = 0 to Matrix.columns x - 1 do
        Reference_backend.Accumulator.add prediction
          (coefficients.(column) *. Matrix.get x row column)
      done;
      let residual =
        Reference_backend.Accumulator.value prediction -. Vector.get target row
      in
      let weight = Linear.weight sample_weight row in
      Reference_backend.Accumulator.add accumulated
        (0.5 *. weight *. residual *. residual);
      Reference_backend.Accumulator.add total_weight weight
    done;
    let l1_share, l2_share = penalty_shares params in
    let l1 = ref 0.0 in
    let l2 = ref 0.0 in
    Array.iter
      (fun coefficient ->
        l1 := !l1 +. Float.abs coefficient;
        l2 := !l2 +. (coefficient *. coefficient))
      coefficients;
    let value =
      Reference_backend.Accumulator.value accumulated
      /. Reference_backend.Accumulator.value total_weight
      +. (params.alpha *. l1_share *. !l1)
      +. (0.5 *. params.alpha *. l2_share *. !l2)
    in
    if Float.is_finite value then Ok value
    else
      Error
        (Linear.numerical ~operation:"SGD regression"
           ~reason:"the objective is not finite"
           ~remediation:
             "rescale the features or target, or reduce the learning rate")

  let update_batch checkpoint ?sample_weight ~feature_schema ~x ~y () =
    let* () =
      Linear.validate_prediction_input ~schema:checkpoint.checkpoint_schema
        feature_schema x
    in
    let* () =
      if Matrix.rows x > 0 then Ok ()
      else
        Error
          (validation ~name:"partial_fit batch"
             ~reason:"must contain at least one sample"
             ~remediation:"skip empty batches or provide training samples")
    in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let params = checkpoint.checkpoint_params in
    let coefficients = Array.copy checkpoint.checkpoint_coefficients in
    let intercept = ref checkpoint.checkpoint_intercept in
    let updates = ref checkpoint.checkpoint_updates in
    let order = Array.init (Matrix.rows x) Fun.id in
    let rng =
      if params.shuffle then shuffle checkpoint.checkpoint_rng order
      else checkpoint.checkpoint_rng
    in
    let target = Target.regression_values y in
    let l1_share, l2_share = penalty_shares params in
    let maximum_step = ref 0.0 in
    let invalid = ref false in
    Array.iter
      (fun row ->
        if not !invalid then (
          let prediction = Reference_backend.Accumulator.create () in
          Reference_backend.Accumulator.add prediction !intercept;
          for column = 0 to Array.length coefficients - 1 do
            Reference_backend.Accumulator.add prediction
              (coefficients.(column) *. Matrix.get x row column)
          done;
          let residual =
            Reference_backend.Accumulator.value prediction
            -. Vector.get target row
          in
          let eta = learning_rate params !updates in
          let gradient_scale = Linear.weight sample_weight row *. residual in
          for column = 0 to Array.length coefficients - 1 do
            let previous = coefficients.(column) in
            let gradient =
              (gradient_scale *. Matrix.get x row column)
              +. (params.alpha *. l2_share *. previous)
            in
            let candidate = previous -. (eta *. gradient) in
            let updated =
              soft_threshold candidate (eta *. params.alpha *. l1_share)
            in
            if Float.is_finite updated then (
              coefficients.(column) <- updated;
              maximum_step :=
                Float.max !maximum_step (Float.abs (updated -. previous)))
            else invalid := true
          done;
          (if params.fit_intercept then
             let previous = !intercept in
             let updated = previous -. (eta *. gradient_scale) in
             if Float.is_finite updated then (
               intercept := updated;
               maximum_step :=
                 Float.max !maximum_step (Float.abs (updated -. previous)))
             else invalid := true);
          if !updates = Int.max_int then invalid := true
          else updates := !updates + 1))
      order;
    if !invalid then
      Error
        (Linear.numerical ~operation:"SGD regression"
           ~reason:"a coefficient, intercept, or update counter overflowed"
           ~remediation:
             "rescale the features or target, reduce the learning rate, or \
              restart from a fresh checkpoint")
    else
      let* objective =
        objective params coefficients !intercept sample_weight x target
      in
      Ok
        ( {
            checkpoint_params = params;
            checkpoint_coefficients = coefficients;
            checkpoint_intercept = !intercept;
            checkpoint_schema = checkpoint.checkpoint_schema;
            checkpoint_rng = rng;
            checkpoint_updates = !updates;
            checkpoint_batches = checkpoint.checkpoint_batches + 1;
            checkpoint_objective = Some objective;
          },
          !maximum_step )

  let partial_fit checkpoint ?sample_weight ~feature_schema ~x ~y () =
    let* checkpoint, _ =
      update_batch checkpoint ?sample_weight ~feature_schema ~x ~y ()
    in
    Ok checkpoint

  let report_for checkpoint ~converged ~stopping_reason =
    {
      converged;
      batches_processed = checkpoint.checkpoint_batches;
      updates = checkpoint.checkpoint_updates;
      objective = Option.get checkpoint.checkpoint_objective;
      stopping_reason;
    }

  let to_fitted checkpoint =
    match checkpoint.checkpoint_objective with
    | None ->
        Error
          (validation ~name:"checkpoint" ~reason:"has not processed a batch"
             ~remediation:
               "call partial_fit with a non-empty batch before requesting a \
                fitted model")
    | Some _ ->
        Ok
          {
            fitted_checkpoint = checkpoint;
            fitted_report =
              report_for checkpoint ~converged:false
                ~stopping_reason:Partial_fit;
          }

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let rec epochs remaining checkpoint =
      let* next, maximum_step =
        update_batch checkpoint ?sample_weight ~feature_schema ~x ~y ()
      in
      match specification.tolerance with
      | Some tolerance ->
          let scale =
            Array.fold_left
              (fun maximum coefficient ->
                Float.max maximum (Float.abs coefficient))
              (Float.max 1.0 (Float.abs next.checkpoint_intercept))
              next.checkpoint_coefficients
          in
          if maximum_step <= tolerance *. scale then
            Ok
              {
                fitted_checkpoint = next;
                fitted_report =
                  report_for next ~converged:true
                    ~stopping_reason:Step_tolerance;
              }
          else if remaining = 1 then
            Error
              (Error.make
                 ~remediation:
                   "increase max_epochs, loosen tolerance, rescale the data, \
                    or reduce eta0"
                 (Error.Convergence
                    {
                      algorithm = "SGD regression";
                      reason =
                        Format.sprintf
                          "the maximum parameter step %.17g exceeded tolerance \
                           %.17g after %d epochs"
                          maximum_step (tolerance *. scale)
                          specification.max_epochs;
                    }))
          else epochs (remaining - 1) next
      | None ->
          if remaining = 1 then
            Ok
              {
                fitted_checkpoint = next;
                fitted_report =
                  report_for next ~converged:false ~stopping_reason:Epoch_limit;
              }
          else epochs (remaining - 1) next
    in
    epochs specification.max_epochs (start specification ~rng ~feature_schema)

  let coefficients fitted =
    Vector.of_array fitted.fitted_checkpoint.checkpoint_coefficients

  let intercept fitted = fitted.fitted_checkpoint.checkpoint_intercept
  let report fitted = fitted.fitted_report

  let checkpoint fitted =
    let checkpoint = fitted.fitted_checkpoint in
    {
      checkpoint with
      checkpoint_coefficients = Array.copy checkpoint.checkpoint_coefficients;
    }

  let checkpoint_updates checkpoint = checkpoint.checkpoint_updates
  let checkpoint_batches_processed checkpoint = checkpoint.checkpoint_batches
  let fitted_params fitted = fitted.fitted_checkpoint.checkpoint_params
  let feature_schema fitted = fitted.fitted_checkpoint.checkpoint_schema

  let predict fitted ~feature_schema ~x =
    Linear.regression_prediction ~operation:"SGD regression prediction"
      ~schema:fitted.fitted_checkpoint.checkpoint_schema
      ~coefficients:fitted.fitted_checkpoint.checkpoint_coefficients
      ~intercept:fitted.fitted_checkpoint.checkpoint_intercept feature_schema x
end
