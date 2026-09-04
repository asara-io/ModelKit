open Modelkit_data
open Modelkit_protocols
module Linear = Modelkit_linear_models.Linear_model_internal

(* Pure pieces shared by every stochastic-gradient estimator: parameter
   validation, penalty arithmetic, learning-rate schedules, deterministic
   shuffling, and the epoch loop that turns per-batch updates into either a
   fitted value or a typed convergence failure. *)
module Sgd_common = struct
  let ( let* ) = Result.bind

  type penalty = No_penalty | L1 | L2 | Elastic_net
  type learning_rate = Constant | Inverse_scaling of { power_t : float }
  type stopping_reason = Epoch_limit | Step_tolerance | Partial_fit

  let validate ~family ~alpha ~l1_ratio ~eta0 ~max_epochs ~learning_rate
      ~tolerance =
    let validation ~name ~reason ~remediation =
      Linear.validation ~name:(family ^ " " ^ name) ~reason ~remediation
    in
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
      match tolerance with
      | None -> Ok ()
      | Some value when Float.is_finite value && value > 0.0 -> Ok ()
      | Some _ ->
          Error
            (validation ~name:"tolerance"
               ~reason:"must be finite and positive when supplied"
               ~remediation:
                 "omit tolerance for a fixed epoch budget or choose a positive \
                  threshold")

  let soft_threshold value threshold =
    if value > threshold then value -. threshold
    else if value < -.threshold then value +. threshold
    else 0.0

  let penalty_shares ~penalty ~l1_ratio =
    match penalty with
    | No_penalty -> (0.0, 0.0)
    | L1 -> (1.0, 0.0)
    | L2 -> (0.0, 1.0)
    | Elastic_net -> (l1_ratio, 1.0 -. l1_ratio)

  let learning_rate ~schedule ~eta0 updates =
    match schedule with
    | Constant -> eta0
    | Inverse_scaling { power_t } ->
        eta0 /. ((Float.of_int updates +. 1.0) ** power_t)

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

  let batch_order ~shuffle:enabled rng rows =
    let order = Array.init rows Fun.id in
    let rng = if enabled then shuffle rng order else rng in
    (order, rng)

  let penalty_value ~alpha ~l1_share ~l2_share coefficient_rows =
    let l1 = ref 0.0 in
    let l2 = ref 0.0 in
    Array.iter
      (Array.iter (fun coefficient ->
           l1 := !l1 +. Float.abs coefficient;
           l2 := !l2 +. (coefficient *. coefficient)))
      coefficient_rows;
    (alpha *. l1_share *. !l1) +. (0.5 *. alpha *. l2_share *. !l2)

  (* Applies one proximal gradient step to [coefficients] in place and returns
     the largest absolute parameter change, or [None] when a value stopped being
     finite. [gradient_scale] multiplies the feature row. *)
  let update_coefficients ~eta ~alpha ~l1_share ~l2_share ~gradient_scale x row
      coefficients =
    let maximum_step = ref 0.0 in
    let finite = ref true in
    let threshold = eta *. alpha *. l1_share in
    for column = 0 to Array.length coefficients - 1 do
      if !finite then
        let previous = coefficients.(column) in
        let gradient =
          (gradient_scale *. Matrix.get x row column)
          +. (alpha *. l2_share *. previous)
        in
        let updated =
          soft_threshold (previous -. (eta *. gradient)) threshold
        in
        if Float.is_finite updated then (
          coefficients.(column) <- updated;
          maximum_step :=
            Float.max !maximum_step (Float.abs (updated -. previous)))
        else finite := false
    done;
    if !finite then Some !maximum_step else None

  let linear_score coefficients intercept x row =
    let accumulator = Reference_backend.Accumulator.create () in
    Reference_backend.Accumulator.add accumulator intercept;
    for column = 0 to Array.length coefficients - 1 do
      Reference_backend.Accumulator.add accumulator
        (coefficients.(column) *. Matrix.get x row column)
    done;
    Reference_backend.Accumulator.value accumulator

  let parameter_scale coefficient_rows intercepts =
    Array.fold_left
      (fun scale row ->
        Array.fold_left
          (fun scale coefficient -> Float.max scale (Float.abs coefficient))
          scale row)
      (Array.fold_left
         (fun scale intercept -> Float.max scale (Float.abs intercept))
         1.0 intercepts)
      coefficient_rows

  let validate_batch ~family checkpoint_schema ?sample_weight ~feature_schema ~x
      ~target_length () =
    let* () =
      Linear.validate_prediction_input ~schema:checkpoint_schema feature_schema
        x
    in
    let* () =
      if Matrix.rows x > 0 then Ok ()
      else
        Error
          (Linear.validation
             ~name:(family ^ " partial_fit batch")
             ~reason:"must contain at least one sample"
             ~remediation:"skip empty batches or provide training samples")
    in
    let* () = Linear.validate_target_length x target_length in
    Linear.validate_sample_weight x sample_weight

  let overflow ~family =
    Linear.numerical ~operation:family
      ~reason:"a coefficient, intercept, or update counter overflowed"
      ~remediation:
        "rescale the features or target, reduce the learning rate, or restart \
         from a fresh checkpoint"

  let non_finite_objective ~family =
    Linear.numerical ~operation:family ~reason:"the objective is not finite"
      ~remediation:"rescale the features or target, or reduce the learning rate"

  let unprocessed_checkpoint ~family =
    Linear.validation ~name:(family ^ " checkpoint")
      ~reason:"has not processed a batch"
      ~remediation:
        "call partial_fit with a non-empty batch before requesting a fitted \
         model"

  (* [update] processes one epoch and reports the largest parameter step;
     [scale] normalizes the tolerance test; [finish] freezes a checkpoint. *)
  let run_epochs ~family ~max_epochs ~tolerance ~update ~scale ~finish initial =
    let rec epochs remaining checkpoint =
      let* next, maximum_step = update checkpoint in
      match tolerance with
      | Some tolerance ->
          let threshold = tolerance *. scale next in
          if maximum_step <= threshold then
            Ok (finish next ~converged:true ~stopping_reason:Step_tolerance)
          else if remaining = 1 then
            Error
              (Error.make
                 ~remediation:
                   "increase max_epochs, loosen tolerance, rescale the data, \
                    or reduce eta0"
                 (Error.Convergence
                    {
                      algorithm = family;
                      reason =
                        Format.sprintf
                          "the maximum parameter step %.17g exceeded tolerance \
                           %.17g after %d epochs"
                          maximum_step threshold max_epochs;
                    }))
          else epochs (remaining - 1) next
      | None ->
          if remaining = 1 then
            Ok (finish next ~converged:false ~stopping_reason:Epoch_limit)
          else epochs (remaining - 1) next
    in
    epochs max_epochs initial
end

module Sgd_regressor = struct
  let ( let* ) = Result.bind

  type penalty = Sgd_common.penalty = No_penalty | L1 | L2 | Elastic_net

  type learning_rate = Sgd_common.learning_rate =
    | Constant
    | Inverse_scaling of { power_t : float }

  type stopping_reason = Sgd_common.stopping_reason =
    | Epoch_limit
    | Step_tolerance
    | Partial_fit

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

  let family = "SGD regression"

  let create ?(penalty = L2) ?(alpha = 0.0001) ?(l1_ratio = 0.15)
      ?(fit_intercept = true)
      ?(learning_rate = Inverse_scaling { power_t = 0.25 }) ?(eta0 = 0.01)
      ?(max_epochs = 1000) ?tolerance ?(shuffle = true) () =
    let* () =
      Sgd_common.validate ~family ~alpha ~l1_ratio ~eta0 ~max_epochs
        ~learning_rate ~tolerance
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

  let shares params =
    Sgd_common.penalty_shares ~penalty:params.penalty ~l1_ratio:params.l1_ratio

  let objective params coefficients intercept sample_weight x target =
    let accumulated = Reference_backend.Accumulator.create () in
    let total_weight = Reference_backend.Accumulator.create () in
    for row = 0 to Matrix.rows x - 1 do
      let residual =
        Sgd_common.linear_score coefficients intercept x row
        -. Vector.get target row
      in
      let weight = Linear.weight sample_weight row in
      Reference_backend.Accumulator.add accumulated
        (0.5 *. weight *. residual *. residual);
      Reference_backend.Accumulator.add total_weight weight
    done;
    let l1_share, l2_share = shares params in
    let value =
      Reference_backend.Accumulator.value accumulated
      /. Reference_backend.Accumulator.value total_weight
      +. Sgd_common.penalty_value ~alpha:params.alpha ~l1_share ~l2_share
           [| coefficients |]
    in
    if Float.is_finite value then Ok value
    else Error (Sgd_common.non_finite_objective ~family)

  let update_batch checkpoint ?sample_weight ~feature_schema ~x ~y () =
    let* () =
      Sgd_common.validate_batch ~family checkpoint.checkpoint_schema
        ?sample_weight ~feature_schema ~x ~target_length:(Target.length y) ()
    in
    let params = checkpoint.checkpoint_params in
    let coefficients = Array.copy checkpoint.checkpoint_coefficients in
    let intercept = ref checkpoint.checkpoint_intercept in
    let updates = ref checkpoint.checkpoint_updates in
    let order, rng =
      Sgd_common.batch_order ~shuffle:params.shuffle checkpoint.checkpoint_rng
        (Matrix.rows x)
    in
    let target = Target.regression_values y in
    let l1_share, l2_share = shares params in
    let maximum_step = ref 0.0 in
    let invalid = ref false in
    Array.iter
      (fun row ->
        if not !invalid then (
          let residual =
            Sgd_common.linear_score coefficients !intercept x row
            -. Vector.get target row
          in
          let eta =
            Sgd_common.learning_rate ~schedule:params.learning_rate
              ~eta0:params.eta0 !updates
          in
          let gradient_scale = Linear.weight sample_weight row *. residual in
          (match
             Sgd_common.update_coefficients ~eta ~alpha:params.alpha ~l1_share
               ~l2_share ~gradient_scale x row coefficients
           with
          | Some step -> maximum_step := Float.max !maximum_step step
          | None -> invalid := true);
          (if params.fit_intercept && not !invalid then
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
    if !invalid then Error (Sgd_common.overflow ~family)
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

  let finish checkpoint ~converged ~stopping_reason =
    {
      fitted_checkpoint = checkpoint;
      fitted_report =
        {
          converged;
          batches_processed = checkpoint.checkpoint_batches;
          updates = checkpoint.checkpoint_updates;
          objective = Option.get checkpoint.checkpoint_objective;
          stopping_reason;
        };
    }

  let to_fitted checkpoint =
    match checkpoint.checkpoint_objective with
    | None -> Error (Sgd_common.unprocessed_checkpoint ~family)
    | Some _ ->
        Ok (finish checkpoint ~converged:false ~stopping_reason:Partial_fit)

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    Sgd_common.run_epochs ~family ~max_epochs:specification.max_epochs
      ~tolerance:specification.tolerance
      ~update:(fun checkpoint ->
        update_batch checkpoint ?sample_weight ~feature_schema ~x ~y ())
      ~scale:(fun checkpoint ->
        Sgd_common.parameter_scale
          [| checkpoint.checkpoint_coefficients |]
          [| checkpoint.checkpoint_intercept |])
      ~finish
      (start specification ~rng ~feature_schema)

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

module Sgd_classifier = struct
  let ( let* ) = Result.bind

  type penalty = Sgd_common.penalty = No_penalty | L1 | L2 | Elastic_net

  type learning_rate = Sgd_common.learning_rate =
    | Constant
    | Inverse_scaling of { power_t : float }

  type stopping_reason = Sgd_common.stopping_reason =
    | Epoch_limit
    | Step_tolerance
    | Partial_fit

  type loss = Hinge | Log_loss

  type report = {
    converged : bool;
    batches_processed : int;
    updates : int;
    objective : float;
    stopping_reason : stopping_reason;
  }

  type params = {
    loss : loss;
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
  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  (* Binary problems train one model scoring the higher class; larger problems
     train one one-versus-rest model per ascending class. *)
  type checkpoint = {
    checkpoint_params : params;
    checkpoint_classes : int array;
    checkpoint_coefficients : float array array;
    checkpoint_intercepts : float array;
    checkpoint_schema : Feature_schema.t;
    checkpoint_rng : Rng.t;
    checkpoint_updates : int;
    checkpoint_batches : int;
    checkpoint_objective : float option;
  }

  type fitted = { fitted_checkpoint : checkpoint; fitted_report : report }

  let family = "SGD classification"

  let create ?(loss = Hinge) ?(penalty = L2) ?(alpha = 0.0001)
      ?(l1_ratio = 0.15) ?(fit_intercept = true)
      ?(learning_rate = Inverse_scaling { power_t = 0.25 }) ?(eta0 = 0.01)
      ?(max_epochs = 1000) ?tolerance ?(shuffle = true) () =
    let* () =
      Sgd_common.validate ~family ~alpha ~l1_ratio ~eta0 ~max_epochs
        ~learning_rate ~tolerance
    in
    Ok
      {
        loss;
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

  let validate_classes classes =
    let sorted = Array.copy classes in
    Array.sort Int.compare sorted;
    let distinct =
      let rec check index =
        index >= Array.length sorted
        || (sorted.(index - 1) <> sorted.(index) && check (index + 1))
      in
      check 1
    in
    if Array.length sorted < 2 then
      Error
        (Linear.validation ~name:(family ^ " classes")
           ~reason:"at least two classes must be registered"
           ~remediation:"register every class label the stream can produce")
    else if not distinct then
      Error
        (Linear.validation ~name:(family ^ " classes")
           ~reason:"registered class labels must be distinct"
           ~remediation:"remove duplicate class labels")
    else Ok sorted

  let model_count classes =
    if Array.length classes = 2 then 1 else Array.length classes

  let start checkpoint_params ~rng ~feature_schema ~classes =
    let* classes = validate_classes classes in
    let features = Feature_schema.feature_count feature_schema in
    let models = model_count classes in
    Ok
      {
        checkpoint_params;
        checkpoint_classes = classes;
        checkpoint_coefficients =
          Array.init models (fun _ -> Array.make features 0.0);
        checkpoint_intercepts = Array.make models 0.0;
        checkpoint_schema = feature_schema;
        checkpoint_rng = rng;
        checkpoint_updates = 0;
        checkpoint_batches = 0;
        checkpoint_objective = None;
      }

  let shares params =
    Sgd_common.penalty_shares ~penalty:params.penalty ~l1_ratio:params.l1_ratio

  (* The positive class of model [model]; every other registered class is the
     negative side of that model. *)
  let positive_class classes model =
    if Array.length classes = 2 then classes.(1) else classes.(model)

  let loss_value params ~positive score =
    match params.loss with
    | Hinge ->
        let sign = if positive then 1.0 else -1.0 in
        Float.max 0.0 (1.0 -. (sign *. score))
    | Log_loss -> Linear.softplus score -. if positive then score else 0.0

  let loss_gradient params ~positive score =
    match params.loss with
    | Hinge ->
        let sign = if positive then 1.0 else -1.0 in
        if sign *. score <= 1.0 then -.sign else 0.0
    | Log_loss -> Linear.stable_sigmoid score -. if positive then 1.0 else 0.0

  (* Maps every positively weighted row to its registered class index; rows
     with zero weight never influence the parameters and are left unchecked. *)
  let class_indices checkpoint sample_weight labels =
    let classes = checkpoint.checkpoint_classes in
    let lookup = Hashtbl.create (Array.length classes) in
    Array.iteri (fun index label -> Hashtbl.replace lookup label index) classes;
    let indices = Array.make (Array.length labels) (-1) in
    let rec assign row =
      if row = Array.length labels then Ok indices
      else
        match Hashtbl.find_opt lookup labels.(row) with
        | Some index ->
            indices.(row) <- index;
            assign (row + 1)
        | None when Linear.weight sample_weight row <= 0.0 -> assign (row + 1)
        | None ->
            Error
              (Linear.validation ~name:(family ^ " batch labels")
                 ~reason:
                   (Format.sprintf
                      "row %d has class %d, which was not registered when the \
                       checkpoint started"
                      row labels.(row))
                 ~remediation:
                   "register every class label at start or drop rows from \
                    unregistered classes")
    in
    assign 0

  let objective params ~classes coefficients intercepts sample_weight x
      class_index =
    let accumulated = Reference_backend.Accumulator.create () in
    let total_weight = Reference_backend.Accumulator.create () in
    for row = 0 to Matrix.rows x - 1 do
      let weight = Linear.weight sample_weight row in
      if weight > 0.0 then
        Array.iteri
          (fun model model_coefficients ->
            let score =
              Sgd_common.linear_score model_coefficients intercepts.(model) x
                row
            in
            let positive =
              classes.(class_index.(row)) = positive_class classes model
            in
            Reference_backend.Accumulator.add accumulated
              (weight *. loss_value params ~positive score))
          coefficients;
      Reference_backend.Accumulator.add total_weight weight
    done;
    let l1_share, l2_share = shares params in
    let value =
      Reference_backend.Accumulator.value accumulated
      /. Reference_backend.Accumulator.value total_weight
      +. Sgd_common.penalty_value ~alpha:params.alpha ~l1_share ~l2_share
           coefficients
    in
    if Float.is_finite value then Ok value
    else Error (Sgd_common.non_finite_objective ~family)

  let update_batch checkpoint ?sample_weight ~feature_schema ~x ~y () =
    let* () =
      Sgd_common.validate_batch ~family checkpoint.checkpoint_schema
        ?sample_weight ~feature_schema ~x ~target_length:(Target.length y) ()
    in
    let labels = Target.classification_values y in
    let* class_index = class_indices checkpoint sample_weight labels in
    let params = checkpoint.checkpoint_params in
    let classes = checkpoint.checkpoint_classes in
    let coefficients =
      Array.map Array.copy checkpoint.checkpoint_coefficients
    in
    let intercepts = Array.copy checkpoint.checkpoint_intercepts in
    let updates = ref checkpoint.checkpoint_updates in
    let order, rng =
      Sgd_common.batch_order ~shuffle:params.shuffle checkpoint.checkpoint_rng
        (Matrix.rows x)
    in
    let l1_share, l2_share = shares params in
    let maximum_step = ref 0.0 in
    let invalid = ref false in
    Array.iter
      (fun row ->
        if not !invalid then (
          let eta =
            Sgd_common.learning_rate ~schedule:params.learning_rate
              ~eta0:params.eta0 !updates
          in
          let weight = Linear.weight sample_weight row in
          Array.iteri
            (fun model model_coefficients ->
              if not !invalid then (
                let score =
                  Sgd_common.linear_score model_coefficients intercepts.(model)
                    x row
                in
                let gradient_scale =
                  if weight > 0.0 then
                    let positive =
                      classes.(class_index.(row)) = positive_class classes model
                    in
                    weight *. loss_gradient params ~positive score
                  else 0.0
                in
                (match
                   Sgd_common.update_coefficients ~eta ~alpha:params.alpha
                     ~l1_share ~l2_share ~gradient_scale x row
                     model_coefficients
                 with
                | Some step -> maximum_step := Float.max !maximum_step step
                | None -> invalid := true);
                if params.fit_intercept && not !invalid then
                  let previous = intercepts.(model) in
                  let updated = previous -. (eta *. gradient_scale) in
                  if Float.is_finite updated then (
                    intercepts.(model) <- updated;
                    maximum_step :=
                      Float.max !maximum_step (Float.abs (updated -. previous)))
                  else invalid := true))
            coefficients;
          if !updates = Int.max_int then invalid := true
          else updates := !updates + 1))
      order;
    if !invalid then Error (Sgd_common.overflow ~family)
    else
      let* objective =
        objective params ~classes coefficients intercepts sample_weight x
          class_index
      in
      Ok
        ( {
            checkpoint with
            checkpoint_coefficients = coefficients;
            checkpoint_intercepts = intercepts;
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

  let finish checkpoint ~converged ~stopping_reason =
    {
      fitted_checkpoint = checkpoint;
      fitted_report =
        {
          converged;
          batches_processed = checkpoint.checkpoint_batches;
          updates = checkpoint.checkpoint_updates;
          objective = Option.get checkpoint.checkpoint_objective;
          stopping_reason;
        };
    }

  let to_fitted checkpoint =
    match checkpoint.checkpoint_objective with
    | None -> Error (Sgd_common.unprocessed_checkpoint ~family)
    | Some _ ->
        Ok (finish checkpoint ~converged:false ~stopping_reason:Partial_fit)

  let effective_classes target sample_weight =
    let labels = Target.classification_values target in
    let distinct = Hashtbl.create (Array.length labels) in
    Array.iteri
      (fun row label ->
        if Linear.weight sample_weight row > 0.0 then
          Hashtbl.replace distinct label ())
      labels;
    let classes = Hashtbl.to_seq_keys distinct |> Array.of_seq in
    if Array.length classes >= 2 then validate_classes classes
    else
      Error
        (Linear.validation ~name:(family ^ " classes")
           ~reason:"at least two positively weighted classes are required"
           ~remediation:"provide training rows from at least two classes")

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let* () = Linear.validate_matrix feature_schema x in
    let* () = Linear.validate_target_length x (Target.length y) in
    let* () = Linear.validate_sample_weight x sample_weight in
    let* classes = effective_classes y sample_weight in
    let* initial = start specification ~rng ~feature_schema ~classes in
    Sgd_common.run_epochs ~family ~max_epochs:specification.max_epochs
      ~tolerance:specification.tolerance
      ~update:(fun checkpoint ->
        update_batch checkpoint ?sample_weight ~feature_schema ~x ~y ())
      ~scale:(fun checkpoint ->
        Sgd_common.parameter_scale checkpoint.checkpoint_coefficients
          checkpoint.checkpoint_intercepts)
      ~finish initial

  let matrix_of_rows ~remediation rows ~columns =
    match Matrix.of_arrays rows with
    | Ok matrix when Matrix.shape matrix = (Array.length rows, columns) ->
        Ok matrix
    | Ok _ -> assert false
    | Error error -> Error (Error.of_data_error ~remediation error)

  let decision_function fitted ~feature_schema ~x =
    let checkpoint = fitted.fitted_checkpoint in
    let* () =
      Linear.validate_prediction_input ~schema:checkpoint.checkpoint_schema
        feature_schema x
    in
    let rows = Matrix.rows x in
    let models = Array.length checkpoint.checkpoint_coefficients in
    let values = Array.make_matrix rows models 0.0 in
    let rec score row model =
      if row = rows then Ok ()
      else if model = models then score (row + 1) 0
      else
        let value =
          Sgd_common.linear_score
            checkpoint.checkpoint_coefficients.(model)
            checkpoint.checkpoint_intercepts.(model)
            x row
        in
        if Float.is_finite value then (
          values.(row).(model) <- value;
          score row (model + 1))
        else
          Error
            (Linear.numerical
               ~operation:(family ^ " decision function")
               ~reason:
                 (Format.sprintf "score for row %d and class %d is not finite"
                    row
                    (positive_class checkpoint.checkpoint_classes model))
               ~remediation:"rescale the prediction features")
    in
    let* () = score 0 0 in
    match
      Matrix.init ~rows ~columns:models (fun row model -> values.(row).(model))
    with
    | Ok matrix -> Ok matrix
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"report invalid SGD-classifier score dimensions" error)

  let is_binary fitted =
    Array.length fitted.fitted_checkpoint.checkpoint_classes = 2

  let binary_decision_function fitted ~feature_schema ~x =
    if not (is_binary fitted) then
      Error
        (Error.make
           ~remediation:
             "use decision_function for one score column per registered class"
           (Error.Compatibility
              {
                component = family ^ " binary decision function";
                reason =
                  Format.sprintf "the model registered %d classes"
                    (Array.length fitted.fitted_checkpoint.checkpoint_classes);
              }))
    else
      let* decisions = decision_function fitted ~feature_schema ~x in
      Ok
        (Vector.unsafe_init (Matrix.rows decisions) (fun row ->
             Matrix.get decisions row 0))

  let predict_proba fitted ~feature_schema ~x =
    match fitted.fitted_checkpoint.checkpoint_params.loss with
    | Hinge ->
        Error
          (Error.make
             ~remediation:
               "configure Log_loss or calibrate decision scores separately"
             (Error.Compatibility
                {
                  component = family ^ " predict_proba";
                  reason = "hinge loss does not produce probability estimates";
                }))
    | Log_loss -> (
        let* decisions = decision_function fitted ~feature_schema ~x in
        let rows = Matrix.rows decisions in
        let class_count =
          Array.length fitted.fitted_checkpoint.checkpoint_classes
        in
        let values = Array.make_matrix rows class_count 0.0 in
        for row = 0 to rows - 1 do
          if is_binary fitted then (
            let positive = Linear.stable_sigmoid (Matrix.get decisions row 0) in
            values.(row).(0) <- 1.0 -. positive;
            values.(row).(1) <- positive)
          else
            (* One-versus-rest sigmoids normalized to the simplex; a row whose
               every sigmoid underflowed to zero becomes uniform. *)
            let total = ref 0.0 in
            for model = 0 to class_count - 1 do
              let probability =
                Linear.stable_sigmoid (Matrix.get decisions row model)
              in
              values.(row).(model) <- probability;
              total := !total +. probability
            done;
            if !total = 0.0 then
              Array.fill values.(row) 0 class_count
                (1.0 /. Float.of_int class_count)
            else
              for model = 0 to class_count - 1 do
                values.(row).(model) <- values.(row).(model) /. !total
              done
        done;
        match
          Matrix.init ~rows ~columns:class_count (fun row model ->
              values.(row).(model))
        with
        | Ok matrix -> Ok matrix
        | Error error ->
            Error
              (Error.of_data_error
                 ~remediation:
                   "report invalid SGD-classifier probability dimensions"
                 error))

  let predict fitted ~feature_schema ~x =
    let classes = fitted.fitted_checkpoint.checkpoint_classes in
    let* decisions = decision_function fitted ~feature_schema ~x in
    let predictions =
      Array.init (Matrix.rows decisions) (fun row ->
          if is_binary fitted then
            if Matrix.get decisions row 0 > 0.0 then classes.(1)
            else classes.(0)
          else
            let best = ref 0 in
            for model = 1 to Matrix.columns decisions - 1 do
              if Matrix.get decisions row model > Matrix.get decisions row !best
              then best := model
            done;
            classes.(!best))
    in
    Ok (Target.classification predictions)

  let coefficients fitted =
    let checkpoint = fitted.fitted_checkpoint in
    match
      matrix_of_rows
        ~remediation:"report invalid SGD-classifier coefficient dimensions"
        (Array.map Array.copy checkpoint.checkpoint_coefficients)
        ~columns:(Feature_schema.feature_count checkpoint.checkpoint_schema)
    with
    | Ok matrix -> matrix
    | Error _ -> assert false

  let intercepts fitted =
    Vector.of_array fitted.fitted_checkpoint.checkpoint_intercepts

  let classes fitted = Array.copy fitted.fitted_checkpoint.checkpoint_classes
  let report fitted = fitted.fitted_report

  let checkpoint fitted =
    let checkpoint = fitted.fitted_checkpoint in
    {
      checkpoint with
      checkpoint_coefficients =
        Array.map Array.copy checkpoint.checkpoint_coefficients;
      checkpoint_intercepts = Array.copy checkpoint.checkpoint_intercepts;
    }

  let checkpoint_classes checkpoint = Array.copy checkpoint.checkpoint_classes
  let checkpoint_updates checkpoint = checkpoint.checkpoint_updates
  let checkpoint_batches_processed checkpoint = checkpoint.checkpoint_batches
  let fitted_params fitted = fitted.fitted_checkpoint.checkpoint_params
  let feature_schema fitted = fitted.fitted_checkpoint.checkpoint_schema
end
