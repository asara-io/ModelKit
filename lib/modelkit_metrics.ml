open Modelkit_data
open Modelkit_protocols

module Undefined_metric_policy = struct
  type t = Error | Return_nan | Use_fallback
end

module Metric_internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let numerical ~operation ~reason =
    Error.make ~remediation:"rescale inputs or inspect the metric inputs"
      (Error.Numerical { operation; reason })

  let validate_nonempty ~name length =
    if length > 0 then Ok ()
    else
      Error
        (validation ~name ~reason:"metric inputs are empty"
           ~remediation:"provide at least one observed and predicted value")

  let validate_length ~name ~expected ~observed =
    if expected = observed then Ok ()
    else
      Error
        (Error.make ~remediation:"provide aligned truth and prediction values"
           (Error.Shape_mismatch
              { name; expected = [ expected ]; observed = [ observed ] }))

  let validate_weights ~expected = function
    | None -> Ok ()
    | Some weights ->
        validate_length ~name:"metric sample weights" ~expected
          ~observed:(Sample_weight.length weights)

  let weight sample_weight index =
    match sample_weight with
    | None -> 1.0
    | Some weights -> Sample_weight.get weights index

  let sum length f =
    let accumulator = Reference_backend.Accumulator.create () in
    for index = 0 to length - 1 do
      Reference_backend.Accumulator.add accumulator (f index)
    done;
    Reference_backend.Accumulator.value accumulator

  let finite ~operation value =
    if Float.is_finite value then Ok value
    else
      Error
        (numerical ~operation
           ~reason:"the result is not representable as a finite float64 value")

  let total_weight sample_weight length =
    sum length (weight sample_weight) |> finite ~operation:"metric weighting"

  let average ~operation ~sample_weight length f =
    let numerator =
      sum length (fun index -> weight sample_weight index *. f index)
    in
    let* numerator = finite ~operation numerator in
    let* denominator = total_weight sample_weight length in
    finite ~operation (numerator /. denominator)

  let undefined policy ~name ~reason ~fallback =
    match policy with
    | Undefined_metric_policy.Error ->
        Error
          (validation ~name ~reason
             ~remediation:
               "choose Return_nan or Use_fallback, or provide metric inputs \
                with the required variation")
    | Undefined_metric_policy.Return_nan -> Ok Float.nan
    | Undefined_metric_policy.Use_fallback -> Ok fallback

  let validate_probabilities ~expected probabilities =
    let observed = Vector.length probabilities in
    let* () =
      validate_length ~name:"positive-class probabilities" ~expected ~observed
    in
    let rec loop index =
      if index = observed then Ok ()
      else
        let probability = Vector.get probabilities index in
        if not (Float.is_finite probability) then
          Error
            (validation ~name:"positive-class probabilities"
               ~reason:
                 (Format.sprintf "value %g at index %d is not finite"
                    probability index)
               ~remediation:
                 "provide finite probabilities in the interval [0, 1]")
        else if probability < 0.0 || probability > 1.0 then
          Error
            (validation ~name:"positive-class probabilities"
               ~reason:
                 (Format.sprintf "value %g at index %d lies outside [0, 1]"
                    probability index)
               ~remediation:"provide probabilities in the interval [0, 1]")
        else loop (index + 1)
    in
    loop 0

  let validate_at_most_two_labels arrays =
    let labels = Hashtbl.create 3 in
    Array.iter
      (Array.iter (fun label -> Hashtbl.replace labels label ()))
      arrays;
    if Hashtbl.length labels <= 2 then Ok ()
    else
      Error
        (validation ~name:"binary classification labels"
           ~reason:"more than two distinct labels were observed"
           ~remediation:"provide binary truth and prediction labels")

  let validate_positive_label ~positive_label arrays =
    let negative = ref None in
    let rec validate_array collection index =
      if index = Array.length collection then Ok ()
      else
        let label = collection.(index) in
        if label = positive_label then validate_array collection (index + 1)
        else
          match !negative with
          | None ->
              negative := Some label;
              validate_array collection (index + 1)
          | Some expected when label = expected ->
              validate_array collection (index + 1)
          | Some expected ->
              Error
                (validation ~name:"binary classification labels"
                   ~reason:
                     (Format.sprintf
                        "label %d is neither positive label %d nor negative \
                         label %d"
                        label positive_label expected)
                   ~remediation:
                     "choose the intended positive label and provide binary \
                      labels")
    in
    let rec loop index =
      if index = Array.length arrays then Ok ()
      else
        let* () = validate_array arrays.(index) 0 in
        loop (index + 1)
    in
    loop 0
end

module Regression_metrics = struct
  type residual_curve = { predictions : Vector.t; residuals : Vector.t }

  let prepare ?sample_weight ~truth ~prediction () =
    let truth = Target.regression_values truth in
    let prediction = Target.regression_values prediction in
    let length = Vector.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"regression metric" length
    in
    let* () =
      Metric_internal.validate_length ~name:"regression prediction"
        ~expected:length ~observed:(Vector.length prediction)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    Ok (truth, prediction, length)

  let mean_absolute_error ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    Metric_internal.average ~operation:"mean absolute error" ~sample_weight
      length (fun index ->
        Float.abs (Vector.get truth index -. Vector.get prediction index))

  let mean_squared_error ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    Metric_internal.average ~operation:"mean squared error" ~sample_weight
      length (fun index ->
        let residual = Vector.get truth index -. Vector.get prediction index in
        residual *. residual)

  let root_mean_squared_error ?sample_weight ~truth ~prediction () =
    mean_squared_error ?sample_weight ~truth ~prediction ()
    |> Result.map Float.sqrt

  let r2 ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare ?sample_weight ~truth ~prediction ()
    in
    let* mean =
      Metric_internal.average ~operation:"R-squared truth mean" ~sample_weight
        length (Vector.get truth)
    in
    let residual_sum =
      Metric_internal.sum length (fun index ->
          let residual =
            Vector.get truth index -. Vector.get prediction index
          in
          Metric_internal.weight sample_weight index *. residual *. residual)
    in
    let total_sum =
      Metric_internal.sum length (fun index ->
          let centered = Vector.get truth index -. mean in
          Metric_internal.weight sample_weight index *. centered *. centered)
    in
    let* residual_sum =
      Metric_internal.finite ~operation:"R-squared residual sum" residual_sum
    in
    let* total_sum =
      Metric_internal.finite ~operation:"R-squared total sum" total_sum
    in
    if total_sum = 0.0 then
      Metric_internal.undefined undefined ~name:"R-squared"
        ~reason:"the weighted truth values are constant"
        ~fallback:(if residual_sum = 0.0 then 1.0 else 0.0)
    else
      Metric_internal.finite ~operation:"R-squared"
        (1.0 -. (residual_sum /. total_sum))

  let residual_curve ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length = prepare ~truth ~prediction () in
    let residuals = Array.make length 0.0 in
    let rec fill index =
      if index = length then Ok ()
      else
        let residual = Vector.get truth index -. Vector.get prediction index in
        if Float.is_finite residual then (
          residuals.(index) <- residual;
          fill (index + 1))
        else
          Error
            (Metric_internal.numerical ~operation:"residual curve"
               ~reason:
                 (Format.sprintf
                    "the residual at index %d is not representable as float64"
                    index))
    in
    let* () = fill 0 in
    Ok { predictions = prediction; residuals = Vector.of_array residuals }
end

module Binary_prediction = struct
  type t = {
    binary_labels : Target.classification Target.t option;
    binary_positive_probabilities : Vector.t option;
    binary_length : int;
  }

  let create ?labels ?positive_probabilities () =
    let ( let* ) = Result.bind in
    match (labels, positive_probabilities) with
    | None, None ->
        Error
          (Metric_internal.validation ~name:"binary prediction"
             ~reason:
               "neither labels nor positive-class probabilities were provided"
             ~remediation:"provide at least one classifier response")
    | Some labels, None ->
        Ok
          {
            binary_labels = Some labels;
            binary_positive_probabilities = None;
            binary_length = Target.length labels;
          }
    | None, Some probabilities ->
        let length = Vector.length probabilities in
        let* () =
          Metric_internal.validate_probabilities ~expected:length probabilities
        in
        Ok
          {
            binary_labels = None;
            binary_positive_probabilities = Some probabilities;
            binary_length = length;
          }
    | Some labels, Some probabilities ->
        let length = Target.length labels in
        let* () =
          Metric_internal.validate_probabilities ~expected:length probabilities
        in
        Ok
          {
            binary_labels = Some labels;
            binary_positive_probabilities = Some probabilities;
            binary_length = length;
          }

  let length prediction = prediction.binary_length
  let labels prediction = prediction.binary_labels

  let positive_probabilities prediction =
    prediction.binary_positive_probabilities
end

module Binary_classification_metrics = struct
  type roc_curve = {
    thresholds : Vector.t;
    false_positive_rates : Vector.t;
    true_positive_rates : Vector.t;
  }

  type precision_recall_curve = {
    decision_thresholds : Vector.t;
    precisions : Vector.t;
    recalls : Vector.t;
  }

  type confusion = { tp : float; tn : float; fp : float; fn : float }

  let prepare_labels ?sample_weight ~positive_label ~truth ~prediction () =
    let truth = Target.classification_values truth in
    let prediction = Target.classification_values prediction in
    let length = Array.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"binary classification metric"
        length
    in
    let* () =
      Metric_internal.validate_length ~name:"classification prediction"
        ~expected:length ~observed:(Array.length prediction)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_positive_label ~positive_label
        [| truth; prediction |]
    in
    Ok (truth, prediction, length)

  let prepare_probabilities ?sample_weight ~positive_label ~truth
      ~positive_probabilities () =
    let truth = Target.classification_values truth in
    let length = Array.length truth in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"binary classification metric"
        length
    in
    let* () =
      Metric_internal.validate_probabilities ~expected:length
        positive_probabilities
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_positive_label ~positive_label [| truth |]
    in
    Ok (truth, length)

  let accuracy ?sample_weight ~truth ~prediction () =
    let truth_values = Target.classification_values truth in
    let prediction_values = Target.classification_values prediction in
    let length = Array.length truth_values in
    let ( let* ) = Result.bind in
    let* () =
      Metric_internal.validate_nonempty ~name:"classification accuracy" length
    in
    let* () =
      Metric_internal.validate_length ~name:"classification prediction"
        ~expected:length
        ~observed:(Array.length prediction_values)
    in
    let* () = Metric_internal.validate_weights ~expected:length sample_weight in
    let* () =
      Metric_internal.validate_at_most_two_labels
        [| truth_values; prediction_values |]
    in
    Metric_internal.average ~operation:"classification accuracy" ~sample_weight
      length (fun index ->
        if truth_values.(index) = prediction_values.(index) then 1.0 else 0.0)

  let confusion ?sample_weight ~positive_label ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* truth, prediction, length =
      prepare_labels ?sample_weight ~positive_label ~truth ~prediction ()
    in
    let tp = Reference_backend.Accumulator.create () in
    let tn = Reference_backend.Accumulator.create () in
    let fp = Reference_backend.Accumulator.create () in
    let fn = Reference_backend.Accumulator.create () in
    for index = 0 to length - 1 do
      let weight = Metric_internal.weight sample_weight index in
      match
        (truth.(index) = positive_label, prediction.(index) = positive_label)
      with
      | true, true -> Reference_backend.Accumulator.add tp weight
      | false, false -> Reference_backend.Accumulator.add tn weight
      | false, true -> Reference_backend.Accumulator.add fp weight
      | true, false -> Reference_backend.Accumulator.add fn weight
    done;
    let* tp =
      Reference_backend.Accumulator.value tp
      |> Metric_internal.finite ~operation:"binary true-positive weight"
    in
    let* tn =
      Reference_backend.Accumulator.value tn
      |> Metric_internal.finite ~operation:"binary true-negative weight"
    in
    let* fp =
      Reference_backend.Accumulator.value fp
      |> Metric_internal.finite ~operation:"binary false-positive weight"
    in
    let* fn =
      Reference_backend.Accumulator.value fn
      |> Metric_internal.finite ~operation:"binary false-negative weight"
    in
    Ok { tp; tn; fp; fn }

  let ratio ~undefined ~name ~reason ~fallback numerator denominator =
    if denominator = 0.0 then
      Metric_internal.undefined undefined ~name ~reason ~fallback
    else Ok (numerator /. denominator)

  let balanced_accuracy ?(positive_label = 1)
      ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    let positive_support = values.tp +. values.fn in
    let negative_support = values.tn +. values.fp in
    if positive_support = 0.0 || negative_support = 0.0 then
      Metric_internal.undefined undefined ~name:"balanced accuracy"
        ~reason:"positive and negative weighted support are both required"
        ~fallback:0.0
    else
      Ok
        (0.5
        *. ((values.tp /. positive_support) +. (values.tn /. negative_support))
        )

  let precision ?(positive_label = 1)
      ?(undefined = Undefined_metric_policy.Error) ?sample_weight ~truth
      ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"precision"
      ~reason:"no positive prediction has positive sample weight" ~fallback:0.0
      values.tp (values.tp +. values.fp)

  let recall ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"recall"
      ~reason:"no positive truth label has positive sample weight" ~fallback:0.0
      values.tp (values.tp +. values.fn)

  let f1 ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~prediction () =
    let ( let* ) = Result.bind in
    let* values =
      confusion ?sample_weight ~positive_label ~truth ~prediction ()
    in
    ratio ~undefined ~name:"F1"
      ~reason:
        "true-positive, false-positive, and false-negative weights are zero"
      ~fallback:0.0 (2.0 *. values.tp)
      ((2.0 *. values.tp) +. values.fp +. values.fn)

  let log_loss ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* truth, length =
      prepare_probabilities ?sample_weight ~positive_label ~truth
        ~positive_probabilities ()
    in
    let epsilon = Float.epsilon in
    Metric_internal.average ~operation:"binary log loss" ~sample_weight length
      (fun index ->
        let probability = Vector.get positive_probabilities index in
        let probability =
          Float.max epsilon (Float.min (1.0 -. epsilon) probability)
        in
        if truth.(index) = positive_label then -.Float.log probability
        else -.Float.log1p (-.probability))

  type curve_counts = {
    curve_thresholds : float array;
    true_positives : float array;
    false_positives : float array;
    positive_weight : float;
    negative_weight : float;
  }

  type scored = {
    score : float;
    positive : bool;
    weight : float;
    source_index : int;
  }

  let curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
      () =
    let ( let* ) = Result.bind in
    let* truth, length =
      prepare_probabilities ?sample_weight ~positive_label ~truth
        ~positive_probabilities ()
    in
    let included_count = ref 0 in
    for index = 0 to length - 1 do
      if Metric_internal.weight sample_weight index > 0.0 then
        incr included_count
    done;
    let included =
      Array.make !included_count
        { score = 0.0; positive = false; weight = 0.0; source_index = 0 }
    in
    let position = ref 0 in
    for index = 0 to length - 1 do
      let weight = Metric_internal.weight sample_weight index in
      if weight > 0.0 then (
        included.(!position) <-
          {
            score = Vector.get positive_probabilities index;
            positive = truth.(index) = positive_label;
            weight;
            source_index = index;
          };
        incr position)
    done;
    Array.stable_sort
      (fun left right ->
        let by_score = Float.compare right.score left.score in
        if by_score <> 0 then by_score
        else Int.compare left.source_index right.source_index)
      included;
    let maximum = Array.length included in
    let thresholds = Array.make maximum 0.0 in
    let true_positives = Array.make maximum 0.0 in
    let false_positives = Array.make maximum 0.0 in
    let tp = Reference_backend.Accumulator.create () in
    let fp = Reference_backend.Accumulator.create () in
    let group_count = ref 0 in
    let index = ref 0 in
    while !index < maximum do
      let threshold = included.(!index).score in
      while !index < maximum && included.(!index).score = threshold do
        let item = included.(!index) in
        Reference_backend.Accumulator.add
          (if item.positive then tp else fp)
          item.weight;
        incr index
      done;
      thresholds.(!group_count) <- threshold;
      true_positives.(!group_count) <- Reference_backend.Accumulator.value tp;
      false_positives.(!group_count) <- Reference_backend.Accumulator.value fp;
      incr group_count
    done;
    let thresholds = Array.sub thresholds 0 !group_count in
    let true_positives = Array.sub true_positives 0 !group_count in
    let false_positives = Array.sub false_positives 0 !group_count in
    let positive_weight = true_positives.(!group_count - 1) in
    let negative_weight = false_positives.(!group_count - 1) in
    let* positive_weight =
      Metric_internal.finite ~operation:"ROC positive support" positive_weight
    in
    let* negative_weight =
      Metric_internal.finite ~operation:"ROC negative support" negative_weight
    in
    if positive_weight = 0.0 || negative_weight = 0.0 then
      Error
        (Metric_internal.validation ~name:"binary ranking curve"
           ~reason:"positive and negative weighted support are both required"
           ~remediation:"provide a fold containing both binary classes")
    else
      Ok
        {
          curve_thresholds = thresholds;
          true_positives;
          false_positives;
          positive_weight;
          negative_weight;
        }

  let roc_curve ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* counts =
      curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
        ()
    in
    let points = Array.length counts.curve_thresholds + 1 in
    Ok
      ({
         thresholds =
           Vector.unsafe_init points (fun index ->
               if index = 0 then Float.infinity
               else counts.curve_thresholds.(index - 1));
         false_positive_rates =
           Vector.unsafe_init points (fun index ->
               if index = 0 then 0.0
               else counts.false_positives.(index - 1) /. counts.negative_weight);
         true_positive_rates =
           Vector.unsafe_init points (fun index ->
               if index = 0 then 0.0
               else counts.true_positives.(index - 1) /. counts.positive_weight);
       }
        : roc_curve)

  let precision_recall_curve ?(positive_label = 1) ?sample_weight ~truth
      ~positive_probabilities () =
    let ( let* ) = Result.bind in
    let* counts =
      curve_counts ?sample_weight ~positive_label ~truth ~positive_probabilities
        ()
    in
    let threshold_count = Array.length counts.curve_thresholds in
    Ok
      ({
         decision_thresholds =
           Vector.unsafe_init threshold_count (fun index ->
               counts.curve_thresholds.(threshold_count - index - 1));
         precisions =
           Vector.unsafe_init (threshold_count + 1) (fun index ->
               if index = threshold_count then 1.0
               else
                 let source = threshold_count - index - 1 in
                 let positives = counts.true_positives.(source) in
                 positives /. (positives +. counts.false_positives.(source)));
         recalls =
           Vector.unsafe_init (threshold_count + 1) (fun index ->
               if index = threshold_count then 0.0
               else
                 counts.true_positives.(threshold_count - index - 1)
                 /. counts.positive_weight);
       }
        : precision_recall_curve)

  let roc_auc ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      ?sample_weight ~truth ~positive_probabilities () =
    match
      roc_curve ~positive_label ?sample_weight ~truth ~positive_probabilities ()
    with
    | Error error -> (
        match Error.kind error with
        | Error.Validation { name = "binary ranking curve"; reason } ->
            Metric_internal.undefined undefined ~name:"ROC AUC" ~reason
              ~fallback:0.5
        | Error.Data _ | Error.Shape_mismatch _
        | Error.Feature_schema_mismatch _ | Error.Validation _
        | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
        | Error.Artifact _ | Error.Cancelled ->
            Error error)
    | Ok curve ->
        let length = Vector.length curve.false_positive_rates in
        let area =
          Metric_internal.sum (length - 1) (fun index ->
              let left_x = Vector.get curve.false_positive_rates index in
              let right_x = Vector.get curve.false_positive_rates (index + 1) in
              let left_y = Vector.get curve.true_positive_rates index in
              let right_y = Vector.get curve.true_positive_rates (index + 1) in
              (right_x -. left_x) *. (left_y +. right_y) *. 0.5)
        in
        Metric_internal.finite ~operation:"ROC AUC" area
end

module Regression_scorer = struct
  type metric =
    | Mean_absolute_error
    | Mean_squared_error
    | Root_mean_squared_error
    | R2

  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t = params
  type truth = Target.regression Target.t
  type prediction = Target.regression Target.t

  let create ?(undefined = Undefined_metric_policy.Error) metric =
    { metric; undefined }

  let neg_mean_absolute_error = create Mean_absolute_error
  let neg_mean_squared_error = create Mean_squared_error
  let neg_root_mean_squared_error = create Root_mean_squared_error
  let r2 ?undefined () = create ?undefined R2
  let clone specification = specification
  let params specification = specification

  let name specification =
    match specification.metric with
    | Mean_absolute_error -> "neg_mean_absolute_error"
    | Mean_squared_error -> "neg_mean_squared_error"
    | Root_mean_squared_error -> "neg_root_mean_squared_error"
    | R2 -> "r2"

  let score specification ?sample_weight ~truth ~prediction () =
    match specification.metric with
    | Mean_absolute_error ->
        Regression_metrics.mean_absolute_error ?sample_weight ~truth ~prediction
          ()
        |> Result.map Float.neg
    | Mean_squared_error ->
        Regression_metrics.mean_squared_error ?sample_weight ~truth ~prediction
          ()
        |> Result.map Float.neg
    | Root_mean_squared_error ->
        Regression_metrics.root_mean_squared_error ?sample_weight ~truth
          ~prediction ()
        |> Result.map Float.neg
    | R2 ->
        Regression_metrics.r2 ~undefined:specification.undefined ?sample_weight
          ~truth ~prediction ()
end

module Binary_classification_scorer = struct
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision
    | Recall
    | F1
    | Log_loss
    | Roc_auc

  type response = Labels | Positive_probabilities

  type params = {
    metric : metric;
    positive_label : int;
    undefined : Undefined_metric_policy.t;
  }

  type t = params
  type truth = Target.classification Target.t
  type prediction = Binary_prediction.t

  let create ?(positive_label = 1) ?(undefined = Undefined_metric_policy.Error)
      metric =
    { metric; positive_label; undefined }

  let accuracy = create Accuracy

  let balanced_accuracy ?positive_label ?undefined () =
    create ?positive_label ?undefined Balanced_accuracy

  let precision ?positive_label ?undefined () =
    create ?positive_label ?undefined Precision

  let recall ?positive_label ?undefined () =
    create ?positive_label ?undefined Recall

  let f1 ?positive_label ?undefined () = create ?positive_label ?undefined F1
  let neg_log_loss ?positive_label () = create ?positive_label Log_loss

  let roc_auc ?positive_label ?undefined () =
    create ?positive_label ?undefined Roc_auc

  let clone specification = specification
  let params specification = specification

  let response specification =
    match specification.metric with
    | Accuracy | Balanced_accuracy | Precision | Recall | F1 -> Labels
    | Log_loss | Roc_auc -> Positive_probabilities

  let name specification =
    match specification.metric with
    | Accuracy -> "accuracy"
    | Balanced_accuracy -> "balanced_accuracy"
    | Precision -> "precision"
    | Recall -> "recall"
    | F1 -> "f1"
    | Log_loss -> "neg_log_loss"
    | Roc_auc -> "roc_auc"

  let missing_response name =
    Error
      (Metric_internal.validation ~name:"binary scorer prediction"
         ~reason:(name ^ " are required by this scorer")
         ~remediation:"provide the classifier response requested by the scorer")

  let score specification ?sample_weight ~truth ~prediction () =
    let score_labels metric =
      match Binary_prediction.labels prediction with
      | None -> missing_response "predicted labels"
      | Some prediction -> metric prediction
    in
    let score_probabilities metric =
      match Binary_prediction.positive_probabilities prediction with
      | None -> missing_response "positive-class probabilities"
      | Some prediction -> metric prediction
    in
    match specification.metric with
    | Accuracy ->
        score_labels (fun prediction ->
            Binary_classification_metrics.accuracy ?sample_weight ~truth
              ~prediction ())
    | Balanced_accuracy ->
        score_labels (fun prediction ->
            Binary_classification_metrics.balanced_accuracy
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Precision ->
        score_labels (fun prediction ->
            Binary_classification_metrics.precision
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Recall ->
        score_labels (fun prediction ->
            Binary_classification_metrics.recall
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | F1 ->
        score_labels (fun prediction ->
            Binary_classification_metrics.f1
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~prediction ())
    | Log_loss ->
        score_probabilities (fun positive_probabilities ->
            Binary_classification_metrics.log_loss
              ~positive_label:specification.positive_label ?sample_weight ~truth
              ~positive_probabilities ()
            |> Result.map Float.neg)
    | Roc_auc ->
        score_probabilities (fun positive_probabilities ->
            Binary_classification_metrics.roc_auc
              ~positive_label:specification.positive_label
              ~undefined:specification.undefined ?sample_weight ~truth
              ~positive_probabilities ())
end

module Score_aggregation = struct
  type t = {
    count : int;
    mean : float;
    standard_deviation : float;
    minimum : float;
    maximum : float;
  }

  let summarize ?(undefined = Undefined_metric_policy.Error) values =
    let count = Array.length values in
    if count = 0 then
      Error
        (Metric_internal.validation ~name:"score aggregation"
           ~reason:"no scores were provided"
           ~remediation:"provide at least one fold score")
    else
      let has_nan = ref false in
      let invalid_infinity = ref None in
      Array.iteri
        (fun index value ->
          if Float.is_nan value then has_nan := true
          else if not (Float.is_finite value) then
            invalid_infinity := Some index)
        values;
      match !invalid_infinity with
      | Some index ->
          Error
            (Metric_internal.validation ~name:"score aggregation"
               ~reason:(Format.sprintf "score at index %d is infinite" index)
               ~remediation:"provide finite scores or resolve the failed fold")
      | None when !has_nan && undefined = Undefined_metric_policy.Error ->
          Error
            (Metric_internal.validation ~name:"score aggregation"
               ~reason:"at least one score is undefined (NaN)"
               ~remediation:
                 "resolve undefined fold metrics or choose an explicit \
                  aggregation policy")
      | None when !has_nan && undefined = Undefined_metric_policy.Return_nan ->
          Ok
            {
              count;
              mean = Float.nan;
              standard_deviation = Float.nan;
              minimum = Float.nan;
              maximum = Float.nan;
            }
      | None ->
          let value index =
            let observed = values.(index) in
            if Float.is_nan observed then 0.0 else observed
          in
          let mean = Metric_internal.sum count value /. Float.of_int count in
          let variance =
            Metric_internal.sum count (fun index ->
                let difference = value index -. mean in
                difference *. difference)
            /. Float.of_int count
          in
          let minimum = ref (value 0) in
          let maximum = ref (value 0) in
          for index = 1 to count - 1 do
            minimum := Float.min !minimum (value index);
            maximum := Float.max !maximum (value index)
          done;
          if Float.is_finite mean && Float.is_finite variance then
            Ok
              {
                count;
                mean;
                standard_deviation = Float.sqrt (Float.max 0.0 variance);
                minimum = !minimum;
                maximum = !maximum;
              }
          else
            Error
              (Metric_internal.numerical ~operation:"score aggregation"
                 ~reason:"the aggregate is not representable as float64")
end
