open Modelkit_data
open Modelkit_protocols

module Undefined_metric_policy : sig
  type t = Error | Return_nan | Use_fallback
end

(** Regression metrics over aligned finite targets.

    Metrics accept optional non-negative sample weights. Empty inputs, shape
    mismatches, and non-finite numerical results are typed failures. R-squared
    is undefined for a constant truth vector: its finite fallback is [1.] for
    perfect predictions and [0.] otherwise. All operations are [O(samples)] with
    [O(1)] scratch; [residual_curve] additionally allocates its returned
    residual vector. *)
module Regression_metrics : sig
  type residual_curve = { predictions : Vector.t; residuals : Vector.t }

  val mean_absolute_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val mean_squared_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val root_mean_squared_error :
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val r2 :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (float, Error.t) result

  val residual_curve :
    truth:Target.regression Target.t ->
    prediction:Target.regression Target.t ->
    unit ->
    (residual_curve, Error.t) result
end

(** Aligned binary classifier outputs used by scorer specifications.

    At least one response must be present. Positive-class probabilities are
    finite and lie in [[0., 1.]]. When both responses are supplied, their
    lengths must agree. *)
module Binary_prediction : sig
  type t

  val create :
    ?labels:Target.classification Target.t ->
    ?positive_probabilities:Vector.t ->
    unit ->
    (t, Error.t) result

  val length : t -> int
  val labels : t -> Target.classification Target.t option
  val positive_probabilities : t -> Vector.t option
end

(** Binary classification metrics and plotting-neutral curve data.

    [positive_label] defaults to [1]. Observed labels must contain at most one
    other integer label. Curve thresholds are deterministic: ROC thresholds are
    descending and begin with infinity; precision-recall thresholds are
    ascending. ROC and precision-recall curves require positive and negative
    weighted support. Scalar fallbacks are zero for undefined precision, recall,
    F1, and balanced accuracy, and [0.5] for ROC AUC. Scalar label and loss
    metrics are [O(samples)] with [O(1)] scratch. Ranking curves are
    [O(samples * log samples)] time and [O(samples)] space. *)
module Binary_classification_metrics : sig
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

  val accuracy :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val balanced_accuracy :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val precision :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val recall :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val f1 :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val log_loss :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (float, Error.t) result

  val roc_auc :
    ?positive_label:int ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (float, Error.t) result

  val roc_curve :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (roc_curve, Error.t) result

  val precision_recall_curve :
    ?positive_label:int ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    positive_probabilities:Vector.t ->
    unit ->
    (precision_recall_curve, Error.t) result
end

(** Higher-is-better regression scorer specifications.

    Loss scorers negate their corresponding metric, following scikit-learn's
    selection convention. *)
module Regression_scorer : sig
  type metric =
    | Mean_absolute_error
    | Mean_squared_error
    | Root_mean_squared_error
    | R2

  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t

  val create : ?undefined:Undefined_metric_policy.t -> metric -> t
  val neg_mean_absolute_error : t
  val neg_mean_squared_error : t
  val neg_root_mean_squared_error : t
  val r2 : ?undefined:Undefined_metric_policy.t -> unit -> t

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.regression Target.t
       and type prediction = Target.regression Target.t
end

(** Higher-is-better binary classification scorer specifications.

    Label metrics request {!Binary_prediction.labels}; log loss and ROC AUC
    request positive-class probabilities. Log loss is negated for selection. *)
module Binary_classification_scorer : sig
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

  type t

  val create :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> metric -> t

  val response : t -> response
  val accuracy : t

  val balanced_accuracy :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val precision :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val recall :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val f1 :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  val neg_log_loss : ?positive_label:int -> unit -> t

  val roc_auc :
    ?positive_label:int -> ?undefined:Undefined_metric_policy.t -> unit -> t

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.classification Target.t
       and type prediction = Binary_prediction.t
end

(** Aligned multiclass classifier outputs used by scorer specifications.

    At least one response must be present. Probabilities require their class
    order, must be finite values in [[0., 1.]], and each row must sum to one
    within [1e-6]. When both responses are supplied, their lengths must agree.
*)
module Multiclass_prediction : sig
  type t

  val create :
    ?labels:Target.classification Target.t ->
    ?classes:int array ->
    ?probabilities:Matrix.t ->
    unit ->
    (t, Error.t) result

  val length : t -> int
  val labels : t -> Target.classification Target.t option
  val classes : t -> int array option
  val probabilities : t -> Matrix.t option
end

(** Confusion-matrix and averaged classification metrics for any label count.

    The label set defaults to the ascending union of observed truth and
    prediction labels; an explicit [labels] array fixes the confusion-matrix
    order and drops rows whose labels fall outside it. [Micro] pools weighted
    counts before dividing, [Macro] averages per-class ratios equally, and
    [Weighted] averages them by weighted truth support. A per-class ratio with a
    zero denominator follows the undefined policy, with a fallback of zero,
    matching scikit-learn's default zero-division handling. Balanced accuracy
    averages recall over classes with positive support. Log loss clips
    probabilities to the machine epsilon and requires every truth label to have
    a probability column. Label metrics are [O(samples + classes squared)] with
    [O(classes squared)] scratch. *)
module Multiclass_classification_metrics : sig
  type average = Micro | Macro | Weighted
  type confusion_matrix = { labels : int array; counts : Matrix.t }

  type class_scores = {
    class_labels : int array;
    precisions : Vector.t;
    recalls : Vector.t;
    f1_scores : Vector.t;
    supports : Vector.t;
  }

  val confusion_matrix :
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (confusion_matrix, Error.t) result
  (** Rows are truth labels and columns are predicted labels. *)

  val accuracy :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val balanced_accuracy :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val class_scores :
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (class_scores, Error.t) result

  val precision :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val recall :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val f1 :
    ?average:average ->
    ?undefined:Undefined_metric_policy.t ->
    ?sample_weight:Sample_weight.t ->
    ?labels:int array ->
    truth:Target.classification Target.t ->
    prediction:Target.classification Target.t ->
    unit ->
    (float, Error.t) result

  val log_loss :
    ?sample_weight:Sample_weight.t ->
    truth:Target.classification Target.t ->
    classes:int array ->
    probabilities:Matrix.t ->
    unit ->
    (float, Error.t) result
end

(** Higher-is-better multiclass scorer specifications.

    Label metrics request {!Multiclass_prediction.labels}; log loss requests
    class probabilities and is negated for selection. Averaged metrics default
    to [Macro] and carry the averaging mode in their name, for example
    [f1_weighted]. *)
module Multiclass_classification_scorer : sig
  type metric =
    | Accuracy
    | Balanced_accuracy
    | Precision of Multiclass_classification_metrics.average
    | Recall of Multiclass_classification_metrics.average
    | F1 of Multiclass_classification_metrics.average
    | Log_loss

  type response = Labels | Class_probabilities
  type params = { metric : metric; undefined : Undefined_metric_policy.t }
  type t

  val create : ?undefined:Undefined_metric_policy.t -> metric -> t
  val response : t -> response
  val accuracy : t
  val balanced_accuracy : ?undefined:Undefined_metric_policy.t -> unit -> t

  val precision :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val recall :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val f1 :
    ?undefined:Undefined_metric_policy.t ->
    ?average:Multiclass_classification_metrics.average ->
    unit ->
    t

  val neg_log_loss : t

  include
    SCORER
      with type t := t
       and type params := params
       and type truth = Target.classification Target.t
       and type prediction = Multiclass_prediction.t
end

(** Stable [O(scores)] population aggregation with [O(1)] scratch.

    Empty arrays and infinities are typed failures. For NaN values, [Error]
    fails, [Return_nan] returns NaN summary statistics, and [Use_fallback]
    substitutes zero. *)
module Score_aggregation : sig
  type t = {
    count : int;
    mean : float;
    standard_deviation : float;
    minimum : float;
    maximum : float;
  }

  val summarize :
    ?undefined:Undefined_metric_policy.t -> float array -> (t, Error.t) result
end
