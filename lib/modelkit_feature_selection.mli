open Modelkit_data
open Modelkit_protocols

(** Dense univariate feature selection by a target association score. *)
module Univariate_selection : sig
  type selection = Count of int | Percentile of float

  (** Regression selection using the squared Pearson-correlation F statistic.

      This is a ranking statistic, not an inferential test: p-values and
      multiple-testing corrections are outside this contract. Inputs must be
      finite and unweighted, with at least three training rows. Constant
      features or targets score zero; perfect correlation scores
      [Float.max_float]. *)
  module Regression : sig
    type params = { selection : selection }
    type t
    type fitted

    val create : selection -> (t, Error.t) result
    val scores : fitted -> Vector.t
    val selected_indices : fitted -> int array

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Target.regression Target.t
         and type fitted := fitted
         and type rng = Rng.t
  end

  (** Classification selection using the one-way ANOVA F statistic.

      This is a ranking statistic, not an inferential test: p-values and
      multiple-testing corrections are outside this contract. Inputs must be
      finite and unweighted. Fitting requires at least two observed classes and
      at least one residual degree of freedom. Constant features score zero;
      nonzero between-class variance with zero within-class variance scores
      [Float.max_float]. *)
  module Classification : sig
    type params = { selection : selection }
    type t
    type fitted

    val create : selection -> (t, Error.t) result
    val scores : fitted -> Vector.t
    val selected_indices : fitted -> int array

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Target.classification Target.t
         and type fitted := fitted
         and type rng = Rng.t
  end
end

(** Conversion of fitted linear-model coefficients into non-negative feature
    importances. *)
module Feature_importance : sig
  type coefficient_norm = L1 | L2 | Max

  val absolute_coefficients : Vector.t -> (Vector.t, Error.t) result
  (** Returns the elementwise absolute coefficient values. *)

  val coefficient_norms :
    ?norm:coefficient_norm -> Matrix.t -> (Vector.t, Error.t) result
  (** Reduces coefficient rows into one importance per column. The default is
      [L1], matching the conventional multiclass model-selection reduction. At
      least one coefficient row is required. *)
end

(** Dense selection using importances extracted from a fitted estimator.

    Fit and transform inputs must be finite. Optional sample weights are passed
    to the importance estimator after row-alignment validation. *)
module Select_from_model : sig
  type threshold = Mean | Median | Value of float

  module Make (Estimator : IMPORTANCE_ESTIMATOR with type rng = Rng.t) : sig
    type params = {
      threshold : threshold;
      max_features : int option;
      estimator_params : Estimator.params;
    }

    type t
    type fitted

    val create :
      ?threshold:threshold ->
      ?max_features:int ->
      Estimator.t ->
      (t, Error.t) result

    val importances : fitted -> Vector.t
    val threshold_value : fitted -> float
    val selected_indices : fitted -> int array

    val fitted_estimator : fitted -> Estimator.fitted
    (** The fitted estimator uses the complete selector input schema and exists
        to derive importances; it is distinct from a pipeline's downstream
        estimator over selected columns. *)

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Estimator.target
         and type fitted := fitted
         and type rng = Rng.t
  end
end

(** Dense recursive feature elimination using fitted estimator importances. *)
module Recursive_feature_elimination : sig
  type step = Count of int | Fraction of float

  module Make (Estimator : IMPORTANCE_ESTIMATOR with type rng = Rng.t) : sig
    type params = {
      feature_count : int;
      step : step;
      estimator_params : Estimator.params;
    }

    type t
    type fitted

    val create :
      ?step:step -> feature_count:int -> Estimator.t -> (t, Error.t) result

    val selected_indices : fitted -> int array
    val ranking : fitted -> int array
    val final_importances : fitted -> Vector.t
    val fitted_estimator : fitted -> Estimator.fitted

    include
      TRANSFORMER
        with type t := t
         and type params := params
         and type target = Estimator.target
         and type fitted := fitted
         and type rng = Rng.t
  end
end
