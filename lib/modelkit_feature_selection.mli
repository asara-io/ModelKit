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
