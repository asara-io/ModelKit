open Modelkit_data
open Modelkit_protocols

(** User-defined test folds with explicit always-training rows. *)
module Predefined_split : sig
  type params = { test_folds : int array }
  type t

  val create : test_folds:int array -> unit -> (t, Error.t) result
  (** Copies one assignment per source row. [-1] means always in training;
      nonnegative IDs identify test folds and need not be contiguous. Other
      negative IDs, no test folds, and an empty training partition are rejected.
  *)

  val fold_ids : t -> int array
  (** A fresh array of distinct test IDs in ascending emission order. *)

  (** [params] returns a fresh assignment array. [split] checks source length;
      each assigned row is tested once, while [-1] rows are never tested. Each
      training partition contains every source row outside its test fold. Row
      views retain source order. RNG, targets, and groups are ignored;
      caller-defined assignments are not checked for group exclusion. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Exhaustive single-row testing in source order. *)
module Leave_one_out : sig
  type t
  type params = unit

  val create : unit -> t

  (** Requires at least two rows. Each row is tested once against all remaining
      training rows. RNG, targets, and groups are ignored. Views are emitted
      eagerly and require quadratic total index storage. Single-row scores such
      as R-squared can be undefined; select an appropriate scorer. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Exhaustive single-group testing with complete group exclusion. *)
module Leave_one_group_out : sig
  type t
  type params = unit

  val create : unit -> t

  (** Requires aligned groups and at least two distinct group IDs. Test groups
      are emitted in ascending integer-ID order, with rows in source order.
      Every row is tested once; each training partition contains all other
      groups. RNG and targets are ignored. Eager views require storage
      proportional to row count times distinct group count. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Greedy class balancing while keeping every group intact. *)
module Stratified_group_k_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  (** Defaults to five folds without shuffling. Requires at least two folds,
      aligned classification labels and groups, and at least as many distinct
      groups as folds. Every fold is nonempty, every row is tested exactly once,
      and a group never crosses training/test boundaries within a fold.

      Groups are considered in descending standard deviation of their class
      counts. Ties use ascending group IDs, or seeded shuffled order when
      [shuffle=true]; unequal dispersions retain their ordering. Classes are
      accumulated in ascending label order. Each group minimizes the mean across
      classes of the standard deviation of per-fold fractions of that class.
      Objectives within [1e-12] tie on fewer rows already allocated to the fold,
      then the lowest fold index. Remaining groups fill empty folds when
      necessary to guarantee nonempty partitions. Output row views retain source
      order.

      Class balance is a heuristic, not an optimum or a guarantee of class
      coverage. Classes confined to too few groups may be absent from training
      or test partitions. This remains a valid split; estimator/scorer
      requirements are checked downstream. RNG behavior is portable but does not
      reproduce NumPy streams. Sparse group histograms use space linear in the
      observed group/class pairs, plus folds times class count; returned row
      indices require space proportional to folds times row count. *)
  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end
