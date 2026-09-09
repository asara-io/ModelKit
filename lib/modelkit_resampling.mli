open Modelkit_data
open Modelkit_protocols

(** Partition sizes shared by resampling and train/test splitting. *)
module Split_size : sig
  type t =
    | Count of int
    | Fraction of float
        (** Counts must be positive; fractions must be finite and strictly
            between zero and one. Training fractions round down, test fractions
            round up. A missing size is the complement of the other. Supplying
            both may leave unused rows; neither partition may be empty and their
            sum cannot exceed the source size. *)
end

(** Independent shuffled partitions; defaults to ten splits and a 10% test
    fraction. Rows are disjoint within each split but may recur across splits.
    Output rows retain permutation order, not sorted source order. Random
    streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Shuffle_split : sig
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t

  val create :
    ?splits:int ->
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Shuffled partitions with proportional class allocation; defaults to ten
    splits and a 10% test fraction. Requires aligned classification labels, at
    least two rows per class, and at least as many rows per partition as
    classes. Largest-remainder allocation fills training first, then test from
    remaining rows, with seeded tie breaking. Extreme imbalance can still omit a
    rare class from a partition. Classes follow first appearance order. Random
    streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Stratified_shuffle_split : sig
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t

  val create :
    ?splits:int ->
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** One train/test partition; defaults to shuffling and a 25% test fraction.
    With [shuffle=false], training takes the first rows, test takes the next
    rows, and any unused rows follow. Without shuffling the RNG is ignored.
    Random streams are deterministic for the same input and seed, independent of
    execution scheduling; they do not reproduce NumPy random streams. Groups do
    not constrain these splits; use group-aware splitting when group exclusion
    is required. *)
module Holdout : sig
  type params = {
    train_size : Split_size.t option;
    test_size : Split_size.t option;
    shuffle : bool;
  }

  type t

  val create :
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    ?shuffle:bool ->
    unit ->
    (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Repeated shuffled K-fold, defaulting to five folds and ten repetitions. Each
    repetition tests every row exactly once. Splits are emitted in
    repetition-major, then fold-major order; row indices retain source order.
    Extending the repetition count preserves earlier repetitions. Random streams
    are deterministic for the same input and seed, independent of execution
    scheduling; they do not reproduce NumPy random streams. Groups do not
    constrain these splits; use group-aware splitting when group exclusion is
    required. *)
module Repeated_k_fold : sig
  type params = { folds : int; repeats : int }
  type t

  val create : ?folds:int -> ?repeats:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

(** Repeated shuffled stratified K-fold, with five folds and ten repetitions by
    default. Uses the class allocation and feasibility rules of
    {!Stratified_k_fold}; rare classes can be absent in a fold. Splits are
    repetition-major, then fold-major, and indices retain source order.
    Extending the repetition count preserves earlier repetitions. Random streams
    are deterministic for the same input and seed, independent of execution
    scheduling; they do not reproduce NumPy random streams. Groups do not
    constrain these splits; use group-aware splitting when group exclusion is
    required. *)
module Repeated_stratified_k_fold : sig
  type params = { folds : int; repeats : int }
  type t

  val create : ?folds:int -> ?repeats:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

(** Materialized train/test datasets with all row-aligned fields selected
    together, preserving feature schema and finiteness policy. *)
module Train_test_split : sig
  val split :
    ?train_size:Split_size.t ->
    ?test_size:Split_size.t ->
    ?shuffle:bool ->
    ?stratify:Target.classification Target.t ->
    rng:Rng.t ->
    'kind Dataset.t ->
    unit ->
    ('kind Dataset.t * 'kind Dataset.t, Error.t) result
  (** Defaults to shuffling and a 25% test fraction, like {!Holdout}. Optional
      classification labels must match the source row count and require
      [shuffle=true]; stratification uses {!Stratified_shuffle_split}'s
      allocation rules. Dataset groups are copied, not kept exclusive. All
      fields follow the exact selected order; source data is immutable.
      Materialization can fail if a selected sample-weight partition has zero
      total weight. For row views instead of copies, use the splitter modules
      and {!Split.of_views}. *)
end
