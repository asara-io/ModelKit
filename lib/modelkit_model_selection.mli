open Modelkit_data
open Modelkit_metadata
open Modelkit_protocols
open Modelkit_pipeline
open Modelkit_metrics

module Cross_validation : sig
  (** [metadata] defaults to {!Metadata.of_dataset}: dataset weights and groups
      are selected with each fold's exact training/test row views, including
      inference. An explicit carrier replaces that default without merging; its
      fields must match the complete dataset's row count. Splitters still use
      dataset groups and scorers still use dataset weights. Search refit
      receives the complete carrier. These inputs are never inferred from a
      previously fitted model.

      A supplied callback receives evaluation lifecycle events and is delivered
      to nested consumers only when their per-method request opts in. Fold
      events are buffered and dispatched on the caller domain in logical order;
      see {!Callback} for bounds, cancellation, and failure semantics. *)

  type failure_policy = Abort | Record
  type partition = Train | Test

  type failure_phase =
    | Materialization
    | Fitting
    | Prediction of partition
    | Scoring of { partition : partition; scorer : string }

  type failure = { phase : failure_phase; error : Error.t }

  type score = {
    name : string;
    train_score : (float, Error.t) result option;
    test_score : (float, Error.t) result option;
  }

  type 'model fold = {
    fold_index : int;
    fit_time : float;
    score_time : float;
    scores : score array;
    model : 'model option;
    train_indices : int array option;
    test_indices : int array option;
    failures : failure array;
  }

  type 'model report
  type 'target splitter

  val target_independent_splitter :
    (module SPLITTER
       with type t = 'specification
        and type target = unit
        and type rng = Rng.t) ->
    'specification ->
    'target splitter
  (** Adapts a target-independent splitter such as {!K_fold}. *)

  val target_aware_splitter :
    (module SPLITTER
       with type t = 'specification
        and type target = 'target
        and type rng = Rng.t) ->
    'specification ->
    'target splitter
  (** Adapts a target-aware splitter such as {!Stratified_k_fold}. *)

  val folds : 'model report -> 'model fold array
  val successful_fold_count : 'model report -> int

  module Regression : sig
    type model =
      (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      splitter:Target.regression Target.t splitter ->
      scorers:Regression_scorer.t array ->
      seed:Seed.t ->
      (Target.regression Target.t, Target.regression Target.t) Pipeline.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      splitter:Target.classification Target.t splitter ->
      scorers:Binary_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    val cross_validate :
      ?return_train_score:bool ->
      ?return_models:bool ->
      ?return_indices:bool ->
      ?failure_policy:failure_policy ->
      ?fit_seed:Seed.t ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      splitter:Target.classification Target.t splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      seed:Seed.t ->
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Typed exhaustive search over finite immutable configuration grids.

    Axes retain declaration order and their values retain caller order. The
    Cartesian product varies the last axis fastest. Each candidate is evaluated
    on identical split membership, while fitted fold RNGs derive from the
    logical candidate and fold identities. Named refitting ranks by the [refit]
    scorer's mean test score in descending order; equal scores receive equal
    competition ranks and the lowest candidate index wins a tie.

    [Record] keeps failed candidates and selects from candidates whose primary
    test score aggregates successfully. [Abort] returns the first failure in
    candidate order. With refitting enabled, the winning immutable specification
    is fitted once on the complete dataset. [search_with_policy] also supports
    disabled refitting and custom multi-metric selection through
    {!Grid_search.refit_policy}. For [c] candidates, [f] folds, and [s] scorers,
    search performs at most [c * f + 1] fits and retains [O(c * (f + s))] report
    data. An empty axis array evaluates the base configuration once. [execution]
    controls each candidate's fold evaluation and defaults to sequential
    execution; candidates themselves are evaluated in stable sequential order.
*)
module Grid_search : sig
  (** [metadata] defaults to {!Metadata.of_dataset}: dataset weights and groups
      are selected with each fold's exact training/test row views, including
      inference. An explicit carrier replaces that default without merging; its
      fields must match the complete dataset's row count. Splitters still use
      dataset groups and scorers still use dataset weights. Search refit
      receives the complete carrier. These inputs are never inferred from a
      previously fitted model.

      A supplied callback receives evaluation lifecycle events and is delivered
      to nested consumers only when their per-method request opts in. Fold
      events are buffered and dispatched on the caller domain in logical order;
      see {!Callback} for bounds, cancellation, and failure semantics. *)

  type parameter_value =
    | Bool of bool
    | Int of int
    | Float of float
    | String of string

  type parameter = {
    parameter_name : string;
    parameter_value : parameter_value;
  }

  type 'configuration axis

  val axis :
    name:string ->
    values:'value array ->
    encode:('value -> parameter_value) ->
    set:('configuration -> 'value -> ('configuration, Error.t) result) ->
    ('configuration axis, Error.t) result
  (** Creates one non-empty typed axis. [set] must return a new configuration
      without mutating its input. *)

  type ('configuration, 'target, 'prediction) grid

  val create :
    base:'configuration ->
    build:
      ('configuration -> (('target, 'prediction) Pipeline.t, Error.t) result) ->
    'configuration axis array ->
    (('configuration, 'target, 'prediction) grid, Error.t) result

  val candidate_count : ('configuration, 'target, 'prediction) grid -> int

  type score_summary = {
    scorer_name : string;
    train : (Score_aggregation.t, Error.t) result option;
    test : (Score_aggregation.t, Error.t) result;
  }

  type 'model candidate = {
    candidate_index : int;
    parameters : parameter array;
    rank : int option;
    mean_fit_time : float;
    mean_score_time : float;
    scores : score_summary array;
    evaluation : 'model Cross_validation.report option;
    build_error : Error.t option;
  }

  type 'model refit_policy =
    | No_refit
    | Best_score of string
    | Custom of ('model candidate array -> (int, Error.t) result)
        (** [No_refit] evaluates candidates without ranking or fitting a winner.
            [Best_score name] preserves named-scorer ranking and refitting.
            [Custom select] receives copied candidate arrays after evaluation
            and returns an array index. The chosen candidate must have built and
            have at least one successful test-score aggregate; the selector
            decides which metrics it requires. Custom selection leaves ranks
            unset. Selectors must not depend on timings if reproducibility
            across execution backends matters. User exceptions propagate;
            returned errors follow the failure policy, while callback/control
            errors always abort. *)

  type 'model selected = {
    selected_candidate_index : int;
    selected_model : 'model;
  }

  type 'model report

  val candidates : 'model report -> 'model candidate array

  val selection : 'model report -> ('model selected, Error.t) result
  (** Returns the fitted winner; disabled refitting returns a typed error. *)

  val refit_result : 'model report -> ('model selected option, Error.t) result
  (** [Ok None] identifies an intentional no-refit run, including a recorded
      report where all candidates failed. [Error] identifies selection/refit
      failure. Candidate failures remain available independently. *)

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        grid ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        grid ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      grid:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        grid ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      policy:model refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end

(** Typed immutable parameter distributions with explicit random streams. *)
module Parameter_distribution : sig
  type 'a t

  val choice : 'a array -> ('a t, Error.t) result
  (** Copies a nonempty array. Values themselves must be immutable. *)

  val uniform : low:float -> high:float -> unit -> (float t, Error.t) result
  val log_uniform : low:float -> high:float -> unit -> (float t, Error.t) result

  val int_uniform : low:int -> high:int -> unit -> (int t, Error.t) result
  (** Bounds are lower-inclusive and upper-exclusive. Float bounds must be
      finite; log-uniform bounds must also be positive. Integer sampling avoids
      modulo bias and supports the complete OCaml integer range of bounds. *)

  val custom : (Rng.t -> ('a, Error.t) result) -> 'a t
  (** Samplers must be deterministic for their RNG and must not retain mutable
      random state. Returned errors are recorded as candidate build failures
      under [Record]; exceptions propagate. *)

  val sample : rng:Rng.t -> 'a t -> ('a, Error.t) result
end

(** Randomized search using the same evaluation, metadata, callback and refit
    contracts as {!Grid_search}. Finite choice-only spaces sample Cartesian
    positions without replacement, capping iterations at the product size.
    Duplicate choice values can still yield equal configurations. If any axis
    uses a distribution, all axes sample with replacement. Candidate and axis
    identities derive deterministic sampling streams, separately from fit RNGs.
    Increasing iterations preserves the sampled prefix. Random identities do not
    reproduce NumPy streams. *)
module Randomized_search : sig
  type 'configuration axis

  val axis :
    name:string ->
    distribution:'value Parameter_distribution.t ->
    encode:('value -> Grid_search.parameter_value) ->
    set:('configuration -> 'value -> ('configuration, Error.t) result) ->
    ('configuration axis, Error.t) result
  (** Setters return new immutable configurations. Axis names must be nonblank
      and unique. Encoders and setters must be deterministic and must not mutate
      inputs. *)

  type ('configuration, 'target, 'prediction) space

  val create :
    ?iterations:int ->
    base:'configuration ->
    build:
      ('configuration -> (('target, 'prediction) Pipeline.t, Error.t) result) ->
    'configuration axis array ->
    (('configuration, 'target, 'prediction) space, Error.t) result
  (** Defaults to ten iterations. An empty axis array samples the base once.
      Iterations must fit an array; finite Cartesian products must fit an OCaml
      integer. Sampling finite spaces uses storage proportional to iterations,
      without expanding the complete Cartesian product. *)

  val candidate_count : ('configuration, 'target, 'prediction) space -> int

  type 'configuration sampled_candidate = {
    sampled_parameters : Grid_search.parameter array;
    sampled_configuration : ('configuration, Error.t) result;
  }

  val sample :
    seed:Seed.t ->
    ('configuration, 'target, 'prediction) space ->
    'configuration sampled_candidate array
  (** Previews typed configurations and encoded parameters without building or
      fitting pipelines. Failed draws omit that axis's parameter; subsequent
      axes still draw after ordinary failures. The first ordinary failure is
      retained with candidate/axis context; control errors override it and stop
      remaining axis draws. Search draws candidates lazily inside candidate
      callbacks, so cancellation prevents sampling later candidates. *)

  type 'model report = 'model Grid_search.report

  val candidates : 'model report -> 'model Grid_search.candidate array
  val selection : 'model report -> ('model Grid_search.selected, Error.t) result

  val refit_result :
    'model report -> ('model Grid_search.selected option, Error.t) result

  module Regression : sig
    type model = Cross_validation.Regression.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        space ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.regression Target.t,
          Target.regression Target.t )
        space ->
      splitter:Target.regression Target.t Cross_validation.splitter ->
      scorers:Regression_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.regression Dataset.t ->
      (model report, Error.t) result
  end

  module Binary_classification : sig
    type model = Cross_validation.Binary_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Binary_classification_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end

  module Multiclass_classification : sig
    type model = Cross_validation.Multiclass_classification.model

    val search :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      refit:string ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result

    val search_with_policy :
      ?return_train_score:bool ->
      ?failure_policy:Cross_validation.failure_policy ->
      ?execution:Execution.t ->
      ?metadata:Metadata.t ->
      space:
        ( 'configuration,
          Target.classification Target.t,
          Target.classification Target.t )
        space ->
      splitter:Target.classification Target.t Cross_validation.splitter ->
      scorers:Multiclass_classification_scorer.t array ->
      policy:model Grid_search.refit_policy ->
      seed:Seed.t ->
      Target.classification Dataset.t ->
      (model report, Error.t) result
  end
end
