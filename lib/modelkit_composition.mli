open Modelkit_data
open Modelkit_protocols
open Modelkit_pipeline

(** Immutable ordered column selections. Duplicate indices or names within a
    selector are rejected; different branches may select the same column. *)
module Column_selector : sig
  type t

  val all : t

  val indices : int array -> (t, Error.t) result
  (** Copies non-negative indices, preserving caller order. Bounds are checked
      when the selector is resolved against an input schema. *)

  val names : string array -> (t, Error.t) result
  (** Copies unique feature names. Non-empty name selection requires a named
      input schema; absent names fail before any branch fits. *)

  val resolve : t -> Feature_schema.t -> (int array, Error.t) result
  (** Returns a fresh array of positions. Empty selections are valid. *)
end

(** Dense column-wise composition of unsupervised transformer stages.

    Branches fit independently on the same training rows and concatenate in
    declaration order. Selectors retain their own order. Unselected columns are
    dropped by default or appended in source order as [remainder]. Columns
    explicitly selected by a dropped branch are excluded from the remainder.
    Empty selections skip fitting and inference, including for custom stages.

    Branch inputs receive named schemas: original feature names when available,
    otherwise [x0], [x1], etc. using original column positions. Output names are
    [branch__feature], using the child's output names or branch-local [x0],
    [x1], etc. for anonymous child outputs. Name collisions are typed errors.
    Inference requires the same complete ordered input schema as fitting.

    Composition validates structure; each child owns its finiteness policy.
    Passthrough preserves values, including NaNs. All-dropped output is a dense
    matrix with the original row count and zero columns; downstream estimators
    may reject it. There is no implicit sparse conversion or artifact codec. *)
module Column_transformer : sig
  type remainder = Drop | Passthrough
  type branch
  type t
  type params = t
  type fitted

  val transformer : columns:Column_selector.t -> Pipeline.transformer -> branch
  (** Uses the packaged stage's name, weight-routing policy, and transform. *)

  val passthrough :
    name:string -> columns:Column_selector.t -> (branch, Error.t) result

  val drop :
    name:string -> columns:Column_selector.t -> (branch, Error.t) result

  val create :
    ?remainder:remainder ->
    ?max_output_features:int ->
    branch array ->
    (t, Error.t) result
  (** Copies the branch array. Names must be non-blank, unique, and different
      from the reserved name [remainder]. [max_output_features] defaults to
      [100_000] and must lie between zero and [Sys.max_array_length]. It bounds
      combined output width before final concatenation; child transforms must
      enforce their own allocation limits. *)

  type branch_info = {
    name : string;
    input_indices : int array;
    output_start : int;
    output_count : int;
  }

  val branches : fitted -> branch_info array
  (** Returns defensive copies of resolved selections and half-open output
      ranges, including dropped/empty branches and any passthrough remainder. *)

  type allocation = { selected_input_bytes : int64; output_bytes : int64 }
  (** Dense payload bytes allocated by the composition operation: selected
      inputs copied for active transform branches, and the final concatenated
      matrix. Passthrough copies directly into the final matrix. Child-owned
      outputs/scratch, schemas, indices, and OCaml allocation overhead are
      excluded; these figures are neither total allocation nor peak memory. *)

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result
  (** Fits each selected child once and reuses its training output. Weights are
      checked for row alignment and supplied to child packages, which route them
      only when explicitly configured. Child RNGs derive from branch names and
      positions, independently of sibling random consumption. *)

  val transform_with_report :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages the composition into a pipeline, reusing training outputs and
      forwarding sample weights to the child packages. This avoids an extra
      transform pass during fitting. Use [Pipeline.Supervised.unsupervised] to
      include it in a target-aware pipeline. *)

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t
    type 'kind branch

    val transformer :
      columns:Column_selector.t ->
      'kind Pipeline.Supervised.stage ->
      'kind branch

    val passthrough :
      name:string -> columns:Column_selector.t -> ('kind branch, Error.t) result

    val drop :
      name:string -> columns:Column_selector.t -> ('kind branch, Error.t) result

    val create :
      ?remainder:remainder ->
      ?max_output_features:int ->
      'kind branch array ->
      ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Sequential preprocessing that can itself be used as a transformer stage.

    Each child fits only on the output of earlier children from the same
    training partition. Fitted schemas propagate without additional name
    prefixes. The empty chain is the identity, including its input schema.
    Stages preserve row count and order. Child RNGs derive from stage names and
    positions; sample weights retain each child's explicit routing policy. The
    chain allocates no additional numeric buffers beyond its children. *)
module Transformer_pipeline : sig
  type t
  type params = t
  type fitted

  val create : Pipeline.transformer array -> (t, Error.t) result
  (** Copies the stage array and rejects duplicate names. *)

  val stage_names : t -> string array

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t, Error.t) result
  (** Fits and transforms each child once, reusing its output for the next
      child. All children are unsupervised and receive no targets. *)

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages a chain for ordinary pipelines, column transformers, or feature
      unions without an extra training transform pass. Sample weights reach only
      children that request them. Composite artifact codecs are not yet
      supported. *)

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t

    val create :
      'kind Pipeline.Supervised.stage array -> ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end

(** Dense concatenation of independently fitted unsupervised branches.

    Every active branch receives the same immutable full input and schema,
    without selection copies. Outputs concatenate in declaration order with
    names [branch__feature]. Anonymous child outputs use branch-local [x0],
    [x1], etc. Duplicate generated names are typed failures.

    Dropped branches do no work. Active branches receive zero-column inputs too,
    leaving their admissibility to each transformer. Empty or all-dropped unions
    produce a zero-column matrix retaining the input row count. Branches run
    sequentially; surrounding CV may own bounded parallelism. This API does not
    add sparse output, branch-output weighting, or composite artifact codecs.
    Target-aware branches are available through [Supervised]. *)
module Feature_union : sig
  type branch
  type t
  type params = t
  type fitted

  val transformer : Pipeline.transformer -> branch
  val passthrough : name:string -> (branch, Error.t) result
  val drop : name:string -> (branch, Error.t) result

  val create : ?max_output_features:int -> branch array -> (t, Error.t) result
  (** Copies the branch array and rejects duplicate or blank names.
      [max_output_features] defaults to [100_000] and must be between zero and
      [Sys.max_array_length]. The combined width is checked before generating
      output names and concatenating; each child owns its own allocation bounds.
  *)

  type branch_info = { name : string; output_start : int; output_count : int }

  val branches : fitted -> branch_info array

  type allocation = { output_bytes : int64 }
  (** Payload bytes of the final concatenated matrix only. Input is shared;
      child allocations, metadata, and scratch are excluded. *)

  val fit_transform :
    t ->
    ?sample_weight:Sample_weight.t ->
    rng:Rng.t ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    y:unit option ->
    unit ->
    (fitted * Matrix.t * allocation, Error.t) result
  (** Fits each active branch once, reuses its training output, and derives
      random streams from branch names and positions. Sample weights are checked
      for row alignment before any branch fits, then routed only to children
      explicitly requesting them. *)

  val transform_with_report :
    fitted ->
    feature_schema:Feature_schema.t ->
    x:Matrix.t ->
    (Matrix.t * allocation, Error.t) result

  val stage : name:string -> t -> (Pipeline.transformer, Error.t) result
  (** Packages the union as an ordinary transformer stage, retaining child
      weight routing and reusing branch outputs during fitting. It can nest
      inside a column transformer or transformer pipeline. *)

  (** Target-aware composition for {!Pipeline.Supervised} pipelines. All active
      children receive the same training targets in row order; sample weights
      retain each child's opt-in policy. Adapt ordinary stages with
      {!Pipeline.Supervised.unsupervised}. The unsupervised composition's
      naming, allocation, empty-input, and deterministic seed rules apply.
      Target and weight length errors fail before any pipeline stage fits.
      Inference uses fitted values without targets. *)
  module Supervised : sig
    type 'kind t
    type 'kind branch

    val transformer : 'kind Pipeline.Supervised.stage -> 'kind branch
    val passthrough : name:string -> ('kind branch, Error.t) result
    val drop : name:string -> ('kind branch, Error.t) result

    val create :
      ?max_output_features:int ->
      'kind branch array ->
      ('kind t, Error.t) result

    val stage :
      name:string ->
      'kind t ->
      ('kind Pipeline.Supervised.stage, Error.t) result
  end

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end
