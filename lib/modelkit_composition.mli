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

  include
    TRANSFORMER
      with type t := t
       and type params := params
       and type target = unit
       and type fitted := fitted
       and type rng = Rng.t
end
