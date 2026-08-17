(** Optional bounded Domainslib execution for ModelKit.

    Requested [domains] include the calling domain. A value of one delegates to
    ModelKit's portable sequential backend without creating a Domainslib pool.
    Parallel execution uses [O(tasks)] atomic result slots, runs at most
    [domains] task bodies concurrently, preserves logical result order, and
    returns the error from the lowest failing task index.

    [inner_threads] describes the maximum threads used inside one task. When it
    is omitted, common BLAS/OpenMP environment limits are inspected and the
    largest valid value is used; otherwise one is assumed. Diagnostics never
    mutate process environment or numerical-library settings. *)

type thread_limit = { variable : string; threads : int }
(** One valid positive limit read from the process environment. *)

(** How {!diagnostics.inner_threads} was determined. *)
type thread_limit_source =
  | Assumed_sequential
  | Explicit
  | Environment of thread_limit array

(** Observable configuration concerns. Warnings do not prevent execution. *)
type warning =
  | Invalid_environment_limit of { variable : string; value : string }
  | Conflicting_environment_limits of thread_limit array
  | Domains_exceed_recommended of {
      requested_domains : int;
      recommended_domains : int;
    }
  | Nested_parallelism of { fold_domains : int; inner_threads : int }
  | Potential_oversubscription of {
      estimated_runnable_threads : int;
      recommended_domains : int;
    }

type diagnostics = {
  fold_domains : int;
  recommended_domains : int;
  inner_threads : int;
  inner_thread_source : thread_limit_source;
  estimated_runnable_threads : int;
  warnings : warning array;
}
(** The effective fold and inner-task concurrency assessment. The runnable
    estimate is saturated at [max_int] rather than overflowing. *)

type t

val create :
  ?inner_threads:int -> domains:int -> unit -> (t, Modelkit.Error.t) result
(** [create ~domains ()] validates a positive total domain count. Supplying
    [inner_threads] overrides environment detection for diagnostics only; the
    backend does not configure numerical libraries. *)

val diagnostics : t -> diagnostics
val warning_to_string : warning -> string

val execution : t -> Modelkit.Execution.t
(** [execution configuration] packages this backend for [cross_validate] and
    finite grid-search calls. *)

include Modelkit.EXECUTION with type t := t
