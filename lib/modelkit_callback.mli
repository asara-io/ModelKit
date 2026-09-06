open Modelkit_data

(** Typed progress and lifecycle notifications. Handlers may continue, cancel,
    or return an explanatory error. Exceptions raised by a handler propagate. *)
module Callback : sig
  (** Lifecycle events enclose evaluation, candidates, folds, refit, and
      requesting consumers' fit/transform methods. [Finished (Failed error)]
      reports ordinary failures, including recorded fold failures. Cancellation,
      callback failure, or an exception may leave a started operation
      unfinished. A handler error becomes [Error.Callback_failure]; [Cancel]
      becomes [Error.Cancelled]. Both abort evaluation even under [Record].
      Handler failures take precedence over an operation failure delivered to
      them. Events after the first handler failure or cancellation are
      discarded.

      CV admits at most [Execution.concurrency] folds per batch. Their events
      are delivered after that batch finishes, in fold order and then emission
      order within each fold. Cancellation prevents subsequent batches,
      candidates, or refit; already-running work may finish. Events are bounded
      per fold, and overflow aborts with [Error.Callback_failure]. Timing is not
      part of an event. Custom consumers with concurrent emitters determine
      their own emission order. No callbacks run while handling an exception. *)
  type operation =
    | Fit
    | Transform
    | Cross_validation
    | Fold
    | Search
    | Candidate
    | Refit

  type outcome = Succeeded | Failed of Error.t

  type status =
    | Started
    | Progress of { completed : int; total : int option }
    | Finished of outcome

  type event = {
    operation : operation;
    context : Error.context list;
    status : status;
  }

  type decision = Continue | Cancel
  type t

  val create :
    ?max_buffered_events:int ->
    (event -> (decision, string) result) ->
    (t, Error.t) result
  (** The positive event bound defaults to [10_000] per buffered fold. Direct
      pipeline calls deliver synchronously; CV buffers fold events and delivers
      them serially on the caller domain in logical fold order, one bounded
      batch at a time. Callbacks need not synchronize their own mutable state
      within one evaluation. Sharing a handler across independent concurrent
      evaluations requires caller synchronization. *)

  val progress :
    t -> completed:int -> ?total:int -> unit -> (unit, Error.t) result
  (** Consumers report progress through the callback delivered in metadata.
      Counts must satisfy [0 <= completed <= total] when a total is given. The
      library supplies the enclosing operation and nested context. Consumers
      must propagate errors from this call and must not retain the delivered
      callback after their operation returns. *)

  val is_control_error : Error.t -> bool
  (** Recognizes [Error.Cancelled] and [Error.Callback_failure]. Evaluators
      always abort on these errors, including under [Record] failure policy. *)

  val scope : Error.context -> t -> t
  val for_operation : operation -> t -> t
  val emit : t -> status -> (unit, Error.t) result

  val run :
    ?outcome:('a -> outcome) ->
    t option ->
    operation:operation ->
    (unit -> ('a, Error.t) result) ->
    ('a, Error.t) result

  val buffer : t -> t * (unit -> (unit, Error.t) result)
end
