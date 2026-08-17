open Modelkit_data
open Modelkit_protocols

module Split : sig
  type t

  val create :
    source_size:int -> train:int array -> test:int array -> (t, Error.t) result

  val of_views : train:Row_view.t -> test:Row_view.t -> (t, Error.t) result
  val train : t -> Row_view.t
  val test : t -> Row_view.t

  val materialize :
    'kind Dataset.t -> t -> ('kind Dataset.t * 'kind Dataset.t, Error.t) result
end

module K_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

module Stratified_k_fold : sig
  type params = { folds : int; shuffle : bool }
  type t

  val create : ?folds:int -> ?shuffle:bool -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = Target.classification Target.t
       and type rng = Rng.t
end

module Group_k_fold : sig
  type params = { folds : int }
  type t

  val create : ?folds:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end

module Time_series_split : sig
  type params = { folds : int; test_size : int option; gap : int }
  type t

  val create :
    ?folds:int -> ?test_size:int -> ?gap:int -> unit -> (t, Error.t) result

  include
    SPLITTER
      with type t := t
       and type params := params
       and type target = unit
       and type rng = Rng.t
end
