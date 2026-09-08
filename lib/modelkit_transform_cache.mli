open Modelkit_data
open Modelkit_protocols

(** Content-addressed transform-cache foundations.

    Stores are explicit values with caller-owned lifetimes; ModelKit does not
    install a process-global cache. Cache payloads are data only and are copied
    at the storage boundary. *)
module Transform_cache : sig
  module Component : sig
    type t

    val create :
      package:string -> name:string -> version:int -> (t, Error.t) result
    (** Names must be nonblank and [version] must be positive. Package-qualified
        identities prevent unrelated extensions from sharing entries. *)

    val package : t -> string
    val name : t -> string
    val version : t -> int
    val equal : t -> t -> bool
    val to_string : t -> string
  end

  module Content_id : sig
    type t

    val of_bytes : bytes -> t
    val of_string : string -> t

    val combine : domain:string -> t array -> (t, Error.t) result
    (** Combines an ordered array under a nonblank domain. Domain and length
        framing make structurally different material distinct. Content IDs are
        deterministic cache identities, not cryptographic authentication. *)

    val equal : t -> t -> bool
    val to_hex : t -> string
  end

  module Key : sig
    type t

    val create :
      component:Component.t ->
      configuration:Content_id.t ->
      training_data:Content_id.t ->
      target:Content_id.t option ->
      routed_metadata:Content_id.t ->
      seed:Seed.t ->
      t
    (** Canonically frames every field. [None] is an explicit no-target marker,
        not the digest of an empty target. *)

    val equal : t -> t -> bool
    val to_hex : t -> string
  end

  (** Stable fitted-state codecs required for cacheable transformers.

      This cache format is independent of the public model-artifact schema.
      Decoders must validate their payload and return typed errors. *)
  module type CACHEABLE_TRANSFORMER = sig
    include TRANSFORMER

    val cache_component : Component.t
    val cache_configuration : t -> Content_id.t
    val encode_fitted : fitted -> (bytes, Error.t) result
    val decode_fitted : bytes -> (fitted, Error.t) result
  end

  module Codec : sig
    type ('specification, 'fitted) t

    type ('specification, 'fitted) support =
      | Unsupported
      | Supported of ('specification, 'fitted) t

    val of_module :
      (module CACHEABLE_TRANSFORMER
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'fitted) t

    val component : ('specification, 'fitted) t -> Component.t

    val configuration :
      ('specification, 'fitted) t -> 'specification -> Content_id.t

    val encode :
      ('specification, 'fitted) t -> 'fitted -> (bytes, Error.t) result

    val decode :
      ('specification, 'fitted) t -> bytes -> ('fitted, Error.t) result

    val require :
      component:string ->
      ('specification, 'fitted) support ->
      (('specification, 'fitted) t, Error.t) result
    (** Returns a typed compatibility error when caching is requested for a
        transformer without a codec. *)
  end

  module Memory : sig
    type limits

    val limits : max_entries:int -> max_bytes:int64 -> (limits, Error.t) result
    (** Both limits must be positive. [max_bytes] counts retained payload bytes,
        excluding keys and OCaml allocation headers. *)

    val default_limits : limits

    type stats = {
      entries : int;
      payload_bytes : int64;
      hits : int64;
      misses : int64;
      evictions : int64;
    }

    type t

    val create : ?limits:limits -> unit -> t

    val get : t -> Key.t -> bytes option
    (** Returns a copy and updates hit/miss counters. *)

    val put : t -> Key.t -> bytes -> (unit, Error.t) result
    (** Copies the payload. Entries larger than the byte limit fail without
        changing the store. Least-recently-written entries are evicted until
        both bounds hold. *)

    val remove : t -> Key.t -> bool
    val clear : t -> unit
    val stats : t -> stats
  end

  (** Portable directory-backed storage for immutable cache entries.

      Entries and temporary publication files contain plaintext fitted state.
      ModelKit requests restrictive permissions for newly created paths but does
      not provide encryption, authenticate content, or override the host
      filesystem's permission semantics. Protect the root, backups, and
      retention policy before caching state derived from secret training data.
  *)
  module Persistent : sig
    type limits

    val limits : max_payload_bytes:int -> (limits, Error.t) result
    (** The limit must be positive and bounds allocation before a payload is
        read. Framing bytes are checked separately. *)

    val default_limits : limits

    (** Corrupt entries are never returned as hits. A subsequent [put] replaces
        a corrupt entry, allowing callers to treat [Corrupt] as an observable
        miss and refit safely. *)
    type lookup = Miss | Hit of bytes | Corrupt of Error.t

    type publication = Published | Already_present
    type t

    val create : ?limits:limits -> root:string -> unit -> (t, Error.t) result
    (** Creates [root] with restrictive requested permissions when absent. Its
        parent must already exist. Existing roots must be directories. *)

    val root : t -> string
    val get : t -> Key.t -> (lookup, Error.t) result

    val put : t -> Key.t -> bytes -> (publication, Error.t) result
    (** Publishes through a temporary file in [root] followed by an atomic
        rename. Concurrent writers for one key must encode identical payloads;
        an already-present different valid payload is a typed failure. *)

    val remove : t -> Key.t -> (bool, Error.t) result
  end
end
