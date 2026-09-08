open Modelkit_data
open Modelkit_protocols

module Transform_cache = struct
  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  module Digest_stream = struct
    type t = { mutable state : string; buffer : Buffer.t }

    let create domain =
      {
        state = Digest.string ("modelkit:" ^ domain);
        buffer = Buffer.create 4096;
      }

    let flush digest =
      if Buffer.length digest.buffer > 0 then (
        digest.state <-
          Digest.string (digest.state ^ Buffer.contents digest.buffer);
        Buffer.clear digest.buffer)

    let emit digest value =
      String.iter
        (fun byte ->
          Buffer.add_char digest.buffer byte;
          if Buffer.length digest.buffer = 4096 then flush digest)
        value

    let token digest value =
      emit digest (string_of_int (String.length value));
      emit digest ":";
      emit digest value

    let finish digest =
      flush digest;
      digest.state
  end

  module Component = struct
    type t = { package_name : string; component_name : string; version : int }

    let create ~package ~name ~version =
      if String.trim package = "" then
        Error
          (validation ~name:"cache component package"
             ~reason:"must not be blank"
             ~remediation:"use a stable package-qualified component identity")
      else if String.trim name = "" then
        Error
          (validation ~name:"cache component name" ~reason:"must not be blank"
             ~remediation:"use a stable component name")
      else if version < 1 then
        Error
          (validation ~name:"cache component version" ~reason:"must be positive"
             ~remediation:"start cache component versions at one")
      else Ok { package_name = package; component_name = name; version }

    let package component = component.package_name
    let name component = component.component_name
    let version component = component.version
    let equal = ( = )

    let to_string component =
      Format.sprintf "%s:%s@%d" component.package_name component.component_name
        component.version
  end

  module Content_id = struct
    type t = string

    let of_bytes value = Digest.bytes value
    let of_string value = Digest.string value

    let combine ~domain values =
      if String.trim domain = "" then
        Error
          (validation ~name:"cache content domain" ~reason:"must not be blank"
             ~remediation:"use a stable domain for this kind of key material")
      else
        let digest = Digest_stream.create "transform-content-v1" in
        Digest_stream.token digest domain;
        Digest_stream.token digest (string_of_int (Array.length values));
        Array.iter (Digest_stream.token digest) values;
        Ok (Digest_stream.finish digest)

    let equal = String.equal
    let to_hex = Digest.to_hex
  end

  module Key = struct
    type t = string

    let create ~component ~configuration ~training_data ~target ~routed_metadata
        ~seed =
      let digest = Digest_stream.create "transform-key-v1" in
      Digest_stream.token digest (Component.package component);
      Digest_stream.token digest (Component.name component);
      Digest_stream.token digest (string_of_int (Component.version component));
      Digest_stream.token digest configuration;
      Digest_stream.token digest training_data;
      (match target with
      | None -> Digest_stream.token digest "no-target"
      | Some target ->
          Digest_stream.token digest "target";
          Digest_stream.token digest target);
      Digest_stream.token digest routed_metadata;
      Digest_stream.token digest (Int64.to_string (Seed.to_int64 seed));
      Digest_stream.finish digest

    let equal = String.equal
    let to_hex = Digest.to_hex
  end

  module type CACHEABLE_TRANSFORMER = sig
    include TRANSFORMER

    val cache_component : Component.t
    val cache_configuration : t -> Content_id.t
    val encode_fitted : fitted -> (bytes, Error.t) result
    val decode_fitted : bytes -> (fitted, Error.t) result
  end

  module Codec = struct
    type ('specification, 'fitted) t = {
      codec_component : Component.t;
      codec_configuration : 'specification -> Content_id.t;
      codec_encode : 'fitted -> (bytes, Error.t) result;
      codec_decode : bytes -> ('fitted, Error.t) result;
    }

    type ('specification, 'fitted) support =
      | Unsupported
      | Supported of ('specification, 'fitted) t

    let of_module (type specification params target fitted rng)
        (module Transformer : CACHEABLE_TRANSFORMER
          with type t = specification
           and type params = params
           and type target = target
           and type fitted = fitted
           and type rng = rng) =
      {
        codec_component = Transformer.cache_component;
        codec_configuration = Transformer.cache_configuration;
        codec_encode = Transformer.encode_fitted;
        codec_decode = Transformer.decode_fitted;
      }

    let component codec = codec.codec_component
    let configuration codec = codec.codec_configuration
    let encode codec = codec.codec_encode
    let decode codec = codec.codec_decode

    let require ~component = function
      | Supported codec -> Ok codec
      | Unsupported ->
          Error
            (Error.make
               ~remediation:
                 "disable caching for this stage or provide a reviewed \
                  fitted-state cache codec"
               (Error.Compatibility
                  {
                    component;
                    reason =
                      "transform caching was requested but no cache codec is \
                       available";
                  }))
  end

  module Memory = struct
    type limits = { max_entries : int; max_bytes : int64 }

    let limits ~max_entries ~max_bytes =
      if max_entries < 1 then
        Error
          (validation ~name:"memory cache entry limit"
             ~reason:"must be positive"
             ~remediation:"configure space for at least one cache entry")
      else if Int64.compare max_bytes 1L < 0 then
        Error
          (validation ~name:"memory cache byte limit" ~reason:"must be positive"
             ~remediation:"configure space for at least one payload byte")
      else Ok { max_entries; max_bytes }

    let default_limits =
      { max_entries = 128; max_bytes = Int64.mul 256L 1024L |> Int64.mul 1024L }

    type stats = {
      entries : int;
      payload_bytes : int64;
      hits : int64;
      misses : int64;
      evictions : int64;
    }

    type entry = { key : Key.t; payload : bytes; size : int64 }

    type state = {
      values : entry list;
      retained_bytes : int64;
      hit_count : int64;
      miss_count : int64;
      eviction_count : int64;
    }

    type t = { limits : limits; state : state Atomic.t }

    let create ?(limits = default_limits) () =
      {
        limits;
        state =
          Atomic.make
            {
              values = [];
              retained_bytes = 0L;
              hit_count = 0L;
              miss_count = 0L;
              eviction_count = 0L;
            };
      }

    let rec get cache key =
      let before = Atomic.get cache.state in
      let found =
        List.find_opt (fun entry -> Key.equal entry.key key) before.values
      in
      let after =
        match found with
        | None -> { before with miss_count = Int64.succ before.miss_count }
        | Some _ -> { before with hit_count = Int64.succ before.hit_count }
      in
      if Atomic.compare_and_set cache.state before after then
        Option.map (fun entry -> Bytes.copy entry.payload) found
      else get cache key

    let rec drop_oldest values =
      match values with
      | [] -> ([], None)
      | [ value ] -> ([], Some value)
      | value :: remaining ->
          let remaining, dropped = drop_oldest remaining in
          (value :: remaining, dropped)

    let trim limits values payload_bytes evictions =
      let rec loop values payload_bytes count evictions =
        if
          count <= limits.max_entries
          && Int64.compare payload_bytes limits.max_bytes <= 0
        then (values, payload_bytes, evictions)
        else
          let values, dropped = drop_oldest values in
          match dropped with
          | None -> ([], 0L, evictions)
          | Some dropped ->
              loop values
                (Int64.sub payload_bytes dropped.size)
                (count - 1) (Int64.succ evictions)
      in
      loop values payload_bytes (List.length values) evictions

    let put cache key payload =
      let size = Int64.of_int (Bytes.length payload) in
      if Int64.compare size cache.limits.max_bytes > 0 then
        Error
          (validation ~name:"memory cache payload"
             ~reason:
               (Format.sprintf "%Ld bytes exceeds the %Ld-byte cache limit" size
                  cache.limits.max_bytes)
             ~remediation:
               "increase the explicit cache limit or leave this result uncached")
      else
        let payload = Bytes.copy payload in
        let rec update () =
          let before = Atomic.get cache.state in
          let replaced_size = ref 0L in
          let retained =
            List.filter
              (fun entry ->
                if Key.equal entry.key key then (
                  replaced_size := entry.size;
                  false)
                else true)
              before.values
          in
          let payload_bytes =
            Int64.add (Int64.sub before.retained_bytes !replaced_size) size
          in
          let values, payload_bytes, evictions =
            trim cache.limits
              ({ key; payload; size } :: retained)
              payload_bytes before.eviction_count
          in
          let after =
            {
              before with
              values;
              retained_bytes = payload_bytes;
              eviction_count = evictions;
            }
          in
          if Atomic.compare_and_set cache.state before after then Ok ()
          else update ()
        in
        update ()

    let rec remove cache key =
      let before = Atomic.get cache.state in
      let removed = ref false in
      let removed_size = ref 0L in
      let values =
        List.filter
          (fun entry ->
            if Key.equal entry.key key then (
              removed := true;
              removed_size := entry.size;
              false)
            else true)
          before.values
      in
      if not !removed then false
      else
        let after =
          {
            before with
            values;
            retained_bytes = Int64.sub before.retained_bytes !removed_size;
          }
        in
        if Atomic.compare_and_set cache.state before after then true
        else remove cache key

    let clear cache =
      let rec update () =
        let before = Atomic.get cache.state in
        let after = { before with values = []; retained_bytes = 0L } in
        if not (Atomic.compare_and_set cache.state before after) then update ()
      in
      update ()

    let stats cache =
      let state = Atomic.get cache.state in
      {
        entries = List.length state.values;
        payload_bytes = state.retained_bytes;
        hits = state.hit_count;
        misses = state.miss_count;
        evictions = state.eviction_count;
      }
  end
end
