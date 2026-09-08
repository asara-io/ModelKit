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

  module Persistent = struct
    let magic = "MDLKTC01"
    let major_version = 1
    let minor_version = 0
    let checksum_algorithm = 1
    let digest_bytes = 16
    let header_bytes = 8 + 1 + 1 + digest_bytes + 8 + 1 + digest_bytes
    let extension = ".mkcache"

    type limits = { max_payload_bytes : int }

    let limits ~max_payload_bytes =
      if max_payload_bytes < 1 then
        Error
          (validation ~name:"persistent cache payload limit"
             ~reason:"must be positive"
             ~remediation:"choose a positive persistent-cache reader limit")
      else if max_payload_bytes > max_int - header_bytes then
        Error
          (validation ~name:"persistent cache payload limit"
             ~reason:"leaves no space for cache-entry framing"
             ~remediation:"choose a smaller persistent-cache reader limit")
      else Ok { max_payload_bytes }

    let default_limits = { max_payload_bytes = 64 * 1024 * 1024 }

    type lookup = Miss | Hit of bytes | Corrupt of Error.t
    type publication = Published | Already_present
    type t = { root_path : string; limits : limits }

    let cache_error ~operation ~reason ~remediation =
      Error.make ~remediation (Error.Artifact { operation; reason })

    let failure ~operation reason =
      cache_error ~operation ~reason
        ~remediation:
          "discard the affected cache entry and recompute it from trusted \
           inputs"

    let io_error ~operation reason =
      Error
        (cache_error ~operation ~reason
           ~remediation:
             "check that the cache root exists and is readable and writable")

    let absolute_path path =
      if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
      else path

    let definitely_absent path =
      try not (Sys.file_exists path) with Sys_error _ -> false

    let create ?(limits = default_limits) ~root () =
      if String.trim root = "" then
        Error
          (validation ~name:"persistent cache root" ~reason:"must not be blank"
             ~remediation:"choose a dedicated cache directory")
      else
        let root_path = absolute_path root in
        try
          if Sys.file_exists root_path then
            if Sys.is_directory root_path then Ok { root_path; limits }
            else
              io_error ~operation:"open transform cache"
                "the configured cache root is not a directory"
          else
            try
              Sys.mkdir root_path 0o700;
              Ok { root_path; limits }
            with Sys_error reason ->
              if Sys.file_exists root_path && Sys.is_directory root_path then
                Ok { root_path; limits }
              else io_error ~operation:"create transform cache" reason
        with Sys_error reason ->
          io_error ~operation:"open transform cache" reason

    let root cache = cache.root_path

    let entry_path cache key =
      Filename.concat cache.root_path (Key.to_hex key ^ extension)

    let set_i64 bytes offset value =
      for index = 0 to 7 do
        let shift = (7 - index) * 8 in
        let byte =
          Int64.(to_int (logand (shift_right_logical value shift) 0xffL))
        in
        Bytes.set bytes (offset + index) (Char.chr byte)
      done

    let get_i64 bytes offset =
      let value = ref 0L in
      for index = 0 to 7 do
        value :=
          Int64.logor
            (Int64.shift_left !value 8)
            (Int64.of_int (Char.code (Bytes.get bytes (offset + index))))
      done;
      !value

    let make_header key payload =
      let header = Bytes.create header_bytes in
      Bytes.blit_string magic 0 header 0 8;
      Bytes.set header 8 (Char.chr major_version);
      Bytes.set header 9 (Char.chr minor_version);
      Bytes.blit_string key 0 header 10 digest_bytes;
      set_i64 header (10 + digest_bytes) (Int64.of_int (Bytes.length payload));
      Bytes.set header (18 + digest_bytes) (Char.chr checksum_algorithm);
      Bytes.blit_string (Digest.bytes payload) 0 header (19 + digest_bytes)
        digest_bytes;
      header

    let parse_header cache key ~file_length header =
      let corrupt reason =
        Error (failure ~operation:"read transform cache" reason)
      in
      if not (String.equal (Bytes.sub_string header 0 8) magic) then
        corrupt "cache-entry magic does not match ModelKit"
      else if
        Char.code (Bytes.get header 8) <> major_version
        || Char.code (Bytes.get header 9) <> minor_version
      then corrupt "unsupported cache-entry container version"
      else if not (String.equal (Bytes.sub_string header 10 digest_bytes) key)
      then corrupt "cache-entry key does not match its requested key"
      else
        let payload_length = get_i64 header (10 + digest_bytes) in
        if Int64.compare payload_length 0L < 0 then
          corrupt "cache-entry payload length is negative"
        else if
          Int64.compare payload_length
            (Int64.of_int cache.limits.max_payload_bytes)
          > 0
        then corrupt "cache entry exceeds the configured payload limit"
        else if
          Char.code (Bytes.get header (18 + digest_bytes)) <> checksum_algorithm
        then corrupt "unsupported cache-entry checksum algorithm"
        else
          let payload_length = Int64.to_int payload_length in
          if payload_length <> file_length - header_bytes then
            corrupt "cache-entry payload length does not match its framing"
          else
            Ok
              ( payload_length,
                Bytes.sub_string header (19 + digest_bytes) digest_bytes )

    let read_open_channel cache key channel =
      try
        let file_length = in_channel_length channel in
        if file_length < header_bytes then
          Ok
            (Corrupt
               (failure ~operation:"read transform cache"
                  "cache entry is truncated"))
        else if file_length > cache.limits.max_payload_bytes + header_bytes then
          Ok
            (Corrupt
               (failure ~operation:"read transform cache"
                  "cache entry exceeds the configured payload limit"))
        else
          let header = Bytes.create header_bytes in
          really_input channel header 0 header_bytes;
          match parse_header cache key ~file_length header with
          | Error error -> Ok (Corrupt error)
          | Ok (payload_length, expected_digest) ->
              let payload = Bytes.create payload_length in
              really_input channel payload 0 payload_length;
              if String.equal expected_digest (Digest.bytes payload) then
                Ok (Hit payload)
              else
                Ok
                  (Corrupt
                     (failure ~operation:"read transform cache"
                        "cache-entry checksum does not match"))
      with End_of_file ->
        Ok
          (Corrupt
             (failure ~operation:"read transform cache"
                "cache entry is truncated"))

    let rec get_with_retries remaining cache key =
      let path = entry_path cache key in
      try
        let channel = open_in_bin path in
        Fun.protect
          ~finally:(fun () -> close_in_noerr channel)
          (fun () -> read_open_channel cache key channel)
      with Sys_error reason ->
        if definitely_absent path then Ok Miss
        else if remaining > 0 then get_with_retries (remaining - 1) cache key
        else io_error ~operation:"read transform cache" reason

    let get cache key = get_with_retries 2 cache key

    let remove_file path =
      try
        Sys.remove path;
        Ok true
      with Sys_error reason ->
        if definitely_absent path then Ok false
        else io_error ~operation:"remove transform cache entry" reason

    let remove cache key = remove_file (entry_path cache key)

    let write_temporary cache key payload =
      try
        let path, channel =
          Filename.open_temp_file ~temp_dir:cache.root_path ~perms:0o600
            "modelkit-transform-cache-" ".tmp"
        in
        let result =
          try
            output_bytes channel (make_header key payload);
            output_bytes channel payload;
            close_out channel;
            Ok path
          with error ->
            close_out_noerr channel;
            ignore (remove_file path);
            raise error
        in
        result
      with Sys_error reason ->
        io_error ~operation:"write transform cache entry" reason

    let payload_conflict () =
      Error
        (cache_error ~operation:"publish transform cache entry"
           ~reason:"a different valid payload already exists for this key"
           ~remediation:
             "version the component codec or correct the nondeterministic \
              fitted-state encoding")

    let finish_publication cache key payload temporary_path =
      let destination = entry_path cache key in
      let clean_temporary () = ignore (remove_file temporary_path) in
      try
        Sys.rename temporary_path destination;
        Ok Published
      with Sys_error rename_reason -> (
        match get cache key with
        | Ok (Hit existing) when Bytes.equal existing payload ->
            clean_temporary ();
            Ok Already_present
        | Ok (Hit _) ->
            clean_temporary ();
            payload_conflict ()
        | Ok (Corrupt _) -> (
            match remove_file destination with
            | Error error ->
                clean_temporary ();
                Error error
            | Ok _ -> (
                try
                  Sys.rename temporary_path destination;
                  Ok Published
                with Sys_error reason ->
                  clean_temporary ();
                  io_error ~operation:"publish transform cache entry" reason))
        | Ok Miss ->
            clean_temporary ();
            io_error ~operation:"publish transform cache entry" rename_reason
        | Error error ->
            clean_temporary ();
            Error error)

    let put cache key payload =
      if Bytes.length payload > cache.limits.max_payload_bytes then
        Error
          (validation ~name:"persistent cache payload"
             ~reason:
               (Format.sprintf "%d bytes exceeds the %d-byte cache limit"
                  (Bytes.length payload) cache.limits.max_payload_bytes)
             ~remediation:
               "increase the explicit cache limit or leave this result uncached")
      else
        match get cache key with
        | Ok (Hit existing) when Bytes.equal existing payload ->
            Ok Already_present
        | Ok (Hit _) -> payload_conflict ()
        | Error error -> Error error
        | Ok Miss | Ok (Corrupt _) ->
            let payload = Bytes.copy payload in
            Result.bind (write_temporary cache key payload)
              (fun temporary_path ->
                finish_publication cache key payload temporary_path)
  end
end
