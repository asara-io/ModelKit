open Ctypes
open Modelkit
module F = Xgboost_bindings.C.Functions
module Next = (val Foreign.dynamic_funptr (ptr void @-> returning int))
module Reset = (val Foreign.dynamic_funptr (ptr void @-> returning void))

let proxy_create =
  Foreign.foreign "XGProxyDMatrixCreate" (ptr (ptr void) @-> returning int)

let proxy_dense =
  Foreign.foreign "XGProxyDMatrixSetDataDense"
    (ptr void @-> string @-> returning int)

let set_info =
  Foreign.foreign "XGDMatrixSetInfoFromInterface"
    (ptr void @-> string @-> string @-> returning int)

let create_iterator =
  Foreign.foreign "XGDMatrixCreateFromCallback"
    (ptr void @-> ptr void @-> Reset.t @-> Next.t @-> string
    @-> ptr (ptr void)
    @-> returning int)

type vector = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

type batch = {
  features : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t;
  labels : vector;
  weights : vector option;
}

exception Failed of Error.t

let native_error operation =
  Error.make ~context:[ Error.Stage operation ]
    ~remediation:"Check the native XGBoost runtime and disk-cache permissions."
    (Error.Numerical { operation; reason = F.xgb_get_last_error () })

let check operation rc = if rc <> 0 then raise (Failed (native_error operation))

let interface pointer shape =
  let address = Nativeint.to_string (raw_address_of_ptr (to_voidp pointer)) in
  let dimensions = String.concat "," (List.map string_of_int shape) in
  Printf.sprintf
    {|{"data":[%s,true],"shape":[%s],"strides":null,"typestr":"%sf4","version":3}|}
    address dimensions
    (if Sys.big_endian then ">" else "<")

let with_handle ~create ~free callback =
  let out = allocate (ptr void) null in
  Fun.protect
    ~finally:(fun () -> if not (is_null !@out) then free !@out)
    (fun () ->
      create out;
      callback !@out)

let train ~parameters ~rounds ~threads ~rows ~columns ~cache_prefix ~next ~reset
    () =
  let failure = ref None in
  let current = ref None in
  let remember error = if Option.is_none !failure then failure := Some error in
  let ensure () =
    match !failure with None -> () | Some error -> raise (Failed error)
  in
  let callback operation f fallback =
    if Option.is_some !failure then fallback
    else
      try f () with
      | Failed error ->
          remember error;
          fallback
      | exn ->
          remember
            (Error.make ~context:[ Error.Stage operation ]
               ~remediation:
                 "Ensure batch callbacks return a replayable dataset."
               (Error.Callback_failure { reason = Printexc.to_string exn }));
          fallback
  in
  let result value =
    match value with Ok value -> value | Error error -> raise (Failed error)
  in
  let cleanup_failure = ref None in
  let free operation release handle =
    let rc = release handle in
    if rc <> 0 && Option.is_none !cleanup_failure then
      cleanup_failure := Some (native_error operation)
  in
  let outcome =
    try
      Ok
        (with_handle
           ~create:(fun out ->
             check "streaming proxy creation" (proxy_create out))
           ~free:(free "streaming proxy cleanup" F.xgdmatrix_free)
           (fun proxy ->
             let next_callback _ =
               callback "streaming next"
                 (fun () ->
                   current := None;
                   match result (next ()) with
                   | None -> 0
                   | Some batch ->
                       (* Pin before any native setter can borrow the buffers. *)
                       current := Some batch;
                       check "streaming proxy features"
                         (proxy_dense proxy
                            (interface
                               (bigarray_start array2 batch.features)
                               [ Bigarray.Array2.dim1 batch.features; columns ]));
                       let info name values =
                         check
                           ("streaming proxy " ^ name)
                           (set_info proxy name
                              (interface
                                 (bigarray_start array1 values)
                                 [ Bigarray.Array1.dim values ]))
                       in
                       info "label" batch.labels;
                       Option.iter (info "weight") batch.weights;
                       1)
                 0
             in
             let reset_callback _ =
               callback "streaming reset"
                 (fun () ->
                   current := None;
                   result (reset ()))
                 ()
             in
             (* Callback trampolines outlive every native training handle. *)
             Next.with_fun next_callback (fun next_pointer ->
                 Reset.with_fun reset_callback (fun reset_pointer ->
                     let config =
                       (* XGBoost accepts NaN; standard JSON writers reject it. *)
                       Printf.sprintf
                         {|{"missing":NaN,"cache_prefix":%s,"nthread":%d}|}
                         (Yojson.Safe.to_string (`String cache_prefix))
                         threads
                     in
                     with_handle
                       ~create:(fun out ->
                         let rc =
                           create_iterator null proxy reset_pointer next_pointer
                             config out
                         in
                         ensure ();
                         check "streaming DMatrix creation" rc)
                       ~free:(free "streaming DMatrix cleanup" F.xgdmatrix_free)
                       (fun matrix ->
                         let dimension name getter expected =
                           let out = allocate uint64_t Unsigned.UInt64.zero in
                           check name (getter matrix out);
                           if !@out <> Unsigned.UInt64.of_int expected then
                             raise
                               (Failed
                                  (Error.make
                                     ~remediation:
                                       "Replay exactly the declared dataset."
                                     (Error.Validation
                                        {
                                          name;
                                          reason =
                                            "native dimension differs from \
                                             declared input";
                                        })))
                         in
                         dimension "streaming rows" F.xgdmatrix_num_row rows;
                         dimension "streaming columns" F.xgdmatrix_num_col
                           columns;
                         let cache = CArray.of_list (ptr void) [ matrix ] in
                         with_handle
                           ~create:(fun out ->
                             check "streaming booster creation"
                               (F.xgbooster_create (CArray.start cache)
                                  (Unsigned.UInt64.of_int 1) out))
                           ~free:
                             (free "streaming booster cleanup" F.xgbooster_free)
                           (fun booster ->
                             List.iter
                               (fun (name, value) ->
                                 check "streaming parameters"
                                   (F.xgbooster_set_param booster name value))
                               parameters;
                             for iter = 0 to rounds - 1 do
                               let rc =
                                 F.xgbooster_update_one_iter booster iter matrix
                               in
                               ensure ();
                               check "streaming training" rc
                             done;
                             let length =
                               allocate uint64_t Unsigned.UInt64.zero
                             in
                             let buffer =
                               allocate (ptr char) (from_voidp char null)
                             in
                             let rc =
                               F.xgbooster_save_model_to_buffer booster
                                 {|{"format":"json"}|} length buffer
                             in
                             ensure ();
                             check "streaming model snapshot" rc;
                             let length = Unsigned.UInt64.to_int !@length in
                             let bytes =
                               Bytes.of_string
                                 (string_from_ptr !@buffer ~length)
                             in
                             ignore (Sys.opaque_identity !current);
                             bytes))))))
    with Failed error -> Error error
  in
  current := None;
  match (outcome, !cleanup_failure) with
  | Error error, Some cleanup_error ->
      Error
        (Error.with_context
           (Error.Stage
              ("streaming native cleanup: " ^ Error.to_string cleanup_error))
           error)
  | Error error, None -> Error error
  | Ok _, Some error -> Error error
  | Ok value, None -> Ok value
