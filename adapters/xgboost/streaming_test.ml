open Modelkit
module X = Modelkit_xgboost
module R = X.Regressor

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let schema names =
  Feature_names.create ~expected_count:(Array.length names) names
  |> data |> Feature_schema.named

let features = schema [| "signal"; "missing"; "category" |]
let rng () = Rng.create (Seed.of_int 193)

let specification () =
  X.Specification.create ~rounds:4 ~max_depth:2 ~max_bin:256 () |> get

let batch ?(weighted = false) ~start ~rows () =
  let x =
    Matrix.init ~rows ~columns:3 (fun row column ->
        let row = start + row in
        if column = 0 then float_of_int (row mod 17) /. 7.
        else if column = 1 && row mod 4 = 0 then Float.nan
        else float_of_int ((row + column) mod 5))
    |> data
  in
  let y =
    Vector.init ~length:rows (fun row ->
        let row = start + row in
        float_of_int ((row mod 17 * 2) + (row mod 5)) /. 3.)
    |> data |> Target.regression |> data
  in
  let sample_weight =
    if weighted then
      Some
        (Vector.init ~length:rows (fun row ->
             let row = start + row in
             if row < 8 || row mod 7 = 0 then 0.
             else 0.3 +. float_of_int (row mod 3))
        |> data)
    else None
  in
  { R.feature_schema = features; x; y; sample_weight }

let with_directory f =
  let path = Filename.temp_file "modelkit-streaming-test-" "" in
  Sys.remove path;
  Unix.mkdir path 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun name -> Sys.remove (Filename.concat path name))
        (Sys.readdir path);
      Unix.rmdir path)
    (fun () -> f path)

let callbacks ?(weighted = false) ~rows ~size () =
  let cursor = ref 0 in
  let next () =
    if !cursor = rows then Ok None
    else
      let length = min size (rows - !cursor) in
      let value = batch ~weighted ~start:!cursor ~rows:length () in
      cursor := !cursor + length;
      Ok (Some value)
  in
  let reset () =
    cursor := 0;
    Ok ()
  in
  (next, reset)

let fit ?(weighted = false) ~rows ~size directory =
  let next, reset = callbacks ~weighted ~rows ~size () in
  R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
    ~expected_rows:rows ~cache_directory:directory ~next ~reset ()
  |> get

let predict fitted x =
  R.predict fitted ~feature_schema:features ~x
  |> get |> Target.regression_values |> Vector.to_array

let empty directory =
  Alcotest.check
    (Alcotest.array Alcotest.string)
    "owned caches removed" [||] (Sys.readdir directory)

let parity weighted () =
  with_directory (fun directory ->
      let streamed = fit ~weighted ~rows:24 ~size:8 directory in
      empty directory;
      let full = batch ~weighted ~start:0 ~rows:24 () in
      let sample_weight =
        Option.map
          (fun weights ->
            Sample_weight.create ~expected_length:24 weights |> data)
          full.R.sample_weight
      in
      let ordinary =
        R.fit (specification ()) ?sample_weight ~rng:(rng ())
          ~feature_schema:features ~x:full.R.x ~y:full.R.y ()
        |> get
      in
      let expected = predict ordinary full.R.x in
      let observed = predict streamed full.R.x in
      Array.iteri
        (fun row value ->
          Alcotest.check
            (Alcotest.float (2e-6 +. (2e-6 *. Float.abs value)))
            "in-memory parity" value observed.(row))
        expected;
      let report = Option.get (R.batches_report streamed) in
      Alcotest.check Alcotest.int "batches" 3 report.R.batches;
      Alcotest.check Alcotest.int "rows" 24 report.R.rows;
      Alcotest.check Alcotest.int "batch rows" 8 report.R.max_batch_rows;
      Alcotest.check Alcotest.bool "native resets observed" true
        (report.R.resets >= 2);
      Alcotest.check Alcotest.bool "native passes observed" true
        (report.R.passes >= 2);
      Alcotest.check Alcotest.int64 "peak feature staging" 96L
        report.R.peak_feature_buffer_bytes;
      let original = R.report ordinary
      and streamed_report = R.report streamed in
      Alcotest.check Alcotest.int "feature rounding counted once"
        (X.Input.rounded_values original.R.feature_conversion)
        (X.Input.rounded_values streamed_report.R.feature_conversion);
      Alcotest.check Alcotest.int "target rounding counted once"
        (X.Input.rounded_values original.R.target_conversion)
        (X.Input.rounded_values streamed_report.R.target_conversion);
      let encoded = X.Artifact.encode_regression streamed |> get in
      let loaded = X.Artifact.decode_regression encoded |> get in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "restored predictions" observed (predict loaded full.R.x);
      Alcotest.check Alcotest.bool "restored batch diagnostics" true
        (R.batches_report streamed = R.batches_report loaded);
      Alcotest.check Alcotest.bool "ordinary fits have no batch report" true
        (Option.is_none (R.batches_report ordinary));
      let ordinary_loaded =
        ordinary |> X.Artifact.encode_regression |> get
        |> X.Artifact.decode_regression |> get
      in
      Alcotest.check Alcotest.bool "version-one payload still loads" true
        (Option.is_none (R.batches_report ordinary_loaded)))

let error_kind check result =
  match result with
  | Ok _ -> Alcotest.fail "expected typed error"
  | Error error ->
      Alcotest.check Alcotest.bool (Error.to_string error) true
        (check (Error.kind error))

let validation = function
  | Error.Data _ | Error.Validation _ | Error.Shape_mismatch _
  | Error.Feature_schema_mismatch _ ->
      true
  | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
  | Error.Artifact _ | Error.Callback_failure _ | Error.Cancelled ->
      false

let rejected name transform () =
  with_directory (fun directory ->
      let delivered = ref false in
      let next () =
        if !delivered then Ok None
        else (
          delivered := true;
          Ok (Some (transform (batch ~start:0 ~rows:8 ()))))
      in
      let reset () =
        delivered := false;
        Ok ()
      in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:8 ~cache_directory:directory ~next ~reset ()
      |> error_kind validation;
      ignore name;
      empty directory)

let callback_failure reset_failure native_failure () =
  with_directory (fun directory ->
      let next, reset = callbacks ~rows:24 ~size:8 () in
      let resets = ref 0 and cursor = ref 0 in
      let active () = (not native_failure) || !resets > 1 in
      let reset () =
        incr resets;
        cursor := 0;
        if reset_failure && active () then failwith "source reset failed"
        else reset ()
      in
      let next () =
        incr cursor;
        if (not reset_failure) && active () && !cursor = 2 then
          failwith "source next failed"
        else next ()
      in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      |> error_kind (function
        | Error.Callback_failure _ -> true
        | Error.Data _ | Error.Shape_mismatch _
        | Error.Feature_schema_mismatch _ | Error.Validation _
        | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
        | Error.Artifact _ | Error.Cancelled ->
            false);
      empty directory)

let replay_failure ?(weighted = false) transform () =
  with_directory (fun directory ->
      let next, reset = callbacks ~weighted ~rows:24 ~size:8 () in
      let resets = ref 0 in
      let reset () =
        incr resets;
        reset ()
      in
      let next () =
        let value = next () in
        if !resets > 1 then Result.map (Option.map transform) value else value
      in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      |> error_kind validation;
      empty directory)

let totals supplied expected () =
  with_directory (fun directory ->
      let next, reset = callbacks ~rows:supplied ~size:8 () in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:expected ~cache_directory:directory ~next ~reset ()
      |> error_kind validation;
      empty directory)

let disk_operation () =
  with_directory (fun directory ->
      let quoted = Filename.concat directory "quoted\" cache" in
      Unix.mkdir quoted 0o700;
      Fun.protect
        ~finally:(fun () -> Unix.rmdir quoted)
        (fun () ->
          let marker = Filename.concat quoted "caller-owned" in
          let output = open_out marker in
          close_out output;
          let next, reset = callbacks ~rows:24 ~size:8 () in
          let saw_pages = ref false in
          let observe () =
            Array.iter
              (fun name ->
                let child = Filename.concat quoted name in
                if Sys.is_directory child then
                  Array.iter
                    (fun name ->
                      if String.starts_with ~prefix:"pages" name then
                        saw_pages := true)
                    (Sys.readdir child))
              (Sys.readdir quoted)
          in
          let next () =
            observe ();
            next ()
          in
          let reset () =
            observe ();
            reset ()
          in
          ignore
            (R.fit_batches (specification ()) ~rng:(rng ())
               ~feature_schema:features ~expected_rows:24
               ~cache_directory:quoted ~next ~reset ()
            |> get);
          Alcotest.check Alcotest.bool "native disk pages observed" true
            !saw_pages;
          Alcotest.check
            (Alcotest.array Alcotest.string)
            "caller file retained" [| "caller-owned" |] (Sys.readdir quoted);
          Sys.remove marker))

let bounded_buffers () =
  with_directory (fun directory ->
      let small = fit ~rows:24 ~size:8 directory in
      let next, reset = callbacks ~rows:240 ~size:8 () in
      let weak = Weak.create 1 in
      let next () =
        Gc.full_major ();
        Alcotest.check Alcotest.bool "previous source matrix released" false
          (Weak.check weak 0);
        let result = next () in
        (match result with
        | Ok (Some batch) -> Weak.set weak 0 (Some batch.R.x)
        | Ok None | Error _ -> ());
        result
      in
      let large =
        R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
          ~expected_rows:240 ~cache_directory:directory ~next ~reset ()
        |> get
      in
      let small = Option.get (R.batches_report small)
      and large = Option.get (R.batches_report large) in
      Alcotest.check Alcotest.int64 "feature peak independent of total rows"
        small.R.peak_feature_buffer_bytes large.R.peak_feature_buffer_bytes;
      Alcotest.check Alcotest.int64 "all staging independent of total rows"
        small.R.peak_staging_buffer_bytes large.R.peak_staging_buffer_bytes;
      Alcotest.check Alcotest.int "larger dataset processed" 240 large.R.rows;
      empty directory)

let typed_callback_error () =
  with_directory (fun directory ->
      let next, reset = callbacks ~rows:24 ~size:8 () in
      let resets = ref 0 in
      let reset () =
        incr resets;
        reset ()
      in
      let next () =
        if !resets > 1 then
          Error (Error.make ~remediation:"Retry the source." Error.Cancelled)
        else next ()
      in
      let result =
        R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
          ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      in
      (match result with
      | Ok _ -> Alcotest.fail "expected callback cancellation"
      | Error error ->
          Alcotest.check Alcotest.bool "callback context retained" true
            (List.mem (Error.Stage "batch next") (Error.context error));
          error_kind
            (function
              | Error.Cancelled -> true
              | Error.Data _ | Error.Shape_mismatch _
              | Error.Feature_schema_mismatch _ | Error.Validation _
              | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
              | Error.Artifact _ | Error.Callback_failure _ ->
                  false)
            result);
      empty directory)

let native_short_input () =
  with_directory (fun directory ->
      let next, reset = callbacks ~rows:24 ~size:8 () in
      let resets = ref 0 and calls = ref 0 in
      let reset () =
        incr resets;
        calls := 0;
        reset ()
      in
      let next () =
        incr calls;
        if !resets > 1 && !calls = 2 then Ok None else next ()
      in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      |> error_kind validation;
      empty directory)

let diagnostics_offset bytes =
  let position = ref 36 in
  let integer () =
    let value = Int64.to_int (Bytes.get_int64_be bytes !position) in
    position := !position + 8;
    value
  in
  let text () =
    let length = integer () in
    position := !position + length
  in
  text ();
  text ();
  Alcotest.check Alcotest.int "streaming payload version" 2 (integer ());
  text ();
  text ();
  text ();
  position := !position + 48 + 8 + 3;
  text ();
  position := !position + 16 + (11 * 8);
  let columns = integer () in
  let named = Bytes.get bytes !position in
  incr position;
  if named = '\001' then
    for _ = 1 to columns do
      text ()
    done;
  text ();
  let classes = integer () in
  position := !position + (classes * 8) + 24;
  let conversion () =
    let dimensions = integer () in
    position := !position + (dimensions * 8) + 24
  in
  conversion ();
  conversion ();
  let weighted = Bytes.get bytes !position in
  incr position;
  if weighted = '\001' then conversion ();
  !position

let invalid_diagnostics () =
  with_directory (fun directory ->
      let fitted = fit ~rows:24 ~size:8 directory in
      let original = X.Artifact.encode_regression fitted |> get in
      let start = diagnostics_offset original in
      List.iteri
        (fun field value ->
          let bytes = Bytes.copy original in
          Bytes.set_int64_be bytes (start + (field * 8)) value;
          Bytes.blit_string
            (Digest.string
               (Bytes.sub_string bytes 36 (Bytes.length bytes - 36)))
            0 bytes 20 16;
          X.Artifact.decode_regression bytes
          |> error_kind (function
            | Error.Artifact _ -> true
            | Error.Data _ | Error.Shape_mismatch _
            | Error.Feature_schema_mismatch _ | Error.Validation _
            | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
            | Error.Callback_failure _ | Error.Cancelled ->
                false))
        [ 0L; 25L; Int64.max_int; 1L; 1L; 97L; -1L; Int64.max_int ];
      empty directory)

let fresh_process () =
  with_directory (fun directory ->
      let fitted = fit ~weighted:true ~rows:24 ~size:8 directory in
      empty directory;
      let path = Filename.concat directory "model.mkxgb" in
      X.Artifact.save_regression ~path fitted |> get;
      let output_path = Filename.concat directory "predictions.txt" in
      let output =
        Unix.openfile output_path
          [ Unix.O_CREAT; Unix.O_WRONLY; Unix.O_TRUNC ]
          0o600
      in
      let variable = "MODELKIT_STREAMING_TEST_ARTIFACT" in
      let environment =
        Unix.environment () |> Array.to_list
        |> List.filter (fun value ->
            not (String.starts_with ~prefix:(variable ^ "=") value))
      in
      let environment =
        Array.of_list ((variable ^ "=" ^ path) :: environment)
      in
      let _, status =
        Fun.protect
          ~finally:(fun () -> Unix.close output)
          (fun () ->
            let pid =
              Unix.create_process_env Sys.executable_name
                [| Sys.executable_name |] environment Unix.stdin output
                Unix.stderr
            in
            Unix.waitpid [] pid)
      in
      Alcotest.check Alcotest.bool "fresh inference process exited successfully"
        true (status = Unix.WEXITED 0);
      let input = open_in output_path in
      let observed =
        Fun.protect
          ~finally:(fun () -> close_in input)
          (fun () ->
            Array.init 24 (fun _ -> input_line input |> float_of_string))
      in
      let x = (batch ~start:0 ~rows:24 ()).R.x in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "fresh-process parity" (predict fitted x) observed)

let invalid_cache () =
  let path = Filename.temp_file "modelkit-invalid-cache-" "" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      List.iter
        (fun cache_directory ->
          let called = ref false in
          let next () =
            called := true;
            Ok None
          in
          let reset () =
            called := true;
            Ok ()
          in
          R.fit_batches (specification ()) ~rng:(rng ())
            ~feature_schema:features ~expected_rows:24 ~cache_directory ~next
            ~reset ()
          |> error_kind validation;
          Alcotest.check Alcotest.bool "cache validation precedes callbacks"
            false !called)
        [ path; path ^ "\000" ])

let mixed_weights () =
  with_directory (fun directory ->
      let next, reset = callbacks ~weighted:true ~rows:24 ~size:8 () in
      let batch_index = ref 0 in
      let reset () =
        batch_index := 0;
        reset ()
      in
      let next () =
        incr batch_index;
        next ()
        |> Result.map
             (Option.map (fun b ->
                  if !batch_index = 2 then { b with R.sample_weight = None }
                  else b))
      in
      R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
        ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      |> error_kind validation;
      empty directory)

let concurrent_fits () =
  with_directory (fun directory ->
      let work () =
        let next, reset = callbacks ~weighted:true ~rows:24 ~size:8 () in
        R.fit_batches (specification ()) ~rng:(rng ()) ~feature_schema:features
          ~expected_rows:24 ~cache_directory:directory ~next ~reset ()
      in
      let worker = Domain.spawn work in
      let first = work () in
      let second = Domain.join worker in
      let first = get first and second = get second in
      let x = (batch ~start:0 ~rows:24 ()).R.x in
      Alcotest.check
        (Alcotest.array (Alcotest.float 0.))
        "independent concurrent fits" (predict first x) (predict second x);
      empty directory)

let suite () =
  let major, minor, patch = X.Native.version () |> get in
  Alcotest.run
    (Printf.sprintf "XGBoost external-memory regression (native %d.%d.%d)" major
       minor patch)
    [
      ( "streaming",
        [
          Alcotest.test_case "multiple batches, missing values, persistence"
            `Quick (parity false);
          Alcotest.test_case "weights including a zero-weight batch" `Quick
            (parity true);
          Alcotest.test_case "native disk cache and caller ownership" `Quick
            disk_operation;
          Alcotest.test_case "bounded buffers and source release" `Quick
            bounded_buffers;
          Alcotest.test_case "typed callback errors retain context" `Quick
            typed_callback_error;
          Alcotest.test_case "premature native replay termination" `Quick
            native_short_input;
          Alcotest.test_case "bounded version-two diagnostics reader" `Quick
            invalid_diagnostics;
          Alcotest.test_case
            "save/load in a fresh process without training caches" `Quick
            fresh_process;
          Alcotest.test_case "invalid disk-cache location" `Quick invalid_cache;
          Alcotest.test_case "mixed batch weight presence" `Quick mixed_weights;
          Alcotest.test_case
            "independent concurrent fits sharing a cache parent" `Quick
            concurrent_fits;
          Alcotest.test_case "feature width" `Quick
            (rejected "width" (fun b ->
                 { b with R.x = Matrix.create ~rows:8 ~columns:2 0. |> data }));
          Alcotest.test_case "empty batch" `Quick
            (rejected "empty" (fun b ->
                 {
                   b with
                   R.x = Matrix.create ~rows:0 ~columns:3 0. |> data;
                   y = Target.regression (Vector.of_array [||]) |> data;
                 }));
          Alcotest.test_case "changed weight values on reset" `Quick
            (replay_failure ~weighted:true (fun b ->
                 {
                   b with
                   R.sample_weight = Some (Vector.create ~length:8 1. |> data);
                 }));
          Alcotest.test_case "short input" `Quick (totals 16 24);
          Alcotest.test_case "excess input" `Quick (totals 24 16);
          Alcotest.test_case "invalid row count" `Quick (totals 8 0);
          Alcotest.test_case "schema order" `Quick
            (rejected "schema" (fun b ->
                 {
                   b with
                   R.feature_schema =
                     schema [| "category"; "missing"; "signal" |];
                 }));
          Alcotest.test_case "target alignment" `Quick
            (rejected "target" (fun b ->
                 {
                   b with
                   R.y = Target.regression (Vector.of_array [| 0. |]) |> data;
                 }));
          Alcotest.test_case "weight alignment" `Quick
            (rejected "weights" (fun b ->
                 { b with R.sample_weight = Some (Vector.of_array [| 1. |]) }));
          Alcotest.test_case "negative weights" `Quick
            (rejected "weights" (fun b ->
                 {
                   b with
                   R.sample_weight = Some (Vector.create ~length:8 (-1.) |> data);
                 }));
          Alcotest.test_case "all-zero weights" `Quick
            (rejected "weights" (fun b ->
                 {
                   b with
                   R.sample_weight = Some (Vector.create ~length:8 0. |> data);
                 }));
          Alcotest.test_case "feature float32 overflow" `Quick
            (rejected "feature" (fun b ->
                 {
                   b with
                   R.x =
                     Matrix.create ~rows:8 ~columns:3 Float.max_float |> data;
                 }));
          Alcotest.test_case "target float32 overflow" `Quick
            (rejected "target" (fun b ->
                 {
                   b with
                   R.y =
                     Target.regression
                       (Vector.create ~length:8 Float.max_float |> data)
                     |> data;
                 }));
          Alcotest.test_case "positive weight float32 underflow" `Quick
            (rejected "weight" (fun b ->
                 {
                   b with
                   R.sample_weight =
                     Some (Vector.create ~length:8 1e-100 |> data);
                 }));
          Alcotest.test_case "nonfinite weights" `Quick
            (rejected "weight" (fun b ->
                 {
                   b with
                   R.sample_weight =
                     Some (Vector.create ~length:8 Float.infinity |> data);
                 }));
          Alcotest.test_case "next exception during validation" `Quick
            (callback_failure false false);
          Alcotest.test_case "reset exception during validation" `Quick
            (callback_failure true false);
          Alcotest.test_case "next exception inside native construction" `Quick
            (callback_failure false true);
          Alcotest.test_case "reset exception inside native construction" `Quick
            (callback_failure true true);
          Alcotest.test_case "changed targets on reset" `Quick
            (replay_failure (fun b ->
                 {
                   b with
                   R.y =
                     Target.regression (Vector.create ~length:8 100. |> data)
                     |> data;
                 }));
          Alcotest.test_case "changed feature values on reset" `Quick
            (replay_failure (fun b ->
                 { b with R.x = Matrix.create ~rows:8 ~columns:3 0. |> data }));
          Alcotest.test_case "changed weight presence on reset" `Quick
            (replay_failure (fun b ->
                 {
                   b with
                   R.sample_weight = Some (Vector.create ~length:8 1. |> data);
                 }));
        ] );
    ]

let () =
  match Sys.getenv_opt "MODELKIT_STREAMING_TEST_ARTIFACT" with
  | None -> suite ()
  | Some path ->
      let fitted = X.Artifact.load_regression ~path () |> get in
      let x = (batch ~start:0 ~rows:24 ()).R.x in
      Array.iter (Printf.printf "%.17g\n") (predict fitted x)
