open Modelkit
module N = Modelkit_xgboost_streaming_native

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let error_kind check = function
  | Ok _ -> Alcotest.fail "expected native error"
  | Error error ->
      Alcotest.check Alcotest.bool (Error.to_string error) true
        (check (Error.kind error))

let empty directory =
  Alcotest.check
    (Alcotest.array Alcotest.string)
    "native pages removed" [||] (Sys.readdir directory)

let with_directory f =
  let directory = Filename.temp_file "modelkit-native-bridge-" "" in
  Sys.remove directory;
  Unix.mkdir directory 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun name -> Sys.remove (Filename.concat directory name))
        (Sys.readdir directory);
      Unix.rmdir directory)
    (fun () -> f directory)

let native_bridge invalid_labels () =
  with_directory (fun directory ->
      let cursor = ref 0 and weak = Weak.create 1 in
      let next () =
        Gc.full_major ();
        Alcotest.check Alcotest.bool "previous float32 staging released" false
          (Weak.check weak 0);
        if !cursor = 3 then Ok None
        else (
          incr cursor;
          let features =
            Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout 8 3
          in
          let labels =
            Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 8
          in
          for row = 0 to 7 do
            for column = 0 to 2 do
              features.{row, column} <- float_of_int (row + column)
            done;
            labels.{row} <-
              (if invalid_labels then Float.nan else float_of_int row)
          done;
          Weak.set weak 0 (Some features);
          Ok (Some { N.features; labels; weights = None }))
      in
      let reset () =
        cursor := 0;
        Ok ()
      in
      let result =
        N.train
          ~parameters:
            (("objective", "reg:squarederror")
            :: [ ("device", "cpu"); ("tree_method", "hist"); ("nthread", "1") ]
            )
          ~rounds:2 ~threads:1 ~rows:24 ~columns:3
          ~cache_prefix:(Filename.concat directory "native")
          ~next ~reset ()
      in
      if invalid_labels then
        result
        |> error_kind (function
          | Error.Numerical _ -> true
          | Error.Data _ | Error.Shape_mismatch _
          | Error.Feature_schema_mismatch _ | Error.Validation _
          | Error.Convergence _ | Error.Compatibility _ | Error.Artifact _
          | Error.Callback_failure _ | Error.Cancelled ->
              false)
      else ignore (get result);
      Gc.full_major ();
      Alcotest.check Alcotest.bool "last float32 staging released" false
        (Weak.check weak 0);
      empty directory)

let () =
  Alcotest.run "XGBoost private iterator bridge"
    [
      ( "lifetime",
        [
          Alcotest.test_case "batch buffers released at native boundary" `Quick
            (native_bridge false);
          Alcotest.test_case "native setter errors contained" `Quick
            (native_bridge true);
        ] );
    ]
