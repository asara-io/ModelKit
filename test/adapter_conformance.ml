open Modelkit
open Admission

(** Source-neutral admission contract shared by every ModelKit adapter.

    Each adapter test instantiates {!Make} with a module that builds its own
    source values from plain OCaml arrays. The harness then checks the semantics
    every adapter must share: row-major logical copies, ordered feature names,
    explicit null identity, numeric-domain rejection, label range checks, weight
    validation, dataset assembly, and payload accounting. Feature names used
    here never collide with the reserved column roles ["target"], ["weight"],
    and ["group"] that table adapters may need. *)
module type ADAPTER = sig
  val name : string

  val features :
    ?null_mask:bool array array ->
    names:string array ->
    float array array ->
    (Admission.features, Error.t) result

  val regression_target :
    float array ->
    (Target.regression Target.t Admission.conversion, Error.t) result

  val classification_target :
    int64 array ->
    (Target.classification Target.t Admission.conversion, Error.t) result

  val sample_weight :
    float array -> (Sample_weight.t Admission.conversion, Error.t) result

  val groups : int64 array -> (Groups.t Admission.conversion, Error.t) result

  val classification_dataset :
    ?null_mask:bool array array ->
    ?sample_weight:float array ->
    ?groups:int64 array ->
    names:string array ->
    x:float array array ->
    y:int64 array ->
    unit ->
    (Target.classification Admission.dataset, Error.t) result

  val regression_dataset :
    ?null_mask:bool array array ->
    ?sample_weight:float array ->
    ?groups:int64 array ->
    names:string array ->
    x:float array array ->
    y:float array ->
    unit ->
    (Target.regression Admission.dataset, Error.t) result
end

module Make (Adapter : ADAPTER) = struct
  let word_bytes = Int64.of_int (Sys.word_size / 8)
  let float_bytes count = Int64.mul 8L (Int64.of_int count)
  let int_bytes count = Int64.mul word_bytes (Int64.of_int count)

  let get_ok = function
    | Ok value -> value
    | Error error -> Alcotest.fail (Error.to_string error)

  let is_data_error kind =
    match kind with
    | Error.Data _ -> true
    | Error.Validation _ | Error.Shape_mismatch _
    | Error.Feature_schema_mismatch _ | Error.Numerical _ | Error.Convergence _
    | Error.Compatibility _ | Error.Artifact _ | Error.Cancelled ->
        false

  let is_validation_error kind =
    match kind with
    | Error.Validation _ -> true
    | Error.Data _ | Error.Shape_mismatch _ | Error.Feature_schema_mismatch _
    | Error.Numerical _ | Error.Convergence _ | Error.Compatibility _
    | Error.Artifact _ | Error.Cancelled ->
        false

  let expect message predicate = function
    | Error error when predicate (Error.kind error) -> ()
    | Error error ->
        Alcotest.failf "%s: unexpected error %s" message (Error.to_string error)
    | Ok _ -> Alcotest.failf "%s: expected an error" message

  let check_report ~message ~source ~dtype ~shape ~temporary ~retained report =
    Alcotest.check Alcotest.string (message ^ " source") source
      (Conversion_report.source report);
    Alcotest.check Alcotest.string (message ^ " dtype") dtype
      (Conversion_report.source_dtype report);
    Alcotest.check
      (Alcotest.array Alcotest.int)
      (message ^ " shape") shape
      (Conversion_report.source_shape report);
    Alcotest.check Alcotest.int64
      (message ^ " temporary payload")
      temporary
      (Conversion_report.temporary_payload_bytes report);
    Alcotest.check Alcotest.int64
      (message ^ " retained payload")
      retained
      (Conversion_report.retained_payload_bytes report);
    Alcotest.check Alcotest.int64
      (message ^ " allocated payload")
      (Int64.add temporary retained)
      (Conversion_report.allocated_payload_bytes report)

  let values = [| [| 1.0; 2.5 |]; [| -3.0; 4.0 |]; [| 5.0; 0.0 |] |]
  let names = [| "alpha"; "beta" |]

  let test_features () =
    let admitted = Adapter.features ~names values |> get_ok in
    Alcotest.check
      (Alcotest.pair Alcotest.int Alcotest.int)
      "shape" (3, 2)
      (Matrix.shape admitted.matrix);
    Array.iteri
      (fun row expected ->
        Array.iteri
          (fun column expected ->
            Alcotest.check (Alcotest.float 0.0)
              (Printf.sprintf "value %d,%d" row column)
              expected
              (Matrix.get admitted.matrix row column))
          expected)
      values;
    let schema_names = Feature_schema.names admitted.schema |> Option.get in
    Alcotest.check
      (Alcotest.array Alcotest.string)
      "ordered names" names
      (Feature_names.to_array schema_names);
    Alcotest.check Alcotest.bool "no mask without nulls" true
      (Option.is_none admitted.null_mask);
    Alcotest.check Alcotest.int "single report" 1
      (List.length admitted.feature_reports);
    check_report ~message:"features" ~source:"features" ~dtype:"float64"
      ~shape:[| 3; 2 |] ~temporary:0L ~retained:(float_bytes 6)
      (List.hd admitted.feature_reports)

  let test_null_identity () =
    let values = [| [| 1.0; infinity |]; [| Float.nan; 4.0 |] |] in
    let null_mask = [| [| false; true |]; [| false; false |] |] in
    let admitted = Adapter.features ~null_mask ~names values |> get_ok in
    let mask = Option.get admitted.null_mask in
    Alcotest.check Alcotest.bool "masked position is null" true
      (Null_mask.get mask 0 1);
    Alcotest.check Alcotest.bool "masked value stored as NaN" true
      (Float.is_nan (Matrix.get admitted.matrix 0 1));
    Alcotest.check Alcotest.bool "genuine NaN is not null" false
      (Null_mask.get mask 1 0);
    Alcotest.check Alcotest.bool "genuine NaN retained" true
      (Float.is_nan (Matrix.get admitted.matrix 1 0));
    Alcotest.check Alcotest.int "null count" 1 (Null_mask.null_count mask);
    Alcotest.check Alcotest.int "value and mask reports" 2
      (List.length admitted.feature_reports);
    check_report ~message:"mask" ~source:"feature null mask" ~dtype:"bool"
      ~shape:[| 2; 2 |] ~temporary:0L ~retained:(int_bytes 4)
      (List.nth admitted.feature_reports 1);
    Alcotest.check Alcotest.int64 "retained total"
      (Int64.add (float_bytes 4) (int_bytes 4))
      (Admission.retained_payload_bytes admitted.feature_reports);
    let all_false = [| [| false; false |]; [| false; false |] |] in
    let admitted =
      Adapter.features ~null_mask:all_false ~names
        [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]
      |> get_ok
    in
    Alcotest.check Alcotest.int "an all-false mask yields no nulls" 0
      (Option.fold ~none:0 ~some:Null_mask.null_count admitted.null_mask)

  let test_feature_rejections () =
    expect "unmasked infinity" is_data_error
      (Adapter.features ~names [| [| 1.0; infinity |] |]);
    expect "unmasked negative infinity" is_data_error
      (Adapter.features ~names [| [| neg_infinity; 1.0 |] |])

  let test_regression_target () =
    let target = Adapter.regression_target [| 1.5; -2.0; 0.0 |] |> get_ok in
    Alcotest.check
      (Alcotest.array (Alcotest.float 0.0))
      "values" [| 1.5; -2.0; 0.0 |]
      (Target.regression_values target.value |> Vector.to_array);
    check_report ~message:"regression target" ~source:"regression target"
      ~dtype:"float64" ~shape:[| 3 |] ~temporary:0L ~retained:(float_bytes 3)
      target.report;
    expect "NaN target" is_data_error
      (Adapter.regression_target [| 1.0; Float.nan |]);
    expect "infinite target" is_data_error
      (Adapter.regression_target [| infinity; 1.0 |])

  let test_classification_target () =
    let target =
      Adapter.classification_target [| 3L; -2L; 3L; 0L |] |> get_ok
    in
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "labels" [| 3; -2; 3; 0 |]
      (Target.classification_values target.value);
    check_report ~message:"classification target"
      ~source:"classification target" ~dtype:"int64" ~shape:[| 4 |]
      ~temporary:(int_bytes 4) ~retained:(int_bytes 4) target.report;
    expect "label overflow" is_validation_error
      (Adapter.classification_target [| 0L; Int64.max_int |]);
    expect "label underflow" is_validation_error
      (Adapter.classification_target [| Int64.min_int |])

  let test_sample_weight () =
    let weights = Adapter.sample_weight [| 1.0; 0.0; 2.5 |] |> get_ok in
    Alcotest.check (Alcotest.float 0.0) "zero weight kept" 0.0
      (Sample_weight.get weights.value 1);
    Alcotest.check (Alcotest.float 0.0) "weight" 2.5
      (Sample_weight.get weights.value 2);
    check_report ~message:"sample weights" ~source:"sample weights"
      ~dtype:"float64" ~shape:[| 3 |] ~temporary:0L ~retained:(float_bytes 3)
      weights.report;
    expect "negative weight" is_data_error
      (Adapter.sample_weight [| 1.0; -0.5 |]);
    expect "all-zero weights" is_data_error
      (Adapter.sample_weight [| 0.0; 0.0 |]);
    expect "NaN weight" is_data_error (Adapter.sample_weight [| Float.nan |])

  let test_groups () =
    let groups = Adapter.groups [| 10L; 10L; -7L |] |> get_ok in
    Alcotest.check Alcotest.int "group" (-7) (Groups.get groups.value 2);
    check_report ~message:"groups" ~source:"groups" ~dtype:"int64"
      ~shape:[| 3 |] ~temporary:(int_bytes 3) ~retained:(int_bytes 3)
      groups.report;
    expect "group overflow" is_validation_error
      (Adapter.groups [| Int64.max_int |])

  let test_datasets () =
    let null_mask =
      [| [| false; true |]; [| false; false |]; [| true; false |] |]
    in
    let admitted =
      Adapter.classification_dataset ~null_mask
        ~sample_weight:[| 1.0; 2.0; 3.0 |] ~groups:[| 5L; 5L; 6L |] ~names
        ~x:values ~y:[| 0L; 1L; 0L |] ()
      |> get_ok
    in
    Alcotest.check Alcotest.int "samples" 3
      (Dataset.sample_count admitted.dataset);
    Alcotest.check Alcotest.int "features" 2
      (Dataset.feature_count admitted.dataset);
    let schema_names =
      Dataset.feature_schema admitted.dataset
      |> Feature_schema.names |> Option.get
    in
    Alcotest.check
      (Alcotest.array Alcotest.string)
      "dataset names" names
      (Feature_names.to_array schema_names);
    Alcotest.check Alcotest.int "null count" 2
      (Null_mask.null_count (Option.get admitted.feature_null_mask));
    Alcotest.check Alcotest.bool "masked feature is NaN in the dataset" true
      (Float.is_nan (Matrix.get (Dataset.features admitted.dataset) 2 0));
    Alcotest.check (Alcotest.float 0.0) "weight" 3.0
      (Sample_weight.get
         (Option.get (Dataset.sample_weight admitted.dataset))
         2);
    Alcotest.check Alcotest.int "group" 6
      (Groups.get (Option.get (Dataset.groups admitted.dataset)) 2);
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "labels" [| 0; 1; 0 |]
      (Target.classification_values (Dataset.target admitted.dataset));
    Alcotest.check Alcotest.int "five reports" 5
      (List.length admitted.dataset_reports);
    Alcotest.check Alcotest.int64 "retained total"
      (List.fold_left Int64.add 0L
         [ float_bytes 6; int_bytes 6; int_bytes 3; float_bytes 3; int_bytes 3 ])
      (Admission.retained_payload_bytes admitted.dataset_reports);
    Alcotest.check Alcotest.int64 "temporary total"
      (Int64.add (int_bytes 3) (int_bytes 3))
      (Admission.temporary_payload_bytes admitted.dataset_reports);
    let regression =
      Adapter.regression_dataset ~names ~x:values ~y:[| 1.0; 2.0; 3.0 |] ()
      |> get_ok
    in
    Alcotest.check Alcotest.bool "no mask" true
      (Option.is_none regression.feature_null_mask);
    Alcotest.check Alcotest.int "two reports" 2
      (List.length regression.dataset_reports);
    Alcotest.check
      (Alcotest.array (Alcotest.float 0.0))
      "regression target" [| 1.0; 2.0; 3.0 |]
      (Target.regression_values (Dataset.target regression.dataset)
      |> Vector.to_array);
    expect "null target rejected through dataset" is_data_error
      (Adapter.regression_dataset ~names ~x:values ~y:[| 1.0; Float.nan; 3.0 |]
         ())

  let tests =
    List.map
      (fun (name, test) ->
        Alcotest.test_case (Adapter.name ^ " " ^ name) `Quick test)
      [
        ("features", test_features);
        ("null identity", test_null_identity);
        ("feature rejections", test_feature_rejections);
        ("regression target", test_regression_target);
        ("classification target", test_classification_target);
        ("sample weights", test_sample_weight);
        ("groups", test_groups);
        ("datasets", test_datasets);
      ]
end
