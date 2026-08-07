open Modelkit

let fail message = raise (Failure message)
let check condition message = if not condition then fail message

let get_ok = function
  | Ok value -> value
  | Error error -> fail (Data_error.to_string error)

type data_error_kind = Negative_dimension | Length_mismatch | Other

let data_error_kind = function
  | Data_error.Negative_dimension _ -> Negative_dimension
  | Data_error.Length_mismatch _ -> Length_mismatch
  | Data_error.Ragged_matrix _
  | Data_error.Index_out_of_bounds _
  | Data_error.Non_finite _
  | Data_error.Negative_weight _
  | Data_error.All_zero_weights
  | Data_error.Empty_feature_name _
  | Data_error.Duplicate_feature_name _ ->
      Other

let expect_data_error expected = function
  | Error error when data_error_kind error = expected -> ()
  | Error error -> fail ("unexpected data error: " ^ Data_error.to_string error)
  | Ok _ -> fail "expected a data error"

let test_feature_schema () =
  expect_data_error Negative_dimension
    (Feature_schema.anonymous ~feature_count:(-1));
  let matrix = get_ok (Matrix.of_arrays [| [| 1.0; 2.0 |] |]) in
  let names =
    get_ok (Feature_names.create ~expected_count:2 [| "height"; "weight" |])
  in
  let same_names =
    get_ok (Feature_names.create ~expected_count:2 [| "height"; "weight" |])
  in
  let reversed_names =
    get_ok (Feature_names.create ~expected_count:2 [| "weight"; "height" |])
  in
  let named = Feature_schema.named names in
  let same = Feature_schema.named same_names in
  let reversed = Feature_schema.named reversed_names in
  let anonymous = get_ok (Feature_schema.of_matrix matrix) in
  check (Feature_schema.feature_count named = 2) "named schema width is wrong";
  check (Feature_schema.equal named same) "equivalent named schemas differ";
  check
    (not (Feature_schema.equal named reversed))
    "feature order is absent from schema equality";
  check
    (not (Feature_schema.equal named anonymous))
    "named and anonymous schemas compare equal";
  check
    (Feature_names.to_array (Option.get (Feature_schema.names named))
    = [| "height"; "weight" |])
    "named schema lost feature names";
  ignore (get_ok (Feature_schema.validate_matrix named matrix));
  let wrong_width = get_ok (Matrix.of_arrays [| [| 1.0 |] |]) in
  expect_data_error Length_mismatch
    (Feature_schema.validate_matrix named wrong_width);
  expect_data_error Length_mismatch
    (Feature_schema.of_matrix ~names
       (get_ok (Matrix.of_arrays [| [| 1.0 |] |])))

let test_error_convention () =
  let error =
    Error.make ~context:[ Error.Stage "fit" ]
      ~remediation:"choose a strictly positive value"
      (Error.Validation { name = "regularization"; reason = "must be positive" })
    |> Error.with_context (Error.Fold 2)
  in
  check
    (Error.context error = [ Error.Fold 2; Error.Stage "fit" ])
    "error context order is wrong";
  check
    (Error.remediation error = "choose a strictly positive value")
    "error remediation changed";
  let validation =
    match Error.kind error with
    | Error.Validation { name; reason } -> Some (name, reason)
    | Error.Data _
    | Error.Shape_mismatch _
    | Error.Feature_schema_mismatch _
    | Error.Numerical _
    | Error.Convergence _
    | Error.Compatibility _
    | Error.Artifact _
    | Error.Cancelled ->
        None
  in
  let name, reason =
    match validation with
    | Some validation -> validation
    | None -> fail "unexpected error kind"
  in
  check (name = "regularization") "validation error name changed";
  check (reason = "must be positive") "validation reason changed";
  check (String.length (Error.to_string error) > 0) "formatted error is empty";
  let data_error =
    Error.of_data_error ~remediation:"provide two values"
      (Data_error.Length_mismatch
         { name = "target"; expected = 2; observed = 1 })
  in
  let wrapped =
    match Error.kind data_error with
    | Error.Data error -> Some error
    | Error.Shape_mismatch _
    | Error.Feature_schema_mismatch _
    | Error.Validation _
    | Error.Numerical _
    | Error.Convergence _
    | Error.Compatibility _
    | Error.Artifact _
    | Error.Cancelled ->
        None
  in
  let dimensions =
    match wrapped with
    | Some (Data_error.Length_mismatch { expected; observed; _ }) ->
        Some (expected, observed)
    | Some
        (Data_error.Negative_dimension _ | Data_error.Ragged_matrix _
        | Data_error.Index_out_of_bounds _ | Data_error.Non_finite _
        | Data_error.Negative_weight _ | Data_error.All_zero_weights
        | Data_error.Empty_feature_name _ | Data_error.Duplicate_feature_name _)
    | None ->
        None
  in
  match dimensions with
  | Some (expected, observed) ->
      check (expected = 2 && observed = 1) "wrapped data error changed"
  | None -> fail "data error was not preserved"

let () =
  test_feature_schema ();
  test_error_convention ()
