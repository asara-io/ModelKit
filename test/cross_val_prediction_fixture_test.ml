open Modelkit

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let fields () =
  In_channel.with_open_text (Sys.getenv "MODELKIT_CROSS_VAL_PREDICTION_FIXTURE")
    (fun channel ->
      In_channel.input_lines channel
      |> List.filter_map (fun line ->
          match String.split_on_char '\t' line with
          | [ key; values ] -> Some (key, String.split_on_char ',' values)
          | _ -> None))

let floats fields name = List.assoc name fields |> List.map float_of_string
let ints fields name = List.assoc name fields |> List.map int_of_string

let k_fold () =
  K_fold.create ~folds:3 () |> get
  |> Cross_validation.target_independent_splitter (module K_fold)

let regression_pipeline () =
  let terminal =
    Pipeline.estimator ~name:"linear"
      (module Linear_regression)
      (Linear_regression.create ())
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

module First_class_classifier = struct
  type t = unit
  type params = unit
  type fitted = { schema : Feature_schema.t; classes : int array }
  type target = Target.classification Target.t
  type prediction = Target.classification Target.t
  type rng = Rng.t

  let clone = Fun.id
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x:_ ~y () =
    let classes =
      Target.classification_values y
      |> Array.to_list |> List.sort_uniq Int.compare |> Array.of_list
    in
    Ok { schema = feature_schema; classes }

  let predict fitted ~feature_schema:_ ~x =
    Ok (Target.classification (Array.make (Matrix.rows x) fitted.classes.(0)))

  let predict_proba fitted ~feature_schema:_ ~x =
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length fitted.classes)
      (fun _ column -> if column = 0 then 1.0 else 0.0)
    |> Result.map_error (fun error ->
        Error.of_data_error ~remediation:"provide valid test features" error)

  let classes fitted = Array.copy fitted.classes
  let fitted_params _ = ()
  let feature_schema fitted = fitted.schema
end

let classification_pipeline () =
  let terminal =
    Pipeline.estimator ~name:"first-class"
      (module First_class_classifier)
      ~predict_proba:First_class_classifier.predict_proba
      ~classes:First_class_classifier.classes ()
    |> get
  in
  Pipeline.set_estimator Pipeline.empty terminal |> get

let test_fixture () =
  let fields = fields () in
  let regression_x = floats fields "regression_x" |> Array.of_list in
  let regression_y = floats fields "regression_y" |> Array.of_list in
  let x =
    Matrix.init ~rows:(Array.length regression_x) ~columns:1 (fun row _ ->
        regression_x.(row))
    |> get_data
  in
  let y = Target.regression (Vector.of_array regression_y) |> get_data in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x ~y () |> get_data
  in
  let regression_prediction =
    Cross_validation.Regression.cross_val_predict ~splitter:(k_fold ())
      ~seed:(Seed.of_int 101) (regression_pipeline ()) dataset
    |> get |> Cross_validation.out_of_fold_predictions |> Result.get_ok
    |> Target.regression_values
  in
  floats fields "regression_prediction"
  |> List.iteri (fun row expected ->
      Alcotest.check (Alcotest.float 1e-9) "regression prediction" expected
        (Vector.get regression_prediction row));
  let labels = ints fields "classification_y" |> Array.of_list in
  let x =
    Matrix.init ~rows:(Array.length labels) ~columns:1 (fun row _ ->
        Float.of_int row)
    |> get_data
  in
  let dataset =
    Dataset.create ~finiteness:Dataset.Require_finite ~x
      ~y:(Target.classification labels)
      ()
    |> get_data
  in
  let prediction =
    Cross_validation.Multiclass_classification.cross_val_predict
      ~response:Cross_validation.Probabilities ~splitter:(k_fold ())
      ~seed:(Seed.of_int 101)
      (classification_pipeline ())
      dataset
    |> get |> Cross_validation.out_of_fold_predictions |> Result.get_ok
  in
  Alcotest.(check (array int))
    "class order"
    (ints fields "classification_classes" |> Array.of_list)
    (Multiclass_prediction.classes prediction |> Option.get);
  let probabilities =
    Multiclass_prediction.probabilities prediction |> Option.get
  in
  for row = 0 to Matrix.rows probabilities - 1 do
    floats fields ("classification_probability_" ^ string_of_int row)
    |> List.iteri (fun column expected ->
        Alcotest.check (Alcotest.float 0.0) "aligned probability" expected
          (Matrix.get probabilities row column))
  done

let () =
  Alcotest.run "Cross-validation prediction reference"
    [ ("sklearn", [ ("out-of-fold parity", `Quick, test_fixture) ]) ]
