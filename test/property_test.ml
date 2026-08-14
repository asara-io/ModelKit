open Modelkit

let vector_values = QCheck.(array nat_small)

let vector_ownership =
  QCheck.Test.make ~count:500
    ~name:"vector admission and export preserve ownership" vector_values
    (fun values ->
      let admitted = Array.map Float.of_int values in
      let source = Array.copy admitted in
      let vector = Vector.of_array source in
      if Array.length source > 0 then source.(0) <- source.(0) +. 1.0;
      let exported = Vector.to_array vector in
      if Array.length exported > 0 then exported.(0) <- exported.(0) +. 1.0;
      Vector.to_array vector = admitted)

let sequential_order =
  QCheck.Test.make ~count:500
    ~name:"sequential execution preserves logical order" vector_values
    (fun values ->
      match
        Sequential_execution.map Sequential_execution.default
          ~f:(fun ~index value -> Ok (index, value))
          values
      with
      | Error _ -> false
      | Ok observed ->
          Array.to_list observed
          = List.mapi (fun index value -> (index, value)) (Array.to_list values))

let seed_derivation =
  QCheck.Test.make ~count:500
    ~name:"logical seed derivation is a pure function of its inputs"
    QCheck.(pair int64 nat_small)
    (fun (root, index) ->
      let root = Seed.of_int64 root in
      Seed.equal
        (Seed.derive root ~operation:"property" ~index)
        (Seed.derive root ~operation:"property" ~index))

let rng_purity =
  QCheck.Test.make ~count:500
    ~name:"random generation does not mutate its input" QCheck.int64
    (fun seed ->
      let state = Rng.create (Seed.of_int64 seed) in
      let first, successor = Rng.next_int64 state in
      let repeated_first, repeated_successor = Rng.next_int64 state in
      let second, _ = Rng.next_int64 successor in
      let repeated_second, _ = Rng.next_int64 repeated_successor in
      Int64.equal first repeated_first && Int64.equal second repeated_second)

let dataset_view_order =
  QCheck.Test.make ~count:500
    ~name:"dataset row views preserve logical order and duplicates"
    QCheck.(pair (array nat_small) (array nat_small))
    (fun (values, requested) ->
      let sample_count = Array.length values in
      if sample_count = 0 then true
      else
        let rows = Array.map (fun index -> index mod sample_count) requested in
        let x =
          Result.get_ok
            (Matrix.init ~rows:sample_count ~columns:1 (fun row _ ->
                 Float.of_int values.(row)))
        in
        let dataset =
          Result.get_ok
            (Dataset.create ~finiteness:Dataset.Require_finite ~x
               ~y:(Target.classification (Array.init sample_count Fun.id))
               ())
        in
        let selection =
          Result.get_ok (Row_view.create ~source_size:sample_count rows)
        in
        let view = Result.get_ok (Dataset.view dataset selection) in
        let rec preserves_order position =
          position = Array.length rows
          ||
          let source = rows.(position) in
          Dataset.source_row view position = source
          && Dataset.feature view ~row:position ~column:0
             = Float.of_int values.(source)
          && Dataset.classification_target view position = source
          && preserves_order (position + 1)
        in
        preserves_order 0)

let preprocessing_rng () = Rng.create (Seed.of_int 17)

let imputer_removes_missing_values =
  QCheck.Test.make ~count:500
    ~name:"mean imputation fills NaN and preserves observed values"
    QCheck.(array nat_small)
    (fun values ->
      if Array.length values = 0 then true
      else
        let input =
          Array.mapi
            (fun row value ->
              [|
                (if row > 0 && row mod 2 = 1 then Float.nan
                 else Float.of_int value);
              |])
            values
        in
        let x = Result.get_ok (Matrix.of_arrays input) in
        let schema = Result.get_ok (Feature_schema.of_matrix x) in
        match
          Simple_imputer.fit (Simple_imputer.mean ())
            ~rng:(preprocessing_rng ()) ~feature_schema:schema ~x ~y:None ()
        with
        | Error _ -> false
        | Ok fitted -> (
            match Simple_imputer.transform fitted ~feature_schema:schema ~x with
            | Error _ -> false
            | Ok transformed ->
                let rec check row =
                  row = Array.length values
                  ||
                  let observed = Matrix.get transformed row 0 in
                  Float.is_finite observed
                  && (Float.is_nan input.(row).(0) || observed = input.(row).(0))
                  && check (row + 1)
                in
                check 0))

let scaler_normalizes_nonconstant_columns =
  QCheck.Test.make ~count:500
    ~name:"standard scaling produces zero mean and unit population variance"
    QCheck.(array nat_small)
    (fun values ->
      let values = Array.map Float.of_int values in
      if Array.length values < 2 || Array.for_all (( = ) values.(0)) values then
        true
      else
        let x =
          Result.get_ok
            (Matrix.of_arrays (Array.map (fun value -> [| value |]) values))
        in
        let schema = Result.get_ok (Feature_schema.of_matrix x) in
        match
          Standard_scaler.fit
            (Standard_scaler.create ())
            ~rng:(preprocessing_rng ()) ~feature_schema:schema ~x ~y:None ()
        with
        | Error _ -> false
        | Ok fitted -> (
            match
              Standard_scaler.transform fitted ~feature_schema:schema ~x
            with
            | Error _ -> false
            | Ok transformed ->
                let rows = Matrix.rows transformed in
                let mean = ref 0.0 in
                for row = 0 to rows - 1 do
                  mean := !mean +. Matrix.get transformed row 0
                done;
                mean := !mean /. Float.of_int rows;
                let variance = ref 0.0 in
                for row = 0 to rows - 1 do
                  let delta = Matrix.get transformed row 0 -. !mean in
                  variance := !variance +. (delta *. delta)
                done;
                variance := !variance /. Float.of_int rows;
                Float.abs !mean <= 1e-10
                && Float.abs (!variance -. 1.0) <= 1e-10))

let variance_threshold_removes_constant_column =
  QCheck.Test.make ~count:500
    ~name:"variance threshold removes constant columns in stable order"
    QCheck.nat_small (fun extra_rows ->
      let rows = 2 + (extra_rows mod 32) in
      let x =
        Result.get_ok
          (Matrix.init ~rows ~columns:2 (fun row column ->
               if column = 0 then 7.0 else Float.of_int row))
      in
      let schema = Result.get_ok (Feature_schema.of_matrix x) in
      let specification = Result.get_ok (Variance_threshold.create ()) in
      match
        Variance_threshold.fit specification ~rng:(preprocessing_rng ())
          ~feature_schema:schema ~x ~y:None ()
      with
      | Error _ -> false
      | Ok fitted -> Variance_threshold.selected_indices fitted = [| 1 |])

type property_passthrough_fitted = { property_schema : Feature_schema.t }

module Property_passthrough :
  ESTIMATOR
    with type t = unit
     and type params = unit
     and type target = unit
     and type prediction = Matrix.t
     and type fitted = property_passthrough_fitted
     and type rng = Rng.t = struct
  type t = unit
  type params = unit
  type target = unit
  type prediction = Matrix.t
  type fitted = property_passthrough_fitted
  type rng = Rng.t

  let clone () = ()
  let params () = ()

  let fit () ?sample_weight:_ ~rng:_ ~feature_schema ~x ~y:() () =
    match Feature_schema.validate_matrix feature_schema x with
    | Ok () -> Ok { property_schema = feature_schema }
    | Error data_error ->
        Error
          (Error.of_data_error ~remediation:"provide aligned property data"
             data_error)

  let predict fitted ~feature_schema ~x =
    if Feature_schema.equal fitted.property_schema feature_schema then Ok x
    else
      Error
        (Error.make ~remediation:"provide the fitted property schema"
           (Error.Feature_schema_mismatch
              { expected = fitted.property_schema; observed = feature_schema }))

  let fitted_params _ = ()
  let feature_schema fitted = fitted.property_schema
end

let pipeline_matches_manual_preprocessing =
  QCheck.Test.make ~count:500
    ~name:"pipeline preprocessing equals the same manually fitted stages"
    QCheck.(array nat_small)
    (fun generated ->
      if Array.length generated = 0 then true
      else
        let values = Array.map Float.of_int generated in
        let x =
          Array.mapi
            (fun row value ->
              [| (if row > 0 && row mod 3 = 0 then Float.nan else value) |])
            values
          |> Matrix.of_arrays |> Result.get_ok
        in
        let schema = Feature_schema.of_matrix x |> Result.get_ok in
        let imputer_specification = Simple_imputer.mean () in
        let scaler_specification = Standard_scaler.create () in
        let manual =
          match
            Simple_imputer.fit imputer_specification ~rng:(preprocessing_rng ())
              ~feature_schema:schema ~x ~y:None ()
          with
          | Error _ -> None
          | Ok imputer -> (
              match
                Simple_imputer.transform imputer ~feature_schema:schema ~x
              with
              | Error _ -> None
              | Ok complete -> (
                  match
                    Standard_scaler.fit scaler_specification
                      ~rng:(preprocessing_rng ()) ~feature_schema:schema
                      ~x:complete ~y:None ()
                  with
                  | Error _ -> None
                  | Ok scaler ->
                      Standard_scaler.transform scaler ~feature_schema:schema
                        ~x:complete
                      |> Result.to_option))
        in
        let pipeline =
          let imputer =
            Pipeline.transformer ~name:"impute"
              (module Simple_imputer)
              imputer_specification
            |> Result.get_ok
          in
          let scaler =
            Pipeline.transformer ~name:"scale"
              (module Standard_scaler)
              scaler_specification
            |> Result.get_ok
          in
          let builder =
            Pipeline.add_transformer Pipeline.empty imputer |> Result.get_ok
          in
          let builder =
            Pipeline.add_transformer builder scaler |> Result.get_ok
          in
          let estimator =
            Pipeline.estimator ~name:"passthrough"
              (module Property_passthrough)
              ()
            |> Result.get_ok
          in
          Pipeline.set_estimator builder estimator |> Result.get_ok
        in
        let pipelined =
          Result.bind
            (Pipeline.fit pipeline ~rng:(preprocessing_rng ())
               ~feature_schema:schema ~x ~y:() ())
            (fun fitted -> Pipeline.predict fitted ~feature_schema:schema ~x)
          |> Result.to_option
        in
        match (manual, pipelined) with
        | Some manual, Some pipelined ->
            Matrix.to_arrays manual = Matrix.to_arrays pipelined
        | None, None -> true
        | Some _, None | None, Some _ -> false)

let linear_model_rng () = Rng.create (Seed.of_int 29)

let ordinary_least_squares_recovers_exact_lines =
  QCheck.Test.make ~count:500
    ~name:"ordinary least squares recovers finite exact one-dimensional lines"
    QCheck.(triple (int_range (-100) 100) (int_range (-100) 100) nat_small)
    (fun (slope, intercept, extra_rows) ->
      let rows = 2 + (extra_rows mod 30) in
      let slope = Float.of_int slope in
      let intercept = Float.of_int intercept in
      let x =
        Result.get_ok
          (Matrix.init ~rows ~columns:1 (fun row _ -> Float.of_int row))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let target =
        Vector.init ~length:rows (fun row ->
            (slope *. Float.of_int row) +. intercept)
        |> Result.get_ok |> Target.regression |> Result.get_ok
      in
      match
        Linear_regression.fit
          (Linear_regression.create ())
          ~rng:(linear_model_rng ()) ~feature_schema ~x ~y:target ()
      with
      | Error _ -> false
      | Ok fitted ->
          Float.abs
            (Vector.get (Linear_regression.coefficients fitted) 0 -. slope)
          <= 1e-9
          && Float.abs (Linear_regression.intercept fitted -. intercept) <= 1e-9)

let logistic_probabilities_are_complementary =
  QCheck.Test.make ~count:200
    ~name:"binary logistic probabilities are finite complements"
    QCheck.(pair int_pos nat_small)
    (fun (raw_c, extra_rows) ->
      let pairs = 1 + (extra_rows mod 12) in
      let rows = pairs * 2 in
      let x =
        Result.get_ok
          (Matrix.init ~rows ~columns:1 (fun row _ ->
               if row < pairs then -.Float.of_int (pairs - row)
               else Float.of_int (row - pairs + 1)))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let specification =
        Logistic_regression.create ~c:(0.1 +. Float.of_int (raw_c mod 20)) ()
        |> Result.get_ok
      in
      let target =
        Target.classification
          (Array.init rows (fun row -> if row < pairs then 0 else 1))
      in
      match
        Logistic_regression.fit specification ~rng:(linear_model_rng ())
          ~feature_schema ~x ~y:target ()
      with
      | Error _ -> false
      | Ok fitted -> (
          match Logistic_regression.predict_proba fitted ~feature_schema ~x with
          | Error _ -> false
          | Ok probabilities ->
              let rec valid row =
                row = rows
                ||
                let left = Matrix.get probabilities row 0 in
                let right = Matrix.get probabilities row 1 in
                Float.is_finite left && Float.is_finite right && left >= 0.0
                && left <= 1.0 && right >= 0.0 && right <= 1.0
                && Float.abs (left +. right -. 1.0) <= 1e-15
                && valid (row + 1)
              in
              valid 0))

let splitter_rng () = Rng.create (Seed.of_int 37)

let k_fold_partitions_every_sample_once =
  QCheck.Test.make ~count:500
    ~name:"K-fold partitions every row and tests each row exactly once"
    QCheck.(pair nat_small nat_small)
    (fun (extra_samples, raw_folds) ->
      let samples = 2 + (extra_samples mod 64) in
      let folds = 2 + (raw_folds mod (samples - 1)) in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:1 (fun row _ -> Float.of_int row))
      in
      match K_fold.create ~folds ~shuffle:true () with
      | Error _ -> false
      | Ok specification -> (
          match
            K_fold.split specification ~rng:(splitter_rng ()) ~x ~y:None ()
          with
          | Error _ -> false
          | Ok splits ->
              let test_occurrences = Array.make samples 0 in
              let valid_partition (train, test) =
                let occurrences = Array.make samples 0 in
                Array.iter
                  (fun row -> occurrences.(row) <- occurrences.(row) + 1)
                  (Row_view.indices train);
                Array.iter
                  (fun row ->
                    occurrences.(row) <- occurrences.(row) + 1;
                    test_occurrences.(row) <- test_occurrences.(row) + 1)
                  (Row_view.indices test);
                Array.for_all (( = ) 1) occurrences
              in
              Array.length splits = folds
              && Array.for_all valid_partition splits
              && Array.for_all (( = ) 1) test_occurrences))

let stratified_folds_balance_each_class =
  QCheck.Test.make ~count:500
    ~name:"stratified K-fold balances every class independently"
    QCheck.(pair nat_small nat_small)
    (fun (raw_per_class, raw_folds) ->
      let per_class = 2 + (raw_per_class mod 32) in
      let folds = 2 + (raw_folds mod (per_class - 1)) in
      let samples = per_class * 3 in
      let labels = Array.init samples (fun row -> row mod 3) in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:1 (fun row _ -> Float.of_int row))
      in
      let specification =
        Stratified_k_fold.create ~folds ~shuffle:true () |> Result.get_ok
      in
      match
        Stratified_k_fold.split specification ~rng:(splitter_rng ()) ~x
          ~y:(Some (Target.classification labels))
          ()
      with
      | Error _ -> false
      | Ok splits ->
          let counts = Array.make_matrix 3 folds 0 in
          Array.iteri
            (fun fold (_, test) ->
              Array.iter
                (fun row ->
                  let label = labels.(row) in
                  counts.(label).(fold) <- counts.(label).(fold) + 1)
                (Row_view.indices test))
            splits;
          Array.for_all
            (fun class_counts ->
              let minimum = Array.fold_left Int.min max_int class_counts in
              let maximum = Array.fold_left Int.max min_int class_counts in
              maximum - minimum <= 1)
            counts)

let time_series_folds_never_train_on_the_future =
  QCheck.Test.make ~count:500
    ~name:"time-series split keeps every training row before its test window"
    QCheck.(pair nat_small nat_small)
    (fun (raw_test_size, raw_gap) ->
      let test_size = 1 + (raw_test_size mod 8) in
      let gap = raw_gap mod 5 in
      let folds = 3 in
      let samples = 1 + gap + (folds * test_size) + 7 in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:1 (fun row _ -> Float.of_int row))
      in
      let specification =
        Time_series_split.create ~folds ~test_size ~gap () |> Result.get_ok
      in
      match
        Time_series_split.split specification ~rng:(splitter_rng ()) ~x ~y:None
          ()
      with
      | Error _ -> false
      | Ok splits ->
          Array.for_all
            (fun (train, test) ->
              let last_train = Row_view.get train (Row_view.length train - 1) in
              let first_test = Row_view.get test 0 in
              last_train + gap < first_test)
            splits)

let () =
  let random = Random.State.make [| 0x4d4f4445; 0x4c4b4954 |] in
  let failures =
    QCheck_base_runner.run_tests ~verbose:true ~rand:random
      [
        vector_ownership;
        sequential_order;
        seed_derivation;
        rng_purity;
        dataset_view_order;
        imputer_removes_missing_values;
        scaler_normalizes_nonconstant_columns;
        variance_threshold_removes_constant_column;
        pipeline_matches_manual_preprocessing;
        ordinary_least_squares_recovers_exact_lines;
        logistic_probabilities_are_complementary;
        k_fold_partitions_every_sample_once;
        stratified_folds_balance_each_class;
        time_series_folds_never_train_on_the_future;
      ]
  in
  if failures <> 0 then exit failures
