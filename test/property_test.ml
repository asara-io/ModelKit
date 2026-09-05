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

let csr_dense_round_trip =
  QCheck.Test.make ~count:500
    ~name:"CSR conversion preserves dense values and portable kernels"
    QCheck.(array (int_range (-100) 100))
    (fun raw ->
      let rows = Array.length raw in
      let columns = 3 in
      let dense =
        Result.get_ok
          (Matrix.init ~rows ~columns (fun row column ->
               let value = raw.(row) + column in
               if (row + column) mod 3 = 0 then 0.0 else Float.of_int value))
      in
      let csr = Csr_matrix.of_dense dense in
      let restored = Csr_matrix.to_dense csr in
      let operand = Vector.of_array [| 1.0; -2.0; 0.5 |] in
      let dense_product =
        Reference_backend.feature_matrix_vector_product
          (Feature_matrix.dense dense)
          operand
      in
      let csr_product =
        Reference_backend.feature_matrix_vector_product (Feature_matrix.csr csr)
          operand
      in
      let transposed_operand = Vector.of_array (Array.map Float.of_int raw) in
      let dense_transposed =
        Reference_backend.transposed_feature_matrix_vector_product
          (Feature_matrix.dense dense)
          transposed_operand
      in
      let csr_transposed =
        Reference_backend.transposed_feature_matrix_vector_product
          (Feature_matrix.csr csr) transposed_operand
      in
      Matrix.to_arrays restored = Matrix.to_arrays dense
      && Result.map Vector.to_array dense_product
         = Result.map Vector.to_array csr_product
      && Result.map Vector.to_array dense_transposed
         = Result.map Vector.to_array csr_transposed)

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

let one_hot_dense_and_csr_agree =
  QCheck.Test.make ~count:300
    ~name:"one-hot dense and CSR outputs agree with one category per feature"
    QCheck.(array (pair (int_range (-3) 3) (int_range (-3) 3)))
    (fun rows ->
      if Array.length rows = 0 then true
      else
        let x =
          Result.get_ok
            (Matrix.of_arrays
               (Array.map
                  (fun (left, right) ->
                    [| Float.of_int left; Float.of_int right |])
                  rows))
        in
        let schema = Result.get_ok (Feature_schema.of_matrix x) in
        match One_hot_encoder.create () with
        | Error _ -> false
        | Ok specification -> (
            match
              One_hot_encoder.fit specification ~rng:(preprocessing_rng ())
                ~feature_schema:schema ~x ~y:None ()
            with
            | Error _ -> false
            | Ok fitted -> (
                match
                  ( One_hot_encoder.transform fitted ~feature_schema:schema ~x,
                    One_hot_encoder.transform_csr fitted ~feature_schema:schema
                      ~x )
                with
                | Ok dense, Ok csr ->
                    let dense_values = Matrix.to_arrays dense in
                    Matrix.to_arrays (Csr_matrix.to_dense csr) = dense_values
                    && Array.for_all
                         (fun row -> Array.fold_left ( +. ) 0.0 row = 2.0)
                         dense_values
                | Error _, _ | _, Error _ -> false)))

let label_encoding_round_trip =
  QCheck.Test.make ~count:500
    ~name:"label encoding is a reversible sorted bijection"
    QCheck.(array (int_range (-100) 100))
    (fun values ->
      let target = Target.classification values in
      match Label_encoder.fit (Label_encoder.create ()) ~y:target with
      | Error _ -> false
      | Ok fitted -> (
          match Label_encoder.transform fitted target with
          | Error _ -> false
          | Ok encoded -> (
              match Label_encoder.inverse_transform fitted encoded with
              | Error _ -> false
              | Ok decoded -> Target.classification_values decoded = values)))

let l2_normalization_is_scale_invariant =
  QCheck.Test.make ~count:300
    ~name:"L2 normalization is invariant under positive row scaling"
    QCheck.(array (int_range (-100) 100))
    (fun values ->
      let values = if Array.length values = 0 then [| 0 |] else values in
      let row = Array.map Float.of_int values in
      let x = Result.get_ok (Matrix.of_arrays [| row |]) in
      let scaled =
        Result.get_ok
          (Matrix.of_arrays [| Array.map (fun value -> value *. 7.0) row |])
      in
      let schema = Result.get_ok (Feature_schema.of_matrix x) in
      match
        Normalizer.fit
          (Normalizer.create ~norm:Normalizer.L2 ())
          ~rng:(preprocessing_rng ()) ~feature_schema:schema ~x ~y:None ()
      with
      | Error _ -> false
      | Ok fitted -> (
          match
            ( Normalizer.transform fitted ~feature_schema:schema ~x,
              Normalizer.transform fitted ~feature_schema:schema ~x:scaled )
          with
          | Ok left, Ok right ->
              let left = Matrix.to_arrays left in
              let right = Matrix.to_arrays right in
              Array.for_all2
                (fun left right -> Float.abs (left -. right) <= 1e-12)
                left.(0) right.(0)
          | Error _, _ | _, Error _ -> false))

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

let ridge_classifier_binary_scores_are_opposites =
  QCheck.Test.make ~count:300
    ~name:"binary ridge-classifier scores are finite opposites"
    QCheck.(pair (array (int_range (-50) 50)) (int_range 0 100))
    (fun (raw, raw_alpha) ->
      let rows = 4 + Array.length raw in
      let x =
        Result.get_ok
          (Matrix.init ~rows ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else (row * 7) - 11
               in
               if column = 0 then Float.of_int value
               else Float.of_int (value * value mod 29)))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let target =
        Target.classification
          (Array.init rows (fun row -> if row mod 2 = 0 then -7 else 12))
      in
      let alpha = 0.1 +. (Float.of_int raw_alpha /. 10.0) in
      match
        Result.bind (Ridge_classifier.create ~alpha ()) (fun specification ->
            Ridge_classifier.fit specification ~rng:(linear_model_rng ())
              ~feature_schema ~x ~y:target ())
      with
      | Error _ -> false
      | Ok fitted -> (
          match
            Ridge_classifier.decision_function fitted ~feature_schema ~x
          with
          | Error _ -> false
          | Ok scores ->
              let rec valid row =
                row = rows
                ||
                let left = Matrix.get scores row 0 in
                let right = Matrix.get scores row 1 in
                Float.is_finite left && Float.is_finite right
                && Float.abs (left +. right) <= 1e-9
                && valid (row + 1)
              in
              Ridge_classifier.classes fitted = [| -7; 12 |] && valid 0))

let ridge_classifier_is_equivariant_to_ordered_label_renaming =
  QCheck.Test.make ~count:200
    ~name:"ridge classification preserves ordered label renaming"
    QCheck.(int_range 0 100)
    (fun raw_alpha ->
      let x =
        Result.get_ok
          (Matrix.of_arrays
             [|
               [| 2.0; 0.0 |];
               [| 3.0; 0.0 |];
               [| 0.0; 2.0 |];
               [| 0.0; 3.0 |];
               [| -2.0; -2.0 |];
               [| -3.0; -3.0 |];
             |])
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let alpha = 0.1 +. (Float.of_int raw_alpha /. 10.0) in
      let specification = Result.get_ok (Ridge_classifier.create ~alpha ()) in
      let fit labels =
        Ridge_classifier.fit specification ~rng:(linear_model_rng ())
          ~feature_schema ~x
          ~y:(Target.classification labels)
          ()
      in
      match (fit [| 0; 0; 1; 1; 2; 2 |], fit [| -5; -5; 7; 7; 99; 99 |]) with
      | Ok original, Ok renamed -> (
          match
            ( Ridge_classifier.predict original ~feature_schema ~x,
              Ridge_classifier.predict renamed ~feature_schema ~x )
          with
          | Ok original, Ok renamed ->
              let original = Target.classification_values original in
              let renamed = Target.classification_values renamed in
              Array.for_all2
                (fun original renamed -> renamed = [| -5; 7; 99 |].(original))
                original renamed
          | Error _, _ | _, Error _ -> false)
      | Error _, _ | _, Error _ -> false)

let multinomial_probabilities_are_simplex_and_scores_are_centered =
  QCheck.Test.make ~count:200
    ~name:"multinomial probabilities form a simplex with centered scores"
    QCheck.(pair (array (int_range (-20) 20)) (int_range 1 100))
    (fun (raw, raw_c) ->
      let rows = 6 + Array.length raw in
      let x =
        Result.get_ok
          (Matrix.init ~rows ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else (row * 5) - 13
               in
               if column = 0 then Float.of_int value
               else Float.of_int (value * value mod 17)))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let target =
        Target.classification
          (Array.init rows (fun row -> [| -7; 3; 12 |].(row mod 3)))
      in
      let c = Float.of_int raw_c /. 10.0 in
      match
        Result.bind (Multinomial_logistic_regression.create ~c ())
          (fun specification ->
            Multinomial_logistic_regression.fit specification
              ~rng:(linear_model_rng ()) ~feature_schema ~x ~y:target ())
      with
      | Error _ -> false
      | Ok fitted -> (
          match
            ( Multinomial_logistic_regression.decision_function fitted
                ~feature_schema ~x,
              Multinomial_logistic_regression.predict_proba fitted
                ~feature_schema ~x )
          with
          | Ok decisions, Ok probabilities ->
              let rec valid row =
                if row = rows then true
                else
                  let score_total = ref 0.0 in
                  let probability_total = ref 0.0 in
                  let finite = ref true in
                  for class_index = 0 to 2 do
                    let score = Matrix.get decisions row class_index in
                    let probability =
                      Matrix.get probabilities row class_index
                    in
                    score_total := !score_total +. score;
                    probability_total := !probability_total +. probability;
                    finite :=
                      !finite && Float.is_finite score
                      && Float.is_finite probability
                      && probability >= 0.0 && probability <= 1.0
                  done;
                  !finite
                  && Float.abs !score_total <= 1e-8
                  && Float.abs (!probability_total -. 1.0) <= 1e-12
                  && valid (row + 1)
              in
              Multinomial_logistic_regression.classes fitted = [| -7; 3; 12 |]
              && valid 0
          | Error _, _ | _, Error _ -> false))

let multinomial_integer_weights_match_row_replication =
  QCheck.Test.make ~count:100
    ~name:"multinomial integer sample weights equal row replication"
    QCheck.(triple (int_range 1 3) (int_range 1 3) (int_range 1 3))
    (fun (first_weight, second_weight, third_weight) ->
      let source =
        [|
          ([| 2.0; 0.0 |], 0, first_weight);
          ([| 3.0; 0.0 |], 0, first_weight);
          ([| 0.0; 2.0 |], 1, second_weight);
          ([| 0.0; 3.0 |], 1, second_weight);
          ([| -2.0; -2.0 |], 2, third_weight);
          ([| -3.0; -3.0 |], 2, third_weight);
        |]
      in
      let x =
        Array.map (fun (features, _, _) -> features) source
        |> Matrix.of_arrays |> Result.get_ok
      in
      let target =
        Target.classification (Array.map (fun (_, label, _) -> label) source)
      in
      let sample_weight =
        Array.map (fun (_, _, weight) -> Float.of_int weight) source
        |> Sample_weight.of_array ~expected_length:(Array.length source)
        |> Result.get_ok
      in
      let replicated =
        Array.to_list source
        |> List.concat_map (fun ((_, _, weight) as row) ->
            List.init weight (fun _ -> row))
        |> Array.of_list
      in
      let replicated_x =
        Array.map (fun (features, _, _) -> features) replicated
        |> Matrix.of_arrays |> Result.get_ok
      in
      let replicated_target =
        Target.classification
          (Array.map (fun (_, label, _) -> label) replicated)
      in
      let specification =
        Multinomial_logistic_regression.create ~c:2.0 ~tolerance:1e-10 ()
        |> Result.get_ok
      in
      let fit ?sample_weight x y =
        Multinomial_logistic_regression.fit specification ?sample_weight
          ~rng:(linear_model_rng ())
          ~feature_schema:(Feature_schema.of_matrix x |> Result.get_ok)
          ~x ~y ()
      in
      match
        (fit ~sample_weight x target, fit replicated_x replicated_target)
      with
      | Ok weighted, Ok duplicated ->
          let weighted =
            Multinomial_logistic_regression.coefficients weighted
            |> Matrix.to_arrays |> Array.to_list |> Array.concat
          in
          let duplicated =
            Multinomial_logistic_regression.coefficients duplicated
            |> Matrix.to_arrays |> Array.to_list |> Array.concat
          in
          Array.for_all2
            (fun left right -> Float.abs (left -. right) <= 1e-8)
            weighted duplicated
      | Error _, _ | _, Error _ -> false)

let poisson_integer_weights_match_row_replication =
  QCheck.Test.make ~count:100
    ~name:"Poisson integer sample weights equal row replication"
    QCheck.(triple (int_range 1 3) (int_range 1 3) (int_range 1 3))
    (fun (first_weight, second_weight, third_weight) ->
      let source =
        [|
          ([| -1.0 |], 0.5, first_weight);
          ([| 0.0 |], 1.0, second_weight);
          ([| 1.0 |], 2.5, third_weight);
        |]
      in
      let x =
        Array.map (fun (features, _, _) -> features) source
        |> Matrix.of_arrays |> Result.get_ok
      in
      let target =
        Array.map (fun (_, value, _) -> value) source
        |> Vector.of_array |> Target.regression |> Result.get_ok
      in
      let sample_weight =
        Array.map (fun (_, _, weight) -> Float.of_int weight) source
        |> Sample_weight.of_array ~expected_length:(Array.length source)
        |> Result.get_ok
      in
      let replicated =
        Array.to_list source
        |> List.concat_map (fun ((_, _, weight) as row) ->
            List.init weight (fun _ -> row))
        |> Array.of_list
      in
      let replicated_x =
        Array.map (fun (features, _, _) -> features) replicated
        |> Matrix.of_arrays |> Result.get_ok
      in
      let replicated_target =
        Array.map (fun (_, value, _) -> value) replicated
        |> Vector.of_array |> Target.regression |> Result.get_ok
      in
      let specification =
        Poisson_regression.create ~alpha:0.3 ~tolerance:1e-10 ()
        |> Result.get_ok
      in
      let fit x y sample_weight =
        let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
        Poisson_regression.fit specification ?sample_weight
          ~rng:(linear_model_rng ()) ~feature_schema ~x ~y ()
      in
      match
        ( fit x target (Some sample_weight),
          fit replicated_x replicated_target None )
      with
      | Ok weighted, Ok repeated ->
          let weighted_coefficient =
            Vector.get (Poisson_regression.coefficients weighted) 0
          in
          let repeated_coefficient =
            Vector.get (Poisson_regression.coefficients repeated) 0
          in
          Float.abs (weighted_coefficient -. repeated_coefficient) <= 1e-7
          && Float.abs
               (Poisson_regression.intercept weighted
               -. Poisson_regression.intercept repeated)
             <= 1e-7
      | Error _, _ | _, Error _ -> false)

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

let regression_losses_are_non_negative_and_consistent =
  QCheck.Test.make ~count:500
    ~name:"regression losses are non-negative and RMSE squares to MSE"
    QCheck.(array nat_small)
    (fun raw ->
      if Array.length raw = 0 then true
      else
        let truth_values = Array.map Float.of_int raw in
        let prediction_values =
          Array.mapi
            (fun index value ->
              Float.of_int value +. Float.of_int ((index mod 3) - 1))
            raw
        in
        let truth =
          Target.regression (Vector.of_array truth_values) |> Result.get_ok
        in
        let prediction =
          Target.regression (Vector.of_array prediction_values) |> Result.get_ok
        in
        match
          ( Regression_metrics.mean_absolute_error ~truth ~prediction (),
            Regression_metrics.mean_squared_error ~truth ~prediction (),
            Regression_metrics.root_mean_squared_error ~truth ~prediction () )
        with
        | Ok mae, Ok mse, Ok rmse ->
            mae >= 0.0 && mse >= 0.0 && rmse >= 0.0
            && Float.abs ((rmse *. rmse) -. mse) <= 1e-12
        | Error _, _, _ | _, Error _, _ | _, _, Error _ -> false)

let binary_scalar_metrics_are_bounded =
  QCheck.Test.make ~count:500
    ~name:"binary scalar metrics remain within documented bounds"
    QCheck.(array nat_small)
    (fun raw ->
      if Array.length raw = 0 then true
      else
        let truth =
          Target.classification (Array.map (fun value -> value mod 2) raw)
        in
        let prediction =
          Target.classification (Array.map (fun value -> value / 2 mod 2) raw)
        in
        let probabilities =
          Vector.of_array
            (Array.map
               (fun value -> Float.of_int (1 + (value mod 98)) /. 100.0)
               raw)
        in
        let undefined = Undefined_metric_policy.Use_fallback in
        let results =
          [|
            Binary_classification_metrics.accuracy ~truth ~prediction ();
            Binary_classification_metrics.balanced_accuracy ~undefined ~truth
              ~prediction ();
            Binary_classification_metrics.precision ~undefined ~truth
              ~prediction ();
            Binary_classification_metrics.recall ~undefined ~truth ~prediction
              ();
            Binary_classification_metrics.f1 ~undefined ~truth ~prediction ();
            Binary_classification_metrics.roc_auc ~undefined ~truth
              ~positive_probabilities:probabilities ();
          |]
        in
        Array.for_all
          (function
            | Ok value -> value >= 0.0 && value <= 1.0 | Error _ -> false)
          results
        &&
        match
          Binary_classification_metrics.log_loss ~truth
            ~positive_probabilities:probabilities ()
        with
        | Ok value -> Float.is_finite value && value >= 0.0
        | Error _ -> false)

let lasso_matches_one_dimensional_soft_threshold =
  QCheck.Test.make ~count:400
    ~name:"lasso matches the analytic one-dimensional soft threshold"
    QCheck.(pair (int_range 1 100) (int_range 0 100))
    (fun (raw_slope, raw_penalty) ->
      let slope = Float.of_int raw_slope /. 10.0 in
      let alpha = Float.of_int raw_penalty /. 20.0 in
      let x =
        Result.get_ok (Matrix.of_arrays [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |])
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let y =
        Result.get_ok
          (Target.regression
             (Vector.of_array [| -.slope +. 3.0; 3.0; slope +. 3.0 |]))
      in
      match
        Result.bind (Lasso_regression.create ~alpha ~tolerance:1e-10 ())
          (fun specification ->
            Lasso_regression.fit specification
              ~rng:(Rng.create (Seed.of_int 11))
              ~feature_schema ~x ~y ())
      with
      | Error _ -> false
      | Ok fitted ->
          let expected = Float.max 0.0 (slope -. (1.5 *. alpha)) in
          Float.abs
            (Vector.get (Lasso_regression.coefficients fitted) 0 -. expected)
          <= 1e-8
          && Float.abs (Lasso_regression.intercept fitted -. 3.0) <= 1e-8)

let regularization_paths_are_descending_and_warm_started =
  QCheck.Test.make ~count:200
    ~name:"regularization paths are descending with aligned fitted models"
    QCheck.(pair (int_range 1 100) (int_range 2 12))
    (fun (raw_slope, count) ->
      let slope = Float.of_int raw_slope /. 10.0 in
      let x =
        Result.get_ok (Matrix.of_arrays [| [| -1.0 |]; [| 0.0 |]; [| 1.0 |] |])
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let y =
        Result.get_ok
          (Target.regression (Vector.of_array [| -.slope; 0.0; slope |]))
      in
      match
        Result.bind (Lasso_path.create ~count ~tolerance:1e-10 ())
          (fun specification ->
            Lasso_path.fit specification
              ~rng:(Rng.create (Seed.of_int 12))
              ~feature_schema ~x ~y ())
      with
      | Error _ -> false
      | Ok path ->
          let alphas = Vector.to_array (Lasso_path.alphas path) in
          let coefficients = Lasso_path.coefficients path in
          let rec aligned index =
            if index = Array.length alphas then true
            else
              let descending =
                index = 0 || alphas.(index - 1) >= alphas.(index)
              in
              match Lasso_path.model path ~index with
              | Error _ -> false
              | Ok model ->
                  descending
                  && Float.abs
                       (Vector.get (Lasso_regression.coefficients model) 0
                       -. Matrix.get coefficients index 0)
                     <= 1e-12
                  && aligned (index + 1)
          in
          Array.length alphas = count
          && Matrix.shape coefficients = (count, 1)
          && Float.abs (Matrix.get coefficients 0 0) <= 1e-10
          && aligned 0)

let sgd_fit_matches_checkpoint_continuation =
  QCheck.Test.make ~count:200
    ~name:"SGD fit equals the same immutable checkpoint stream"
    QCheck.(pair (array nat_small) (int_range 1 6))
    (fun (raw, epochs) ->
      let samples = 1 + Array.length raw in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else 17 + row
               in
               Float.of_int (((value + (column * 13)) mod 21) - 10) /. 10.0))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let y =
        Result.get_ok
          (Target.regression
             (Result.get_ok
                (Vector.init ~length:samples (fun row ->
                     0.5
                     +. (1.2 *. Matrix.get x row 0)
                     -. (0.4 *. Matrix.get x row 1)))))
      in
      let specification =
        Result.get_ok
          (Sgd_regressor.create ~penalty:Sgd_regressor.Elastic_net ~alpha:0.01
             ~l1_ratio:0.25
             ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.2 })
             ~eta0:0.01 ~max_epochs:epochs ~shuffle:true ())
      in
      let rng = Rng.create (Seed.of_int 2026) in
      let rec train remaining checkpoint =
        if remaining = 0 then Ok checkpoint
        else
          Result.bind
            (Sgd_regressor.partial_fit checkpoint ~feature_schema ~x ~y ())
            (train (remaining - 1))
      in
      match
        ( Sgd_regressor.fit specification ~rng ~feature_schema ~x ~y (),
          Result.bind
            (train epochs
               (Sgd_regressor.start specification ~rng ~feature_schema))
            Sgd_regressor.to_fitted )
      with
      | Ok fitted, Ok resumed ->
          Vector.to_array (Sgd_regressor.coefficients fitted)
          = Vector.to_array (Sgd_regressor.coefficients resumed)
          && Sgd_regressor.intercept fitted = Sgd_regressor.intercept resumed
      | Error _, _ | _, Error _ -> false)

let sgd_classifier_fit_matches_checkpoint_continuation =
  QCheck.Test.make ~count:200
    ~name:"SGD classifier fit equals the same immutable checkpoint stream"
    QCheck.(pair (array nat_small) (int_range 1 6))
    (fun (raw, epochs) ->
      let samples = 3 + Array.length raw in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else 17 + row
               in
               Float.of_int (((value + (column * 13)) mod 21) - 10) /. 10.0))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let y =
        Target.classification
          (Array.init samples (fun row -> [| -3; 4; 11 |].(row mod 3)))
      in
      let specification =
        Result.get_ok
          (Sgd_classifier.create ~loss:Sgd_classifier.Log_loss
             ~penalty:Sgd_classifier.Elastic_net ~alpha:0.01 ~l1_ratio:0.25
             ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = 0.2 })
             ~eta0:0.05 ~max_epochs:epochs ~shuffle:true ())
      in
      let rng = Rng.create (Seed.of_int 2026) in
      let rec train remaining checkpoint =
        if remaining = 0 then Ok checkpoint
        else
          Result.bind
            (Sgd_classifier.partial_fit checkpoint ~feature_schema ~x ~y ())
            (train (remaining - 1))
      in
      match
        ( Sgd_classifier.fit specification ~rng ~feature_schema ~x ~y (),
          Result.bind
            (Result.bind
               (Sgd_classifier.start specification ~rng ~feature_schema
                  ~classes:[| 11; -3; 4 |])
               (train epochs))
            Sgd_classifier.to_fitted )
      with
      | Ok fitted, Ok resumed ->
          Matrix.to_arrays (Sgd_classifier.coefficients fitted)
          = Matrix.to_arrays (Sgd_classifier.coefficients resumed)
          && Vector.to_array (Sgd_classifier.intercepts fitted)
             = Vector.to_array (Sgd_classifier.intercepts resumed)
          && Sgd_classifier.classes fitted = [| -3; 4; 11 |]
      | Error _, _ | _, Error _ -> false)

let sgd_log_loss_probabilities_are_simplex =
  QCheck.Test.make ~count:200
    ~name:"SGD log-loss probabilities are bounded and sum to one"
    QCheck.(pair (array nat_small) (int_range 2 4))
    (fun (raw, class_count) ->
      let samples = class_count + Array.length raw in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else 29 + row
               in
               Float.of_int (((value + (column * 7)) mod 41) - 20) /. 4.0))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let y =
        Target.classification
          (Array.init samples (fun row -> row mod class_count))
      in
      let specification =
        Result.get_ok
          (Sgd_classifier.create ~loss:Sgd_classifier.Log_loss
             ~learning_rate:Sgd_classifier.Constant ~eta0:0.5 ~max_epochs:3 ())
      in
      match
        Result.bind
          (Sgd_classifier.fit specification
             ~rng:(Rng.create (Seed.of_int 7))
             ~feature_schema ~x ~y ())
          (fun fitted -> Sgd_classifier.predict_proba fitted ~feature_schema ~x)
      with
      | Ok probabilities ->
          let rec rows row =
            row = Matrix.rows probabilities
            ||
            let total = ref 0.0 in
            let bounded = ref true in
            for column = 0 to class_count - 1 do
              let value = Matrix.get probabilities row column in
              bounded := !bounded && value >= 0.0 && value <= 1.0;
              total := !total +. value
            done;
            !bounded && Float.abs (!total -. 1.0) <= 1e-12 && rows (row + 1)
          in
          Matrix.columns probabilities = class_count && rows 0
      | Error _ -> false)

let sgd_ordered_batches_compose =
  QCheck.Test.make ~count:200
    ~name:"SGD ordered streams give the same parameters however batches are cut"
    QCheck.(pair (array nat_small) (int_range 0 20))
    (fun (raw, cut) ->
      let samples = 4 + Array.length raw in
      let cut = 1 + (cut mod (samples - 1)) in
      let x =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns:2 (fun row column ->
               let value =
                 if row < Array.length raw then raw.(row) else 23 + row
               in
               Float.of_int (((value + (column * 11)) mod 19) - 9) /. 6.0))
      in
      let feature_schema = Result.get_ok (Feature_schema.of_matrix x) in
      let slice from count =
        Result.get_ok
          (Matrix.init ~rows:count ~columns:2 (fun row column ->
               Matrix.get x (from + row) column))
      in
      let labels = Array.init samples (fun row -> row mod 3) in
      let regression_values =
        Array.init samples (fun row ->
            0.3 +. (0.8 *. Matrix.get x row 0) -. (0.6 *. Matrix.get x row 1))
      in
      let regression from count =
        Result.get_ok
          (Target.regression
             (Vector.of_array (Array.sub regression_values from count)))
      in
      let classification from count =
        Target.classification (Array.sub labels from count)
      in
      let rng = Rng.create (Seed.of_int 11) in
      let regressor =
        Result.get_ok
          (Sgd_regressor.create ~penalty:Sgd_regressor.Elastic_net ~alpha:0.01
             ~l1_ratio:0.5
             ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.4 })
             ~eta0:0.05 ~max_epochs:1 ~shuffle:false ())
      in
      let classifier =
        Result.get_ok
          (Sgd_classifier.create ~loss:Sgd_classifier.Hinge
             ~penalty:Sgd_classifier.Elastic_net ~alpha:0.01 ~l1_ratio:0.5
             ~learning_rate:(Sgd_classifier.Inverse_scaling { power_t = 0.4 })
             ~eta0:0.05 ~max_epochs:1 ~shuffle:false ())
      in
      let ( let* ) = Result.bind in
      let regression_state fitted =
        ( Vector.to_array (Sgd_regressor.coefficients fitted),
          Sgd_regressor.intercept fitted )
      in
      let classification_state fitted =
        ( Matrix.to_arrays (Sgd_classifier.coefficients fitted),
          Vector.to_array (Sgd_classifier.intercepts fitted) )
      in
      let regression_result =
        let* whole =
          Sgd_regressor.fit regressor ~rng ~feature_schema ~x
            ~y:(regression 0 samples) ()
        in
        let* first =
          Sgd_regressor.partial_fit
            (Sgd_regressor.start regressor ~rng ~feature_schema)
            ~feature_schema ~x:(slice 0 cut) ~y:(regression 0 cut) ()
        in
        let* second =
          Sgd_regressor.partial_fit first ~feature_schema
            ~x:(slice cut (samples - cut))
            ~y:(regression cut (samples - cut))
            ()
        in
        let* chained = Sgd_regressor.to_fitted second in
        Ok (regression_state whole = regression_state chained)
      in
      let classification_result =
        let* whole =
          Sgd_classifier.fit classifier ~rng ~feature_schema ~x
            ~y:(classification 0 samples) ()
        in
        let* initial =
          Sgd_classifier.start classifier ~rng ~feature_schema
            ~classes:[| 0; 1; 2 |]
        in
        let* first =
          Sgd_classifier.partial_fit initial ~feature_schema ~x:(slice 0 cut)
            ~y:(classification 0 cut) ()
        in
        let* second =
          Sgd_classifier.partial_fit first ~feature_schema
            ~x:(slice cut (samples - cut))
            ~y:(classification cut (samples - cut))
            ()
        in
        let* chained = Sgd_classifier.to_fitted second in
        Ok (classification_state whole = classification_state chained)
      in
      match (regression_result, classification_result) with
      | Ok regression, Ok classification -> regression && classification
      | Error _, _ | _, Error _ -> false)

let balanced_class_weights_equalize_class_mass =
  QCheck.Test.make ~count:300
    ~name:"balanced class weights give every class equal total weight"
    QCheck.(pair (array nat_small) (int_range 2 4))
    (fun (raw, class_count) ->
      let samples = class_count + Array.length raw in
      let labels =
        Array.init samples (fun row ->
            if row < class_count then row
            else raw.(row - class_count) mod class_count)
      in
      let sample_weight =
        Result.get_ok
          (Sample_weight.of_array ~expected_length:samples
             (Array.init samples (fun row -> 1.0 +. Float.of_int (row mod 3))))
      in
      let y = Target.classification labels in
      match Class_weight.resolve Class_weight.balanced ~sample_weight y with
      | Error _ -> false
      | Ok resolved ->
          let totals = Array.make class_count 0.0 in
          let original = ref 0.0 in
          Array.iteri
            (fun row label ->
              totals.(label) <- totals.(label) +. Sample_weight.get resolved row;
              original := !original +. Sample_weight.get sample_weight row)
            labels;
          let resolved_total = Array.fold_left ( +. ) 0.0 totals in
          Array.for_all
            (fun total -> Float.abs (total -. totals.(0)) <= 1e-9 *. totals.(0))
            totals
          && Float.abs (resolved_total -. !original) <= 1e-9 *. !original)

let multiclass_averages_are_consistent =
  QCheck.Test.make ~count:300
    ~name:"multiclass micro scores equal accuracy and weighted recall"
    QCheck.(pair (array nat_small) (int_range 2 5))
    (fun (raw, class_count) ->
      let samples = class_count + Array.length raw in
      let truth =
        Target.classification
          (Array.init samples (fun row -> row mod class_count))
      in
      let prediction =
        Target.classification
          (Array.init samples (fun row ->
               let value = if row < Array.length raw then raw.(row) else row in
               value * 7 mod class_count))
      in
      let open Multiclass_classification_metrics in
      let undefined = Undefined_metric_policy.Use_fallback in
      match
        ( accuracy ~truth ~prediction (),
          precision ~undefined ~average:Micro ~truth ~prediction (),
          recall ~undefined ~average:Weighted ~truth ~prediction (),
          f1 ~undefined ~average:Macro ~truth ~prediction (),
          confusion_matrix ~truth ~prediction () )
      with
      | Ok accuracy, Ok micro, Ok weighted_recall, Ok macro_f1, Ok confusion ->
          let trace = ref 0.0 in
          let total = ref 0.0 in
          for row = 0 to Matrix.rows confusion.counts - 1 do
            trace := !trace +. Matrix.get confusion.counts row row;
            for column = 0 to Matrix.columns confusion.counts - 1 do
              total := !total +. Matrix.get confusion.counts row column
            done
          done;
          Float.abs (accuracy -. micro) <= 1e-12
          && Float.abs (accuracy -. weighted_recall) <= 1e-12
          && Float.abs (accuracy -. (!trace /. !total)) <= 1e-12
          && macro_f1 >= 0.0 && macro_f1 <= 1.0
          && Float.abs (!total -. Float.of_int samples) <= 1e-12
      | Error _, _, _, _, _
      | _, Error _, _, _, _
      | _, _, Error _, _, _
      | _, _, _, Error _, _
      | _, _, _, _, Error _ ->
          false)

let ranking_scores_are_bounded_and_ideal_orderings_are_perfect =
  QCheck.Test.make ~count:300
    ~name:
      "NDCG, average precision, and multiclass AUC are bounded and perfect for \
       ideal rankings"
    QCheck.(pair (array nat_small) (int_range 2 4))
    (fun (raw, columns) ->
      let rows = 1 + (Array.length raw / columns) in
      let value row column =
        let index = (row * columns) + column in
        if index < Array.length raw then raw.(index) else index * 13 mod 7
      in
      let relevance =
        Result.get_ok
          (Matrix.init ~rows ~columns (fun row column ->
               Float.of_int
                 (if column = 0 then 1 + (value row column mod 4)
                  else value row column mod 4)))
      in
      let scores =
        Result.get_ok
          (Matrix.init ~rows ~columns (fun row column ->
               Float.of_int (value (row + 1) column mod 5) /. 5.0))
      in
      let samples = rows * columns in
      let truth =
        Target.classification
          (Array.init samples (fun index -> index mod columns))
      in
      let probabilities =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns (fun row column ->
               let raw_value =
                 Float.of_int (1 + (value (row mod rows) column mod 5))
               in
               let total = ref 0.0 in
               for other = 0 to columns - 1 do
                 total :=
                   !total
                   +. Float.of_int (1 + (value (row mod rows) other mod 5))
               done;
               raw_value /. !total))
      in
      let ideal =
        Result.get_ok
          (Matrix.init ~rows:samples ~columns (fun row column ->
               if column = row mod columns then 0.7
               else 0.3 /. Float.of_int (columns - 1)))
      in
      let classes = Array.init columns Fun.id in
      let bounded value = value >= 0.0 && value <= 1.0 +. 1e-12 in
      match
        ( Ranking_metrics.ndcg ~relevance ~scores (),
          Ranking_metrics.ndcg ~relevance ~scores:relevance (),
          Multiclass_ranking.roc_auc
            ~undefined:Undefined_metric_policy.Use_fallback ~truth ~classes
            ~probabilities (),
          Multiclass_ranking.roc_auc ~strategy:Multiclass_ranking.One_vs_one
            ~truth ~classes ~probabilities:ideal (),
          Multiclass_ranking.top_k_accuracy ~k:1 ~truth ~classes
            ~probabilities:ideal () )
      with
      | Ok ndcg, Ok ideal_ndcg, Ok auc, Ok ideal_auc, Ok ideal_top ->
          bounded ndcg && bounded auc
          && Float.abs (ideal_ndcg -. 1.0) <= 1e-12
          && Float.abs (ideal_auc -. 1.0) <= 1e-12
          && Float.abs (ideal_top -. 1.0) <= 1e-12
      | Error _, _, _, _, _
      | _, Error _, _, _, _
      | _, _, Error _, _, _
      | _, _, _, Error _, _
      | _, _, _, _, Error _ ->
          false)

let ranking_curves_are_monotone =
  QCheck.Test.make ~count:500
    ~name:"ROC and precision-recall curve axes are monotone"
    QCheck.(array nat_small)
    (fun raw ->
      let samples = 4 + Array.length raw in
      let truth =
        Target.classification (Array.init samples (fun index -> index mod 2))
      in
      let probabilities =
        Vector.of_array
          (Array.init samples (fun index ->
               let value =
                 if index < Array.length raw then raw.(index) else index * 17
               in
               Float.of_int (value mod 101) /. 100.0))
      in
      match
        ( Binary_classification_metrics.roc_curve ~truth
            ~positive_probabilities:probabilities (),
          Binary_classification_metrics.precision_recall_curve ~truth
            ~positive_probabilities:probabilities () )
      with
      | Ok roc, Ok precision_recall ->
          let false_positive_rates =
            Vector.to_array
              roc.Binary_classification_metrics.false_positive_rates
          in
          let true_positive_rates =
            Vector.to_array
              roc.Binary_classification_metrics.true_positive_rates
          in
          let recalls =
            Vector.to_array
              precision_recall.Binary_classification_metrics.recalls
          in
          let nondecreasing values =
            let rec loop index =
              index = Array.length values
              || (values.(index - 1) <= values.(index) && loop (index + 1))
            in
            loop 1
          in
          let nonincreasing values =
            let rec loop index =
              index = Array.length values
              || (values.(index - 1) >= values.(index) && loop (index + 1))
            in
            loop 1
          in
          nondecreasing false_positive_rates
          && nondecreasing true_positive_rates
          && nonincreasing recalls
      | Error _, _ | _, Error _ -> false)

let special_float_gen =
  QCheck.Gen.oneof_weighted
    [
      ( 20,
        QCheck.Gen.map
          (fun value -> Float.of_int value /. 7.0)
          QCheck.Gen.int_small );
      ( 5,
        QCheck.Gen.map
          (fun value -> Float.of_int value *. 1e300)
          QCheck.Gen.int_small );
      (1, QCheck.Gen.return Float.infinity);
      (1, QCheck.Gen.return Float.neg_infinity);
      (1, QCheck.Gen.return Float.nan);
    ]

let kernel_operands =
  QCheck.make
    ~print:(fun (operand, cells) ->
      Printf.sprintf "operand=[%s] cells=[%s]"
        (String.concat ";" (List.map Float.to_string operand))
        (String.concat ";" (List.map Float.to_string cells)))
    (QCheck.Gen.pair
       (QCheck.Gen.list_size (QCheck.Gen.int_range 1 6) special_float_gen)
       (QCheck.Gen.list_size (QCheck.Gen.int_range 0 24) special_float_gen))

let same_bits expected observed =
  Int64.equal (Int64.bits_of_float expected) (Int64.bits_of_float observed)

(* Neumaier compensated summation with non-finite tracking, written out
   independently of the kernels so the property checks the documented
   reduction order rather than the implementation against itself. *)
let fold_accumulator values =
  let total = ref 0.0 and correction = ref 0.0 in
  let positive = ref false and negative = ref false and nan = ref false in
  let record value =
    if Float.is_nan value then nan := true
    else if value > 0.0 then positive := true
    else negative := true
  in
  Array.iter
    (fun value ->
      if not (Float.is_finite value) then record value
      else if not (!nan || !positive || !negative) then begin
        let next = !total +. value in
        if not (Float.is_finite next) then record next
        else begin
          let step =
            if Float.abs !total >= Float.abs value then !total -. next +. value
            else value -. next +. !total
          in
          let corrected = !correction +. step in
          if Float.is_finite corrected then correction := corrected
          else record corrected;
          total := next
        end
      end)
    values;
  if !nan || (!positive && !negative) then Float.nan
  else if !positive then Float.infinity
  else if !negative then Float.neg_infinity
  else !total +. !correction

let reference_kernels_match_accumulator_folds =
  QCheck.Test.make ~count:500
    ~name:"reference kernels match accumulator folds bit for bit"
    kernel_operands (fun (operand, cells) ->
      let columns = List.length operand in
      let rows = List.length cells / columns in
      let cells = Array.of_list cells in
      let matrix =
        Matrix.init ~rows ~columns (fun row column ->
            cells.((row * columns) + column))
        |> Result.get_ok
      in
      let operand = Vector.of_array (Array.of_list operand) in
      let row_operand =
        Vector.init ~length:rows (fun row -> Float.of_int (row - 1) /. 3.0)
        |> Result.get_ok
      in
      let csr = Csr_matrix.of_dense matrix in
      let expected_product =
        Array.init rows (fun row ->
            fold_accumulator
              (Array.init columns (fun column ->
                   Matrix.get matrix row column *. Vector.get operand column)))
      in
      let expected_transposed =
        Array.init columns (fun column ->
            fold_accumulator
              (Array.init rows (fun row ->
                   Matrix.get matrix row column *. Vector.get row_operand row)))
      in
      let expected_csr_product =
        Array.init rows (fun row ->
            let products = ref [] in
            Csr_matrix.iter_row csr ~row ~f:(fun ~column ~value ->
                products := (value *. Vector.get operand column) :: !products);
            fold_accumulator (Array.of_list (List.rev !products)))
      in
      let expected_csr_transposed =
        Array.init columns (fun column ->
            let products = ref [] in
            for row = 0 to rows - 1 do
              Csr_matrix.iter_row csr ~row ~f:(fun ~column:entry ~value ->
                  if entry = column then
                    products :=
                      (value *. Vector.get row_operand row) :: !products)
            done;
            fold_accumulator (Array.of_list (List.rev !products)))
      in
      let observed kernel = Result.get_ok kernel |> Vector.to_array in
      let all_same expected observed =
        Array.length expected = Array.length observed
        && Array.for_all2 same_bits expected observed
      in
      same_bits (fold_accumulator cells)
        (Reference_backend.sum (Vector.of_array cells))
      && same_bits
           (fold_accumulator
              (Array.init columns (fun column ->
                   Vector.get operand column *. Vector.get operand column)))
           (Result.get_ok (Reference_backend.dot operand operand))
      && all_same expected_product
           (observed (Reference_backend.matrix_vector_product matrix operand))
      && all_same expected_transposed
           (observed
              (Reference_backend.transposed_matrix_vector_product matrix
                 row_operand))
      && all_same expected_csr_product
           (observed
              (Reference_backend.feature_matrix_vector_product
                 (Feature_matrix.csr csr) operand))
      && all_same expected_csr_transposed
           (observed
              (Reference_backend.transposed_feature_matrix_vector_product
                 (Feature_matrix.csr csr) row_operand)))

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
        csr_dense_round_trip;
        reference_kernels_match_accumulator_folds;
        imputer_removes_missing_values;
        scaler_normalizes_nonconstant_columns;
        one_hot_dense_and_csr_agree;
        label_encoding_round_trip;
        l2_normalization_is_scale_invariant;
        variance_threshold_removes_constant_column;
        pipeline_matches_manual_preprocessing;
        ordinary_least_squares_recovers_exact_lines;
        lasso_matches_one_dimensional_soft_threshold;
        regularization_paths_are_descending_and_warm_started;
        sgd_fit_matches_checkpoint_continuation;
        sgd_classifier_fit_matches_checkpoint_continuation;
        sgd_log_loss_probabilities_are_simplex;
        sgd_ordered_batches_compose;
        balanced_class_weights_equalize_class_mass;
        multiclass_averages_are_consistent;
        ranking_scores_are_bounded_and_ideal_orderings_are_perfect;
        logistic_probabilities_are_complementary;
        ridge_classifier_binary_scores_are_opposites;
        ridge_classifier_is_equivariant_to_ordered_label_renaming;
        multinomial_probabilities_are_simplex_and_scores_are_centered;
        multinomial_integer_weights_match_row_replication;
        poisson_integer_weights_match_row_replication;
        k_fold_partitions_every_sample_once;
        stratified_folds_balance_each_class;
        time_series_folds_never_train_on_the_future;
        regression_losses_are_non_negative_and_consistent;
        binary_scalar_metrics_are_bounded;
        ranking_curves_are_monotone;
      ]
  in
  if failures <> 0 then exit failures
