open Modelkit_data
open Modelkit_protocols

module Univariate_selection = struct
  type selection = Count of int | Percentile of float

  module Internal = struct
    let ( let* ) = Result.bind

    let validation ~remediation name reason =
      Error (Error.make ~remediation (Error.Validation { name; reason }))

    let validate_selection = function
      | Count count when count > 0 -> Ok ()
      | Count _ ->
          validation
            ~remediation:"choose a strictly positive selected-feature count"
            "univariate selection count" "must be positive"
      | Percentile percentile
        when Float.is_finite percentile && percentile > 0.0
             && percentile <= 100.0 ->
          Ok ()
      | Percentile _ ->
          validation
            ~remediation:
              "choose a finite selected-feature percentile in (0, 100]"
            "univariate selection percentile"
            "must be finite, greater than zero, and at most 100"

    let validate_input ~operation feature_schema x =
      let* () =
        Feature_schema.validate_matrix feature_schema x
        |> Result.map_error (fun error ->
            Error.of_data_error
              ~remediation:"provide features matching the declared schema" error)
      in
      let rows = Matrix.rows x and columns = Matrix.columns x in
      let rec loop row column =
        if row = rows then Ok ()
        else if column = columns then loop (row + 1) 0
        else
          let value = Matrix.get x row column in
          if Float.is_finite value then loop row (column + 1)
          else
            Error
              (Error.of_data_error
                 ~remediation:
                   ("impute or remove non-finite values before " ^ operation)
                 (Data_error.Non_finite
                    {
                      name =
                        Printf.sprintf "%s input at row %d, column %d" operation
                          row column;
                      index = (row * columns) + column;
                      value;
                    }))
      in
      loop 0 0

    let validate_target_length ~target_length x y =
      let expected = Matrix.rows x and observed = target_length y in
      if expected = observed then Ok ()
      else
        Error
          (Error.of_data_error
             ~remediation:"provide one target value per training row"
             (Data_error.Length_mismatch
                { name = "univariate selection target"; expected; observed }))

    let reject_sample_weight = function
      | None -> Ok ()
      | Some _ ->
          validation
            ~remediation:
              "do not route sample weights to univariate selection in this \
               release"
            "univariate selection sample_weight"
            "sample weights are not supported"

    let selected_count selection columns =
      match selection with
      | Count count ->
          if count <= columns then Ok count
          else
            validation
              ~remediation:
                "choose a selected-feature count no greater than the input \
                 width"
              "univariate selection count"
              (Printf.sprintf "is %d for an input with %d features" count
                 columns)
      | Percentile percentile ->
          let count =
            int_of_float
              (Float.floor (Float.of_int columns *. percentile /. 100.0))
          in
          if count > 0 then Ok count
          else
            validation
              ~remediation:
                "increase the percentile so at least one input feature is \
                 selected"
              "univariate selection percentile"
              "selects no features at this input width"

    let choose selection scores =
      let columns = Array.length scores in
      let* count = selected_count selection columns in
      let ranked = Array.init columns Fun.id in
      Array.sort
        (fun left right ->
          let by_score = Float.compare scores.(right) scores.(left) in
          if by_score <> 0 then by_score else Int.compare left right)
        ranked;
      let selected = Array.sub ranked 0 count in
      Array.sort Int.compare selected;
      Ok selected

    let subset_schema schema selected =
      match Feature_schema.names schema with
      | None ->
          Feature_schema.anonymous ~feature_count:(Array.length selected)
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"provide a representable selected feature count"
                error)
      | Some names ->
          let values =
            Array.map
              (fun column ->
                Feature_names.get names column |> Feature_name.to_string)
              selected
          in
          Feature_names.create ~expected_count:(Array.length values) values
          |> Result.map Feature_schema.named
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"provide valid selected feature names" error)

    let transform ~operation ~expected_schema ~selected ~feature_schema ~x =
      let* () =
        if Feature_schema.equal expected_schema feature_schema then Ok ()
        else
          Error
            (Error.make
               ~remediation:"provide features with the fitted input schema"
               (Error.Feature_schema_mismatch
                  { expected = expected_schema; observed = feature_schema }))
      in
      let* () = validate_input ~operation feature_schema x in
      Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length selected)
        (fun row output_column -> Matrix.get x row selected.(output_column))
      |> Result.map_error (fun error ->
          Error.of_data_error
            ~remediation:"provide representable selected matrix dimensions"
            error)

    let finite_score value =
      if Float.is_finite value then Float.max 0.0 value else Float.max_float
  end

  module type SCORE = sig
    type target

    val operation : string
    val target_length : target -> int
    val score : Matrix.t -> target -> (float array, Error.t) result
  end

  module Make (Score : SCORE) = struct
    type params = { selection : selection }
    type t = params

    type fitted = {
      params : params;
      scores : Vector.t;
      selected : int array;
      input_schema : Feature_schema.t;
      output_schema : Feature_schema.t;
    }

    type target = Score.target
    type rng = Rng.t

    let create selection =
      let open Internal in
      let* () = validate_selection selection in
      Ok { selection }

    let clone specification = specification
    let params specification = specification

    let fit specification ?sample_weight ~rng:_ ~feature_schema ~x ~y () =
      let open Internal in
      let* () = reject_sample_weight sample_weight in
      let* () = validate_input ~operation:Score.operation feature_schema x in
      let* target =
        match y with
        | Some target -> Ok target
        | None ->
            validation
              ~remediation:
                "package this selector with Pipeline.Supervised.transformer \
                 and provide training targets"
              "univariate selection target" "is required"
      in
      let* () =
        validate_target_length ~target_length:Score.target_length x target
      in
      let* scores = Score.score x target in
      let* selected = choose specification.selection scores in
      let* output_schema = subset_schema feature_schema selected in
      Ok
        {
          params = specification;
          scores = Vector.of_array scores;
          selected;
          input_schema = feature_schema;
          output_schema;
        }

    let transform fitted ~feature_schema ~x =
      Internal.transform ~operation:Score.operation
        ~expected_schema:fitted.input_schema ~selected:fitted.selected
        ~feature_schema ~x

    let fitted_params fitted = fitted.params
    let input_schema fitted = fitted.input_schema
    let output_schema fitted = fitted.output_schema
    let scores fitted = fitted.scores
    let selected_indices fitted = Array.copy fitted.selected
  end

  module Regression_score = struct
    type target = Target.regression Target.t

    let operation = "univariate regression selection"
    let target_length = Target.length

    let score x target =
      let open Internal in
      let rows = Matrix.rows x and columns = Matrix.columns x in
      if rows < 3 then
        validation
          ~remediation:
            "provide at least three training rows for regression F-scores"
          "univariate regression samples"
          "fewer than three rows leave no residual degrees of freedom"
      else
        let y = Target.regression_values target in
        let y_scale = ref 0.0 in
        for row = 0 to rows - 1 do
          y_scale := Float.max !y_scale (Float.abs (Vector.get y row))
        done;
        let degrees_of_freedom = Float.of_int (rows - 2) in
        Ok
          (Array.init columns (fun column ->
               let x_scale = ref 0.0 in
               for row = 0 to rows - 1 do
                 x_scale :=
                   Float.max !x_scale (Float.abs (Matrix.get x row column))
               done;
               if !x_scale = 0.0 || !y_scale = 0.0 then 0.0
               else
                 let mean_x = ref 0.0
                 and mean_y = ref 0.0
                 and sum_xx = ref 0.0
                 and sum_yy = ref 0.0
                 and sum_xy = ref 0.0 in
                 for row = 0 to rows - 1 do
                   let count = Float.of_int (row + 1) in
                   let x_value = Matrix.get x row column /. !x_scale in
                   let y_value = Vector.get y row /. !y_scale in
                   let delta_x = x_value -. !mean_x in
                   let delta_y = y_value -. !mean_y in
                   mean_x := !mean_x +. (delta_x /. count);
                   mean_y := !mean_y +. (delta_y /. count);
                   sum_xx := !sum_xx +. (delta_x *. (x_value -. !mean_x));
                   sum_yy := !sum_yy +. (delta_y *. (y_value -. !mean_y));
                   sum_xy := !sum_xy +. (delta_x *. (y_value -. !mean_y))
                 done;
                 if !sum_xx <= 0.0 || !sum_yy <= 0.0 then 0.0
                 else
                   let squared_correlation =
                     Float.min 1.0
                       (Float.max 0.0
                          (!sum_xy *. !sum_xy /. (!sum_xx *. !sum_yy)))
                   in
                   if squared_correlation >= 1.0 then Float.max_float
                   else
                     finite_score
                       (degrees_of_freedom *. squared_correlation
                       /. (1.0 -. squared_correlation))))
  end

  module Classification_score = struct
    type target = Target.classification Target.t

    let operation = "univariate classification selection"
    let target_length = Target.length

    let score x target =
      let open Internal in
      let labels = Target.classification_values target in
      let rows = Matrix.rows x and columns = Matrix.columns x in
      let sorted = Array.copy labels in
      Array.sort Int.compare sorted;
      let classes =
        Array.fold_left
          (fun classes label ->
            match classes with
            | previous :: _ when previous = label -> classes
            | _ -> label :: classes)
          [] sorted
        |> List.rev |> Array.of_list
      in
      let class_count = Array.length classes in
      if class_count < 2 then
        validation
          ~remediation:
            "provide training rows from at least two distinct classes"
          "univariate classification classes" "fewer than two classes observed"
      else if rows <= class_count then
        validation
          ~remediation:
            "provide more training rows than observed classes for ANOVA \
             F-scores"
          "univariate classification samples"
          "no residual degrees of freedom remain"
      else
        let class_indices = Hashtbl.create class_count in
        Array.iteri
          (fun index label -> Hashtbl.add class_indices label index)
          classes;
        let class_of_row =
          Array.map (fun label -> Hashtbl.find class_indices label) labels
        in
        let between_df = Float.of_int (class_count - 1) in
        let within_df = Float.of_int (rows - class_count) in
        Ok
          (Array.init columns (fun column ->
               let scale = ref 0.0 in
               for row = 0 to rows - 1 do
                 scale := Float.max !scale (Float.abs (Matrix.get x row column))
               done;
               if !scale = 0.0 then 0.0
               else
                 let counts = Array.make class_count 0 in
                 let means = Array.make class_count 0.0 in
                 let within = Array.make class_count 0.0 in
                 for row = 0 to rows - 1 do
                   let group = class_of_row.(row) in
                   counts.(group) <- counts.(group) + 1;
                   let count = Float.of_int counts.(group) in
                   let value = Matrix.get x row column /. !scale in
                   let delta = value -. means.(group) in
                   means.(group) <- means.(group) +. (delta /. count);
                   within.(group) <-
                     within.(group) +. (delta *. (value -. means.(group)))
                 done;
                 let grand_mean =
                   Array.fold_left
                     (fun total group ->
                       total +. (Float.of_int counts.(group) *. means.(group)))
                     0.0
                     (Array.init class_count Fun.id)
                   /. Float.of_int rows
                 in
                 let between_sum = ref 0.0 and within_sum = ref 0.0 in
                 for group = 0 to class_count - 1 do
                   let delta = means.(group) -. grand_mean in
                   between_sum :=
                     !between_sum
                     +. (Float.of_int counts.(group) *. delta *. delta);
                   within_sum := !within_sum +. within.(group)
                 done;
                 if !between_sum <= 0.0 then 0.0
                 else if !within_sum <= 0.0 then Float.max_float
                 else
                   finite_score
                     (!between_sum /. between_df /. (!within_sum /. within_df))))
  end

  module Regression = Make (Regression_score)
  module Classification = Make (Classification_score)
end

module Feature_importance = struct
  type coefficient_norm = L1 | L2 | Max

  let numerical reason =
    Error
      (Error.make
         ~remediation:
           "provide finite coefficients whose reduced importance is \
            representable as float64"
         (Error.Numerical { operation = "feature importance"; reason }))

  let validate_coefficient ~row ~column value =
    if Float.is_finite value then Ok ()
    else
      numerical
        (Printf.sprintf "coefficient at row %d, column %d is not finite" row
           column)

  let absolute_coefficients coefficients =
    let values = Vector.to_array coefficients in
    let rec validate index =
      if index = Array.length values then Ok (Vector.of_array values)
      else
        let value = values.(index) in
        if not (Float.is_finite value) then
          numerical
            (Printf.sprintf "coefficient at index %d is not finite" index)
        else (
          values.(index) <- Float.abs value;
          validate (index + 1))
    in
    validate 0

  let l1 matrix column =
    let total = ref 0.0 in
    let rec loop row =
      if row = Matrix.rows matrix then
        if Float.is_finite !total then Ok !total
        else numerical (Printf.sprintf "L1 norm overflowed at column %d" column)
      else
        let value = Matrix.get matrix row column in
        match validate_coefficient ~row ~column value with
        | Error _ as error -> error
        | Ok () ->
            total := !total +. Float.abs value;
            loop (row + 1)
    in
    loop 0

  let l2 matrix column =
    let scale = ref 0.0 and sum_squares = ref 1.0 in
    let rec loop row =
      if row = Matrix.rows matrix then
        let value =
          if !scale = 0.0 then 0.0 else !scale *. Float.sqrt !sum_squares
        in
        if Float.is_finite value then Ok value
        else numerical (Printf.sprintf "L2 norm overflowed at column %d" column)
      else
        let value = Matrix.get matrix row column in
        match validate_coefficient ~row ~column value with
        | Error _ as error -> error
        | Ok () ->
            let absolute = Float.abs value in
            (if absolute <> 0.0 then
               if !scale < absolute then (
                 let ratio = !scale /. absolute in
                 sum_squares := 1.0 +. (!sum_squares *. ratio *. ratio);
                 scale := absolute)
               else
                 let ratio = absolute /. !scale in
                 sum_squares := !sum_squares +. (ratio *. ratio));
            loop (row + 1)
    in
    loop 0

  let maximum matrix column =
    let result = ref 0.0 in
    let rec loop row =
      if row = Matrix.rows matrix then Ok !result
      else
        let value = Matrix.get matrix row column in
        match validate_coefficient ~row ~column value with
        | Error _ as error -> error
        | Ok () ->
            result := Float.max !result (Float.abs value);
            loop (row + 1)
    in
    loop 0

  let coefficient_norms ?(norm = L1) coefficients =
    let rows = Matrix.rows coefficients
    and columns = Matrix.columns coefficients in
    if rows = 0 then
      Error
        (Error.make ~remediation:"provide at least one fitted coefficient row"
           (Error.Validation
              {
                name = "coefficient importance rows";
                reason = "must not be empty";
              }))
    else
      let values = Array.make columns 0.0 in
      let rec reduce column =
        if column = columns then Ok (Vector.of_array values)
        else
          let result =
            match norm with
            | L1 -> l1 coefficients column
            | L2 -> l2 coefficients column
            | Max -> maximum coefficients column
          in
          match result with
          | Error _ as error -> error
          | Ok value ->
              values.(column) <- value;
              reduce (column + 1)
      in
      reduce 0
end

module Select_from_model = struct
  type threshold = Mean | Median | Value of float

  module Make (Estimator : IMPORTANCE_ESTIMATOR with type rng = Rng.t) = struct
    type params = {
      threshold : threshold;
      max_features : int option;
      estimator_params : Estimator.params;
    }

    type t = {
      specification_params : params;
      estimator_specification : Estimator.t;
    }

    type fitted = {
      fitted_params_value : params;
      fitted_model : Estimator.fitted;
      importances : Vector.t;
      threshold_value : float;
      selected : int array;
      input_schema : Feature_schema.t;
      output_schema : Feature_schema.t;
    }

    type target = Estimator.target
    type rng = Rng.t

    let ( let* ) = Result.bind

    let validation ~remediation name reason =
      Error (Error.make ~remediation (Error.Validation { name; reason }))

    let validate_threshold = function
      | Mean | Median -> Ok ()
      | Value value when Float.is_finite value && value >= 0.0 -> Ok ()
      | Value _ ->
          validation
            ~remediation:"choose a finite, non-negative importance threshold"
            "model selection threshold" "must be finite and non-negative"

    let validate_max_features = function
      | None -> Ok ()
      | Some count when count > 0 -> Ok ()
      | Some _ ->
          validation
            ~remediation:"choose a strictly positive maximum feature count"
            "model selection max_features" "must be positive"

    let create ?(threshold = Mean) ?max_features estimator =
      let* () = validate_threshold threshold in
      let* () = validate_max_features max_features in
      Ok
        ({
           specification_params =
             {
               threshold;
               max_features;
               estimator_params = Estimator.params estimator;
             };
           estimator_specification = estimator;
         }
          : t)

    let clone (specification : t) =
      let estimator = Estimator.clone specification.estimator_specification in
      ({
         specification_params =
           {
             specification.specification_params with
             estimator_params = Estimator.params estimator;
           };
         estimator_specification = estimator;
       }
        : t)

    let params (specification : t) = specification.specification_params

    let validate_importances ~columns importances =
      let observed = Vector.length importances in
      if observed <> columns then
        Error
          (Error.make
             ~remediation:
               "return exactly one importance for every fitted input feature"
             (Error.Shape_mismatch
                {
                  name = "fitted feature importances";
                  expected = [ columns ];
                  observed = [ observed ];
                }))
      else
        let rec loop column =
          if column = columns then Ok ()
          else
            let value = Vector.get importances column in
            if Float.is_finite value && value >= 0.0 then loop (column + 1)
            else
              validation
                ~remediation:
                  "return finite, non-negative fitted feature importances"
                "fitted feature importance"
                (Printf.sprintf "value at column %d is %g" column value)
        in
        loop 0

    let resolve_threshold threshold importances =
      let values = Vector.to_array importances in
      match threshold with
      | Value value -> value
      | Mean ->
          let mean = ref 0.0 in
          Array.iteri
            (fun index value ->
              mean := !mean +. ((value -. !mean) /. Float.of_int (index + 1)))
            values;
          !mean
      | Median ->
          Array.sort Float.compare values;
          let length = Array.length values in
          if length mod 2 = 1 then values.(length / 2)
          else
            let lower = values.((length / 2) - 1) in
            let upper = values.(length / 2) in
            lower +. ((upper -. lower) /. 2.0)

    let select ~threshold ~max_features importances =
      let eligible = ref [] in
      for column = 0 to Vector.length importances - 1 do
        if Vector.get importances column >= threshold then
          eligible := column :: !eligible
      done;
      let eligible = Array.of_list (List.rev !eligible) in
      if Array.length eligible = 0 then
        validation
          ~remediation:
            "lower the threshold or fit an estimator with nonzero feature \
             importances"
          "model-based feature selection"
          "no fitted feature importance meets the threshold"
      else
        let retained =
          match max_features with
          | None -> eligible
          | Some count when Array.length eligible <= count -> eligible
          | Some count ->
              Array.sort
                (fun left right ->
                  let by_importance =
                    Float.compare
                      (Vector.get importances right)
                      (Vector.get importances left)
                  in
                  if by_importance <> 0 then by_importance
                  else Int.compare left right)
                eligible;
              Array.sub eligible 0 count
        in
        Array.sort Int.compare retained;
        Ok retained

    let fit (specification : t) ?sample_weight ~rng ~feature_schema ~x ~y () =
      let module Internal = Univariate_selection.Internal in
      let* () =
        Internal.validate_input ~operation:"model-based feature selection"
          feature_schema x
      in
      let* () =
        if Matrix.columns x > 0 then Ok ()
        else
          validation
            ~remediation:
              "provide at least one input feature for model-based selection"
            "model-based selection features" "input width is zero"
      in
      let* () =
        match sample_weight with
        | None -> Ok ()
        | Some weights ->
            let expected = Matrix.rows x in
            let observed = Sample_weight.length weights in
            if expected = observed then Ok ()
            else
              Error
                (Error.of_data_error
                   ~remediation:
                     "provide one sample weight per model-selection training \
                      row"
                   (Data_error.Length_mismatch
                      {
                        name = "model-based selection sample weights";
                        expected;
                        observed;
                      }))
      in
      let* target =
        match y with
        | Some target -> Ok target
        | None ->
            validation
              ~remediation:
                "package this selector with Pipeline.Supervised.transformer \
                 and provide training targets"
              "model-based selection target" "is required"
      in
      let* estimator =
        Estimator.fit
          (Estimator.clone specification.estimator_specification)
          ?sample_weight ~rng ~feature_schema ~x ~y:target ()
      in
      let estimator_schema = Estimator.feature_schema estimator in
      let* () =
        if Feature_schema.equal estimator_schema feature_schema then Ok ()
        else
          Error
            (Error.make
               ~remediation:
                 "ensure the importance estimator reports its fitted input \
                  schema"
               (Error.Compatibility
                  {
                    component = "importance estimator";
                    reason = "reported a different fitted feature schema";
                  }))
      in
      let* importances = Estimator.feature_importances estimator in
      let* () = validate_importances ~columns:(Matrix.columns x) importances in
      let threshold_value =
        resolve_threshold specification.specification_params.threshold
          importances
      in
      let* selected =
        select ~threshold:threshold_value
          ~max_features:specification.specification_params.max_features
          importances
      in
      let* output_schema = Internal.subset_schema feature_schema selected in
      Ok
        ({
           fitted_params_value =
             {
               specification.specification_params with
               estimator_params = Estimator.fitted_params estimator;
             };
           fitted_model = estimator;
           importances;
           threshold_value;
           selected;
           input_schema = feature_schema;
           output_schema;
         }
          : fitted)

    let transform (fitted : fitted) ~feature_schema ~x =
      Univariate_selection.Internal.transform
        ~operation:"model-based feature selection"
        ~expected_schema:fitted.input_schema ~selected:fitted.selected
        ~feature_schema ~x

    let fitted_params (fitted : fitted) = fitted.fitted_params_value
    let input_schema (fitted : fitted) = fitted.input_schema
    let output_schema (fitted : fitted) = fitted.output_schema
    let importances (fitted : fitted) = fitted.importances
    let threshold_value (fitted : fitted) = fitted.threshold_value
    let selected_indices (fitted : fitted) = Array.copy fitted.selected
    let fitted_estimator (fitted : fitted) = fitted.fitted_model
  end
end

module Recursive_feature_elimination = struct
  type step = Count of int | Fraction of float

  module Make (Estimator : IMPORTANCE_ESTIMATOR with type rng = Rng.t) = struct
    type params = {
      feature_count : int;
      step : step;
      estimator_params : Estimator.params;
    }

    type t = {
      specification_params : params;
      estimator_specification : Estimator.t;
    }

    type fitted = {
      fitted_params_value : params;
      fitted_model : Estimator.fitted;
      final_importances : Vector.t;
      selected : int array;
      feature_ranking : int array;
      input_schema : Feature_schema.t;
      output_schema : Feature_schema.t;
    }

    type path_point = {
      path_estimator : Estimator.fitted;
      path_importances : Vector.t;
      path_selected : int array;
      path_schema : Feature_schema.t;
    }

    type target = Estimator.target
    type rng = Rng.t

    let ( let* ) = Result.bind

    let validation ~remediation name reason =
      Error (Error.make ~remediation (Error.Validation { name; reason }))

    let validate_feature_count count =
      if count > 0 then Ok ()
      else
        validation
          ~remediation:"choose a strictly positive selected-feature count"
          "recursive feature elimination feature_count" "must be positive"

    let validate_step = function
      | Count count when count > 0 -> Ok ()
      | Count _ ->
          validation
            ~remediation:"choose a strictly positive integer elimination step"
            "recursive feature elimination step" "must be positive"
      | Fraction fraction
        when Float.is_finite fraction && fraction > 0.0 && fraction < 1.0 ->
          Ok ()
      | Fraction _ ->
          validation
            ~remediation:
              "choose a finite fractional elimination step strictly between 0 \
               and 1"
            "recursive feature elimination step"
            "must be finite and strictly between zero and one"

    let create ?(step = Count 1) ~feature_count estimator =
      let* () = validate_feature_count feature_count in
      let* () = validate_step step in
      Ok
        ({
           specification_params =
             {
               feature_count;
               step;
               estimator_params = Estimator.params estimator;
             };
           estimator_specification = estimator;
         }
          : t)

    let clone (specification : t) =
      let estimator = Estimator.clone specification.estimator_specification in
      ({
         specification_params =
           {
             specification.specification_params with
             estimator_params = Estimator.params estimator;
           };
         estimator_specification = estimator;
       }
        : t)

    let params (specification : t) = specification.specification_params

    let validate_importances ~columns importances =
      let observed = Vector.length importances in
      if observed <> columns then
        Error
          (Error.make
             ~remediation:
               "return exactly one importance for every fitted input feature"
             (Error.Shape_mismatch
                {
                  name = "recursive elimination fitted feature importances";
                  expected = [ columns ];
                  observed = [ observed ];
                }))
      else
        let rec loop column =
          if column = columns then Ok ()
          else
            let value = Vector.get importances column in
            if Float.is_finite value && value >= 0.0 then loop (column + 1)
            else
              validation
                ~remediation:
                  "return finite, non-negative fitted feature importances"
                "recursive elimination fitted feature importance"
                (Printf.sprintf "value at column %d is %g" column value)
        in
        loop 0

    let validate_estimator_schema ~expected estimator =
      let observed = Estimator.feature_schema estimator in
      if Feature_schema.equal expected observed then Ok ()
      else
        Error
          (Error.make
             ~remediation:
               "ensure the importance estimator reports its fitted input schema"
             (Error.Compatibility
                {
                  component = "recursive elimination importance estimator";
                  reason = "reported a different fitted feature schema";
                }))

    let validate_sample_weight ~rows = function
      | None -> Ok ()
      | Some weights ->
          let observed = Sample_weight.length weights in
          if rows = observed then Ok ()
          else
            Error
              (Error.of_data_error
                 ~remediation:
                   "provide one sample weight per recursive-elimination \
                    training row"
                 (Data_error.Length_mismatch
                    {
                      name = "recursive feature elimination sample weights";
                      expected = rows;
                      observed;
                    }))

    let resolved_step step columns =
      match step with
      | Count count -> count
      | Fraction fraction ->
          Int.max 1
            (int_of_float (Float.floor (fraction *. Float.of_int columns)))

    let remove_weakest ~count ~active ~importances =
      let ranked = Array.init (Array.length active) Fun.id in
      Array.sort
        (fun left right ->
          let by_importance =
            Float.compare
              (Vector.get importances left)
              (Vector.get importances right)
          in
          if by_importance <> 0 then by_importance
          else Int.compare active.(left) active.(right))
        ranked;
      let removed = Array.make (Array.length active) false in
      for rank = 0 to count - 1 do
        removed.(ranked.(rank)) <- true
      done;
      let retained = Array.make (Array.length active - count) 0 in
      let output = ref 0 in
      Array.iteri
        (fun position column ->
          if not removed.(position) then (
            retained.(!output) <- column;
            incr output))
        active;
      retained

    let fit_path (specification : t) ?sample_weight ~rng ~feature_schema ~x ~y
        () =
      let module Internal = Univariate_selection.Internal in
      let operation = "recursive feature elimination" in
      let* () = Internal.validate_input ~operation feature_schema x in
      let columns = Matrix.columns x in
      let* () =
        if columns > 0 then Ok ()
        else
          validation
            ~remediation:
              "provide at least one input feature for recursive elimination"
            "recursive feature elimination features" "input width is zero"
      in
      let desired = specification.specification_params.feature_count in
      let* () =
        if desired <= columns then Ok ()
        else
          validation
            ~remediation:
              "choose a selected-feature count no greater than the input width"
            "recursive feature elimination feature_count"
            (Printf.sprintf "is %d for an input with %d features" desired
               columns)
      in
      let* () = validate_sample_weight ~rows:(Matrix.rows x) sample_weight in
      let* target =
        match y with
        | Some target -> Ok target
        | None ->
            validation
              ~remediation:
                "package this selector with Pipeline.Supervised.transformer \
                 and provide training targets"
              "recursive feature elimination target" "is required"
      in
      let step =
        resolved_step specification.specification_params.step columns
      in
      let ranking = Array.make columns 1 in
      let root_seed = Rng.to_seed rng in
      let rec eliminate round reversed active =
        let* active_schema =
          if Array.length active = columns then Ok feature_schema
          else Internal.subset_schema feature_schema active
        in
        let* active_x =
          if Array.length active = columns then Ok x
          else
            Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length active)
              (fun row output_column -> Matrix.get x row active.(output_column))
            |> Result.map_error (fun error ->
                Error.of_data_error
                  ~remediation:
                    "provide representable recursive-elimination matrix \
                     dimensions"
                  error)
        in
        let round_rng =
          Rng.create
            (Seed.derive root_seed
               ~operation:"recursive-feature-elimination-round" ~index:round)
        in
        let* estimator =
          Estimator.fit
            (Estimator.clone specification.estimator_specification)
            ?sample_weight ~rng:round_rng ~feature_schema:active_schema
            ~x:active_x ~y:target ()
          |> Result.map_error
               (Error.with_context
                  (Error.Stage
                     (Printf.sprintf "recursive elimination round %d" round)))
        in
        let* () = validate_estimator_schema ~expected:active_schema estimator in
        let* importances = Estimator.feature_importances estimator in
        let* () =
          validate_importances ~columns:(Array.length active) importances
        in
        let point =
          {
            path_estimator = estimator;
            path_importances = importances;
            path_selected = active;
            path_schema = active_schema;
          }
        in
        if Array.length active = desired then
          Ok (Array.of_list (List.rev (point :: reversed)), ranking)
        else
          let remove_count = Int.min step (Array.length active - desired) in
          let next = remove_weakest ~count:remove_count ~active ~importances in
          let retained = Array.make columns false in
          Array.iter (fun column -> retained.(column) <- true) next;
          for column = 0 to columns - 1 do
            if not retained.(column) then
              ranking.(column) <- ranking.(column) + 1
          done;
          eliminate (round + 1) (point :: reversed) next
      in
      eliminate 0 [] (Array.init columns Fun.id)

    let fit (specification : t) ?sample_weight ~rng ~feature_schema ~x ~y () =
      let* path, ranking =
        fit_path specification ?sample_weight ~rng ~feature_schema ~x ~y ()
      in
      let final = path.(Array.length path - 1) in
      let selected = final.path_selected in
      let output_schema = final.path_schema in
      let estimator = final.path_estimator in
      let final_importances = final.path_importances in
      Ok
        ({
           fitted_params_value =
             {
               specification.specification_params with
               estimator_params = Estimator.fitted_params estimator;
             };
           fitted_model = estimator;
           final_importances;
           selected;
           feature_ranking = ranking;
           input_schema = feature_schema;
           output_schema;
         }
          : fitted)

    module Internal = struct
      type nonrec path_point = path_point

      let fit_path = fit_path
      let estimator point = point.path_estimator
      let importances point = point.path_importances
      let selected_indices point = Array.copy point.path_selected
      let feature_schema point = point.path_schema
    end

    let transform (fitted : fitted) ~feature_schema ~x =
      Univariate_selection.Internal.transform
        ~operation:"recursive feature elimination"
        ~expected_schema:fitted.input_schema ~selected:fitted.selected
        ~feature_schema ~x

    let fitted_params (fitted : fitted) = fitted.fitted_params_value
    let input_schema (fitted : fitted) = fitted.input_schema
    let output_schema (fitted : fitted) = fitted.output_schema
    let selected_indices (fitted : fitted) = Array.copy fitted.selected
    let ranking (fitted : fitted) = Array.copy fitted.feature_ranking
    let final_importances (fitted : fitted) = fitted.final_importances
    let fitted_estimator (fitted : fitted) = fitted.fitted_model
  end
end
