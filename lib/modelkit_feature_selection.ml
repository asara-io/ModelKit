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
