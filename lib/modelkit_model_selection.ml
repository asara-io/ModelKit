open Modelkit_data
open Modelkit_metadata
module Callback = Modelkit_callback.Callback
open Modelkit_protocols
open Modelkit_pipeline
open Modelkit_metrics
open Modelkit_splitting

module Cross_validation = struct
  type failure_policy = Abort | Record
  type partition = Train | Test

  type failure_phase =
    | Materialization
    | Fitting
    | Prediction of partition
    | Scoring of { partition : partition; scorer : string }

  type failure = { phase : failure_phase; error : Error.t }

  type score = {
    name : string;
    train_score : (float, Error.t) result option;
    test_score : (float, Error.t) result option;
  }

  type 'model fold = {
    fold_index : int;
    fit_time : float;
    score_time : float;
    scores : score array;
    model : 'model option;
    train_indices : int array option;
    test_indices : int array option;
    failures : failure array;
  }

  type 'model report = { report_folds : 'model fold array }

  type 'target splitter = {
    run_splitter :
      rng:Rng.t ->
      groups:Groups.t option ->
      x:Matrix.t ->
      y:'target ->
      ((Row_view.t * Row_view.t) array, Error.t) result;
  }

  let target_independent_splitter (type specification)
      (module Splitter : SPLITTER
        with type t = specification
         and type target = unit
         and type rng = Rng.t) (specification : specification) =
    {
      run_splitter =
        (fun ~rng ~groups ~x ~y:_ ->
          Splitter.split specification ~rng ?groups ~x ~y:None ());
    }

  let target_aware_splitter (type specification target)
      (module Splitter : SPLITTER
        with type t = specification
         and type target = target
         and type rng = Rng.t) (specification : specification) =
    {
      run_splitter =
        (fun ~rng ~groups ~x ~(y : target) ->
          Splitter.split specification ~rng ?groups ~x ~y:(Some y) ());
    }

  let copy_fold fold =
    {
      fold with
      scores = Array.copy fold.scores;
      train_indices = Option.map Array.copy fold.train_indices;
      test_indices = Option.map Array.copy fold.test_indices;
      failures = Array.copy fold.failures;
    }

  let folds report = Array.map copy_fold report.report_folds

  let successful_fold_count report =
    Array.fold_left
      (fun count fold ->
        let scores_succeeded =
          Array.for_all
            (fun score ->
              match score.test_score with
              | Some (Ok _) -> true
              | None | Some (Error _) -> false)
            fold.scores
        in
        if Array.length fold.failures = 0 && scores_succeeded then count + 1
        else count)
      0 report.report_folds

  let[@warning "-4"] contextualize fold error =
    match Error.context error with
    | Error.Fold index :: _ when index = fold -> error
    | _ -> Error.with_context (Error.Fold fold) error

  let scorer_error fold scorer error =
    error
    |> Error.with_context (Error.Stage scorer)
    |> Error.with_context (Error.Fold fold)

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let validate_scorers names =
    if Array.length names = 0 then
      Error
        (validation ~name:"cross-validation scorers"
           ~reason:"at least one scorer is required"
           ~remediation:"provide one or more uniquely named scorers")
    else
      let seen = Hashtbl.create (Array.length names) in
      let rec loop index =
        if index = Array.length names then Ok ()
        else
          let name = names.(index) in
          if String.length (String.trim name) = 0 then
            Error
              (validation ~name:"cross-validation scorer name"
                 ~reason:"scorer names must not be blank"
                 ~remediation:"use a scorer with a non-empty stable name")
          else if Hashtbl.mem seen name then
            Error
              (validation ~name:"cross-validation scorers"
                 ~reason:(Format.sprintf "scorer name %S is duplicated" name)
                 ~remediation:
                   "provide at most one scorer for each report field name")
          else (
            Hashtbl.add seen name ();
            loop (index + 1))
      in
      loop 0

  let empty_scores ~return_train_score:_ names =
    Array.map
      (fun name -> { name; train_score = None; test_score = None })
      names

  let retain_indices return_indices split =
    if return_indices then
      ( Some (Row_view.indices (Split.train split)),
        Some (Row_view.indices (Split.test split)) )
    else (None, None)

  let timed operation =
    let started = Sys.time () in
    let result = operation () in
    (Sys.time () -. started, result)

  let run ~return_train_score ~return_models ~return_indices ~failure_policy
      ~fit_seed ~execution ~metadata ~splitter ~scorer_names ~seed ~score_model
      pipeline dataset =
    let ( let* ) = Result.bind in
    let* () = validate_scorers scorer_names in
    let* () = Metadata.validate ~rows:(Dataset.sample_count dataset) metadata in
    Callback.run
      ~outcome:(fun report ->
        let failure =
          Array.to_list report.report_folds
          |> List.find_map (fun fold ->
              if Array.length fold.failures = 0 then None
              else Some fold.failures.(0).error)
        in
        match failure with
        | None -> Callback.Succeeded
        | Some error -> Callback.Failed error)
      (Metadata.callback metadata)
      ~operation:Callback.Cross_validation
      (fun () ->
        let splitter_rng =
          Seed.derive seed ~operation:"cross-validation-splitter" ~index:0
          |> Rng.create
        in
        let* view_pairs =
          splitter.run_splitter ~rng:splitter_rng
            ~groups:(Dataset.groups dataset) ~x:(Dataset.features dataset)
            ~y:(Dataset.target dataset)
        in
        let* splits =
          let rec validate index reversed =
            if index = Array.length view_pairs then
              Ok (Array.of_list (List.rev reversed))
            else
              let train, test = view_pairs.(index) in
              match Split.of_views ~train ~test with
              | Ok split -> validate (index + 1) (split :: reversed)
              | Error error -> Error (contextualize index error)
          in
          validate 0 []
        in
        let evaluate metadata ~index:fold_index split =
          let train_indices, test_indices =
            retain_indices return_indices split
          in
          let materialized =
            let* train, test = Split.materialize dataset split in
            let* train_metadata =
              Metadata.select metadata (Split.train split)
            in
            let* test_metadata = Metadata.select metadata (Split.test split) in
            Ok (train, test, train_metadata, test_metadata)
          in
          match materialized with
          | Error error ->
              let error = contextualize fold_index error in
              if failure_policy = Abort || Callback.is_control_error error then
                Error error
              else
                Ok
                  {
                    fold_index;
                    fit_time = 0.0;
                    score_time = 0.0;
                    scores = empty_scores ~return_train_score scorer_names;
                    model = None;
                    train_indices;
                    test_indices;
                    failures = [| { phase = Materialization; error } |];
                  }
          | Ok (train, test, train_metadata, test_metadata) -> (
              let fold_rng =
                Seed.derive fit_seed ~operation:"cross-validation-fold"
                  ~index:fold_index
                |> Rng.create
              in
              let fit_time, fitted =
                timed (fun () ->
                    Pipeline.fit_with_metadata (Pipeline.clone pipeline)
                      ~metadata:train_metadata ~rng:fold_rng
                      ~feature_schema:(Dataset.feature_schema train)
                      ~x:(Dataset.features train) ~y:(Dataset.target train) ())
              in
              match fitted with
              | Error error ->
                  let error = contextualize fold_index error in
                  if failure_policy = Abort || Callback.is_control_error error
                  then Error error
                  else
                    Ok
                      {
                        fold_index;
                        fit_time;
                        score_time = 0.0;
                        scores = empty_scores ~return_train_score scorer_names;
                        model = None;
                        train_indices;
                        test_indices;
                        failures = [| { phase = Fitting; error } |];
                      }
              | Ok fitted ->
                  let score_time, scored =
                    timed (fun () ->
                        score_model ~fold_index ~return_train_score
                          ~failure_policy ~train_metadata ~test_metadata fitted
                          train test)
                  in
                  let* scores, failures = scored in
                  Ok
                    {
                      fold_index;
                      fit_time;
                      score_time;
                      scores;
                      model = (if return_models then Some fitted else None);
                      train_indices;
                      test_indices;
                      failures;
                    })
        in
        let run_fold metadata ~index split =
          let metadata = Metadata.scope (Error.Fold index) metadata in
          Callback.run
            ~outcome:(fun fold ->
              if Array.length fold.failures = 0 then Callback.Succeeded
              else Callback.Failed fold.failures.(0).error)
            (Metadata.callback metadata)
            ~operation:Callback.Fold
            (fun () -> evaluate metadata ~index split)
          |> Result.map_error (contextualize index)
        in
        let* report_folds =
          match Metadata.callback metadata with
          | None -> Execution.map execution ~f:(run_fold metadata) splits
          | Some callback ->
              let batch_size = max 1 (Execution.concurrency execution) in
              let rec batches offset reversed =
                if offset = Array.length splits then
                  Ok (Array.of_list (List.rev reversed))
                else
                  let count = min batch_size (Array.length splits - offset) in
                  let batch = Array.sub splits offset count in
                  let* outcomes =
                    Execution.map execution batch ~f:(fun ~index split ->
                        let buffered, flush = Callback.buffer callback in
                        let scoped =
                          Metadata.with_callback metadata (Some buffered)
                        in
                        let result =
                          run_fold scoped ~index:(offset + index) split
                        in
                        Ok (result, flush))
                  in
                  let* reversed =
                    Array.fold_left
                      (fun accumulated (result, flush) ->
                        let* accumulated = accumulated in
                        let* () = flush () in
                        let* fold = result in
                        Ok (fold :: accumulated))
                      (Ok reversed) outcomes
                  in
                  batches (offset + count) reversed
              in
              batches 0 []
        in
        Ok { report_folds })

  let failure ~phase error = { phase; error }

  let finish_scoring failure_policy scores failures =
    match
      List.find_opt
        (fun failure -> Callback.is_control_error failure.error)
        failures
    with
    | Some failure -> Error failure.error
    | None -> (
        match (failure_policy, failures) with
        | Abort, first :: _ -> Error first.error
        | (Abort | Record), _ -> Ok (scores, Array.of_list failures))

  module Regression = struct
    type model =
      (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

    let score_partition ~fold_index ~partition scorers dataset prediction =
      Array.map
        (fun scorer ->
          let name = Regression_scorer.name scorer in
          let result =
            Regression_scorer.score scorer
              ?sample_weight:(Dataset.sample_weight dataset)
              ~truth:(Dataset.target dataset) ~prediction ()
            |> Result.map_error (scorer_error fold_index name)
          in
          let failures =
            match result with
            | Ok _ -> []
            | Error error ->
                [ failure ~phase:(Scoring { partition; scorer = name }) error ]
          in
          (result, failures))
        scorers

    let score_model scorers ~fold_index ~return_train_score ~failure_policy
        ~train_metadata ~test_metadata fitted train test =
      let predict partition metadata dataset =
        Pipeline.predict_with_metadata fitted ~metadata
          ~feature_schema:(Dataset.feature_schema dataset)
          ~x:(Dataset.features dataset)
        |> Result.map_error (contextualize fold_index)
        |> fun result ->
        let failures =
          match result with
          | Ok _ -> []
          | Error error -> [ failure ~phase:(Prediction partition) error ]
        in
        (result, failures)
      in
      let train_prediction, train_prediction_failures =
        if return_train_score then predict Train train_metadata train
        else
          ( Error
              (validation ~name:"unused training prediction"
                 ~reason:"not requested"
                 ~remediation:
                   "request train scores to compute training predictions"),
            [] )
      in
      let test_prediction, test_prediction_failures =
        predict Test test_metadata test
      in
      let train_results =
        match train_prediction with
        | Ok prediction ->
            Some
              (score_partition ~fold_index ~partition:Train scorers train
                 prediction)
        | Error _ -> None
      in
      let test_results =
        match test_prediction with
        | Ok prediction ->
            Some
              (score_partition ~fold_index ~partition:Test scorers test
                 prediction)
        | Error _ -> None
      in
      let scores =
        Array.mapi
          (fun index scorer ->
            let name = Regression_scorer.name scorer in
            let train_score =
              if not return_train_score then None
              else
                match train_results with
                | Some results -> Some (fst results.(index))
                | None -> (
                    match train_prediction with
                    | Error error -> Some (Error error)
                    | Ok _ -> assert false)
            in
            let test_score =
              match test_results with
              | Some results -> Some (fst results.(index))
              | None -> (
                  match test_prediction with
                  | Error error -> Some (Error error)
                  | Ok _ -> assert false)
            in
            { name; train_score; test_score })
          scorers
      in
      let scoring_failures results =
        match results with
        | None -> []
        | Some results ->
            Array.fold_left
              (fun accumulated (_, failures) -> accumulated @ failures)
              [] results
      in
      finish_scoring failure_policy scores
        (train_prediction_failures @ test_prediction_failures
        @ scoring_failures train_results
        @ scoring_failures test_results)

    let cross_validate ?(return_train_score = false) ?(return_models = false)
        ?(return_indices = false) ?(failure_policy = Abort) ?fit_seed
        ?(execution = Execution.sequential) ?metadata ~splitter ~scorers ~seed
        pipeline dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let fit_seed = Option.value fit_seed ~default:seed in
      let scorer_names = Array.map Regression_scorer.name scorers in
      run ~return_train_score ~return_models ~return_indices ~failure_policy
        ~fit_seed ~execution ~metadata ~splitter ~scorer_names ~seed
        ~score_model:(score_model scorers) pipeline dataset
  end

  (* Classification evaluation shared by the binary and multiclass variants:
     the scorer family decides which responses a fold needs and how a scorer's
     prediction value is assembled from them. *)
  module type CLASSIFICATION_SCORING = sig
    type scorer
    type prediction

    val name : scorer -> string
    val needs_labels : scorer -> bool
    val needs_probabilities : scorer -> bool
    val validate_classes : int array -> Matrix.t -> (unit, Error.t) result

    val prediction :
      scorer ->
      labels:Target.classification Target.t option ->
      probabilities:(Matrix.t * int array) option ->
      (prediction, Error.t) result

    val score :
      scorer ->
      ?sample_weight:Sample_weight.t ->
      truth:Target.classification Target.t ->
      prediction:prediction ->
      unit ->
      (float, Error.t) result
  end

  module Classification_evaluation (Scoring : CLASSIFICATION_SCORING) = struct
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    type responses = {
      labels : (Target.classification Target.t, Error.t) result option;
      probabilities : (Matrix.t * int array, Error.t) result option;
    }

    let probability_data ~fold_index ~metadata fitted dataset =
      let ( let* ) = Result.bind in
      (let* classes = Pipeline.classes fitted in
       let* probabilities =
         Pipeline.predict_proba_with_metadata fitted ~metadata
           ~feature_schema:(Dataset.feature_schema dataset)
           ~x:(Dataset.features dataset)
       in
       let* () = Scoring.validate_classes classes probabilities in
       Ok (probabilities, classes))
      |> Result.map_error (contextualize fold_index)

    let responses ~fold_index ~metadata scorers fitted dataset =
      let labels =
        if Array.exists Scoring.needs_labels scorers then
          Some
            (Pipeline.predict_with_metadata fitted ~metadata
               ~feature_schema:(Dataset.feature_schema dataset)
               ~x:(Dataset.features dataset)
            |> Result.map_error (contextualize fold_index))
        else None
      in
      let probabilities =
        if Array.exists Scoring.needs_probabilities scorers then
          Some (probability_data ~fold_index ~metadata fitted dataset)
        else None
      in
      { labels; probabilities }

    let response_failures partition responses =
      let collect accumulated = function
        | None | Some (Ok _) -> accumulated
        | Some (Error error) ->
            accumulated @ [ failure ~phase:(Prediction partition) error ]
      in
      collect (collect [] responses.labels) responses.probabilities

    let response_failed scorer responses =
      let failed = function
        | Some (Error _) -> true
        | None | Some (Ok _) -> false
      in
      (Scoring.needs_labels scorer && failed responses.labels)
      || (Scoring.needs_probabilities scorer && failed responses.probabilities)

    let required needed = function
      | Some result when needed -> Result.map Option.some result
      | None | Some _ -> Ok None

    let prediction_for_scorer scorer responses =
      let ( let* ) = Result.bind in
      let* labels = required (Scoring.needs_labels scorer) responses.labels in
      let* probabilities =
        required (Scoring.needs_probabilities scorer) responses.probabilities
      in
      Scoring.prediction scorer ~labels ~probabilities

    let score_partition ~fold_index ~partition scorers dataset responses =
      Array.map
        (fun scorer ->
          let name = Scoring.name scorer in
          match prediction_for_scorer scorer responses with
          | Error error when response_failed scorer responses ->
              (Error error, [])
          | Error error ->
              let error = scorer_error fold_index name error in
              ( Error error,
                [ failure ~phase:(Scoring { partition; scorer = name }) error ]
              )
          | Ok prediction ->
              let result =
                Scoring.score scorer
                  ?sample_weight:(Dataset.sample_weight dataset)
                  ~truth:(Dataset.target dataset) ~prediction ()
                |> Result.map_error (scorer_error fold_index name)
              in
              let failures =
                match result with
                | Ok _ -> []
                | Error error ->
                    [
                      failure
                        ~phase:(Scoring { partition; scorer = name })
                        error;
                    ]
              in
              (result, failures))
        scorers

    let score_model scorers ~fold_index ~return_train_score ~failure_policy
        ~train_metadata ~test_metadata fitted train test =
      let train_responses =
        if return_train_score then
          Some
            (responses ~fold_index ~metadata:train_metadata scorers fitted train)
        else None
      in
      let test_responses =
        responses ~fold_index ~metadata:test_metadata scorers fitted test
      in
      let train_results =
        Option.map
          (score_partition ~fold_index ~partition:Train scorers train)
          train_responses
      in
      let test_results =
        score_partition ~fold_index ~partition:Test scorers test test_responses
      in
      let scores =
        Array.mapi
          (fun index scorer ->
            {
              name = Scoring.name scorer;
              train_score =
                Option.map (fun results -> fst results.(index)) train_results;
              test_score = Some (fst test_results.(index));
            })
          scorers
      in
      let scoring_failures results =
        Array.fold_left
          (fun accumulated (_, failures) -> accumulated @ failures)
          [] results
      in
      let failures =
        (match train_responses with
          | None -> []
          | Some values -> response_failures Train values)
        @ response_failures Test test_responses
        @ (match train_results with
          | None -> []
          | Some values -> scoring_failures values)
        @ scoring_failures test_results
      in
      finish_scoring failure_policy scores failures

    let cross_validate ?(return_train_score = false) ?(return_models = false)
        ?(return_indices = false) ?(failure_policy = Abort) ?fit_seed
        ?(execution = Execution.sequential) ?metadata ~splitter ~scorers ~seed
        pipeline dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let fit_seed = Option.value fit_seed ~default:seed in
      let scorer_names = Array.map Scoring.name scorers in
      run ~return_train_score ~return_models ~return_indices ~failure_policy
        ~fit_seed ~execution ~metadata ~splitter ~scorer_names ~seed
        ~score_model:(score_model scorers) pipeline dataset
  end

  let compatibility ~component ~reason ~remediation =
    Error.make ~remediation (Error.Compatibility { component; reason })

  let validate_probability_columns classes probabilities =
    if Matrix.columns probabilities <> Array.length classes then
      Error
        (compatibility ~component:"pipeline probability output"
           ~reason:
             (Format.sprintf "%d columns were returned for %d classes"
                (Matrix.columns probabilities)
                (Array.length classes))
           ~remediation:"return one probability column for every declared class")
    else Ok ()

  module Binary_scoring = struct
    type scorer = Binary_classification_scorer.t
    type prediction = Binary_prediction.t

    let name = Binary_classification_scorer.name

    let needs_labels scorer =
      Binary_classification_scorer.response scorer
      = Binary_classification_scorer.Labels

    let needs_probabilities scorer =
      Binary_classification_scorer.response scorer
      = Binary_classification_scorer.Positive_probabilities

    let validate_classes classes probabilities =
      if Array.length classes <> 2 then
        Error
          (compatibility ~component:"pipeline probability class order"
             ~reason:
               (Format.sprintf "%d classes were declared" (Array.length classes))
             ~remediation:"declare exactly two distinct binary classes")
      else if classes.(0) = classes.(1) then
        Error
          (compatibility ~component:"pipeline probability class order"
             ~reason:"the two declared class labels are identical"
             ~remediation:"declare each binary class exactly once")
      else validate_probability_columns classes probabilities

    let find_class_column classes positive_label =
      let rec loop column =
        if column = Array.length classes then None
        else if classes.(column) = positive_label then Some column
        else loop (column + 1)
      in
      loop 0

    let prediction scorer ~labels ~probabilities =
      match (needs_labels scorer, labels, probabilities) with
      | true, Some labels, _ -> Binary_prediction.create ~labels ()
      | false, _, Some (probabilities, classes) -> (
          let params = Binary_classification_scorer.params scorer in
          match
            find_class_column classes
              params.Binary_classification_scorer.positive_label
          with
          | None ->
              Error
                (validation ~name:"positive probability class"
                   ~reason:
                     (Format.sprintf
                        "label %d is absent from the declared class order"
                        params.Binary_classification_scorer.positive_label)
                   ~remediation:
                     "configure the scorer positive label to match the \
                      classifier")
          | Some column ->
              let positive_probabilities =
                Vector.unsafe_init (Matrix.rows probabilities) (fun row ->
                    Matrix.get probabilities row column)
              in
              Binary_prediction.create ~positive_probabilities ())
      | true, None, _ | false, _, None -> assert false

    let score = Binary_classification_scorer.score
  end

  module Multiclass_scoring = struct
    type scorer = Multiclass_classification_scorer.t
    type prediction = Multiclass_prediction.t

    let name = Multiclass_classification_scorer.name

    let needs_labels scorer =
      Multiclass_classification_scorer.response scorer
      = Multiclass_classification_scorer.Labels

    let needs_probabilities scorer =
      Multiclass_classification_scorer.response scorer
      = Multiclass_classification_scorer.Class_probabilities

    let validate_classes classes probabilities =
      let sorted = Array.copy classes in
      Array.sort Int.compare sorted;
      let rec distinct index =
        index >= Array.length sorted
        || (sorted.(index - 1) <> sorted.(index) && distinct (index + 1))
      in
      if Array.length classes < 2 then
        Error
          (compatibility ~component:"pipeline probability class order"
             ~reason:
               (Format.sprintf "%d classes were declared" (Array.length classes))
             ~remediation:"declare at least two distinct classes")
      else if not (distinct 1) then
        Error
          (compatibility ~component:"pipeline probability class order"
             ~reason:"a declared class label repeats"
             ~remediation:"declare each class exactly once")
      else validate_probability_columns classes probabilities

    let prediction scorer ~labels ~probabilities =
      match (needs_labels scorer, labels, probabilities) with
      | true, Some labels, _ -> Multiclass_prediction.create ~labels ()
      | false, _, Some (probabilities, classes) ->
          Multiclass_prediction.create ~classes ~probabilities ()
      | true, None, _ | false, _, None -> assert false

    let score = Multiclass_classification_scorer.score
  end

  module Binary_classification = Classification_evaluation (Binary_scoring)

  module Multiclass_classification =
    Classification_evaluation (Multiclass_scoring)
end

module Grid_search = struct
  type parameter_value =
    | Bool of bool
    | Int of int
    | Float of float
    | String of string

  type parameter = {
    parameter_name : string;
    parameter_value : parameter_value;
  }

  type 'configuration axis =
    | Axis : {
        name : string;
        values : 'value array;
        encode : 'value -> parameter_value;
        set : 'configuration -> 'value -> ('configuration, Error.t) result;
      }
        -> 'configuration axis

  type ('configuration, 'target, 'prediction) grid = {
    base : 'configuration;
    build :
      'configuration -> (('target, 'prediction) Pipeline.t, Error.t) result;
    axes : 'configuration axis array;
    candidate_count : int;
  }

  type score_summary = {
    scorer_name : string;
    train : (Score_aggregation.t, Error.t) result option;
    test : (Score_aggregation.t, Error.t) result;
  }

  type 'model candidate = {
    candidate_index : int;
    parameters : parameter array;
    rank : int option;
    mean_fit_time : float;
    mean_score_time : float;
    scores : score_summary array;
    evaluation : 'model Cross_validation.report option;
    build_error : Error.t option;
  }

  type 'model refit_policy =
    | No_refit
    | Best_score of string
    | Custom of ('model candidate array -> (int, Error.t) result)

  type 'model selected = {
    selected_candidate_index : int;
    selected_model : 'model;
  }

  type 'model report = {
    report_candidates : 'model candidate array;
    report_selection : ('model selected option, Error.t) result;
  }

  type 'configuration partial = {
    configuration : ('configuration, Error.t) result;
    reversed_parameters : parameter list;
  }

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let axis ~name ~values ~encode ~set =
    if String.length (String.trim name) = 0 then
      Error
        (validation ~name:"grid-search axis name" ~reason:"must not be blank"
           ~remediation:"choose a non-empty unique parameter name")
    else if Array.length values = 0 then
      Error
        (validation
           ~name:("grid-search axis " ^ name)
           ~reason:"contains no values"
           ~remediation:"provide at least one finite candidate value")
    else Ok (Axis { name; values = Array.copy values; encode; set })

  let create ~base ~build axes =
    let seen = Hashtbl.create (Array.length axes) in
    let rec validate index count =
      if index = Array.length axes then Ok count
      else
        let (Axis axis) = axes.(index) in
        if Hashtbl.mem seen axis.name then
          Error
            (validation ~name:"grid-search axes"
               ~reason:
                 (Format.sprintf "parameter name %S is duplicated" axis.name)
               ~remediation:"use each parameter name at most once")
        else if count > Sys.max_array_length / Array.length axis.values then
          Error
            (validation ~name:"grid-search candidate count"
               ~reason:"the Cartesian product exceeds the array size limit"
               ~remediation:"reduce the number of axes or candidate values")
        else (
          Hashtbl.add seen axis.name ();
          validate (index + 1) (count * Array.length axis.values))
    in
    match validate 0 1 with
    | Error _ as error -> error
    | Ok candidate_count ->
        Ok { base; build; axes = Array.copy axes; candidate_count }

  let candidate_count grid = grid.candidate_count

  let copy_candidate candidate =
    {
      candidate with
      parameters = Array.copy candidate.parameters;
      scores = Array.copy candidate.scores;
    }

  let candidates report = Array.map copy_candidate report.report_candidates
  let refit_result report = report.report_selection

  let selection report =
    match report.report_selection with
    | Ok (Some selected) -> Ok selected
    | Error error -> Error error
    | Ok None ->
        Error
          (validation ~name:"search selection" ~reason:"refitting was disabled"
             ~remediation:"inspect candidate reports or run with a refit policy")

  let expand_axis partial (Axis axis) =
    Array.to_list axis.values
    |> List.map (fun value ->
        let configuration =
          match partial.configuration with
          | Error _ as error -> error
          | Ok configuration -> axis.set configuration value
        in
        {
          configuration;
          reversed_parameters =
            { parameter_name = axis.name; parameter_value = axis.encode value }
            :: partial.reversed_parameters;
        })

  let expand grid =
    Array.fold_left
      (fun partials axis ->
        List.fold_right
          (fun partial accumulated -> expand_axis partial axis @ accumulated)
          partials [])
      [ { configuration = Ok grid.base; reversed_parameters = [] } ]
      grid.axes
    |> Array.of_list

  let[@warning "-4"] with_candidate candidate error =
    match Error.context error with
    | Error.Candidate index :: _ when index = candidate -> error
    | _ -> Error.with_context (Error.Candidate candidate) error

  let missing_score candidate scorer partition =
    validation ~name:"search score"
      ~reason:(Format.sprintf "%s score %S is unavailable" partition scorer)
      ~remediation:"inspect the candidate's fold failures"
    |> with_candidate candidate

  let aggregate candidate scorer partition extract folds =
    let values = Array.make (Array.length folds) 0.0 in
    let rec collect fold_index =
      if fold_index = Array.length folds then
        Score_aggregation.summarize values
        |> Result.map_error (with_candidate candidate)
      else
        let score = folds.(fold_index).Cross_validation.scores.(scorer) in
        match extract score with
        | Some (Ok value) ->
            values.(fold_index) <- value;
            collect (fold_index + 1)
        | Some (Error error) -> Error (with_candidate candidate error)
        | None ->
            Error
              (missing_score candidate score.Cross_validation.name partition)
    in
    collect 0

  let summarize candidate scorer_names ~return_train_score evaluation =
    let folds = Cross_validation.folds evaluation in
    Array.mapi
      (fun scorer name ->
        {
          scorer_name = name;
          train =
            (if return_train_score then
               Some
                 (aggregate candidate scorer "training"
                    (fun score -> score.Cross_validation.train_score)
                    folds)
             else None);
          test =
            aggregate candidate scorer "test"
              (fun score -> score.Cross_validation.test_score)
              folds;
        })
      scorer_names

  let mean_time select evaluation =
    let folds = Cross_validation.folds evaluation in
    if Array.length folds = 0 then 0.0
    else
      Array.fold_left (fun total fold -> total +. select fold) 0.0 folds
      /. Float.of_int (Array.length folds)

  let failed_summaries ~return_train_score scorer_names error =
    Array.map
      (fun name ->
        {
          scorer_name = name;
          train = (if return_train_score then Some (Error error) else None);
          test = Error error;
        })
      scorer_names

  let validate_refit scorer_names refit =
    let ( let* ) = Result.bind in
    let* () = Cross_validation.validate_scorers scorer_names in
    let rec find index =
      if index = Array.length scorer_names then
        Error
          (validation ~name:"search refit scorer"
             ~reason:(Format.sprintf "scorer %S was not provided" refit)
             ~remediation:"choose one of the configured scorer names")
      else if String.equal scorer_names.(index) refit then Ok index
      else find (index + 1)
    in
    find 0

  let rank_candidates primary candidates =
    let eligible =
      candidates |> Array.to_list
      |> List.filter_map (fun (candidate : _ candidate) ->
          match candidate.scores.(primary).test with
          | Ok summary ->
              Some (candidate.candidate_index, summary.Score_aggregation.mean)
          | Error _ -> None)
      |> Array.of_list
    in
    Array.sort
      (fun (left_index, left_score) (right_index, right_score) ->
        let score_order = Float.compare right_score left_score in
        if score_order <> 0 then score_order
        else Int.compare left_index right_index)
      eligible;
    let ranks = Array.make (Array.length candidates) None in
    let previous_score = ref None in
    let previous_rank = ref 0 in
    Array.iteri
      (fun position (candidate, score) ->
        let rank =
          match !previous_score with
          | Some previous when Float.compare previous score = 0 ->
              !previous_rank
          | None | Some _ -> position + 1
        in
        ranks.(candidate) <- Some rank;
        previous_score := Some score;
        previous_rank := rank)
      eligible;
    ( Array.mapi
        (fun index candidate -> { candidate with rank = ranks.(index) })
        candidates,
      if Array.length eligible = 0 then None else Some (fst eligible.(0)) )

  let search_candidates ~return_train_score ~failure_policy ~cross_validate
      ~scorer_names ~metadata ~policy ~seed ~operation ~prepare ~build dataset =
    let ( let* ) = Result.bind in
    let* primary =
      match policy with
      | Best_score refit ->
          Result.map Option.some (validate_refit scorer_names refit)
      | No_refit | Custom _ ->
          Result.map
            (fun () -> None)
            (Cross_validation.validate_scorers scorer_names)
    in
    let* () = Metadata.validate ~rows:(Dataset.sample_count dataset) metadata in
    Callback.run
      ~outcome:(fun report ->
        match report.report_selection with
        | Ok _ -> Callback.Succeeded
        | Error error -> Callback.Failed error)
      (Metadata.callback metadata)
      ~operation:Callback.Search
      (fun () ->
        let count, partial_at = prepare () in
        let pipelines = Array.make count None in
        let evaluate_candidate candidate_index =
          let candidate_metadata =
            Metadata.scope (Error.Candidate candidate_index) metadata
          in
          Callback.run
            ~outcome:(fun candidate ->
              match candidate.build_error with
              | Some error -> Callback.Failed error
              | None -> (
                  match
                    Array.to_list candidate.scores
                    |> List.find_map (fun score ->
                        match score.test with
                        | Ok _ -> None
                        | Error error -> Some error)
                  with
                  | None -> Callback.Succeeded
                  | Some error -> Callback.Failed error))
            (Metadata.callback candidate_metadata)
            ~operation:Callback.Candidate
            (fun () ->
              let partial = partial_at candidate_index in
              let parameters =
                partial.reversed_parameters |> List.rev |> Array.of_list
              in
              let built =
                match partial.configuration with
                | Error error -> Error error
                | Ok configuration -> build configuration
              in
              match built with
              | Error error ->
                  let error = with_candidate candidate_index error in
                  if
                    failure_policy = Cross_validation.Abort
                    || Callback.is_control_error error
                  then Error error
                  else
                    let candidate : _ candidate =
                      {
                        candidate_index;
                        parameters;
                        rank = None;
                        mean_fit_time = 0.0;
                        mean_score_time = 0.0;
                        scores =
                          failed_summaries ~return_train_score scorer_names
                            error;
                        evaluation = None;
                        build_error = Some error;
                      }
                    in
                    Ok candidate
              | Ok pipeline ->
                  pipelines.(candidate_index) <- Some pipeline;
                  let fit_seed =
                    Seed.derive seed ~operation:(operation ^ "-candidate")
                      ~index:candidate_index
                  in
                  let* evaluation =
                    cross_validate ~metadata:candidate_metadata
                      ~return_train_score ~failure_policy ~fit_seed pipeline
                      dataset
                    |> Result.map_error (with_candidate candidate_index)
                  in
                  let candidate : _ candidate =
                    {
                      candidate_index;
                      parameters;
                      rank = None;
                      mean_fit_time =
                        mean_time
                          (fun fold -> fold.Cross_validation.fit_time)
                          evaluation;
                      mean_score_time =
                        mean_time
                          (fun fold -> fold.Cross_validation.score_time)
                          evaluation;
                      scores =
                        summarize candidate_index scorer_names
                          ~return_train_score evaluation;
                      evaluation = Some evaluation;
                      build_error = None;
                    }
                  in
                  Ok candidate)
          |> Result.map_error (with_candidate candidate_index)
        in
        let rec evaluate candidate_index reversed =
          if candidate_index = count then Ok (Array.of_list (List.rev reversed))
          else
            let* candidate = evaluate_candidate candidate_index in
            evaluate (candidate_index + 1) (candidate :: reversed)
        in
        let* evaluated = evaluate 0 [] in
        let ranked, best =
          match primary with
          | Some primary -> rank_candidates primary evaluated
          | None -> (evaluated, None)
        in
        let failure error =
          if
            failure_policy = Cross_validation.Abort
            || Callback.is_control_error error
          then Error error
          else Ok { report_candidates = ranked; report_selection = Error error }
        in
        let invalid reason =
          validation ~name:"search selection" ~reason
            ~remediation:
              "choose a built candidate with at least one successful \
               test-score aggregate"
        in
        let chosen =
          match policy with
          | No_refit -> Ok None
          | Best_score _ -> (
              match best with
              | Some index -> Ok (Some index)
              | None ->
                  Error
                    (invalid
                       "no candidate produced an aggregatable primary test \
                        score"))
          | Custom select ->
              Result.map Option.some (select (Array.map copy_candidate ranked))
        in
        match chosen with
        | Error error -> failure error
        | Ok None ->
            Ok { report_candidates = ranked; report_selection = Ok None }
        | Ok (Some candidate_index) -> (
            if candidate_index < 0 || candidate_index >= count then
              failure
                (invalid "selector returned an out-of-range candidate index")
            else if
              not
                (Array.exists
                   (fun score -> Result.is_ok score.test)
                   ranked.(candidate_index).scores)
            then
              failure
                (invalid
                   "selected candidate has no successful test-score aggregate"
                |> with_candidate candidate_index)
            else
              match pipelines.(candidate_index) with
              | None ->
                  failure
                    (invalid "selected candidate did not build"
                    |> with_candidate candidate_index)
              | Some pipeline -> (
                  let refit_seed =
                    Seed.derive seed ~operation:(operation ^ "-refit")
                      ~index:candidate_index
                    |> Rng.create
                  in
                  let refit_metadata =
                    Metadata.scope (Error.Candidate candidate_index) metadata
                  in
                  let refitted =
                    Callback.run (Metadata.callback refit_metadata)
                      ~operation:Callback.Refit (fun () ->
                        Pipeline.fit_with_metadata (Pipeline.clone pipeline)
                          ~metadata:refit_metadata ~rng:refit_seed
                          ~feature_schema:(Dataset.feature_schema dataset)
                          ~x:(Dataset.features dataset)
                          ~y:(Dataset.target dataset) ())
                    |> Result.map_error (with_candidate candidate_index)
                  in
                  match refitted with
                  | Error error -> failure error
                  | Ok model ->
                      Ok
                        {
                          report_candidates = ranked;
                          report_selection =
                            Ok
                              (Some
                                 {
                                   selected_candidate_index = candidate_index;
                                   selected_model = model;
                                 });
                        })))

  let search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
      ~metadata ~policy ~seed grid dataset =
    let prepare () =
      let partials = expand grid in
      (Array.length partials, Array.get partials)
    in
    search_candidates ~return_train_score ~failure_policy ~cross_validate
      ~scorer_names ~metadata ~policy ~seed ~operation:"grid-search" ~prepare
      ~build:grid.build dataset

  module Regression = struct
    type model = Cross_validation.Regression.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~grid ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names = Array.map Regression_scorer.name scorers in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Regression.cross_validate ~return_train_score
          ~failure_policy ~fit_seed ~execution ~metadata ~splitter ~scorers
          ~seed pipeline dataset
      in
      search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
        ~metadata ~policy ~seed grid dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~grid
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~grid ~splitter ~scorers ~policy:(Best_score refit) ~seed
        dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~grid ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Binary_classification.cross_validate
          ~return_train_score ~failure_policy ~fit_seed ~execution ~metadata
          ~splitter ~scorers ~seed pipeline dataset
      in
      search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
        ~metadata ~policy ~seed grid dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~grid
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~grid ~splitter ~scorers ~policy:(Best_score refit) ~seed
        dataset
  end

  module Multiclass_classification = struct
    type model = Cross_validation.Multiclass_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~grid ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names =
        Array.map Multiclass_classification_scorer.name scorers
      in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Multiclass_classification.cross_validate
          ~return_train_score ~failure_policy ~fit_seed ~execution ~metadata
          ~splitter ~scorers ~seed pipeline dataset
      in
      search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
        ~metadata ~policy ~seed grid dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~grid
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~grid ~splitter ~scorers ~policy:(Best_score refit) ~seed
        dataset
  end
end

module Parameter_distribution = struct
  type 'a t = Choices of 'a array | Sample of (Rng.t -> ('a, Error.t) result)

  let invalid reason =
    Error
      (Error.make
         (Error.Validation { name = "parameter distribution"; reason })
         ~remediation:
           "supply nonempty immutable choices or finite ordered distribution \
            bounds")

  let choice values =
    if Array.length values = 0 then invalid "choices must not be empty"
    else Ok (Choices (Array.copy values))

  let custom sampler = Sample sampler

  let rec bounded rng bound =
    let bits, next = Rng.next_int64 rng in
    let draw = Int64.shift_right_logical bits 1 in
    let limit = Int64.sub Int64.max_int (Int64.rem Int64.max_int bound) in
    if draw >= limit then bounded next bound else (Int64.rem draw bound, next)

  let sample ~rng = function
    | Choices values ->
        let index, _ = bounded rng (Int64.of_int (Array.length values)) in
        Ok values.(Int64.to_int index)
    | Sample draw -> draw rng

  let uniform ~low ~high () =
    if not (Float.is_finite low && Float.is_finite high && low < high) then
      invalid "uniform bounds must be finite with low < high"
    else
      Ok
        (Sample
           (fun rng ->
             let u, _ = Rng.next_float rng in
             let value = ((1. -. u) *. low) +. (u *. high) in
             Ok (max low (min (Float.next_after high neg_infinity) value))))

  let log_uniform ~low ~high () =
    if
      not (Float.is_finite low && Float.is_finite high && low > 0. && low < high)
    then
      invalid "log-uniform bounds must be positive and finite with low < high"
    else
      Ok
        (Sample
           (fun rng ->
             let u, _ = Rng.next_float rng in
             let value = exp (((1. -. u) *. log low) +. (u *. log high)) in
             Ok (max low (min (Float.next_after high neg_infinity) value))))

  let int_uniform ~low ~high () =
    if low >= high then invalid "integer bounds require low < high"
    else
      Ok
        (Sample
           (fun rng ->
             let lower = Int64.of_int low in
             let width = Int64.sub (Int64.of_int high) lower in
             let offset, _ = bounded rng width in
             Ok (Int64.to_int (Int64.add lower offset))))
end

module Randomized_search = struct
  type 'configuration axis =
    | Axis : {
        sample_name : string;
        distribution : 'value Parameter_distribution.t;
        sample_encode : 'value -> Grid_search.parameter_value;
        sample_set :
          'configuration -> 'value -> ('configuration, Error.t) result;
      }
        -> 'configuration axis

  type ('configuration, 'target, 'prediction) space = {
    sample_base : 'configuration;
    sample_build :
      'configuration -> (('target, 'prediction) Pipeline.t, Error.t) result;
    sample_axes : 'configuration axis array;
    sample_count : int;
    finite_count : int option;
  }

  type 'configuration sampled_candidate = {
    sampled_parameters : Grid_search.parameter array;
    sampled_configuration : ('configuration, Error.t) result;
  }

  type 'model report = 'model Grid_search.report

  let candidates = Grid_search.candidates
  let selection = Grid_search.selection
  let refit_result = Grid_search.refit_result

  let invalid reason =
    Error
      (Error.make
         (Error.Validation { name = "randomized search"; reason })
         ~remediation:
           "use unique parameter names, positive iterations, and a finite \
            choice product that fits an OCaml integer")

  let axis ~name ~distribution ~encode ~set =
    if String.trim name = "" then invalid "parameter names must not be blank"
    else
      Ok
        (Axis
           {
             sample_name = name;
             distribution;
             sample_encode = encode;
             sample_set = set;
           })

  let create ?(iterations = 10) ~base ~build axes =
    let ( let* ) = Result.bind in
    let* () =
      if iterations > 0 && iterations <= Sys.max_array_length then Ok ()
      else invalid "iteration count must be positive and fit an array"
    in
    let seen = Hashtbl.create (Array.length axes) in
    let rec names i =
      if i = Array.length axes then Ok ()
      else
        let (Axis axis) = axes.(i) in
        if Hashtbl.mem seen axis.sample_name then
          invalid ("duplicate parameter name: " ^ axis.sample_name)
        else (
          Hashtbl.add seen axis.sample_name ();
          names (i + 1))
    in
    let* () = names 0 in
    let finite =
      Array.for_all
        (fun (Axis axis) ->
          match axis.distribution with
          | Parameter_distribution.Choices _ -> true
          | Parameter_distribution.Sample _ -> false)
        axes
    in
    let* finite_count =
      if not finite then Ok None
      else
        Array.fold_left
          (fun result (Axis axis) ->
            let* count = result in
            match axis.distribution with
            | Parameter_distribution.Sample _ -> assert false
            | Parameter_distribution.Choices values ->
                if count > max_int / Array.length values then
                  invalid "finite choice product exceeds the integer limit"
                else Ok (count * Array.length values))
          (Ok 1) axes
        |> Result.map Option.some
    in
    let sample_count =
      Option.fold ~none:iterations ~some:(min iterations) finite_count
    in
    Ok
      {
        sample_base = base;
        sample_build = build;
        sample_axes = Array.copy axes;
        sample_count;
        finite_count;
      }

  let candidate_count space = space.sample_count

  let prepare ~seed space =
    let index_at =
      Option.map
        (fun total ->
          let swaps = Hashtbl.create space.sample_count in
          let indices = Array.make space.sample_count 0 in
          let generated = ref 0 in
          let rng =
            ref
              (Rng.create
                 (Seed.derive seed ~operation:"randomized-search-choices"
                    ~index:0))
          in
          fun index ->
            while !generated <= index do
              let remaining = total - !generated in
              let value, next =
                Parameter_distribution.bounded !rng (Int64.of_int remaining)
              in
              rng := next;
              let position = Int64.to_int value in
              let at index =
                Option.value (Hashtbl.find_opt swaps index) ~default:index
              in
              let selected = at position in
              let last = at (remaining - 1) in
              Hashtbl.remove swaps (remaining - 1);
              if position <> remaining - 1 then
                Hashtbl.replace swaps position last;
              indices.(!generated) <- selected;
              incr generated
            done;
            indices.(index))
        space.finite_count
    in
    let partial_at candidate_index =
      let remaining =
        ref (Option.fold ~none:0 ~some:(fun at -> at candidate_index) index_at)
      in
      let coordinates = Array.make (Array.length space.sample_axes) 0 in
      (match index_at with
      | None -> ()
      | Some _ ->
          for i = Array.length space.sample_axes - 1 downto 0 do
            let (Axis axis) = space.sample_axes.(i) in
            match axis.distribution with
            | Parameter_distribution.Sample _ -> assert false
            | Parameter_distribution.Choices values ->
                coordinates.(i) <- !remaining mod Array.length values;
                remaining := !remaining / Array.length values
          done);
      Array.fold_left
        (fun (i, partial) (Axis axis) ->
          match partial.Grid_search.configuration with
          | Error error when Callback.is_control_error error -> (i + 1, partial)
          | Ok _ | Error _ ->
              let draw =
                match (axis.distribution, index_at) with
                | Parameter_distribution.Choices values, Some _ ->
                    Ok values.(coordinates.(i))
                | ( ( Parameter_distribution.Choices _
                    | Parameter_distribution.Sample _ ),
                    None )
                | Parameter_distribution.Sample _, Some _ ->
                    let rng =
                      Seed.derive seed
                        ~operation:
                          ("randomized-search-parameter:" ^ axis.sample_name)
                        ~index:candidate_index
                      |> Rng.create
                    in
                    Parameter_distribution.sample ~rng axis.distribution
              in
              let contextual error =
                error
                |> Error.with_context (Error.Stage axis.sample_name)
                |> Grid_search.with_candidate candidate_index
              in
              let configuration, reversed_parameters =
                match draw with
                | Error error ->
                    ( (if Callback.is_control_error error then
                         Error (contextual error)
                       else
                         match partial.Grid_search.configuration with
                         | Error _ as error -> error
                         | Ok _ -> Error (contextual error)),
                      partial.Grid_search.reversed_parameters )
                | Ok value ->
                    let configuration =
                      match partial.Grid_search.configuration with
                      | Error _ as error -> error
                      | Ok configuration ->
                          axis.sample_set configuration value
                          |> Result.map_error contextual
                    in
                    ( configuration,
                      {
                        Grid_search.parameter_name = axis.sample_name;
                        parameter_value = axis.sample_encode value;
                      }
                      :: partial.Grid_search.reversed_parameters )
              in
              (i + 1, { Grid_search.configuration; reversed_parameters }))
        ( 0,
          {
            Grid_search.configuration = Ok space.sample_base;
            reversed_parameters = [];
          } )
        space.sample_axes
      |> snd
    in
    (space.sample_count, partial_at)

  let sample ~seed space =
    let count, partial_at = prepare ~seed space in
    Array.init count (fun index ->
        let partial = partial_at index in
        {
          sampled_parameters =
            List.rev partial.Grid_search.reversed_parameters |> Array.of_list;
          sampled_configuration = partial.Grid_search.configuration;
        })

  module Regression = struct
    type model = Cross_validation.Regression.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~space ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Regression_scorer.name scorers in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Regression.cross_validate ~metadata ~return_train_score
          ~failure_policy ~fit_seed ~execution ~splitter ~scorers ~seed pipeline
          dataset
      in
      Grid_search.search_candidates ~return_train_score ~failure_policy
        ~cross_validate ~scorer_names ~metadata ~policy ~seed
        ~operation:"randomized-search"
        ~prepare:(fun () -> prepare ~seed space)
        ~build:space.sample_build dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~space
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~space ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Binary_classification.cross_validate ~metadata
          ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
          ~scorers ~seed pipeline dataset
      in
      Grid_search.search_candidates ~return_train_score ~failure_policy
        ~cross_validate ~scorer_names ~metadata ~policy ~seed
        ~operation:"randomized-search"
        ~prepare:(fun () -> prepare ~seed space)
        ~build:space.sample_build dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~space
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end

  module Multiclass_classification = struct
    type model = Cross_validation.Multiclass_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ~space ~splitter ~scorers
        ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names =
        Array.map Multiclass_classification_scorer.name scorers
      in
      let cross_validate ~metadata ~return_train_score ~failure_policy ~fit_seed
          pipeline dataset =
        Cross_validation.Multiclass_classification.cross_validate ~metadata
          ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
          ~scorers ~seed pipeline dataset
      in
      Grid_search.search_candidates ~return_train_score ~failure_policy
        ~cross_validate ~scorer_names ~metadata ~policy ~seed
        ~operation:"randomized-search"
        ~prepare:(fun () -> prepare ~seed space)
        ~build:space.sample_build dataset

    let search ?return_train_score ?failure_policy ?execution ?metadata ~space
        ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end
end
