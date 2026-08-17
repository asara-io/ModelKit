module Data_error = Modelkit_data.Data_error
module Vector = Modelkit_data.Vector
module Matrix = Modelkit_data.Matrix
module Row_view = Modelkit_data.Row_view
module Target = Modelkit_data.Target
module Feature_name = Modelkit_data.Feature_name
module Feature_names = Modelkit_data.Feature_names
module Sample_weight = Modelkit_data.Sample_weight
module Groups = Modelkit_data.Groups
module Schema_fingerprint = Modelkit_data.Schema_fingerprint
module Feature_schema = Modelkit_data.Feature_schema
module Dataset = Modelkit_data.Dataset
module Error = Modelkit_data.Error

module type SPECIFICATION = Modelkit_protocols.SPECIFICATION
module type ESTIMATOR = Modelkit_protocols.ESTIMATOR
module type CLASSIFIER = Modelkit_protocols.CLASSIFIER
module type REGRESSOR = Modelkit_protocols.REGRESSOR
module type TRANSFORMER = Modelkit_protocols.TRANSFORMER
module type SCORER = Modelkit_protocols.SCORER
module type SPLITTER = Modelkit_protocols.SPLITTER
module type EXECUTION = Modelkit_protocols.EXECUTION
module type RNG = Modelkit_protocols.RNG
module type NUMERICAL_BACKEND = Modelkit_protocols.NUMERICAL_BACKEND

module Seed = Modelkit_protocols.Seed
module Rng = Modelkit_protocols.Rng
module Sequential_execution = Modelkit_protocols.Sequential_execution
module Execution = Modelkit_protocols.Execution
module Reference_backend = Modelkit_protocols.Reference_backend
module Preprocessing_internal = Modelkit_preprocessing.Preprocessing_internal
module Simple_imputer = Modelkit_preprocessing.Simple_imputer
module Standard_scaler = Modelkit_preprocessing.Standard_scaler
module Variance_threshold = Modelkit_preprocessing.Variance_threshold
module Pipeline = Modelkit_pipeline.Pipeline
module Solver_report = Modelkit_linear_models.Solver_report
module Linear_regression = Modelkit_linear_models.Linear_regression
module Ridge_regression = Modelkit_linear_models.Ridge_regression
module Logistic_regression = Modelkit_linear_models.Logistic_regression
module Split = Modelkit_splitting.Split
module K_fold = Modelkit_splitting.K_fold
module Stratified_k_fold = Modelkit_splitting.Stratified_k_fold
module Group_k_fold = Modelkit_splitting.Group_k_fold
module Time_series_split = Modelkit_splitting.Time_series_split
module Undefined_metric_policy = Modelkit_metrics.Undefined_metric_policy
module Regression_metrics = Modelkit_metrics.Regression_metrics
module Binary_prediction = Modelkit_metrics.Binary_prediction

module Binary_classification_metrics =
  Modelkit_metrics.Binary_classification_metrics

module Regression_scorer = Modelkit_metrics.Regression_scorer

module Binary_classification_scorer =
  Modelkit_metrics.Binary_classification_scorer

module Score_aggregation = Modelkit_metrics.Score_aggregation

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

  let contextualize fold error = Error.with_context (Error.Fold fold) error

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
      ~fit_seed ~execution ~splitter ~scorer_names ~seed ~score_model pipeline
      dataset =
    let ( let* ) = Result.bind in
    let* () = validate_scorers scorer_names in
    let splitter_rng =
      Seed.derive seed ~operation:"cross-validation-splitter" ~index:0
      |> Rng.create
    in
    let* view_pairs =
      splitter.run_splitter ~rng:splitter_rng ~groups:(Dataset.groups dataset)
        ~x:(Dataset.features dataset) ~y:(Dataset.target dataset)
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
    let evaluate ~index:fold_index split =
      let train_indices, test_indices = retain_indices return_indices split in
      match Split.materialize dataset split with
      | Error error ->
          let error = contextualize fold_index error in
          if failure_policy = Abort then Error error
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
      | Ok (train, test) -> (
          let fold_rng =
            Seed.derive fit_seed ~operation:"cross-validation-fold"
              ~index:fold_index
            |> Rng.create
          in
          let fit_time, fitted =
            timed (fun () ->
                Pipeline.fit (Pipeline.clone pipeline)
                  ?sample_weight:(Dataset.sample_weight train)
                  ~rng:fold_rng
                  ~feature_schema:(Dataset.feature_schema train)
                  ~x:(Dataset.features train) ~y:(Dataset.target train) ())
          in
          match fitted with
          | Error error ->
              let error = contextualize fold_index error in
              if failure_policy = Abort then Error error
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
                    score_model ~fold_index ~return_train_score ~failure_policy
                      fitted train test)
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
    let* report_folds = Execution.map execution ~f:evaluate splits in
    Ok { report_folds }

  let failure ~phase error = { phase; error }

  let finish_scoring failure_policy scores failures =
    match (failure_policy, failures) with
    | Abort, first :: _ -> Error first.error
    | (Abort | Record), _ -> Ok (scores, Array.of_list failures)

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
        fitted train test =
      let predict partition dataset =
        Pipeline.predict fitted
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
        if return_train_score then predict Train train
        else
          ( Error
              (validation ~name:"unused training prediction"
                 ~reason:"not requested"
                 ~remediation:
                   "request train scores to compute training predictions"),
            [] )
      in
      let test_prediction, test_prediction_failures = predict Test test in
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
        ?(execution = Execution.sequential) ~splitter ~scorers ~seed pipeline
        dataset =
      let fit_seed = Option.value fit_seed ~default:seed in
      let scorer_names = Array.map Regression_scorer.name scorers in
      run ~return_train_score ~return_models ~return_indices ~failure_policy
        ~fit_seed ~execution ~splitter ~scorer_names ~seed
        ~score_model:(score_model scorers) pipeline dataset
  end

  module Binary_classification = struct
    type model =
      ( Target.classification Target.t,
        Target.classification Target.t )
      Pipeline.fitted

    type responses = {
      labels : (Target.classification Target.t, Error.t) result option;
      probabilities : (Matrix.t * int array, Error.t) result option;
    }

    let needs response scorers =
      Array.exists
        (fun scorer -> Binary_classification_scorer.response scorer = response)
        scorers

    let probability_data ~fold_index fitted dataset =
      let ( let* ) = Result.bind in
      (let* classes = Pipeline.classes fitted in
       let* probabilities =
         Pipeline.predict_proba fitted
           ~feature_schema:(Dataset.feature_schema dataset)
           ~x:(Dataset.features dataset)
       in
       if Array.length classes <> 2 then
         Error
           (Error.make
              ~remediation:"declare exactly two distinct binary classes"
              (Error.Compatibility
                 {
                   component = "pipeline probability class order";
                   reason =
                     Format.sprintf "%d classes were declared"
                       (Array.length classes);
                 }))
       else if classes.(0) = classes.(1) then
         Error
           (Error.make ~remediation:"declare each binary class exactly once"
              (Error.Compatibility
                 {
                   component = "pipeline probability class order";
                   reason = "the two declared class labels are identical";
                 }))
       else if Matrix.columns probabilities <> Array.length classes then
         Error
           (Error.make
              ~remediation:
                "return one probability column for every declared class"
              (Error.Compatibility
                 {
                   component = "pipeline probability output";
                   reason =
                     Format.sprintf "%d columns were returned for %d classes"
                       (Matrix.columns probabilities)
                       (Array.length classes);
                 }))
       else Ok (probabilities, classes))
      |> Result.map_error (contextualize fold_index)

    let responses ~fold_index scorers fitted dataset =
      let labels =
        if needs Binary_classification_scorer.Labels scorers then
          Some
            (Pipeline.predict fitted
               ~feature_schema:(Dataset.feature_schema dataset)
               ~x:(Dataset.features dataset)
            |> Result.map_error (contextualize fold_index))
        else None
      in
      let probabilities =
        if needs Binary_classification_scorer.Positive_probabilities scorers
        then Some (probability_data ~fold_index fitted dataset)
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

    let find_class_column classes positive_label =
      let rec loop column =
        if column = Array.length classes then None
        else if classes.(column) = positive_label then Some column
        else loop (column + 1)
      in
      loop 0

    let prediction_for_scorer scorer responses =
      match Binary_classification_scorer.response scorer with
      | Binary_classification_scorer.Labels -> (
          match responses.labels with
          | Some (Ok labels) -> Binary_prediction.create ~labels ()
          | Some (Error error) -> Error error
          | None -> assert false)
      | Binary_classification_scorer.Positive_probabilities -> (
          match responses.probabilities with
          | Some (Error error) -> Error error
          | None -> assert false
          | Some (Ok (probabilities, classes)) -> (
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
                  Binary_prediction.create ~positive_probabilities ()))

    let response_failed scorer responses =
      match Binary_classification_scorer.response scorer with
      | Binary_classification_scorer.Labels -> (
          match responses.labels with
          | Some (Error _) -> true
          | None | Some (Ok _) -> false)
      | Binary_classification_scorer.Positive_probabilities -> (
          match responses.probabilities with
          | Some (Error _) -> true
          | None | Some (Ok _) -> false)

    let score_partition ~fold_index ~partition scorers dataset responses =
      Array.map
        (fun scorer ->
          let name = Binary_classification_scorer.name scorer in
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
                Binary_classification_scorer.score scorer
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
        fitted train test =
      let train_responses =
        if return_train_score then
          Some (responses ~fold_index scorers fitted train)
        else None
      in
      let test_responses = responses ~fold_index scorers fitted test in
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
              name = Binary_classification_scorer.name scorer;
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
        ?(execution = Execution.sequential) ~splitter ~scorers ~seed pipeline
        dataset =
      let fit_seed = Option.value fit_seed ~default:seed in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      run ~return_train_score ~return_models ~return_indices ~failure_policy
        ~fit_seed ~execution ~splitter ~scorer_names ~seed
        ~score_model:(score_model scorers) pipeline dataset
  end
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

  type 'model selected = {
    selected_candidate_index : int;
    selected_model : 'model;
  }

  type 'model report = {
    report_candidates : 'model candidate array;
    report_selection : ('model selected, Error.t) result;
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
  let selection report = report.report_selection

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

  let with_candidate candidate error =
    Error.with_context (Error.Candidate candidate) error

  let missing_score candidate scorer partition =
    validation ~name:"grid-search score"
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
          (validation ~name:"grid-search refit scorer"
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

  let search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
      ~refit ~seed grid dataset =
    let ( let* ) = Result.bind in
    let* primary = validate_refit scorer_names refit in
    let partials = expand grid in
    let pipelines = Array.make grid.candidate_count None in
    let rec evaluate candidate_index reversed =
      if candidate_index = Array.length partials then
        Ok (Array.of_list (List.rev reversed))
      else
        let partial = partials.(candidate_index) in
        let parameters =
          partial.reversed_parameters |> List.rev |> Array.of_list
        in
        let built =
          match partial.configuration with
          | Error error -> Error error
          | Ok configuration -> grid.build configuration
        in
        match built with
        | Error error ->
            let error = with_candidate candidate_index error in
            if failure_policy = Cross_validation.Abort then Error error
            else
              let candidate : _ candidate =
                {
                  candidate_index;
                  parameters;
                  rank = None;
                  mean_fit_time = 0.0;
                  mean_score_time = 0.0;
                  scores =
                    failed_summaries ~return_train_score scorer_names error;
                  evaluation = None;
                  build_error = Some error;
                }
              in
              evaluate (candidate_index + 1) (candidate :: reversed)
        | Ok pipeline ->
            pipelines.(candidate_index) <- Some pipeline;
            let fit_seed =
              Seed.derive seed ~operation:"grid-search-candidate"
                ~index:candidate_index
            in
            let* evaluation =
              cross_validate ~return_train_score ~failure_policy ~fit_seed
                pipeline dataset
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
                  summarize candidate_index scorer_names ~return_train_score
                    evaluation;
                evaluation = Some evaluation;
                build_error = None;
              }
            in
            evaluate (candidate_index + 1) (candidate :: reversed)
    in
    let* evaluated = evaluate 0 [] in
    let ranked, best = rank_candidates primary evaluated in
    let unavailable () =
      validation ~name:"grid-search selection"
        ~reason:"no candidate produced an aggregatable primary test score"
        ~remediation:"inspect candidate build, fold, and scorer failures"
    in
    match best with
    | None ->
        Ok
          {
            report_candidates = ranked;
            report_selection = Error (unavailable ());
          }
    | Some candidate_index -> (
        match pipelines.(candidate_index) with
        | None -> assert false
        | Some pipeline -> (
            let refit_seed =
              Seed.derive seed ~operation:"grid-search-refit"
                ~index:candidate_index
              |> Rng.create
            in
            let refitted =
              Pipeline.fit (Pipeline.clone pipeline)
                ?sample_weight:(Dataset.sample_weight dataset)
                ~rng:refit_seed
                ~feature_schema:(Dataset.feature_schema dataset)
                ~x:(Dataset.features dataset) ~y:(Dataset.target dataset) ()
              |> Result.map_error (with_candidate candidate_index)
            in
            match (failure_policy, refitted) with
            | Cross_validation.Abort, Error error -> Error error
            | (Cross_validation.Abort | Cross_validation.Record), Ok model ->
                Ok
                  {
                    report_candidates = ranked;
                    report_selection =
                      Ok
                        {
                          selected_candidate_index = candidate_index;
                          selected_model = model;
                        };
                  }
            | Cross_validation.Record, Error error ->
                Ok
                  { report_candidates = ranked; report_selection = Error error }
            ))

  module Regression = struct
    type model = Cross_validation.Regression.model

    let search ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ~grid ~splitter ~scorers ~refit
        ~seed dataset =
      let scorer_names = Array.map Regression_scorer.name scorers in
      let cross_validate ~return_train_score ~failure_policy ~fit_seed pipeline
          dataset =
        Cross_validation.Regression.cross_validate ~return_train_score
          ~failure_policy ~fit_seed ~execution ~splitter ~scorers ~seed pipeline
          dataset
      in
      search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
        ~refit ~seed grid dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ~grid ~splitter ~scorers ~refit
        ~seed dataset =
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let cross_validate ~return_train_score ~failure_policy ~fit_seed pipeline
          dataset =
        Cross_validation.Binary_classification.cross_validate
          ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
          ~scorers ~seed pipeline dataset
      in
      search ~return_train_score ~failure_policy ~cross_validate ~scorer_names
        ~refit ~seed grid dataset
  end
end

module Artifact = struct
  type limits = {
    max_bytes : int;
    max_components : int;
    max_features : int;
    max_string_bytes : int;
    max_metadata_entries : int;
  }

  type metadata = {
    training_rows : int option;
    root_seed : Seed.t option;
    sample_weighted : bool option;
    labels : (string * string) array;
  }

  type 'model loaded = {
    model : 'model;
    metadata : metadata;
    producer_version : string;
    producer_ocaml_version : string option;
  }

  type regression_model =
    (Target.regression Target.t, Target.regression Target.t) Pipeline.fitted

  type binary_classification_model =
    ( Target.classification Target.t,
      Target.classification Target.t )
    Pipeline.fitted

  let ( let* ) = Result.bind

  let artifact_error ~operation ~reason ~remediation =
    Error.make ~remediation (Error.Artifact { operation; reason })

  let failure ~operation reason =
    Error
      (artifact_error ~operation ~reason
         ~remediation:
           "use a valid ModelKit artifact produced by a supported codec")

  let default_limits =
    {
      max_bytes = 64 * 1024 * 1024;
      max_components = 1_024;
      max_features = 1_000_000;
      max_string_bytes = 1_048_576;
      max_metadata_entries = 128;
    }

  let current_producer_version = "0.3.0"

  let limits ?(max_bytes = default_limits.max_bytes)
      ?(max_components = default_limits.max_components)
      ?(max_features = default_limits.max_features)
      ?(max_string_bytes = default_limits.max_string_bytes)
      ?(max_metadata_entries = default_limits.max_metadata_entries) () =
    let values =
      [
        ("max_bytes", max_bytes);
        ("max_components", max_components);
        ("max_features", max_features);
        ("max_string_bytes", max_string_bytes);
        ("max_metadata_entries", max_metadata_entries);
      ]
    in
    match List.find_opt (fun (_, value) -> value <= 0) values with
    | Some (name, _) ->
        Error
          (Error.make ~remediation:"choose positive artifact reader limits"
             (Error.Validation
                { name = "artifact " ^ name; reason = "must be positive" }))
    | None ->
        Ok
          {
            max_bytes;
            max_components;
            max_features;
            max_string_bytes;
            max_metadata_entries;
          }

  let valid_utf8 value =
    let length = String.length value in
    let continuation index =
      index < length && Char.code value.[index] land 0xc0 = 0x80
    in
    let rec loop index =
      if index = length then true
      else
        let byte = Char.code value.[index] in
        if byte < 0x80 then loop (index + 1)
        else if byte >= 0xc2 && byte <= 0xdf then
          continuation (index + 1) && loop (index + 2)
        else if byte = 0xe0 then
          index + 2 < length
          && Char.code value.[index + 1] >= 0xa0
          && Char.code value.[index + 1] <= 0xbf
          && continuation (index + 2)
          && loop (index + 3)
        else if (byte >= 0xe1 && byte <= 0xec) || (byte >= 0xee && byte <= 0xef)
        then
          continuation (index + 1)
          && continuation (index + 2)
          && loop (index + 3)
        else if byte = 0xed then
          index + 2 < length
          && Char.code value.[index + 1] >= 0x80
          && Char.code value.[index + 1] <= 0x9f
          && continuation (index + 2)
          && loop (index + 3)
        else if byte = 0xf0 then
          index + 3 < length
          && Char.code value.[index + 1] >= 0x90
          && Char.code value.[index + 1] <= 0xbf
          && continuation (index + 2)
          && continuation (index + 3)
          && loop (index + 4)
        else if byte >= 0xf1 && byte <= 0xf3 then
          continuation (index + 1)
          && continuation (index + 2)
          && continuation (index + 3)
          && loop (index + 4)
        else if byte = 0xf4 then
          index + 3 < length
          && Char.code value.[index + 1] >= 0x80
          && Char.code value.[index + 1] <= 0x8f
          && continuation (index + 2)
          && continuation (index + 3)
          && loop (index + 4)
        else false
    in
    loop 0

  let validate_text ~name value =
    if valid_utf8 value then Ok ()
    else
      Error
        (artifact_error ~operation:"encode"
           ~reason:(name ^ " is not valid UTF-8")
           ~remediation:"use valid UTF-8 for artifact names and metadata")

  let empty_metadata =
    {
      training_rows = None;
      root_seed = None;
      sample_weighted = None;
      labels = [||];
    }

  let metadata ?training_rows ?root_seed ?sample_weighted ?(labels = [||]) () =
    match training_rows with
    | Some rows when rows <= 0 ->
        Error
          (Error.make
             ~remediation:"provide a positive training row count or omit it"
             (Error.Validation
                { name = "artifact training_rows"; reason = "must be positive" }))
    | _ ->
        let rec validate index =
          if index = Array.length labels then Ok ()
          else
            let key, value = labels.(index) in
            let* () = validate_text ~name:"metadata key" key in
            let* () = validate_text ~name:"metadata value" value in
            validate (index + 1)
        in
        let* () = validate 0 in
        Ok
          {
            training_rows;
            root_seed;
            sample_weighted;
            labels = Array.copy labels;
          }

  let model loaded = loaded.model
  let metadata_of_loaded loaded = loaded.metadata
  let producer_version loaded = loaded.producer_version
  let producer_ocaml_version loaded = loaded.producer_ocaml_version
  let training_rows metadata = metadata.training_rows
  let root_seed metadata = metadata.root_seed
  let sample_weighted metadata = metadata.sample_weighted
  let labels metadata = Array.copy metadata.labels

  module Writer = struct
    let create () = Buffer.create 256
    let u8 buffer value = Buffer.add_char buffer (Char.chr value)

    let i64 buffer value =
      for shift = 7 downto 0 do
        u8 buffer
          (Int64.to_int
             (Int64.logand (Int64.shift_right_logical value (shift * 8)) 0xffL))
      done

    let length buffer value = i64 buffer (Int64.of_int value)
    let bool buffer value = u8 buffer (if value then 1 else 0)
    let float buffer value = i64 buffer (Int64.bits_of_float value)

    let string buffer value =
      length buffer (String.length value);
      Buffer.add_string buffer value

    let bytes buffer value =
      length buffer (Bytes.length value);
      Buffer.add_bytes buffer value

    let int buffer value = i64 buffer (Int64.of_int value)

    let float_array buffer values =
      length buffer (Array.length values);
      Array.iter (float buffer) values

    let int_array buffer values =
      length buffer (Array.length values);
      Array.iter (int buffer) values

    let contents buffer = Bytes.of_string (Buffer.contents buffer)
  end

  module Reader = struct
    type t = { bytes : bytes; limits : limits; mutable position : int }

    let create ~limits bytes = { bytes; limits; position = 0 }
    let remaining reader = Bytes.length reader.bytes - reader.position

    let require reader count =
      if count < 0 || count > remaining reader then
        failure ~operation:"decode" "artifact is truncated"
      else Ok ()

    let u8 reader =
      let* () = require reader 1 in
      let value = Char.code (Bytes.get reader.bytes reader.position) in
      reader.position <- reader.position + 1;
      Ok value

    let i64 reader =
      let* () = require reader 8 in
      let value = ref 0L in
      for _ = 0 to 7 do
        value :=
          Int64.logor
            (Int64.shift_left !value 8)
            (Int64.of_int (Char.code (Bytes.get reader.bytes reader.position)));
        reader.position <- reader.position + 1
      done;
      Ok !value

    let bounded_length reader ~name ~maximum =
      let* value = i64 reader in
      if value < 0L || value > Int64.of_int maximum then
        failure ~operation:"decode"
          (Format.sprintf "%s exceeds its configured limit" name)
      else Ok (Int64.to_int value)

    let bool reader =
      let* value = u8 reader in
      match value with
      | 0 -> Ok false
      | 1 -> Ok true
      | _ -> failure ~operation:"decode" "invalid Boolean encoding"

    let float reader =
      let* bits = i64 reader in
      Ok (Int64.float_of_bits bits)

    let string reader =
      let* length =
        bounded_length reader ~name:"string length"
          ~maximum:reader.limits.max_string_bytes
      in
      let* () = require reader length in
      let value = Bytes.sub_string reader.bytes reader.position length in
      reader.position <- reader.position + length;
      if valid_utf8 value then Ok value
      else failure ~operation:"decode" "artifact string is not valid UTF-8"

    let bytes reader =
      let* length =
        bounded_length reader ~name:"component payload length"
          ~maximum:reader.limits.max_bytes
      in
      let* () = require reader length in
      let value = Bytes.sub reader.bytes reader.position length in
      reader.position <- reader.position + length;
      Ok value

    let int reader =
      let* value = i64 reader in
      if value < Int64.of_int min_int || value > Int64.of_int max_int then
        failure ~operation:"decode"
          "integer is not representable on this target"
      else Ok (Int64.to_int value)

    let float_array reader =
      let* length =
        bounded_length reader ~name:"numeric vector length"
          ~maximum:reader.limits.max_features
      in
      let* () =
        if length > remaining reader / 8 then
          failure ~operation:"decode" "numeric vector is truncated"
        else Ok ()
      in
      let values = Array.make length 0.0 in
      let rec loop index =
        if index = length then Ok values
        else
          let* value = float reader in
          values.(index) <- value;
          loop (index + 1)
      in
      loop 0

    let int_array reader =
      let* length =
        bounded_length reader ~name:"index vector length"
          ~maximum:reader.limits.max_features
      in
      let* () =
        if length > remaining reader / 8 then
          failure ~operation:"decode" "index vector is truncated"
        else Ok ()
      in
      let values = Array.make length 0 in
      let rec loop index =
        if index = length then Ok values
        else
          let* value = int reader in
          values.(index) <- value;
          loop (index + 1)
      in
      loop 0

    let finish reader =
      if remaining reader = 0 then Ok ()
      else failure ~operation:"decode" "artifact contains trailing data"
  end

  let validate_encode_text limits ~name value =
    if String.length value > limits.max_string_bytes then
      failure ~operation:"encode" (name ^ " exceeds the reader limit")
    else validate_text ~name value

  let write_schema writer schema =
    Writer.length writer (Feature_schema.feature_count schema);
    (match Feature_schema.names schema with
    | None -> Writer.bool writer false
    | Some names ->
        Writer.bool writer true;
        Writer.length writer (Feature_names.length names);
        for index = 0 to Feature_names.length names - 1 do
          Writer.string writer
            (Feature_name.to_string (Feature_names.get names index))
        done);
    Writer.string writer
      (Schema_fingerprint.to_string (Feature_schema.fingerprint schema))

  let validate_schema_for_encode limits schema =
    if Feature_schema.feature_count schema > limits.max_features then
      failure ~operation:"encode" "feature count exceeds the reader limit"
    else
      match Feature_schema.names schema with
      | None -> Ok ()
      | Some names ->
          let rec validate index =
            if index = Feature_names.length names then Ok ()
            else
              let name =
                Feature_name.to_string (Feature_names.get names index)
              in
              let* () = validate_encode_text limits ~name:"feature name" name in
              validate (index + 1)
          in
          validate 0

  let read_schema reader =
    let* feature_count =
      Reader.bounded_length reader ~name:"feature count"
        ~maximum:reader.Reader.limits.max_features
    in
    let* named = Reader.bool reader in
    let* schema =
      if not named then
        Feature_schema.anonymous ~feature_count
        |> Result.map_error (fun error ->
            Error.of_data_error
              ~remediation:"use a valid artifact feature schema" error)
      else
        let* name_count =
          Reader.bounded_length reader ~name:"feature-name count"
            ~maximum:reader.Reader.limits.max_features
        in
        if name_count <> feature_count then
          failure ~operation:"decode"
            "feature-name count does not match feature count"
        else
          let names = Array.make name_count "" in
          let rec read index =
            if index = name_count then Ok ()
            else
              let* name = Reader.string reader in
              names.(index) <- name;
              read (index + 1)
          in
          let* () = read 0 in
          let* names =
            Feature_names.create ~expected_count:feature_count names
            |> Result.map_error (fun error ->
                Error.of_data_error
                  ~remediation:"use unique, non-empty artifact feature names"
                  error)
          in
          Ok (Feature_schema.named names)
    in
    let* fingerprint = Reader.string reader in
    let expected =
      Schema_fingerprint.to_string (Feature_schema.fingerprint schema)
    in
    if String.equal fingerprint expected then Ok schema
    else failure ~operation:"decode" "feature-schema fingerprint does not match"

  let write_report writer report =
    Writer.bool writer (Solver_report.converged report);
    Writer.int writer (Solver_report.iterations report);
    Writer.float writer (Solver_report.objective report);
    Writer.u8 writer
      (match Solver_report.stopping_reason report with
      | Solver_report.Direct_solution -> 0
      | Solver_report.Gradient_tolerance -> 1
      | Solver_report.Step_tolerance -> 2);
    match Solver_report.rank report with
    | None -> Writer.bool writer false
    | Some rank ->
        Writer.bool writer true;
        Writer.int writer rank

  let read_report reader =
    let* converged = Reader.bool reader in
    let* iterations = Reader.int reader in
    let* objective = Reader.float reader in
    let* stopping_tag = Reader.u8 reader in
    let* stopping_reason =
      match stopping_tag with
      | 0 -> Ok Solver_report.Direct_solution
      | 1 -> Ok Solver_report.Gradient_tolerance
      | 2 -> Ok Solver_report.Step_tolerance
      | _ -> failure ~operation:"decode" "unknown solver stopping reason"
    in
    let* has_rank = Reader.bool reader in
    let* rank =
      if has_rank then Result.map Option.some (Reader.int reader) else Ok None
    in
    if (not converged) || iterations < 0 || not (Float.is_finite objective) then
      failure ~operation:"decode" "invalid solver report"
    else
      match rank with
      | Some value when value < 0 ->
          failure ~operation:"decode" "solver rank is negative"
      | _ ->
          Ok
            {
              Solver_report.converged;
              iterations;
              objective;
              stopping_reason;
              rank;
            }

  let component tag payload =
    {
      Pipeline.component_tag = tag;
      component_version = 1;
      component_payload = payload;
    }

  let encode_simple_imputer fitted =
    let writer = Writer.create () in
    let params = Simple_imputer.fitted_params fitted in
    (match params.Simple_imputer.strategy with
    | Simple_imputer.Mean -> Writer.u8 writer 0
    | Simple_imputer.Median -> Writer.u8 writer 1
    | Simple_imputer.Constant value ->
        Writer.u8 writer 2;
        Writer.float writer value);
    Writer.float_array writer
      (Vector.to_array (Simple_imputer.statistics fitted));
    Ok (component 1 (Writer.contents writer))

  let encode_standard_scaler fitted =
    let writer = Writer.create () in
    let params = Standard_scaler.fitted_params fitted in
    Writer.bool writer params.Standard_scaler.with_mean;
    Writer.bool writer params.Standard_scaler.with_std;
    Writer.float_array writer (Vector.to_array (Standard_scaler.mean fitted));
    Writer.float_array writer
      (Vector.to_array (Standard_scaler.variance fitted));
    Writer.float_array writer (Vector.to_array (Standard_scaler.scale fitted));
    Ok (component 2 (Writer.contents writer))

  let encode_variance_threshold fitted =
    let writer = Writer.create () in
    let params = Variance_threshold.fitted_params fitted in
    Writer.float writer params.Variance_threshold.threshold;
    Writer.float_array writer
      (Vector.to_array (Variance_threshold.variances fitted));
    Writer.int_array writer (Variance_threshold.selected_indices fitted);
    Ok (component 3 (Writer.contents writer))

  let encode_linear_regression fitted =
    let writer = Writer.create () in
    let params = Linear_regression.fitted_params fitted in
    Writer.bool writer params.Linear_regression.fit_intercept;
    Writer.float_array writer
      (Vector.to_array (Linear_regression.coefficients fitted));
    Writer.float writer (Linear_regression.intercept fitted);
    write_report writer (Linear_regression.report fitted);
    Ok (component 16 (Writer.contents writer))

  let encode_ridge_regression fitted =
    let writer = Writer.create () in
    let params = Ridge_regression.fitted_params fitted in
    Writer.float writer params.Ridge_regression.alpha;
    Writer.bool writer params.Ridge_regression.fit_intercept;
    Writer.float_array writer
      (Vector.to_array (Ridge_regression.coefficients fitted));
    Writer.float writer (Ridge_regression.intercept fitted);
    write_report writer (Ridge_regression.report fitted);
    Ok (component 17 (Writer.contents writer))

  let encode_logistic_regression fitted =
    let writer = Writer.create () in
    let params = Logistic_regression.fitted_params fitted in
    Writer.float writer params.Logistic_regression.c;
    Writer.bool writer params.Logistic_regression.fit_intercept;
    Writer.float writer params.Logistic_regression.tolerance;
    Writer.int writer params.Logistic_regression.max_iterations;
    Writer.float_array writer
      (Vector.to_array (Logistic_regression.coefficients fitted));
    Writer.float writer (Logistic_regression.intercept fitted);
    Writer.int_array writer (Logistic_regression.classes fitted);
    write_report writer (Logistic_regression.report fitted);
    Ok (component 18 (Writer.contents writer))

  let simple_imputer_stage ~name specification =
    Pipeline.transformer_internal ~encode:encode_simple_imputer ~name
      (module Simple_imputer)
      specification

  let standard_scaler_stage ~name specification =
    Pipeline.transformer_internal ~encode:encode_standard_scaler ~name
      (module Standard_scaler)
      specification

  let variance_threshold_stage ~name specification =
    Pipeline.transformer_internal ~encode:encode_variance_threshold ~name
      (module Variance_threshold)
      specification

  let linear_regression_estimator ~name specification =
    Pipeline.estimator_internal ~encode:encode_linear_regression ~name
      (module Linear_regression)
      specification

  let ridge_regression_estimator ~name specification =
    Pipeline.estimator_internal ~encode:encode_ridge_regression ~name
      (module Ridge_regression)
      specification

  let logistic_regression_estimator ~name specification =
    Pipeline.estimator_internal ~encode:encode_logistic_regression ~name
      (module Logistic_regression)
      ~decision_function:Logistic_regression.decision_function
      ~predict_proba:Logistic_regression.predict_proba
      ~classes:Logistic_regression.classes specification

  let validate_metadata limits metadata =
    if Array.length metadata.labels > limits.max_metadata_entries then
      failure ~operation:"encode"
        "metadata entry count exceeds the reader limit"
    else
      let rec validate index =
        if index = Array.length metadata.labels then Ok ()
        else
          let key, value = metadata.labels.(index) in
          if
            String.length key > limits.max_string_bytes
            || String.length value > limits.max_string_bytes
          then
            failure ~operation:"encode"
              "metadata string exceeds the reader limit"
          else
            let* () = validate_text ~name:"metadata key" key in
            let* () = validate_text ~name:"metadata value" value in
            validate (index + 1)
      in
      validate 0

  let write_metadata writer metadata =
    (match metadata.training_rows with
    | None -> Writer.bool writer false
    | Some rows ->
        Writer.bool writer true;
        Writer.length writer rows);
    (match metadata.root_seed with
    | None -> Writer.bool writer false
    | Some seed ->
        Writer.bool writer true;
        Writer.i64 writer (Seed.to_int64 seed));
    (match metadata.sample_weighted with
    | None -> Writer.u8 writer 0
    | Some false -> Writer.u8 writer 1
    | Some true -> Writer.u8 writer 2);
    Writer.length writer (Array.length metadata.labels);
    Array.iter
      (fun (key, value) ->
        Writer.string writer key;
        Writer.string writer value)
      metadata.labels

  let read_metadata reader =
    let* has_rows = Reader.bool reader in
    let* training_rows =
      if has_rows then
        let* rows =
          Reader.bounded_length reader ~name:"training row count"
            ~maximum:max_int
        in
        if rows = 0 then
          failure ~operation:"decode" "training row count is zero"
        else Ok (Some rows)
      else Ok None
    in
    let* has_seed = Reader.bool reader in
    let* root_seed =
      if has_seed then
        Result.map (fun value -> Some (Seed.of_int64 value)) (Reader.i64 reader)
      else Ok None
    in
    let* weighted_tag = Reader.u8 reader in
    let* sample_weighted =
      match weighted_tag with
      | 0 -> Ok None
      | 1 -> Ok (Some false)
      | 2 -> Ok (Some true)
      | _ -> failure ~operation:"decode" "invalid sample-weight metadata"
    in
    let* count =
      Reader.bounded_length reader ~name:"metadata entry count"
        ~maximum:reader.Reader.limits.max_metadata_entries
    in
    let labels = Array.make count ("", "") in
    let rec read index =
      if index = count then Ok ()
      else
        let* key = Reader.string reader in
        let* value = Reader.string reader in
        labels.(index) <- (key, value);
        read (index + 1)
    in
    let* () = read 0 in
    Ok { training_rows; root_seed; sample_weighted; labels }

  let write_component writer encoded =
    Writer.u8 writer encoded.Pipeline.component_tag;
    Writer.length writer encoded.Pipeline.component_version;
    Writer.bytes writer encoded.Pipeline.component_payload

  let read_component reader =
    let* component_tag = Reader.u8 reader in
    let* component_version =
      Reader.bounded_length reader ~name:"component version" ~maximum:65535
    in
    let* component_payload = Reader.bytes reader in
    if component_version <> 1 then
      failure ~operation:"decode" "unsupported component codec version"
    else Ok { Pipeline.component_tag; component_version; component_payload }

  let encode_payload ~limits ~task ?(metadata = empty_metadata) fitted =
    let* () = validate_metadata limits metadata in
    let* () =
      validate_schema_for_encode limits fitted.Pipeline.pipeline_input_schema
    in
    let* () =
      validate_schema_for_encode limits fitted.Pipeline.pipeline_output_schema
    in
    let transformer_count = Array.length fitted.Pipeline.fitted_transformers in
    if transformer_count >= limits.max_components then
      failure ~operation:"encode"
        "pipeline component count exceeds the reader limit"
    else
      let writer = Writer.create () in
      Writer.string writer current_producer_version;
      (* Omitting compiler diagnostics keeps identical fitted data byte-stable across supported OCaml versions. *)
      Writer.string writer "";
      write_metadata writer metadata;
      write_schema writer fitted.Pipeline.pipeline_input_schema;
      write_schema writer fitted.Pipeline.pipeline_output_schema;
      Writer.length writer transformer_count;
      let rec write_transformer index =
        if index = transformer_count then Ok ()
        else
          let transformer = fitted.Pipeline.fitted_transformers.(index) in
          let* () =
            validate_encode_text limits ~name:"pipeline stage name"
              transformer.Pipeline.stage_name
          in
          let* () =
            validate_schema_for_encode limits
              transformer.Pipeline.transform_input_schema
          in
          let* () =
            validate_schema_for_encode limits
              transformer.Pipeline.transform_output_schema
          in
          let* encoded =
            match transformer.Pipeline.encode_transformer with
            | Some encode -> encode ()
            | None ->
                failure ~operation:"encode"
                  (Format.sprintf "pipeline stage %S has no artifact codec"
                     transformer.Pipeline.stage_name)
          in
          Writer.string writer transformer.Pipeline.stage_name;
          write_schema writer transformer.Pipeline.transform_input_schema;
          write_schema writer transformer.Pipeline.transform_output_schema;
          write_component writer encoded;
          write_transformer (index + 1)
      in
      let* () = write_transformer 0 in
      let terminal = fitted.Pipeline.fitted_estimator in
      let* () =
        validate_encode_text limits ~name:"terminal estimator name"
          terminal.Pipeline.terminal_name
      in
      let* encoded =
        match terminal.Pipeline.encode_estimator with
        | Some encode -> encode ()
        | None ->
            failure ~operation:"encode"
              (Format.sprintf "terminal estimator %S has no artifact codec"
                 terminal.Pipeline.terminal_name)
      in
      Writer.string writer terminal.Pipeline.terminal_name;
      write_component writer encoded;
      let payload = Writer.contents writer in
      let envelope_size = 36 in
      if Bytes.length payload > limits.max_bytes - envelope_size then
        failure ~operation:"encode" "artifact exceeds the configured byte limit"
      else
        let envelope = Writer.create () in
        Buffer.add_string envelope "MDLKIT01";
        Writer.u8 envelope 1;
        Writer.u8 envelope 0;
        Writer.u8 envelope task;
        Writer.length envelope (Bytes.length payload);
        Writer.u8 envelope 1;
        Buffer.add_string envelope (Digest.string (Bytes.to_string payload));
        Buffer.add_bytes envelope payload;
        Ok (Writer.contents envelope)

  let encode_regression ?metadata ?(limits = default_limits) fitted =
    encode_payload ~limits ~task:1 ?metadata fitted

  let encode_binary_classification ?metadata ?(limits = default_limits) fitted =
    encode_payload ~limits ~task:2 ?metadata fitted

  let finite_array values = Array.for_all Float.is_finite values

  let validate_component_reader reader =
    let* () = Reader.finish reader in
    Ok ()

  let decode_transformer ~limits ~name ~input_schema ~output_schema encoded =
    let reader = Reader.create ~limits encoded.Pipeline.component_payload in
    match encoded.Pipeline.component_tag with
    | 1 ->
        let* strategy_tag = Reader.u8 reader in
        let* strategy =
          match strategy_tag with
          | 0 -> Ok Simple_imputer.Mean
          | 1 -> Ok Simple_imputer.Median
          | 2 ->
              Result.map
                (fun value -> Simple_imputer.Constant value)
                (Reader.float reader)
          | _ -> failure ~operation:"decode" "unknown imputer strategy"
        in
        let* statistics = Reader.float_array reader in
        let* () = validate_component_reader reader in
        let valid_strategy =
          match strategy with
          | Simple_imputer.Constant value -> Float.is_finite value
          | Simple_imputer.Mean | Simple_imputer.Median -> true
        in
        if
          Array.length statistics <> Feature_schema.feature_count input_schema
          || (not (Feature_schema.equal input_schema output_schema))
          || (not (finite_array statistics))
          || not valid_strategy
        then failure ~operation:"decode" "invalid fitted imputer payload"
        else
          let fitted : Simple_imputer.fitted =
            {
              Simple_imputer.params = { Simple_imputer.strategy };
              statistics = Vector.of_array statistics;
              schema = input_schema;
            }
          in
          Ok
            {
              Pipeline.stage_name = name;
              transform_input_schema = input_schema;
              transform_output_schema = output_schema;
              apply_transform = Simple_imputer.transform fitted;
              encode_transformer = Some (fun () -> encode_simple_imputer fitted);
            }
    | 2 ->
        let* with_mean = Reader.bool reader in
        let* with_std = Reader.bool reader in
        let* mean = Reader.float_array reader in
        let* variance = Reader.float_array reader in
        let* scale = Reader.float_array reader in
        let* () = validate_component_reader reader in
        let width = Feature_schema.feature_count input_schema in
        let valid_scale value = Float.is_finite value && value > 0.0 in
        if
          Array.length mean <> width
          || Array.length variance <> width
          || Array.length scale <> width
          || (not (Feature_schema.equal input_schema output_schema))
          || (not (finite_array mean))
          || (not
                (Array.for_all
                   (fun value -> Float.is_finite value && value >= 0.0)
                   variance))
          || not (Array.for_all valid_scale scale)
        then
          failure ~operation:"decode" "invalid fitted standard-scaler payload"
        else
          let fitted : Standard_scaler.fitted =
            {
              Standard_scaler.params = { Standard_scaler.with_mean; with_std };
              mean = Vector.of_array mean;
              variance = Vector.of_array variance;
              scale = Vector.of_array scale;
              schema = input_schema;
            }
          in
          Ok
            {
              Pipeline.stage_name = name;
              transform_input_schema = input_schema;
              transform_output_schema = output_schema;
              apply_transform = Standard_scaler.transform fitted;
              encode_transformer =
                Some (fun () -> encode_standard_scaler fitted);
            }
    | 3 ->
        let* threshold = Reader.float reader in
        let* variances = Reader.float_array reader in
        let* selected = Reader.int_array reader in
        let* () = validate_component_reader reader in
        let width = Feature_schema.feature_count input_schema in
        let rec valid_indices index =
          index = Array.length selected
          || selected.(index) >= 0
             && selected.(index) < width
             && (index = 0 || selected.(index - 1) < selected.(index))
             && valid_indices (index + 1)
        in
        if
          (not (Float.is_finite threshold))
          || threshold < 0.0
          || Array.length variances <> width
          || Array.length selected = 0
          || Array.length selected <> Feature_schema.feature_count output_schema
          || (not
                (Array.for_all
                   (fun value -> Float.is_finite value && value >= 0.0)
                   variances))
          || not (valid_indices 0)
        then
          failure ~operation:"decode"
            "invalid fitted variance-threshold payload"
        else
          let expected_output =
            Preprocessing_internal.subset_schema input_schema selected
          in
          let* expected_output = expected_output in
          if not (Feature_schema.equal expected_output output_schema) then
            failure ~operation:"decode"
              "variance-threshold output schema does not match selected \
               features"
          else
            let fitted : Variance_threshold.fitted =
              {
                Variance_threshold.params = { Variance_threshold.threshold };
                variances = Vector.of_array variances;
                selected;
                input_schema;
                output_schema;
              }
            in
            Ok
              {
                Pipeline.stage_name = name;
                transform_input_schema = input_schema;
                transform_output_schema = output_schema;
                apply_transform = Variance_threshold.transform fitted;
                encode_transformer =
                  Some (fun () -> encode_variance_threshold fitted);
              }
    | _ -> failure ~operation:"decode" "unknown transformer component tag"

  let validate_estimator_values schema coefficients intercept =
    Array.length coefficients = Feature_schema.feature_count schema
    && finite_array coefficients && Float.is_finite intercept

  let decode_linear ~limits ~name ~schema ~ridge encoded =
    let reader = Reader.create ~limits encoded.Pipeline.component_payload in
    let* alpha = if ridge then Reader.float reader else Ok 0.0 in
    let* fit_intercept = Reader.bool reader in
    let* coefficients = Reader.float_array reader in
    let* intercept = Reader.float reader in
    let* report = read_report reader in
    let* () = Reader.finish reader in
    if
      (not (Float.is_finite alpha))
      || alpha < 0.0
      || (not (validate_estimator_values schema coefficients intercept))
      || Solver_report.stopping_reason report <> Solver_report.Direct_solution
      || Solver_report.rank report = None
      || Solver_report.iterations report <> 1
      || Solver_report.objective report < 0.0
      || Option.get (Solver_report.rank report)
         > Feature_schema.feature_count schema
    then failure ~operation:"decode" "invalid fitted linear-model payload"
    else if ridge then
      let fitted : Ridge_regression.fitted =
        {
          Ridge_regression.ridge_params =
            { Ridge_regression.alpha; fit_intercept };
          ridge_coefficients = coefficients;
          ridge_intercept = intercept;
          ridge_schema = schema;
          ridge_report = report;
        }
      in
      Ok
        {
          Pipeline.terminal_name = name;
          terminal_predict = Ridge_regression.predict fitted;
          terminal_decision_function = None;
          terminal_predict_proba = None;
          terminal_classes = None;
          encode_estimator = Some (fun () -> encode_ridge_regression fitted);
        }
    else
      let fitted : Linear_regression.fitted =
        {
          Linear_regression.linear_params = { Linear_regression.fit_intercept };
          linear_coefficients = coefficients;
          linear_intercept = intercept;
          linear_schema = schema;
          linear_report = report;
        }
      in
      Ok
        {
          Pipeline.terminal_name = name;
          terminal_predict = Linear_regression.predict fitted;
          terminal_decision_function = None;
          terminal_predict_proba = None;
          terminal_classes = None;
          encode_estimator = Some (fun () -> encode_linear_regression fitted);
        }

  let decode_regression_estimator ~limits ~name ~schema encoded =
    match encoded.Pipeline.component_tag with
    | 16 -> decode_linear ~limits ~name ~schema ~ridge:false encoded
    | 17 -> decode_linear ~limits ~name ~schema ~ridge:true encoded
    | _ ->
        failure ~operation:"decode"
          "artifact terminal is not a regression estimator"

  let decode_classification_estimator ~limits ~name ~schema encoded =
    if encoded.Pipeline.component_tag <> 18 then
      failure ~operation:"decode"
        "artifact terminal is not a binary-classification estimator"
    else
      let reader = Reader.create ~limits encoded.Pipeline.component_payload in
      let* c = Reader.float reader in
      let* fit_intercept = Reader.bool reader in
      let* tolerance = Reader.float reader in
      let* max_iterations = Reader.int reader in
      let* coefficients = Reader.float_array reader in
      let* intercept = Reader.float reader in
      let* classes = Reader.int_array reader in
      let* report = read_report reader in
      let* () = Reader.finish reader in
      if
        (not (Float.is_finite c))
        || c <= 0.0
        || (not (Float.is_finite tolerance))
        || tolerance <= 0.0 || max_iterations <= 0
        || Array.length classes <> 2
        || classes.(0) >= classes.(1)
        || (not (validate_estimator_values schema coefficients intercept))
        || Solver_report.rank report <> None
        || Solver_report.stopping_reason report = Solver_report.Direct_solution
        || Solver_report.iterations report > max_iterations
        || Solver_report.objective report < 0.0
      then
        failure ~operation:"decode" "invalid fitted logistic-regression payload"
      else
        let fitted : Logistic_regression.fitted =
          {
            Logistic_regression.logistic_params =
              {
                Logistic_regression.c;
                fit_intercept;
                tolerance;
                max_iterations;
              };
            logistic_coefficients = coefficients;
            logistic_intercept = intercept;
            logistic_classes = (classes.(0), classes.(1));
            logistic_schema = schema;
            logistic_report = report;
          }
        in
        Ok
          {
            Pipeline.terminal_name = name;
            terminal_predict = Logistic_regression.predict fitted;
            terminal_decision_function =
              Some (Logistic_regression.decision_function fitted);
            terminal_predict_proba =
              Some (Logistic_regression.predict_proba fitted);
            terminal_classes =
              Some (fun () -> Logistic_regression.classes fitted);
            encode_estimator =
              Some (fun () -> encode_logistic_regression fitted);
          }

  let parse_envelope ~limits ~expected_task bytes =
    if Bytes.length bytes > limits.max_bytes then
      failure ~operation:"decode" "artifact exceeds the configured byte limit"
    else if Bytes.length bytes < 36 then
      failure ~operation:"decode" "artifact is truncated"
    else if not (String.equal (Bytes.sub_string bytes 0 8) "MDLKIT01") then
      failure ~operation:"decode" "artifact magic does not match ModelKit"
    else
      let reader = Reader.create ~limits bytes in
      reader.Reader.position <- 8;
      let* major = Reader.u8 reader in
      let* minor = Reader.u8 reader in
      let* task = Reader.u8 reader in
      if major <> 1 || minor <> 0 then
        failure ~operation:"decode" "unsupported artifact container version"
      else if task <> expected_task then
        failure ~operation:"decode"
          "artifact task does not match the requested loader"
      else
        let* payload_length =
          Reader.bounded_length reader ~name:"artifact payload length"
            ~maximum:limits.max_bytes
        in
        let* checksum_algorithm = Reader.u8 reader in
        if checksum_algorithm <> 1 then
          failure ~operation:"decode" "unsupported checksum algorithm"
        else
          let* () = Reader.require reader 16 in
          let digest = Bytes.sub_string bytes reader.Reader.position 16 in
          reader.Reader.position <- reader.Reader.position + 16;
          if payload_length <> Reader.remaining reader then
            failure ~operation:"decode"
              "artifact payload length does not match framing"
          else
            let payload =
              Bytes.sub bytes reader.Reader.position payload_length
            in
            if
              not
                (String.equal digest (Digest.string (Bytes.to_string payload)))
            then failure ~operation:"decode" "artifact checksum does not match"
            else Ok payload

  let decode_payload ~limits ~decode_estimator payload =
    let reader = Reader.create ~limits payload in
    let* producer_version = Reader.string reader in
    let* producer_ocaml_version = Reader.string reader in
    let producer_ocaml_version =
      if String.length producer_ocaml_version = 0 then None
      else Some producer_ocaml_version
    in
    let* metadata = read_metadata reader in
    let* pipeline_input_schema = read_schema reader in
    let* pipeline_output_schema = read_schema reader in
    let* transformer_count =
      Reader.bounded_length reader ~name:"pipeline stage count"
        ~maximum:(limits.max_components - 1)
    in
    let transformers = Array.make transformer_count None in
    let stage_names = Hashtbl.create (transformer_count + 1) in
    let rec read_transformer index expected_schema =
      if index = transformer_count then Ok expected_schema
      else
        let* name = Reader.string reader in
        let* input_schema = read_schema reader in
        let* output_schema = read_schema reader in
        let* encoded = read_component reader in
        if String.length (String.trim name) = 0 then
          failure ~operation:"decode" "pipeline stage name is blank"
        else if Hashtbl.mem stage_names name then
          failure ~operation:"decode" "pipeline stage name is duplicated"
        else if not (Feature_schema.equal input_schema expected_schema) then
          failure ~operation:"decode"
            "pipeline transformer schemas are discontinuous"
        else (
          Hashtbl.add stage_names name ();
          let* transformer =
            decode_transformer ~limits ~name ~input_schema ~output_schema
              encoded
          in
          transformers.(index) <- Some transformer;
          read_transformer (index + 1) output_schema)
    in
    let* final_schema = read_transformer 0 pipeline_input_schema in
    if not (Feature_schema.equal final_schema pipeline_output_schema) then
      failure ~operation:"decode" "pipeline output schema is inconsistent"
    else
      let* terminal_name = Reader.string reader in
      let* encoded = read_component reader in
      let* fitted_estimator =
        if String.length (String.trim terminal_name) = 0 then
          failure ~operation:"decode" "terminal estimator name is blank"
        else if Hashtbl.mem stage_names terminal_name then
          failure ~operation:"decode" "terminal estimator name is duplicated"
        else
          decode_estimator ~limits ~name:terminal_name
            ~schema:pipeline_output_schema encoded
      in
      let* () = Reader.finish reader in
      let fitted_transformers =
        Array.map
          (function Some transformer -> transformer | None -> assert false)
          transformers
      in
      let model =
        {
          Pipeline.fitted_transformers;
          fitted_estimator;
          pipeline_input_schema;
          pipeline_output_schema;
        }
      in
      Ok { model; metadata; producer_version; producer_ocaml_version }

  let decode_regression ?(limits = default_limits) bytes =
    let* payload = parse_envelope ~limits ~expected_task:1 bytes in
    decode_payload ~limits ~decode_estimator:decode_regression_estimator payload

  let decode_binary_classification ?(limits = default_limits) bytes =
    let* payload = parse_envelope ~limits ~expected_task:2 bytes in
    decode_payload ~limits ~decode_estimator:decode_classification_estimator
      payload

  let save_bytes ~path bytes =
    try
      let channel = open_out_bin path in
      try
        output_bytes channel bytes;
        close_out channel;
        Ok ()
      with error ->
        close_out_noerr channel;
        raise error
    with Sys_error reason ->
      Error
        (artifact_error ~operation:"save" ~reason
           ~remediation:"choose a writable destination path")

  let load_bytes ~limits ~path =
    try
      let channel = open_in_bin path in
      try
        let length = in_channel_length channel in
        if length > limits.max_bytes then (
          close_in channel;
          failure ~operation:"load" "artifact exceeds the configured byte limit")
        else
          let bytes = Bytes.create length in
          really_input channel bytes 0 length;
          close_in channel;
          Ok bytes
      with error ->
        close_in_noerr channel;
        raise error
    with
    | Sys_error reason ->
        Error
          (artifact_error ~operation:"load" ~reason
             ~remediation:"choose a readable ModelKit artifact path")
    | End_of_file -> failure ~operation:"load" "artifact file is truncated"

  let save_regression ?metadata ?limits ~path fitted =
    let* bytes = encode_regression ?metadata ?limits fitted in
    save_bytes ~path bytes

  let save_binary_classification ?metadata ?limits ~path fitted =
    let* bytes = encode_binary_classification ?metadata ?limits fitted in
    save_bytes ~path bytes

  let load_regression ?(limits = default_limits) ~path () =
    let* bytes = load_bytes ~limits ~path in
    decode_regression ~limits bytes

  let load_binary_classification ?(limits = default_limits) ~path () =
    let* bytes = load_bytes ~limits ~path in
    decode_binary_classification ~limits bytes
end
