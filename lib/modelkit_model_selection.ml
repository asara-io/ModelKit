open Modelkit_data
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
