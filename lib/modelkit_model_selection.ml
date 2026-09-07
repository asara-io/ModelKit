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
  type classification_response = Labels | Probabilities

  type 'prediction prediction_fold = {
    prediction_fold_index : int;
    prediction_fit_time : float;
    predict_time : float;
    prediction_test_indices : int array;
    prediction_result : ('prediction, failure) result;
  }

  type 'prediction prediction_report = {
    prediction_report_folds : 'prediction prediction_fold array;
    assembled_predictions : ('prediction, failure array) result;
  }

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

  let copy_prediction_fold fold =
    {
      fold with
      prediction_test_indices = Array.copy fold.prediction_test_indices;
    }

  let prediction_folds report =
    Array.map copy_prediction_fold report.prediction_report_folds

  let successful_prediction_fold_count report =
    Array.fold_left
      (fun count fold ->
        match fold.prediction_result with Ok _ -> count + 1 | Error _ -> count)
      0 report.prediction_report_folds

  let out_of_fold_predictions report =
    Result.map_error Array.copy report.assembled_predictions

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

  let validate_prediction_coverage ~rows splits =
    if Array.length splits = 0 then
      Error
        (validation ~name:"out-of-fold test coverage"
           ~reason:"the splitter produced no folds"
           ~remediation:
             "use a splitter whose test folds partition every source row")
    else
      let counts = Array.make rows 0 in
      Array.iter
        (fun split ->
          Array.iter
            (fun row -> counts.(row) <- counts.(row) + 1)
            (Row_view.indices (Split.test split)))
        splits;
      let rec check row =
        if row = rows then Ok ()
        else
          match counts.(row) with
          | 1 -> check (row + 1)
          | 0 ->
              Error
                (validation ~name:"out-of-fold test coverage"
                   ~reason:(Format.sprintf "source row %d is never tested" row)
                   ~remediation:
                     "use a splitter whose test folds partition every source \
                      row")
          | count ->
              Error
                (validation ~name:"out-of-fold test coverage"
                   ~reason:
                     (Format.sprintf "source row %d is tested %d times" row
                        count)
                   ~remediation:
                     "use a non-repeated splitter with disjoint test folds")
      in
      check 0

  let run_prediction ~failure_policy ~fit_seed ~execution ~metadata ~splitter
      ~seed ~predict_fold ~assemble pipeline dataset =
    let ( let* ) = Result.bind in
    let rows = Dataset.sample_count dataset in
    let* () = Metadata.validate ~rows metadata in
    Callback.run
      ~outcome:(fun report ->
        match report.assembled_predictions with
        | Ok _ -> Callback.Succeeded
        | Error failures -> Callback.Failed failures.(0).error)
      (Metadata.callback metadata)
      ~operation:Callback.Cross_validation
      (fun () ->
        let splitter_rng =
          Seed.derive seed ~operation:"cross-val-predict-splitter" ~index:0
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
        let* () = validate_prediction_coverage ~rows splits in
        let failed ~fold_index ~fit_time ~predict_time ~test_indices phase error
            =
          let error = contextualize fold_index error in
          if failure_policy = Abort || Callback.is_control_error error then
            Error error
          else
            Ok
              {
                prediction_fold_index = fold_index;
                prediction_fit_time = fit_time;
                predict_time;
                prediction_test_indices = test_indices;
                prediction_result = Error { phase; error };
              }
        in
        let evaluate metadata ~index:fold_index split =
          let test_indices = Row_view.indices (Split.test split) in
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
              failed ~fold_index ~fit_time:0.0 ~predict_time:0.0 ~test_indices
                Materialization error
          | Ok (train, test, train_metadata, test_metadata) -> (
              let fold_rng =
                Seed.derive fit_seed ~operation:"cross-val-predict-fold"
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
                  failed ~fold_index ~fit_time ~predict_time:0.0 ~test_indices
                    Fitting error
              | Ok fitted -> (
                  let predict_time, prediction =
                    timed (fun () ->
                        predict_fold ~fold_index ~metadata:test_metadata fitted
                          test)
                  in
                  match prediction with
                  | Error error ->
                      failed ~fold_index ~fit_time ~predict_time ~test_indices
                        (Prediction Test) error
                  | Ok prediction ->
                      Ok
                        {
                          prediction_fold_index = fold_index;
                          prediction_fit_time = fit_time;
                          predict_time;
                          prediction_test_indices = test_indices;
                          prediction_result = Ok prediction;
                        }))
        in
        let run_fold metadata ~index split =
          let metadata = Metadata.scope (Error.Fold index) metadata in
          Callback.run
            ~outcome:(fun fold ->
              match fold.prediction_result with
              | Ok _ -> Callback.Succeeded
              | Error failure -> Callback.Failed failure.error)
            (Metadata.callback metadata)
            ~operation:Callback.Fold
            (fun () -> evaluate metadata ~index split)
          |> Result.map_error (contextualize index)
        in
        let* prediction_report_folds =
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
        let assembled_predictions = assemble prediction_report_folds in
        Ok { prediction_report_folds; assembled_predictions })

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

  let prediction_length_error ~name ~expected ~observed =
    Error.make
      ~remediation:"return exactly one prediction for every selected test row"
      (Error.Shape_mismatch
         { name; expected = [ expected ]; observed = [ observed ] })

  let collected_prediction_failures folds =
    Array.fold_left
      (fun reversed fold ->
        match fold.prediction_result with
        | Ok _ -> reversed
        | Error failure -> failure :: reversed)
      [] folds
    |> List.rev |> Array.of_list

  let successful_predictions folds =
    let failures = collected_prediction_failures folds in
    if Array.length failures = 0 then Ok () else Error failures

  let assemble_regression ~rows folds =
    let ( let* ) = Result.bind in
    let* () = successful_predictions folds in
    let values = Array.make rows 0.0 in
    Array.iter
      (fun fold ->
        match fold.prediction_result with
        | Error _ -> assert false
        | Ok prediction ->
            let predicted = Target.regression_values prediction in
            Array.iteri
              (fun position row ->
                values.(row) <- Vector.get predicted position)
              fold.prediction_test_indices)
      folds;
    match Target.regression (Vector.of_array values) with
    | Ok prediction -> Ok prediction
    | Error _ -> assert false

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

    let cross_val_predict ?(failure_policy = Abort) ?fit_seed
        ?(execution = Execution.sequential) ?metadata ~splitter ~seed pipeline
        dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let fit_seed = Option.value fit_seed ~default:seed in
      let predict_fold ~fold_index:_ ~metadata fitted test =
        let ( let* ) = Result.bind in
        let expected = Dataset.sample_count test in
        let* prediction =
          Pipeline.predict_with_metadata fitted ~metadata
            ~feature_schema:(Dataset.feature_schema test)
            ~x:(Dataset.features test)
        in
        let observed = Target.length prediction in
        if observed = expected then Ok prediction
        else
          Error
            (prediction_length_error ~name:"out-of-fold regression prediction"
               ~expected ~observed)
      in
      run_prediction ~failure_policy ~fit_seed ~execution ~metadata ~splitter
        ~seed ~predict_fold
        ~assemble:(assemble_regression ~rows:(Dataset.sample_count dataset))
        pipeline dataset
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

  let sorted_unique_labels target =
    let values = Target.classification_values target in
    Array.sort Int.compare values;
    if Array.length values = 0 then [||]
    else
      let reversed = ref [ values.(0) ] in
      for index = 1 to Array.length values - 1 do
        if values.(index) <> values.(index - 1) then
          reversed := values.(index) :: !reversed
      done;
      Array.of_list (List.rev !reversed)

  let validate_prediction_classes ~binary classes =
    let count = Array.length classes in
    if (binary && count = 2) || ((not binary) && count >= 2) then Ok ()
    else
      Error
        (validation ~name:"out-of-fold classification classes"
           ~reason:
             (if binary then
                Format.sprintf "binary data contains %d distinct classes" count
              else
                Format.sprintf
                  "classification data contains %d distinct classes" count)
           ~remediation:
             (if binary then "provide targets containing exactly two classes"
              else "provide targets containing at least two classes"))

  let class_index classes =
    let index = Hashtbl.create (Array.length classes) in
    Array.iteri (fun column label -> Hashtbl.add index label column) classes;
    index

  let validate_predicted_labels ~classes prediction =
    let index = class_index classes in
    let labels = Target.classification_values prediction in
    let rec check row =
      if row = Array.length labels then Ok ()
      else if Hashtbl.mem index labels.(row) then check (row + 1)
      else
        Error
          (compatibility ~component:"out-of-fold classifier labels"
             ~reason:
               (Format.sprintf
                  "predicted label %d at row %d is absent from the dataset"
                  labels.(row) row)
             ~remediation:
               "return labels drawn from the complete dataset class set")
    in
    check 0

  let align_probability_columns ~classes ~fold_classes probabilities =
    let ( let* ) = Result.bind in
    let rows = Matrix.rows probabilities in
    let* () = validate_probability_columns fold_classes probabilities in
    let global = class_index classes in
    let destinations = Array.make (Array.length fold_classes) 0 in
    let seen = Hashtbl.create (Array.length fold_classes) in
    let rec map_columns column =
      if column = Array.length fold_classes then Ok ()
      else
        let label = fold_classes.(column) in
        if Hashtbl.mem seen label then
          Error
            (compatibility ~component:"out-of-fold probability class order"
               ~reason:
                 (Format.sprintf "class label %d is declared more than once"
                    label)
               ~remediation:"declare each fitted class exactly once")
        else
          match Hashtbl.find_opt global label with
          | None ->
              Error
                (compatibility ~component:"out-of-fold probability class order"
                   ~reason:
                     (Format.sprintf
                        "fitted class label %d is absent from the dataset" label)
                   ~remediation:
                     "declare only classes present in the complete dataset")
          | Some destination ->
              Hashtbl.add seen label ();
              destinations.(column) <- destination;
              map_columns (column + 1)
    in
    let* () = map_columns 0 in
    let sources = Array.make (Array.length classes) (-1) in
    Array.iteri
      (fun source destination -> sources.(destination) <- source)
      destinations;
    Matrix.init ~rows ~columns:(Array.length classes) (fun row column ->
        let source = sources.(column) in
        if source < 0 then 0.0 else Matrix.get probabilities row source)
    |> Result.map_error (fun error ->
        Error.of_data_error
          ~remediation:"return a valid classifier probability matrix" error)

  let assemble_classification ~response ~classes ~rows folds =
    let ( let* ) = Result.bind in
    let* () = successful_predictions folds in
    match response with
    | Labels -> (
        let labels = Array.make rows 0 in
        Array.iter
          (fun fold ->
            match fold.prediction_result with
            | Error _ -> assert false
            | Ok prediction ->
                let predicted =
                  Multiclass_prediction.labels prediction
                  |> Option.get |> Target.classification_values
                in
                Array.iteri
                  (fun position row -> labels.(row) <- predicted.(position))
                  fold.prediction_test_indices)
          folds;
        Multiclass_prediction.create ~labels:(Target.classification labels) ()
        |> function
        | Ok prediction -> Ok prediction
        | Error _ -> assert false)
    | Probabilities -> (
        let row_predictions = Array.make rows None in
        let row_positions = Array.make rows 0 in
        Array.iter
          (fun fold ->
            match fold.prediction_result with
            | Error _ -> assert false
            | Ok prediction ->
                let predicted =
                  Multiclass_prediction.probabilities prediction |> Option.get
                in
                Array.iteri
                  (fun position row ->
                    row_predictions.(row) <- Some predicted;
                    row_positions.(row) <- position)
                  fold.prediction_test_indices)
          folds;
        let probabilities =
          match
            Matrix.init ~rows ~columns:(Array.length classes) (fun row column ->
                Matrix.get
                  (Option.get row_predictions.(row))
                  row_positions.(row) column)
          with
          | Ok probabilities -> probabilities
          | Error _ -> assert false
        in
        Multiclass_prediction.create ~classes ~probabilities () |> function
        | Ok prediction -> Ok prediction
        | Error _ -> assert false)

  let classification_cross_val_predict ~binary ?(failure_policy = Abort)
      ?fit_seed ?(execution = Execution.sequential) ?metadata ~response
      ~splitter ~seed pipeline dataset =
    let ( let* ) = Result.bind in
    let classes = sorted_unique_labels (Dataset.target dataset) in
    let* () = validate_prediction_classes ~binary classes in
    let metadata =
      match metadata with
      | Some metadata -> metadata
      | None -> Metadata.of_dataset dataset
    in
    let fit_seed = Option.value fit_seed ~default:seed in
    let predict_fold ~fold_index:_ ~metadata fitted test =
      let expected = Dataset.sample_count test in
      match response with
      | Labels ->
          let* labels =
            Pipeline.predict_with_metadata fitted ~metadata
              ~feature_schema:(Dataset.feature_schema test)
              ~x:(Dataset.features test)
          in
          let observed = Target.length labels in
          let* () =
            if observed = expected then Ok ()
            else
              Error
                (prediction_length_error
                   ~name:"out-of-fold classification labels" ~expected ~observed)
          in
          let* () = validate_predicted_labels ~classes labels in
          Multiclass_prediction.create ~labels ()
      | Probabilities ->
          let* fold_classes = Pipeline.classes fitted in
          let* probabilities =
            Pipeline.predict_proba_with_metadata fitted ~metadata
              ~feature_schema:(Dataset.feature_schema test)
              ~x:(Dataset.features test)
          in
          let observed = Matrix.rows probabilities in
          let* () =
            if observed = expected then Ok ()
            else
              Error
                (prediction_length_error ~name:"out-of-fold class probabilities"
                   ~expected ~observed)
          in
          let* probabilities =
            align_probability_columns ~classes ~fold_classes probabilities
          in
          Multiclass_prediction.create ~classes ~probabilities ()
    in
    run_prediction ~failure_policy ~fit_seed ~execution ~metadata ~splitter
      ~seed ~predict_fold
      ~assemble:
        (assemble_classification ~response ~classes
           ~rows:(Dataset.sample_count dataset))
      pipeline dataset

  module Binary_classification = struct
    include Classification_evaluation (Binary_scoring)

    let cross_val_predict = classification_cross_val_predict ~binary:true
  end

  module Multiclass_classification = struct
    include Classification_evaluation (Multiclass_scoring)

    let cross_val_predict = classification_cross_val_predict ~binary:false
  end
end

module Search_checkpoint = struct
  module Wire = Modelkit_checkpoint_codec

  type entry = {
    stage : string;
    candidate_index : int;
    configuration_id : string;
    evaluation : (unit Cross_validation.report, Error.t) result;
  }

  type snapshot = {
    specification_id : string;
    identity : string option;
    plans : (string * string) array;
    entries : entry array;
  }

  type 'configuration t = {
    identify : 'configuration -> string;
    mutable state : snapshot;
    busy : bool Atomic.t;
    cached : (string * int, entry) Hashtbl.t;
    mutable reversed_entries : entry list;
  }

  let incompatible reason =
    Error
      (Error.make
         (Error.Compatibility { component = "search checkpoint"; reason })
         ~remediation:
           "resume with identical data, splits, seeds, options, and versioned \
            specifications, or start a new checkpoint")

  let copy_evaluation evaluation =
    Result.map
      (fun report ->
        { Cross_validation.report_folds = Cross_validation.folds report })
      evaluation

  let copy_entry entry =
    { entry with evaluation = copy_evaluation entry.evaluation }

  let copy state =
    {
      state with
      plans = Array.copy state.plans;
      entries = Array.map copy_entry state.entries;
    }

  let snapshot session =
    copy
      {
        session.state with
        entries = Array.of_list (List.rev session.reversed_entries);
      }

  let completed state = Array.map copy_entry state.entries

  let create ?resume ~specification_id ~configuration_id () =
    if String.trim specification_id = "" then
      incompatible "specification ID must not be blank"
    else
      match resume with
      | Some state when state.specification_id <> specification_id ->
          incompatible "specification ID differs"
      | None | Some _ ->
          let state =
            Option.fold
              ~none:
                {
                  specification_id;
                  identity = None;
                  plans = [||];
                  entries = [||];
                }
              ~some:copy resume
          in
          let cached = Hashtbl.create (Array.length state.entries) in
          Array.iter
            (fun entry ->
              Hashtbl.add cached (entry.stage, entry.candidate_index) entry)
            state.entries;
          Ok
            {
              identify = configuration_id;
              busy = Atomic.make false;
              cached;
              reversed_entries = List.rev (Array.to_list state.entries);
              state = { state with entries = [||] };
            }

  let report_without_models report =
    let report_folds =
      Cross_validation.folds report
      |> Array.map (fun fold ->
          {
            Cross_validation.fold_index = fold.Cross_validation.fold_index;
            fit_time = fold.Cross_validation.fit_time;
            score_time = fold.Cross_validation.score_time;
            scores = fold.Cross_validation.scores;
            model = None;
            train_indices = fold.Cross_validation.train_indices;
            test_indices = fold.Cross_validation.test_indices;
            failures = fold.Cross_validation.failures;
          })
    in
    { Cross_validation.report_folds }

  let find session stage index =
    Option.bind session (fun session ->
        Hashtbl.find_opt session.cached (stage, index))

  let record session entry =
    match session with
    | None -> ()
    | Some session ->
        if
          Option.is_none (find (Some session) entry.stage entry.candidate_index)
        then (
          let entry = copy_entry entry in
          Hashtbl.add session.cached (entry.stage, entry.candidate_index) entry;
          session.reversed_entries <- entry :: session.reversed_entries)

  let plan session stage identity =
    match session with
    | None -> Ok ()
    | Some session -> (
        match
          Array.find_opt (fun (key, _) -> key = stage) session.state.plans
        with
        | Some (_, previous) when previous <> identity ->
            incompatible
              "candidate configurations or parameter encodings differ"
        | Some _ -> Ok ()
        | None ->
            session.state <-
              {
                session.state with
                plans = Array.append session.state.plans [| (stage, identity) |];
              };
            Ok ())

  let partition emit = function
    | Cross_validation.Train -> Wire.int emit 0
    | Cross_validation.Test -> Wire.int emit 1

  let read_partition reader =
    match Wire.read_int reader with
    | 0 -> Cross_validation.Train
    | 1 -> Cross_validation.Test
    | _ -> Wire.invalid ()

  let failure emit failure =
    (match failure.Cross_validation.phase with
    | Cross_validation.Materialization -> Wire.int emit 0
    | Cross_validation.Fitting -> Wire.int emit 1
    | Cross_validation.Prediction part ->
        Wire.int emit 2;
        partition emit part
    | Cross_validation.Scoring { partition = part; scorer } ->
        Wire.int emit 3;
        partition emit part;
        Wire.token emit scorer);
    Wire.error emit failure.Cross_validation.error

  let read_failure reader =
    let phase =
      match Wire.read_int reader with
      | 0 -> Cross_validation.Materialization
      | 1 -> Cross_validation.Fitting
      | 2 -> Cross_validation.Prediction (read_partition reader)
      | 3 ->
          let partition = read_partition reader in
          let scorer = Wire.read_token reader in
          Cross_validation.Scoring { partition; scorer }
      | _ -> Wire.invalid ()
    in
    let error = Wire.read_error reader in
    { Cross_validation.phase; error }

  let score emit score =
    Wire.token emit score.Cross_validation.name;
    Wire.option (Wire.result Wire.float) emit score.Cross_validation.train_score;
    Wire.option (Wire.result Wire.float) emit score.Cross_validation.test_score

  let read_score reader =
    let name = Wire.read_token reader in
    let train_score =
      Wire.read_option (Wire.read_result Wire.read_float) reader
    in
    let test_score =
      Wire.read_option (Wire.read_result Wire.read_float) reader
    in
    { Cross_validation.name; train_score; test_score }

  let fold emit fold =
    Wire.int emit fold.Cross_validation.fold_index;
    Wire.float emit fold.Cross_validation.fit_time;
    Wire.float emit fold.Cross_validation.score_time;
    Wire.array score emit fold.Cross_validation.scores;
    Wire.option (Wire.array Wire.int) emit fold.Cross_validation.train_indices;
    Wire.option (Wire.array Wire.int) emit fold.Cross_validation.test_indices;
    Wire.array failure emit fold.Cross_validation.failures

  let read_fold reader =
    let fold_index = Wire.read_int reader in
    let fit_time = Wire.read_float reader in
    let score_time = Wire.read_float reader in
    if
      fold_index < 0
      || (not (Float.is_finite fit_time && Float.is_finite score_time))
      || fit_time < 0. || score_time < 0.
    then Wire.invalid ();
    let scores = Wire.read_array read_score reader in
    let train_indices =
      Wire.read_option (Wire.read_array Wire.read_int) reader
    in
    let test_indices =
      Wire.read_option (Wire.read_array Wire.read_int) reader
    in
    let failures = Wire.read_array read_failure reader in
    {
      Cross_validation.fold_index;
      fit_time;
      score_time;
      scores;
      train_indices;
      test_indices;
      failures;
      model = None;
    }

  let evaluation emit report =
    Wire.array fold emit report.Cross_validation.report_folds

  let read_evaluation reader =
    let report_folds = Wire.read_array read_fold reader in
    Array.iteri
      (fun i fold ->
        if fold.Cross_validation.fold_index <> i then Wire.invalid ())
      report_folds;
    { Cross_validation.report_folds }

  let entry emit entry =
    Wire.token emit entry.stage;
    Wire.int emit entry.candidate_index;
    Wire.token emit entry.configuration_id;
    Wire.result evaluation emit entry.evaluation

  let read_entry reader =
    let stage = Wire.read_token reader in
    let candidate_index = Wire.read_int reader in
    if candidate_index < 0 then Wire.invalid ();
    let configuration_id = Wire.read_token reader in
    let evaluation = Wire.read_result read_evaluation reader in
    { stage; candidate_index; configuration_id; evaluation }

  let pair emit (left, right) =
    Wire.token emit left;
    Wire.token emit right

  let read_pair reader =
    let left = Wire.read_token reader in
    let right = Wire.read_token reader in
    (left, right)

  let payload emit state =
    Wire.token emit "modelkit-search-checkpoint-v1";
    Wire.token emit state.specification_id;
    Wire.option Wire.token emit state.identity;
    Wire.array pair emit state.plans;
    Wire.array entry emit state.entries

  let encode state =
    Wire.protect (fun () ->
        let payload = Wire.encode payload state in
        Bytes.of_string
          (Wire.encode
             (fun emit () ->
               Wire.token emit payload;
               Wire.token emit (Digest.to_hex (Digest.string payload)))
             ()))

  let decode bytes =
    Wire.protect (fun () ->
        if Bytes.length bytes > Wire.limit then Wire.invalid ();
        let outer = Wire.reader (Bytes.to_string bytes) in
        let payload = Wire.read_token outer in
        let digest = Wire.read_token outer in
        Wire.finish outer;
        if digest <> Digest.to_hex (Digest.string payload) then Wire.invalid ();
        let reader = Wire.reader payload in
        if Wire.read_token reader <> "modelkit-search-checkpoint-v1" then
          Wire.invalid ();
        let specification_id = Wire.read_token reader in
        let identity = Wire.read_option Wire.read_token reader in
        let plans = Wire.read_array read_pair reader in
        let entries = Wire.read_array read_entry reader in
        Wire.finish reader;
        let keys = Hashtbl.create (Array.length plans) in
        Array.iter
          (fun (key, _) ->
            if Hashtbl.mem keys key then Wire.invalid ();
            Hashtbl.add keys key ())
          plans;
        let seen = Hashtbl.create (Array.length entries) in
        Array.iter
          (fun entry ->
            let key = (entry.stage, entry.candidate_index) in
            if (not (Hashtbl.mem keys entry.stage)) || Hashtbl.mem seen key then
              Wire.invalid ();
            Hashtbl.add seen key ())
          entries;
        if
          String.trim specification_id = ""
          || identity = None
             && (Array.length plans > 0 || Array.length entries > 0)
        then Wire.invalid ();
        { specification_id; identity; plans; entries })

  let vector emit values =
    Wire.int emit (Vector.length values);
    for i = 0 to Vector.length values - 1 do
      Wire.float emit (Vector.get values i)
    done

  let weights emit value = vector emit (Sample_weight.to_vector value)
  let groups emit value = Wire.array Wire.int emit (Groups.to_array value)

  let regression emit target =
    Wire.token emit "regression";
    vector emit (Target.regression_values target)

  let classification emit target =
    Wire.token emit "classification";
    Wire.array Wire.int emit (Target.classification_values target)

  let with_run session ~algorithm ~settings ~seed ~metadata ~target ~splitter
      dataset f =
    match session with
    | None -> f splitter
    | Some session ->
        if not (Atomic.compare_and_set session.busy false true) then
          incompatible "checkpoint is already in use"
        else
          Fun.protect
            ~finally:(fun () -> Atomic.set session.busy false)
            (fun () ->
              let ( let* ) = Result.bind in
              let rng =
                Seed.derive seed ~operation:"cross-validation-splitter" ~index:0
                |> Rng.create
              in
              let* pairs =
                splitter.Cross_validation.run_splitter ~rng
                  ~groups:(Dataset.groups dataset) ~x:(Dataset.features dataset)
                  ~y:(Dataset.target dataset)
              in
              let identity =
                Wire.digest
                  (fun emit () ->
                    Wire.token emit algorithm;
                    Wire.token emit settings;
                    Wire.token emit (Seed.to_string seed);
                    Wire.schema emit (Dataset.feature_schema dataset);
                    Wire.bool emit
                      (Dataset.finiteness dataset = Dataset.Require_finite);
                    let x = Dataset.features dataset in
                    Wire.int emit (Matrix.rows x);
                    Wire.int emit (Matrix.columns x);
                    for row = 0 to Matrix.rows x - 1 do
                      for col = 0 to Matrix.columns x - 1 do
                        Wire.float emit (Matrix.get x row col)
                      done
                    done;
                    target emit (Dataset.target dataset);
                    Wire.option weights emit (Dataset.sample_weight dataset);
                    Wire.option groups emit (Dataset.groups dataset);
                    Wire.option weights emit (Metadata.sample_weight metadata);
                    Wire.option groups emit (Metadata.groups metadata);
                    Wire.bool emit (Option.is_some (Metadata.callback metadata));
                    Wire.array
                      (fun emit (train, test) ->
                        Wire.int emit (Row_view.source_size train);
                        Wire.array Wire.int emit (Row_view.indices train);
                        Wire.int emit (Row_view.source_size test);
                        Wire.array Wire.int emit (Row_view.indices test))
                      emit pairs)
                  ()
              in
              let* () =
                match session.state.identity with
                | Some previous when previous <> identity ->
                    incompatible
                      "data, splits, seed, task, or search options differ"
                | Some _ -> Ok ()
                | None ->
                    session.state <-
                      { session.state with identity = Some identity };
                    Ok ()
              in
              f
                {
                  Cross_validation.run_splitter =
                    (fun ~rng:_ ~groups:_ ~x:_ ~y:_ -> Ok pairs);
                })
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
      candidates
      |> Array.mapi (fun position candidate ->
          match candidate.scores.(primary).test with
          | Ok summary ->
              Some
                ( position,
                  candidate.candidate_index,
                  summary.Score_aggregation.mean )
          | Error _ -> None)
      |> Array.to_list |> List.filter_map Fun.id |> Array.of_list
    in
    Array.sort
      (fun (_, left, left_score) (_, right, right_score) ->
        let order = Float.compare right_score left_score in
        if order <> 0 then order else Int.compare left right)
      eligible;
    let ranks = Array.make (Array.length candidates) None in
    let previous_score = ref None and previous_rank = ref 0 in
    Array.iteri
      (fun position (candidate, _, score) ->
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
        (fun position candidate -> { candidate with rank = ranks.(position) })
        candidates,
      if Array.length eligible = 0 then None
      else
        let position, _, _ = eligible.(0) in
        Some position )

  let settings ~return_train_score ~failure_policy ~scorer_names ~policy =
    let module Wire = Modelkit_checkpoint_codec in
    Wire.encode
      (fun emit () ->
        Wire.bool emit return_train_score;
        Wire.bool emit (failure_policy = Cross_validation.Abort);
        Wire.array Wire.token emit scorer_names;
        match policy with
        | No_refit -> Wire.token emit "none"
        | Best_score name ->
            Wire.token emit "best";
            Wire.token emit name
        | Custom _ -> Wire.token emit "custom")
      ()

  let checkpoint_plan checkpoint ~stage ~candidate_id partials =
    match checkpoint with
    | None -> Ok (Array.make (Array.length partials) "")
    | Some session ->
        let module Wire = Modelkit_checkpoint_codec in
        let identities =
          Array.map
            (fun partial ->
              match partial.configuration with
              | Ok configuration ->
                  session.Search_checkpoint.identify configuration
              | Error error -> Wire.digest Wire.error error)
            partials
        in
        if Array.exists (fun identity -> String.trim identity = "") identities
        then
          Search_checkpoint.incompatible "configuration IDs must not be blank"
        else
          let identity =
            Wire.digest
              (fun emit () ->
                Wire.int emit (Array.length partials);
                Array.iteri
                  (fun position partial ->
                    Wire.int emit (candidate_id position);
                    Wire.token emit identities.(position);
                    Wire.result
                      (fun emit _ -> Wire.token emit "configured")
                      emit partial.configuration;
                    Wire.array
                      (fun emit parameter ->
                        Wire.token emit parameter.parameter_name;
                        match parameter.parameter_value with
                        | Bool value ->
                            Wire.int emit 0;
                            Wire.bool emit value
                        | Int value ->
                            Wire.int emit 1;
                            Wire.int emit value
                        | Float value ->
                            Wire.int emit 2;
                            Wire.float emit value
                        | String value ->
                            Wire.int emit 3;
                            Wire.token emit value)
                      emit
                      (Array.of_list (List.rev partial.reversed_parameters)))
                  partials)
              ()
          in
          Result.map
            (fun () -> identities)
            (Search_checkpoint.plan checkpoint stage identity)

  let search_candidates ?checkpoint ?(checkpoint_stage = "candidates")
      ?(candidate_id = Fun.id) ?(search_callback = true) ~return_train_score
      ~failure_policy ~cross_validate ~scorer_names ~metadata ~policy ~seed
      ~operation ~prepare ~build dataset =
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
    let with_search f =
      if not search_callback then f ()
      else
        Callback.run
          ~outcome:(fun report ->
            match report.report_selection with
            | Ok _ -> Callback.Succeeded
            | Error error -> Callback.Failed error)
          (Metadata.callback metadata)
          ~operation:Callback.Search f
    in
    with_search (fun () ->
        let count, original_at = prepare () in
        let* partials =
          match checkpoint with
          | None -> Ok None
          | Some _ ->
              let rec collect position reversed =
                if position = count then
                  Ok (Some (Array.of_list (List.rev reversed)))
                else
                  let partial = original_at position in
                  match partial.configuration with
                  | Error error when Callback.is_control_error error ->
                      Error (with_candidate (candidate_id position) error)
                  | Ok _ | Error _ ->
                      collect (position + 1) (partial :: reversed)
              in
              collect 0 []
        in
        let partial_at =
          match partials with
          | None -> original_at
          | Some values -> Array.get values
        in
        let* identities =
          match partials with
          | None -> Ok [||]
          | Some values ->
              checkpoint_plan checkpoint ~stage:checkpoint_stage ~candidate_id
                values
        in
        let pipelines = Array.make count None in
        let evaluate_candidate position =
          let candidate_index = candidate_id position in
          let cached =
            Search_checkpoint.find checkpoint checkpoint_stage candidate_index
          in
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
              let partial = partial_at position in
              let parameters =
                partial.reversed_parameters |> List.rev |> Array.of_list
              in
              let built =
                match cached with
                | Some { Search_checkpoint.evaluation = Error error; _ } ->
                    Error error
                | None | Some { Search_checkpoint.evaluation = Ok _; _ } -> (
                    match partial.configuration with
                    | Error error -> Error error
                    | Ok configuration -> build configuration)
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
                  pipelines.(position) <- Some pipeline;
                  let fit_seed =
                    Seed.derive seed ~operation:(operation ^ "-candidate")
                      ~index:candidate_index
                  in
                  let* evaluation =
                    match cached with
                    | Some { Search_checkpoint.evaluation = Ok report; _ } ->
                        Ok (Search_checkpoint.report_without_models report)
                    | Some { Search_checkpoint.evaluation = Error error; _ } ->
                        Error error
                    | None ->
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
            (match checkpoint with
            | None -> ()
            | Some _ ->
                let evaluation =
                  match (candidate.evaluation, candidate.build_error) with
                  | Some report, _ ->
                      Ok (Search_checkpoint.report_without_models report)
                  | None, Some error -> Error error
                  | None, None -> assert false
                in
                Search_checkpoint.record checkpoint
                  {
                    Search_checkpoint.stage = checkpoint_stage;
                    candidate_index = candidate.candidate_index;
                    configuration_id = identities.(candidate_index);
                    evaluation;
                  });
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
            else
              let position = candidate_index in
              let candidate_index = candidate_id position in
              if
                not
                  (Array.exists
                     (fun score -> Result.is_ok score.test)
                     ranked.(position).scores)
              then
                failure
                  (invalid
                     "selected candidate has no successful test-score aggregate"
                  |> with_candidate candidate_index)
              else
                match pipelines.(position) with
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

  let search ?checkpoint ~return_train_score ~failure_policy ~cross_validate
      ~scorer_names ~metadata ~policy ~seed grid dataset =
    let prepare () =
      let partials = expand grid in
      (Array.length partials, Array.get partials)
    in
    search_candidates ?checkpoint ~return_train_score ~failure_policy
      ~cross_validate ~scorer_names ~metadata ~policy ~seed
      ~operation:"grid-search" ~prepare ~build:grid.build dataset

  module Regression = struct
    type model = Cross_validation.Regression.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~grid
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names = Array.map Regression_scorer.name scorers in
      let settings =
        settings ~return_train_score ~failure_policy ~scorer_names ~policy
      in
      Search_checkpoint.with_run checkpoint ~algorithm:"grid-search:Regression"
        ~settings ~seed ~metadata ~target:Search_checkpoint.regression ~splitter
        dataset (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Regression.cross_validate ~return_train_score
              ~failure_policy ~fit_seed ~execution ~metadata ~splitter ~scorers
              ~seed pipeline dataset
          in
          search ?checkpoint ~return_train_score ~failure_policy ~cross_validate
            ~scorer_names ~metadata ~policy ~seed grid dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~grid ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~grid ~splitter ~scorers
        ~policy:(Best_score refit) ~seed dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~grid
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let settings =
        settings ~return_train_score ~failure_policy ~scorer_names ~policy
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"grid-search:Binary_classification" ~settings ~seed ~metadata
        ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Binary_classification.cross_validate
              ~return_train_score ~failure_policy ~fit_seed ~execution ~metadata
              ~splitter ~scorers ~seed pipeline dataset
          in
          search ?checkpoint ~return_train_score ~failure_policy ~cross_validate
            ~scorer_names ~metadata ~policy ~seed grid dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~grid ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~grid ~splitter ~scorers
        ~policy:(Best_score refit) ~seed dataset
  end

  module Multiclass_classification = struct
    type model = Cross_validation.Multiclass_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~grid
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        match metadata with
        | Some metadata -> metadata
        | None -> Metadata.of_dataset dataset
      in
      let scorer_names =
        Array.map Multiclass_classification_scorer.name scorers
      in
      let settings =
        settings ~return_train_score ~failure_policy ~scorer_names ~policy
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"grid-search:Multiclass_classification" ~settings ~seed
        ~metadata ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Multiclass_classification.cross_validate
              ~return_train_score ~failure_policy ~fit_seed ~execution ~metadata
              ~splitter ~scorers ~seed pipeline dataset
          in
          search ?checkpoint ~return_train_score ~failure_policy ~cross_validate
            ~scorer_names ~metadata ~policy ~seed grid dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~grid ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~grid ~splitter ~scorers
        ~policy:(Best_score refit) ~seed dataset
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
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~space
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Regression_scorer.name scorers in
      let settings =
        Grid_search.settings ~return_train_score ~failure_policy ~scorer_names
          ~policy
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"randomized-search:Regression" ~settings ~seed ~metadata
        ~target:Search_checkpoint.regression ~splitter dataset (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Regression.cross_validate ~metadata
              ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
              ~scorers ~seed pipeline dataset
          in
          Grid_search.search_candidates ?checkpoint ~return_train_score
            ~failure_policy ~cross_validate ~scorer_names ~metadata ~policy
            ~seed ~operation:"randomized-search"
            ~prepare:(fun () -> prepare ~seed space)
            ~build:space.sample_build dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~space ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~space
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let settings =
        Grid_search.settings ~return_train_score ~failure_policy ~scorer_names
          ~policy
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"randomized-search:Binary_classification" ~settings ~seed
        ~metadata ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Binary_classification.cross_validate ~metadata
              ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
              ~scorers ~seed pipeline dataset
          in
          Grid_search.search_candidates ?checkpoint ~return_train_score
            ~failure_policy ~cross_validate ~scorer_names ~metadata ~policy
            ~seed ~operation:"randomized-search"
            ~prepare:(fun () -> prepare ~seed space)
            ~build:space.sample_build dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~space ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end

  module Multiclass_classification = struct
    type model = Cross_validation.Multiclass_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~space
        ~splitter ~scorers ~policy ~seed dataset =
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names =
        Array.map Multiclass_classification_scorer.name scorers
      in
      let settings =
        Grid_search.settings ~return_train_score ~failure_policy ~scorer_names
          ~policy
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"randomized-search:Multiclass_classification" ~settings ~seed
        ~metadata ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate ~metadata ~return_train_score ~failure_policy
              ~fit_seed pipeline dataset =
            Cross_validation.Multiclass_classification.cross_validate ~metadata
              ~return_train_score ~failure_policy ~fit_seed ~execution ~splitter
              ~scorers ~seed pipeline dataset
          in
          Grid_search.search_candidates ?checkpoint ~return_train_score
            ~failure_policy ~cross_validate ~scorer_names ~metadata ~policy
            ~seed ~operation:"randomized-search"
            ~prepare:(fun () -> prepare ~seed space)
            ~build:space.sample_build dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~space ~splitter ~scorers ~refit ~seed dataset =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~space ~splitter ~scorers
        ~policy:(Grid_search.Best_score refit) ~seed dataset
  end
end

module Successive_halving = struct
  type budget = { schedule : int array; factor : int; max_fits : int option }

  type ('configuration, 'target, 'prediction) candidates = {
    count : int;
    prepare : Seed.t -> int -> 'configuration Grid_search.partial;
    build :
      'configuration -> (('target, 'prediction) Pipeline.t, Error.t) result;
  }

  type 'model round = {
    round_index : int;
    training_samples : int;
    candidates : 'model Grid_search.candidate array;
    promoted_candidate_indices : int array;
  }

  type 'model report = {
    completed_rounds : 'model round array;
    final_report : 'model Grid_search.report;
  }

  let invalid reason =
    Error
      (Grid_search.validation ~name:"successive halving" ~reason
         ~remediation:
           "supply feasible training-row budgets, folds, and candidate \
            specifications")

  let budget ?max_fits ~min_samples ~max_samples ~factor () =
    if min_samples < 1 || max_samples < min_samples || factor < 2 then
      invalid "require 1 <= min_samples <= max_samples and factor >= 2"
    else if Option.fold ~none:false ~some:(fun value -> value < 1) max_fits then
      invalid "max_fits must be positive"
    else
      let rec schedule value reversed =
        if value = max_samples then Array.of_list (List.rev (value :: reversed))
        else
          schedule
            (if value > max_samples / factor then max_samples
             else value * factor)
            (value :: reversed)
      in
      Ok { schedule = schedule min_samples []; factor; max_fits }

  let resources budget = Array.copy budget.schedule

  let of_grid grid =
    {
      count = Grid_search.candidate_count grid;
      prepare =
        (fun _ ->
          let partials = Grid_search.expand grid in
          Array.get partials);
      build = grid.Grid_search.build;
    }

  let of_randomized space =
    {
      count = Randomized_search.candidate_count space;
      prepare = (fun seed -> snd (Randomized_search.prepare ~seed space));
      build = space.Randomized_search.sample_build;
    }

  let rounds report =
    Array.map
      (fun round ->
        {
          round with
          candidates = Array.map Grid_search.copy_candidate round.candidates;
          promoted_candidate_indices =
            Array.copy round.promoted_candidate_indices;
        })
      report.completed_rounds

  let selection report = Grid_search.selection report.final_report
  let refit_result report = Grid_search.refit_result report.final_report
  let survivors count factor = 1 + ((count - 1) / factor)

  let fit_bound budget count folds policy =
    let ( let* ) = Result.bind in
    let rec sum index count total =
      if index = Array.length budget.schedule then Ok total
      else if count > (max_int - total) / folds then
        invalid "planned fit count exceeds the integer limit"
      else
        sum (index + 1) (survivors count budget.factor) (total + (count * folds))
    in
    let* total =
      sum 0 count
        (match policy with
        | Grid_search.No_refit -> 0
        | Grid_search.Best_score _ | Grid_search.Custom _ -> 1)
    in
    match budget.max_fits with
    | Some maximum when total > maximum ->
        invalid "planned fit count exceeds max_fits"
    | None | Some _ -> Ok ()

  let data result =
    Result.map_error
      (fun error ->
        Error.of_data_error
          ~remediation:"supply aligned rows with positive total sample weight"
          error)
      result

  let validate_view dataset metadata view =
    let ( let* ) = Result.bind in
    let* _ = Metadata.select metadata view in
    match Dataset.sample_weight dataset with
    | None -> Ok ()
    | Some weight ->
        Result.map (fun _ -> ()) (data (Sample_weight.select weight view))

  let partitions ~budget ~splitter ~labels ~seed ~metadata dataset =
    let ( let* ) = Result.bind in
    let rows = Dataset.sample_count dataset in
    let rng =
      Seed.derive seed ~operation:"cross-validation-splitter" ~index:0
      |> Rng.create
    in
    let* pairs =
      splitter.Cross_validation.run_splitter ~rng
        ~groups:(Dataset.groups dataset) ~x:(Dataset.features dataset)
        ~y:(Dataset.target dataset)
    in
    let* () =
      if Array.length pairs = 0 then
        invalid "at least one validation fold is required"
      else Ok ()
    in
    let classes =
      match labels with
      | None -> [||]
      | Some values ->
          Array.to_list values |> List.sort_uniq Int.compare |> Array.of_list
    in
    let* () =
      if Array.length classes > budget.schedule.(0) then
        invalid "minimum training budget cannot cover every class"
      else Ok ()
    in
    let validate_classes indices =
      match labels with
      | None -> Ok ()
      | Some values ->
          let seen = Hashtbl.create (Array.length classes) in
          Array.iter (fun row -> Hashtbl.replace seen values.(row) ()) indices;
          if Hashtbl.length seen = Array.length classes then Ok ()
          else
            invalid
              "every base training and validation fold must contain every class"
    in
    let rec prepare index reversed =
      if index = Array.length pairs then Ok (Array.of_list (List.rev reversed))
      else
        let train, test = pairs.(index) in
        let checked =
          let* _ = Split.of_views ~train ~test in
          let* () =
            if
              Row_view.source_size train <> rows
              || Row_view.source_size test <> rows
            then invalid "fold source size differs from the dataset"
            else Ok ()
          in
          let order = Row_view.indices train in
          let* () =
            if
              Array.length order
              < budget.schedule.(Array.length budget.schedule - 1)
            then invalid "maximum training budget exceeds a base training fold"
            else Ok ()
          in
          let* () = validate_classes order in
          let* () = validate_classes (Row_view.indices test) in
          let* () = validate_view dataset metadata test in
          let rng =
            Seed.derive seed ~operation:"halving-training-rows" ~index
            |> Rng.create
          in
          Splitter_internal.shuffle rng order;
          let order =
            match labels with
            | None -> order
            | Some values ->
                let seen = Hashtbl.create (Array.length classes) in
                let first, rest =
                  Array.to_list order
                  |> List.partition (fun row ->
                      if Hashtbl.mem seen values.(row) then false
                      else (
                        Hashtbl.add seen values.(row) ();
                        true))
                in
                Array.of_list (first @ rest)
          in
          let* views =
            Array.fold_left
              (fun result resource ->
                let* reversed = result in
                let* train =
                  data
                    (Row_view.create ~source_size:rows
                       (Array.sub order 0 resource))
                in
                let* () = validate_view dataset metadata train in
                Ok ((train, test) :: reversed))
              (Ok []) budget.schedule
          in
          Ok (Array.of_list (List.rev views))
        in
        let* views =
          Result.map_error (Error.with_context (Error.Fold index)) checked
        in
        prepare (index + 1) (views :: reversed)
    in
    prepare 0 []

  let run ?checkpoint ~return_train_score ~failure_policy ~metadata ~budget
      ~candidates:source ~splitter ~scorer_names ~cross_validate ~labels
      ~promotion_score ~policy ~seed dataset =
    let ( let* ) = Result.bind in
    let* primary = Grid_search.validate_refit scorer_names promotion_score in
    let* () =
      match policy with
      | Grid_search.Best_score name ->
          Result.map
            (fun _ -> ())
            (Grid_search.validate_refit scorer_names name)
      | Grid_search.No_refit | Grid_search.Custom _ -> Ok ()
    in
    let* () = Metadata.validate ~rows:(Dataset.sample_count dataset) metadata in
    Callback.run
      (Metadata.callback metadata)
      ~operation:Callback.Search
      ~outcome:(fun report ->
        match refit_result report with
        | Ok _ -> Callback.Succeeded
        | Error error -> Callback.Failed error)
      (fun () ->
        let* partitions =
          partitions ~budget ~splitter ~labels ~seed ~metadata dataset
        in
        let* () =
          fit_bound budget source.count (Array.length partitions) policy
        in
        let at = source.prepare seed in
        let cache = Array.make source.count None in
        let partial index =
          match cache.(index) with
          | Some value -> value
          | None ->
              let value = at index in
              cache.(index) <- Some value;
              value
        in
        let rec loop index active reversed =
          let final = index + 1 = Array.length budget.schedule in
          let pairs = Array.map (fun views -> views.(index)) partitions in
          let splitter =
            {
              Cross_validation.run_splitter =
                (fun ~rng:_ ~groups:_ ~x:_ ~y:_ -> Ok pairs);
            }
          in
          let metadata =
            Metadata.scope
              (Error.Stage ("halving round " ^ string_of_int index))
              metadata
          in
          let* report =
            Grid_search.search_candidates ?checkpoint
              ~checkpoint_stage:(string_of_int index)
              ~candidate_id:(Array.get active) ~search_callback:false
              ~return_train_score ~failure_policy
              ~cross_validate:(cross_validate splitter) ~scorer_names ~metadata
              ~policy:(if final then policy else Grid_search.No_refit)
              ~seed ~operation:"halving-search"
              ~prepare:(fun () ->
                (Array.length active, fun position -> partial active.(position)))
              ~build:source.build dataset
          in
          let ranked, _ =
            Grid_search.rank_candidates primary
              report.Grid_search.report_candidates
          in
          let eligible =
            Array.to_list ranked
            |> List.filter (fun candidate ->
                Option.is_some candidate.Grid_search.rank)
            |> List.sort (fun left right ->
                let order =
                  compare left.Grid_search.rank right.Grid_search.rank
                in
                if order <> 0 then order
                else
                  Int.compare left.Grid_search.candidate_index
                    right.Grid_search.candidate_index)
          in
          let rec take count = function
            | [] -> []
            | _ when count = 0 -> []
            | head :: tail ->
                head.Grid_search.candidate_index :: take (count - 1) tail
          in
          let promoted =
            if final then [||]
            else
              Array.of_list
                (take (survivors (Array.length active) budget.factor) eligible)
          in
          let round =
            {
              round_index = index;
              training_samples = budget.schedule.(index);
              candidates = ranked;
              promoted_candidate_indices = promoted;
            }
          in
          let reversed = round :: reversed in
          let finish report =
            Ok
              {
                completed_rounds = Array.of_list (List.rev reversed);
                final_report = report;
              }
          in
          let* () =
            match Metadata.callback metadata with
            | None -> Ok ()
            | Some callback ->
                Callback.progress
                  (Callback.for_operation Callback.Search callback)
                  ~completed:(index + 1)
                  ~total:(Array.length budget.schedule)
                  ()
          in
          if final then finish report
          else if Array.length promoted = 0 then
            let error =
              match
                invalid "no candidate has an aggregatable promotion score"
              with
              | Error error -> error
              | Ok _ -> assert false
            in
            if failure_policy = Cross_validation.Abort then Error error
            else
              finish { report with Grid_search.report_selection = Error error }
          else
            let active = Array.copy promoted in
            Array.sort Int.compare active;
            loop (index + 1) active reversed
        in
        loop 0 (Array.init source.count Fun.id) [])

  module Regression = struct
    type model = Cross_validation.Regression.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~budget
        ~candidates ~splitter ~scorers ~promotion_score ~policy ~seed dataset =
      let labels = None in
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Regression_scorer.name scorers in
      let settings =
        Modelkit_checkpoint_codec.encode
          (fun emit () ->
            Modelkit_checkpoint_codec.token emit
              (Grid_search.settings ~return_train_score ~failure_policy
                 ~scorer_names ~policy);
            Modelkit_checkpoint_codec.token emit promotion_score;
            Modelkit_checkpoint_codec.array Modelkit_checkpoint_codec.int emit
              budget.schedule;
            Modelkit_checkpoint_codec.int emit budget.factor;
            Modelkit_checkpoint_codec.option Modelkit_checkpoint_codec.int emit
              budget.max_fits)
          ()
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"halving-search:Regression" ~settings ~seed ~metadata
        ~target:Search_checkpoint.regression ~splitter dataset (fun splitter ->
          let cross_validate splitter ~metadata ~return_train_score
              ~failure_policy ~fit_seed pipeline dataset =
            Cross_validation.Regression.cross_validate ~return_indices:true
              ~metadata ~return_train_score ~failure_policy ~fit_seed ~execution
              ~splitter ~scorers ~seed pipeline dataset
          in
          run ?checkpoint ~return_train_score ~failure_policy ~metadata ~budget
            ~candidates ~splitter ~scorer_names ~cross_validate ~labels
            ~promotion_score ~policy ~seed dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~budget ~candidates ~splitter ~scorers ~refit ~seed dataset
        =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~budget ~candidates ~splitter ~scorers
        ~promotion_score:refit ~policy:(Grid_search.Best_score refit) ~seed
        dataset
  end

  module Binary_classification = struct
    type model = Cross_validation.Binary_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~budget
        ~candidates ~splitter ~scorers ~promotion_score ~policy ~seed dataset =
      let ( let* ) = Result.bind in
      let values = Target.classification_values (Dataset.target dataset) in
      let classes =
        List.sort_uniq Int.compare (Array.to_list values) |> List.length
      in
      let* () =
        if classes <> 2 then
          invalid "Binary_classification requires exactly two classes"
        else Ok ()
      in
      let labels = Some values in
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names = Array.map Binary_classification_scorer.name scorers in
      let settings =
        Modelkit_checkpoint_codec.encode
          (fun emit () ->
            Modelkit_checkpoint_codec.token emit
              (Grid_search.settings ~return_train_score ~failure_policy
                 ~scorer_names ~policy);
            Modelkit_checkpoint_codec.token emit promotion_score;
            Modelkit_checkpoint_codec.array Modelkit_checkpoint_codec.int emit
              budget.schedule;
            Modelkit_checkpoint_codec.int emit budget.factor;
            Modelkit_checkpoint_codec.option Modelkit_checkpoint_codec.int emit
              budget.max_fits)
          ()
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"halving-search:Binary_classification" ~settings ~seed
        ~metadata ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate splitter ~metadata ~return_train_score
              ~failure_policy ~fit_seed pipeline dataset =
            Cross_validation.Binary_classification.cross_validate
              ~return_indices:true ~metadata ~return_train_score ~failure_policy
              ~fit_seed ~execution ~splitter ~scorers ~seed pipeline dataset
          in
          run ?checkpoint ~return_train_score ~failure_policy ~metadata ~budget
            ~candidates ~splitter ~scorer_names ~cross_validate ~labels
            ~promotion_score ~policy ~seed dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~budget ~candidates ~splitter ~scorers ~refit ~seed dataset
        =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~budget ~candidates ~splitter ~scorers
        ~promotion_score:refit ~policy:(Grid_search.Best_score refit) ~seed
        dataset
  end

  module Multiclass_classification = struct
    type model = Cross_validation.Multiclass_classification.model

    let search_with_policy ?(return_train_score = false)
        ?(failure_policy = Cross_validation.Record)
        ?(execution = Execution.sequential) ?metadata ?checkpoint ~budget
        ~candidates ~splitter ~scorers ~promotion_score ~policy ~seed dataset =
      let ( let* ) = Result.bind in
      let values = Target.classification_values (Dataset.target dataset) in
      let classes =
        List.sort_uniq Int.compare (Array.to_list values) |> List.length
      in
      let* () =
        if classes < 3 then
          invalid "Multiclass_classification requires at least three classes"
        else Ok ()
      in
      let labels = Some values in
      let metadata =
        Option.value metadata ~default:(Metadata.of_dataset dataset)
      in
      let scorer_names =
        Array.map Multiclass_classification_scorer.name scorers
      in
      let settings =
        Modelkit_checkpoint_codec.encode
          (fun emit () ->
            Modelkit_checkpoint_codec.token emit
              (Grid_search.settings ~return_train_score ~failure_policy
                 ~scorer_names ~policy);
            Modelkit_checkpoint_codec.token emit promotion_score;
            Modelkit_checkpoint_codec.array Modelkit_checkpoint_codec.int emit
              budget.schedule;
            Modelkit_checkpoint_codec.int emit budget.factor;
            Modelkit_checkpoint_codec.option Modelkit_checkpoint_codec.int emit
              budget.max_fits)
          ()
      in
      Search_checkpoint.with_run checkpoint
        ~algorithm:"halving-search:Multiclass_classification" ~settings ~seed
        ~metadata ~target:Search_checkpoint.classification ~splitter dataset
        (fun splitter ->
          let cross_validate splitter ~metadata ~return_train_score
              ~failure_policy ~fit_seed pipeline dataset =
            Cross_validation.Multiclass_classification.cross_validate
              ~return_indices:true ~metadata ~return_train_score ~failure_policy
              ~fit_seed ~execution ~splitter ~scorers ~seed pipeline dataset
          in
          run ?checkpoint ~return_train_score ~failure_policy ~metadata ~budget
            ~candidates ~splitter ~scorer_names ~cross_validate ~labels
            ~promotion_score ~policy ~seed dataset)

    let search ?return_train_score ?failure_policy ?execution ?metadata
        ?checkpoint ~budget ~candidates ~splitter ~scorers ~refit ~seed dataset
        =
      search_with_policy ?return_train_score ?failure_policy ?execution
        ?metadata ?checkpoint ~budget ~candidates ~splitter ~scorers
        ~promotion_score:refit ~policy:(Grid_search.Best_score refit) ~seed
        dataset
  end
end
