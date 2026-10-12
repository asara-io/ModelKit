open Modelkit
module Xgb = Modelkit_xgboost

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let ( let* ) = Result.bind
let seed = Seed.of_int 43

external openmp_limits : unit -> int * int
  = "modelkit_xgboost_test_openmp_limits"

let check_thread_contract () =
  ignore (Xgb.Native.version () |> get);
  let maximum, limit = openmp_limits () in
  if maximum <> 1 || limit <> 1 then
    failwith
      (Format.sprintf
         "OpenMP startup contract requires max=1 and limit=1; observed %d/%d"
         maximum limit)

let bounded call =
  check_thread_contract ();
  let result = call () in
  check_thread_contract ();
  result

type 'target event = {
  event_specification : Xgb.Specification.t;
  event_rng : Rng.t;
  event_schema : Feature_schema.t;
  event_features : Matrix.t;
  event_target : 'target;
  event_weights : Sample_weight.t option;
}

module Audit
    (Estimator :
      ESTIMATOR
        with type t = Xgb.Specification.t
         and type params = Xgb.Specification.params
         and type rng = Rng.t) =
struct
  include Estimator

  let events = Atomic.make []
  let models = Atomic.make []

  let rec record destination event =
    let previous = Atomic.get destination in
    if not (Atomic.compare_and_set destination previous (event :: previous))
    then record destination event

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y () =
    let event =
      {
        event_specification = specification;
        event_rng = rng;
        event_schema = feature_schema;
        event_features = x;
        event_target = y;
        event_weights = sample_weight;
      }
    in
    record events event;
    let result =
      bounded (fun () ->
          Estimator.fit specification ?sample_weight ~rng ~feature_schema ~x ~y
            ())
    in
    (match result with
    | Ok model -> record models (event, model)
    | Error _ -> ());
    result

  let predict fitted ~feature_schema ~x =
    bounded (fun () -> Estimator.predict fitted ~feature_schema ~x)
end

type 'kind model = ('kind Target.t, 'kind Target.t) Pipeline.fitted
type 'kind pipeline = ('kind Target.t, 'kind Target.t) Pipeline.t
type 'kind grid = (int, 'kind Target.t, 'kind Target.t) Grid_search.grid
type 'kind space = (int, 'kind Target.t, 'kind Target.t) Randomized_search.space

type 'kind candidates =
  (int, 'kind Target.t, 'kind Target.t) Successive_halving.candidates

let split () =
  Group_k_fold.create ~folds:3 ()
  |> get
  |> Cross_validation.target_independent_splitter (module Group_k_fold)

let specification depth =
  Xgb.Specification.create ~max_depth:depth ~rounds:5 ~subsample:0.75
    ~colsample_bytree:0.75 ~threads:1 ()
  |> get

let subset source indices =
  Row_view.create ~source_size:(Dataset.sample_count source) indices
  |> data |> Dataset.view source |> data |> Dataset.materialize |> data

let source_features () =
  Matrix.init ~rows:54 ~columns:3 (fun row column ->
      if column = 0 then float_of_int row
      else if column = 1 then float_of_int ((row * 7 mod 19) - 9)
      else if row mod 4 = 0 then Float.nan
      else float_of_int ((row / 6 * 100) + (row mod 5)))
  |> data

let dataset target =
  let x = source_features () in
  let names =
    Feature_names.create ~expected_count:3 [| "id"; "signal"; "missing" |]
    |> data
  in
  let sample_weight =
    Sample_weight.of_array ~expected_length:54
      (Array.init 54 (fun row ->
           if row = 0 then 0. else 1. +. (0.5 *. float_of_int (row mod 4))))
    |> data
  in
  let groups =
    Groups.create ~expected_length:54 (Array.init 54 (fun row -> row / 6))
    |> data
  in
  Dataset.create ~finiteness:Dataset.Allow_nan ~feature_names:names
    ~sample_weight ~groups ~x ~y:target ()
  |> data

let expected_imputation source indices x =
  let raw = Dataset.features source in
  Matrix.init ~rows:(Matrix.rows x) ~columns:3 (fun row column ->
      let value = Matrix.get x row column in
      if not (Float.is_nan value) then value
      else
        let sum, count =
          Array.fold_left
            (fun (sum, count) source_row ->
              let value = Matrix.get raw source_row column in
              if Float.is_nan value then (sum, count)
              else (sum +. value, count + 1))
            (0., 0) indices
        in
        sum /. float_of_int count)
  |> data

let row_ids x =
  Array.init (Matrix.rows x) (fun row -> int_of_float (Matrix.get x row 0))

let fold_snapshot report =
  Cross_validation.folds report
  |> Array.map (fun fold ->
      ( fold.Cross_validation.fold_index,
        fold.Cross_validation.train_indices,
        fold.Cross_validation.test_indices,
        Array.map
          (fun score ->
            ( score.Cross_validation.name,
              Option.map get score.Cross_validation.train_score,
              Option.map get score.Cross_validation.test_score ))
          fold.Cross_validation.scores,
        Array.length fold.Cross_validation.failures ))

let candidate_snapshot candidates =
  Array.map
    (fun candidate ->
      ( candidate.Grid_search.candidate_index,
        candidate.Grid_search.parameters,
        candidate.Grid_search.rank,
        Array.map
          (fun score ->
            ( score.Grid_search.scorer_name,
              Option.map
                (fun aggregate -> (get aggregate).Score_aggregation.mean)
                score.Grid_search.train,
              (get score.Grid_search.test).Score_aggregation.mean ))
          candidate.Grid_search.scores,
        Option.map fold_snapshot candidate.Grid_search.evaluation ))
    candidates

let combined_score candidate =
  let scores = candidate.Grid_search.scores in
  (get scores.(0).Grid_search.test).Score_aggregation.mean
  +. (0.01 *. (get scores.(1).Grid_search.test).Score_aggregation.mean)

let choose candidates =
  let best = ref 0 in
  for index = 1 to Array.length candidates - 1 do
    if combined_score candidates.(index) > combined_score candidates.(!best)
    then best := index
  done;
  Ok !best

module type TASK = sig
  type kind

  module Estimator :
    ESTIMATOR
      with type t = Xgb.Specification.t
       and type params = Xgb.Specification.params
       and type rng = Rng.t
       and type target = kind Target.t
       and type prediction = kind Target.t

  val name : string
  val source : unit -> kind Dataset.t
  val equal_target : kind Target.t -> kind Target.t -> bool

  val package :
    (module ESTIMATOR
       with type t = Xgb.Specification.t
        and type rng = Rng.t
        and type target = kind Target.t
        and type prediction = kind Target.t
        and type fitted = Estimator.fitted) ->
    Xgb.Specification.t ->
    ((kind Target.t, kind Target.t) Pipeline.estimator, Error.t) result

  val fingerprint : kind model -> Feature_schema.t -> Matrix.t -> float array

  val native_fingerprint :
    Estimator.fitted -> Feature_schema.t -> Matrix.t -> float array

  val check_scores :
    kind model -> kind Dataset.t -> Cross_validation.score array -> unit

  val cv :
    Execution.t ->
    kind pipeline ->
    kind Dataset.t ->
    (kind model Cross_validation.report, Error.t) result

  val grid :
    Execution.t ->
    kind grid ->
    kind Dataset.t ->
    (kind model Grid_search.report, Error.t) result

  val random :
    Execution.t ->
    kind space ->
    kind Dataset.t ->
    (kind model Grid_search.report, Error.t) result

  val halving :
    Execution.t ->
    kind candidates ->
    Successive_halving.budget ->
    kind Dataset.t ->
    (kind model Successive_halving.report, Error.t) result
end

module Workflow (Task : TASK) = struct
  module Audited = Audit (Task.Estimator)

  let pipeline depth =
    let* imputer =
      Pipeline.transformer ~name:"impute"
        (module Simple_imputer)
        (Simple_imputer.mean ())
    in
    let* builder = Pipeline.add_transformer Pipeline.empty imputer in
    let* estimator = Task.package (module Audited) (specification depth) in
    Pipeline.set_estimator builder estimator

  let grid () =
    Grid_search.create ~base:1 ~build:pipeline
      [|
        Grid_search.axis ~name:"max_depth" ~values:[| 1; 2; 3 |]
          ~encode:(fun value -> Grid_search.Int value)
          ~set:(fun _ value -> Ok value)
        |> get;
      |]
    |> get

  let space () =
    Randomized_search.create ~iterations:3 ~base:1 ~build:pipeline
      [|
        Randomized_search.axis ~name:"max_depth"
          ~distribution:(Parameter_distribution.choice [| 1; 2; 3 |] |> get)
          ~encode:(fun value -> Grid_search.Int value)
          ~set:(fun _ value -> Ok value)
        |> get;
      |]
    |> get

  let check_events source =
    let events = Atomic.get Audited.events in
    List.iter
      (fun event ->
        let rows = row_ids event.event_features in
        let selected = subset source rows in
        let expected =
          expected_imputation source rows (Dataset.features selected)
        in
        for row = 0 to Matrix.rows expected - 1 do
          for column = 0 to Matrix.columns expected - 1 do
            let value = Matrix.get expected row column in
            Alcotest.check
              (Alcotest.float (1e-12 *. Float.max 1. (Float.abs value)))
              "only training rows determine preprocessing" value
              (Matrix.get event.event_features row column)
          done
        done;
        Alcotest.check Alcotest.bool "targets follow training rows" true
          (Task.equal_target (Dataset.target selected) event.event_target);
        let observed = Option.get event.event_weights in
        let expected = Option.get (Dataset.sample_weight selected) in
        Alcotest.check
          (Alcotest.array (Alcotest.float 0.))
          "weights follow training rows"
          (Vector.to_array (Sample_weight.to_vector expected))
          (Vector.to_array (Sample_weight.to_vector observed));
        Alcotest.check Alcotest.int "tree rounds remain a fresh-fit parameter" 5
          (Xgb.Specification.params event.event_specification)
            .Xgb.Specification.rounds;
        Alcotest.check Alcotest.int "one booster thread" 1
          (Xgb.Specification.params event.event_specification)
            .Xgb.Specification.threads)
      events;
    events

  let check_group_separation source fold =
    let groups = Option.get (Dataset.groups source) in
    let train = Option.get fold.Cross_validation.train_indices in
    let test = Option.get fold.Cross_validation.test_indices in
    Array.iter
      (fun left ->
        Array.iter
          (fun right ->
            Alcotest.check Alcotest.bool "groups never cross fold boundaries"
              true
              (Groups.get groups left <> Groups.get groups right))
          test)
      train

  let replay source events model rows x =
    let event =
      List.find (fun event -> row_ids event.event_features = rows) events
    in
    let expected_x =
      expected_imputation source rows (Dataset.features (subset source rows))
    in
    let native =
      Task.Estimator.fit event.event_specification
        ?sample_weight:event.event_weights ~rng:event.event_rng
        ~feature_schema:event.event_schema ~x:expected_x ~y:event.event_target
        ()
      |> get
    in
    let transformed = expected_imputation source rows x in
    Alcotest.check
      (Alcotest.array (Alcotest.float 0.))
      "manual train-only fit reproduces the model"
      (Task.native_fingerprint native event.event_schema transformed)
      (Task.fingerprint model (Dataset.feature_schema source) x)

  let cv execution =
    Atomic.set Audited.events [];
    Atomic.set Audited.models [];
    let source = Task.source () in
    let report = Task.cv execution (pipeline 2 |> get) source |> get in
    Alcotest.check Alcotest.int "all folds succeed" 3
      (Cross_validation.successful_fold_count report);
    let events = check_events source in
    Alcotest.check Alcotest.int "one native fit per fold" 3 (List.length events);
    let fingerprints =
      Cross_validation.folds report
      |> Array.map (fun fold ->
          check_group_separation source fold;
          let rows = Option.get fold.Cross_validation.train_indices in
          let test =
            subset source (Option.get fold.Cross_validation.test_indices)
          in
          let model = Option.get fold.Cross_validation.model in
          replay source events model rows (Dataset.features test);
          Task.check_scores model test fold.Cross_validation.scores;
          Task.fingerprint model
            (Dataset.feature_schema source)
            (Dataset.features test))
    in
    (fold_snapshot report, fingerprints)

  let check_search source candidates selected expected_fit_count =
    let events = check_events source in
    Alcotest.check Alcotest.int "candidate folds plus full-data refit"
      expected_fit_count (List.length events);
    let models = Atomic.get Audited.models in
    Alcotest.check Alcotest.int "every successful fit owns retained state"
      expected_fit_count (List.length models);
    List.iter
      (fun (event, model) ->
        let independent =
          Task.Estimator.fit event.event_specification
            ?sample_weight:event.event_weights ~rng:event.event_rng
            ~feature_schema:event.event_schema ~x:event.event_features
            ~y:event.event_target ()
          |> get
        in
        Alcotest.check
          (Alcotest.array (Alcotest.float 0.))
          "later candidate/fold/refit work preserves each native model"
          (Task.native_fingerprint independent event.event_schema
             event.event_features)
          (Task.native_fingerprint model event.event_schema event.event_features))
      models;
    Array.iter
      (fun candidate ->
        Alcotest.check Alcotest.int "both metrics retained" 2
          (Array.length candidate.Grid_search.scores);
        let report = Option.get candidate.Grid_search.evaluation in
        Alcotest.check Alcotest.int "candidate folds succeed" 3
          (Cross_validation.successful_fold_count report);
        Array.iter
          (fun fold ->
            match
              ( fold.Cross_validation.train_indices,
                fold.Cross_validation.test_indices )
            with
            | Some _, Some _ -> check_group_separation source fold
            | None, None -> ()
            | _ -> Alcotest.fail "incomplete fold indices")
          (Cross_validation.folds report))
      candidates;
    let full = Array.init (Dataset.sample_count source) Fun.id in
    let refit =
      List.filter
        (fun event ->
          Matrix.rows event.event_features = Dataset.sample_count source)
        events
    in
    Alcotest.check Alcotest.int "exactly one full-data refit" 1
      (List.length refit);
    replay source refit selected.Grid_search.selected_model full
      (Dataset.features source);
    Task.fingerprint selected.Grid_search.selected_model
      (Dataset.feature_schema source)
      (Dataset.features source)

  let search execution randomized =
    Atomic.set Audited.events [];
    Atomic.set Audited.models [];
    let source = Task.source () in
    let report =
      if randomized then Task.random execution (space ()) source |> get
      else Task.grid execution (grid ()) source |> get
    in
    let candidates = Grid_search.candidates report in
    let selected = Grid_search.selection report |> get in
    Alcotest.check Alcotest.int "three immutable candidates" 3
      (Array.length candidates);
    Alcotest.check Alcotest.int
      "multi-metric refit chooses the expected candidate"
      (get (choose candidates))
      selected.Grid_search.selected_candidate_index;
    let fingerprint = check_search source candidates selected 10 in
    let partitions =
      Group_k_fold.split
        (Group_k_fold.create ~folds:3 () |> get)
        ~rng:(Rng.create seed) ?groups:(Dataset.groups source)
        ~x:(Dataset.features source) ~y:None ()
      |> get
    in
    Array.iter
      (fun candidate ->
        Alcotest.check Alcotest.int "one depth parameter" 1
          (Array.length candidate.Grid_search.parameters);
        let depth =
          match
            candidate.Grid_search.parameters.(0).Grid_search.parameter_value
          with
          | Grid_search.Int depth -> depth
          | Grid_search.Bool _ | Grid_search.Float _ | Grid_search.String _ ->
              Alcotest.fail "unexpected candidate parameters"
        in
        Array.iter
          (fun (train, test) ->
            let rows = Row_view.indices train in
            let matches =
              Atomic.get Audited.events
              |> List.filter (fun event ->
                  (Xgb.Specification.params event.event_specification)
                    .Xgb.Specification.max_depth = depth
                  && row_ids event.event_features = rows)
            in
            Alcotest.check Alcotest.int
              "each candidate independently fits each group partition" 1
              (List.length matches);
            let groups = Option.get (Dataset.groups source) in
            Array.iter
              (fun left ->
                Array.iter
                  (fun right ->
                    Alcotest.check Alcotest.bool "search groups remain disjoint"
                      true
                      (Groups.get groups left <> Groups.get groups right))
                  (Row_view.indices test))
              rows)
          partitions)
      candidates;
    ( candidate_snapshot candidates,
      selected.Grid_search.selected_candidate_index,
      fingerprint )

  let halving execution =
    Atomic.set Audited.events [];
    Atomic.set Audited.models [];
    let source = Task.source () in
    let budget =
      Successive_halving.budget ~min_samples:12 ~max_samples:36 ~factor:2
        ~max_fits:19 ()
      |> get
    in
    let report =
      Task.halving execution
        (Successive_halving.of_grid (grid ()))
        budget source
      |> get
    in
    let rounds = Successive_halving.rounds report in
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "training-row budgets" [| 12; 24; 36 |]
      (Array.map
         (fun round -> round.Successive_halving.training_samples)
         rounds);
    let previous = ref None in
    Array.iter
      (fun round ->
        let candidates = round.Successive_halving.candidates in
        let membership =
          Cross_validation.folds
            (Option.get candidates.(0).Grid_search.evaluation)
          |> Array.map (fun fold ->
              ( Option.get fold.Cross_validation.train_indices,
                Option.get fold.Cross_validation.test_indices ))
        in
        Array.iter
          (fun candidate ->
            Cross_validation.folds (Option.get candidate.Grid_search.evaluation)
            |> Array.iteri (fun index fold ->
                check_group_separation source fold;
                let train = Option.get fold.Cross_validation.train_indices in
                Alcotest.check Alcotest.int
                  "fresh fit gets the current row budget"
                  round.Successive_halving.training_samples (Array.length train);
                Alcotest.check Alcotest.bool
                  "same rows for competing candidates" true
                  ((train, Option.get fold.Cross_validation.test_indices)
                  = membership.(index))))
          candidates;
        (match !previous with
        | None -> ()
        | Some earlier ->
            Array.iteri
              (fun index (train, test) ->
                let old_train, old_test = earlier.(index) in
                Alcotest.check
                  (Alcotest.array Alcotest.int)
                  "nested training subset" old_train
                  (Array.sub train 0 (Array.length old_train));
                Alcotest.check
                  (Alcotest.array Alcotest.int)
                  "fixed validation rows" old_test test)
              membership);
        previous := Some membership)
      rounds;
    let selected = Successive_halving.selection report |> get in
    let all_candidates =
      Array.concat
        (Array.to_list
           (Array.map (fun round -> round.Successive_halving.candidates) rounds))
    in
    let fingerprint = check_search source all_candidates selected 19 in
    ( Array.map
        (fun round ->
          ( round.Successive_halving.training_samples,
            round.Successive_halving.promoted_candidate_indices,
            candidate_snapshot round.Successive_halving.candidates ))
        rounds,
      selected.Grid_search.selected_candidate_index,
      fingerprint )

  let tests =
    [
      ( Task.name ^ " CV with train-only preprocessing",
        `Quick,
        fun () -> ignore (cv Execution.sequential) );
      ( Task.name ^ " grid and multi-metric refit",
        `Quick,
        fun () -> ignore (search Execution.sequential false) );
      ( Task.name ^ " randomized search",
        `Quick,
        fun () -> ignore (search Execution.sequential true) );
      ( Task.name ^ " row-budget halving",
        `Quick,
        fun () -> ignore (halving Execution.sequential) );
    ]

  let check_execution execution =
    Alcotest.check Alcotest.bool
      (Task.name ^ " CV schedule independence")
      true
      (cv Execution.sequential = cv execution);
    Alcotest.check Alcotest.bool
      (Task.name ^ " grid schedule independence")
      true
      (search Execution.sequential false = search execution false);
    Alcotest.check Alcotest.bool
      (Task.name ^ " randomized schedule independence")
      true
      (search Execution.sequential true = search execution true);
    Alcotest.check Alcotest.bool
      (Task.name ^ " halving schedule independence")
      true
      (halving Execution.sequential = halving execution)
end

module Regression_task = struct
  type kind = Target.regression

  module Estimator = Xgb.Regressor

  let name = "regression"

  let source () =
    Array.init 54 (fun row ->
        (0.4 *. float_of_int row) +. float_of_int ((row * 7 mod 19) - 9))
    |> Vector.of_array |> Target.regression |> data |> dataset

  let equal_target left right =
    Vector.to_array (Target.regression_values left)
    = Vector.to_array (Target.regression_values right)

  let package implementation specification =
    Pipeline.estimator ~name:"xgboost" implementation specification

  let fingerprint model schema x =
    Pipeline.predict model ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array

  let native_fingerprint model schema x =
    Estimator.predict model ~feature_schema:schema ~x
    |> get |> Target.regression_values |> Vector.to_array

  let scorers =
    [|
      Regression_scorer.neg_mean_squared_error;
      Regression_scorer.neg_mean_absolute_error;
    |]

  let check_scores model source scores =
    let predicted =
      Pipeline.predict model
        ~feature_schema:(Dataset.feature_schema source)
        ~x:(Dataset.features source)
      |> get
    in
    Array.iteri
      (fun index score ->
        Alcotest.check (Alcotest.float 1e-12) "weighted test score"
          (Regression_scorer.score scorers.(index)
             ?sample_weight:(Dataset.sample_weight source)
             ~truth:(Dataset.target source) ~prediction:predicted ()
          |> get)
          (Option.get score.Cross_validation.test_score |> get))
      scores

  let cv execution pipeline source =
    Cross_validation.Regression.cross_validate ~execution
      ~return_train_score:true ~return_models:true ~return_indices:true
      ~splitter:(split ()) ~scorers ~seed pipeline source

  let grid execution grid source =
    Grid_search.Regression.search_with_policy ~execution
      ~return_train_score:true ~grid ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let random execution space source =
    Randomized_search.Regression.search_with_policy ~execution
      ~return_train_score:true ~space ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let halving execution candidates budget source =
    Successive_halving.Regression.search ~execution ~return_train_score:true
      ~candidates ~budget ~splitter:(split ()) ~scorers
      ~refit:"neg_mean_squared_error" ~seed source
end

module Classification_base = struct
  type kind = Target.classification

  module Estimator = Xgb.Classifier

  let source count =
    let vocabulary = [| -7; 19; 61 |] in
    Array.init 54 (fun row -> vocabulary.(row mod count))
    |> Target.classification |> dataset

  let equal_target left right =
    Target.classification_values left = Target.classification_values right

  let probability model schema x =
    bounded (fun () -> Estimator.predict_proba model ~feature_schema:schema ~x)

  let package implementation specification =
    Pipeline.estimator ~name:"xgboost" implementation ~classes:Estimator.classes
      ~predict_proba:(fun model ~feature_schema ~x ->
        probability model feature_schema x)
      specification

  let combine labels classes probabilities =
    Array.concat
      [
        Array.map float_of_int (Target.classification_values labels);
        Array.map float_of_int classes;
        Array.concat (Array.to_list (Matrix.to_arrays probabilities));
      ]

  let fingerprint model schema x =
    combine
      (Pipeline.predict model ~feature_schema:schema ~x |> get)
      (Pipeline.classes model |> get)
      (Pipeline.predict_proba model ~feature_schema:schema ~x |> get)

  let native_fingerprint model schema x =
    combine
      (Estimator.predict model ~feature_schema:schema ~x |> get)
      (Estimator.classes model)
      (Estimator.predict_proba model ~feature_schema:schema ~x |> get)

  let check_scores model source scores =
    let labels =
      Pipeline.predict model
        ~feature_schema:(Dataset.feature_schema source)
        ~x:(Dataset.features source)
      |> get
    in
    let probabilities =
      Pipeline.predict_proba model
        ~feature_schema:(Dataset.feature_schema source)
        ~x:(Dataset.features source)
      |> get
    in
    let classes = Pipeline.classes model |> get in
    let accuracy =
      Multiclass_classification_metrics.accuracy
        ?sample_weight:(Dataset.sample_weight source)
        ~truth:(Dataset.target source) ~prediction:labels ()
      |> get
    in
    let loss =
      Multiclass_classification_metrics.log_loss
        ?sample_weight:(Dataset.sample_weight source)
        ~truth:(Dataset.target source) ~classes ~probabilities ()
      |> get
    in
    Alcotest.check (Alcotest.float 1e-12) "weighted classification accuracy"
      accuracy
      (Option.get scores.(0).Cross_validation.test_score |> get);
    Alcotest.check (Alcotest.float 1e-12) "weighted classification log loss"
      (-.loss)
      (Option.get scores.(1).Cross_validation.test_score |> get)
end

module Binary_task = struct
  include Classification_base

  let name = "binary"
  let source () = source 2

  let scorers =
    [|
      Binary_classification_scorer.accuracy;
      Binary_classification_scorer.neg_log_loss ~positive_label:19 ();
    |]

  let cv execution pipeline source =
    Cross_validation.Binary_classification.cross_validate ~execution
      ~return_train_score:true ~return_models:true ~return_indices:true
      ~splitter:(split ()) ~scorers ~seed pipeline source

  let grid execution grid source =
    Grid_search.Binary_classification.search_with_policy ~execution
      ~return_train_score:true ~grid ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let random execution space source =
    Randomized_search.Binary_classification.search_with_policy ~execution
      ~return_train_score:true ~space ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let halving execution candidates budget source =
    Successive_halving.Binary_classification.search ~execution
      ~return_train_score:true ~candidates ~budget ~splitter:(split ()) ~scorers
      ~refit:"neg_log_loss" ~seed source
end

module Multiclass_task = struct
  include Classification_base

  let name = "multiclass"
  let source () = source 3

  let scorers =
    [|
      Multiclass_classification_scorer.accuracy;
      Multiclass_classification_scorer.neg_log_loss;
    |]

  let cv execution pipeline source =
    Cross_validation.Multiclass_classification.cross_validate ~execution
      ~return_train_score:true ~return_models:true ~return_indices:true
      ~splitter:(split ()) ~scorers ~seed pipeline source

  let grid execution grid source =
    Grid_search.Multiclass_classification.search_with_policy ~execution
      ~return_train_score:true ~grid ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let random execution space source =
    Randomized_search.Multiclass_classification.search_with_policy ~execution
      ~return_train_score:true ~space ~splitter:(split ()) ~scorers
      ~policy:(Grid_search.Custom choose) ~seed source

  let halving execution candidates budget source =
    Successive_halving.Multiclass_classification.search ~execution
      ~return_train_score:true ~candidates ~budget ~splitter:(split ()) ~scorers
      ~refit:"neg_log_loss" ~seed source
end

module Regression = Workflow (Regression_task)
module Binary = Workflow (Binary_task)
module Multiclass = Workflow (Multiclass_task)

let check_degenerate_fold execution =
  Atomic.set Binary.Audited.events [];
  Atomic.set Binary.Audited.models [];
  let source =
    Array.init 54 (fun row -> if row < 6 then 19 else -7)
    |> Target.classification |> dataset
  in
  let report =
    Cross_validation.Binary_classification.cross_validate ~execution
      ~return_models:true ~return_indices:true
      ~failure_policy:Cross_validation.Record ~splitter:(split ())
      ~scorers:[| Binary_classification_scorer.accuracy |]
      ~seed
      (Binary.pipeline 2 |> get)
      source
    |> get
  in
  Alcotest.check Alcotest.int "successful peers survive a single-class fold" 2
    (Cross_validation.successful_fold_count report);
  let failures =
    Cross_validation.folds report
    |> Array.to_list
    |> List.filter (fun fold -> Array.length fold.Cross_validation.failures > 0)
  in
  Alcotest.check Alcotest.int "exactly one failed fold" 1 (List.length failures);
  let fold = List.hd failures in
  let failure = fold.Cross_validation.failures.(0) in
  Alcotest.check Alcotest.bool "single-class failure remains typed" true
    (Error.kind failure.Cross_validation.error
    = Error.Validation
        {
          name = "XGBoost classification classes";
          reason = "at least two positively weighted classes are required";
        });
  Alcotest.check Alcotest.bool "failure occurs during fitting" true
    (failure.Cross_validation.phase = Cross_validation.Fitting);
  Alcotest.check Alcotest.bool "fold context preserved" true
    (List.mem (Error.Fold fold.Cross_validation.fold_index)
       (Error.context failure.Cross_validation.error));
  Alcotest.check Alcotest.bool "estimator stage context preserved" true
    (List.mem (Error.Stage "xgboost")
       (Error.context failure.Cross_validation.error));
  Alcotest.check Alcotest.bool "failed fold has no fitted model" true
    (Option.is_none fold.Cross_validation.model);
  fold_snapshot report

let tests =
  ("native OpenMP startup limits", `Quick, check_thread_contract)
  :: ( "single-effective-class fold recording",
       `Quick,
       fun () -> ignore (check_degenerate_fold Execution.sequential) )
  :: Regression.tests
  @ Binary.tests @ Multiclass.tests

let check_execution execution =
  Regression.check_execution execution;
  Binary.check_execution execution;
  Multiclass.check_execution execution;
  Alcotest.check Alcotest.bool "recorded failures are schedule independent" true
    (check_degenerate_fold Execution.sequential
    = check_degenerate_fold execution)
