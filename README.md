# ModelKit

[![CI](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml/badge.svg)](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml)

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Feature Highlights

The full documentation is available via: [https://ocaml.org/p/modelkit/latest/doc/index.html](https://ocaml.org/p/modelkit/latest/doc/index.html)

- Reproducible foundations with deterministic random streams and stable reference numerical operations across supported platforms, OCaml versions, and execution schedules.
- Typed extension contracts separate immutable estimator specifications from fitted models and return actionable errors.
- Immutable, validated float64 data primitives catch shape, feature-order, and sample-alignment problems before model code runs.
- The optional `modelkit-nx` package admits explicitly typed Nx tensors with checked shapes, names, null masks, groups, weights, and observable copy/allocation behavior without making Raven a core dependency.
- Checked immutable CSR matrices provide canonical sparse storage, zero-copy indexed row views, explicit materialization and payload-memory accounting, and portable dense/CSR numerical-kernel dispatch.
- Dense datasets admit aligned features, targets, weights, groups, and names under an explicit finiteness policy; stable schema fingerprints and copy/view reports make compatibility and allocation behavior observable.
- Immutable preprocessing specifications fit mean, median, or constant imputation, population standardization, and variance-based feature filtering without changing or losing feature identities.
- Portable numeric, categorical, target, interaction, and missingness transforms cover min-max, max-absolute, robust, per-sample normalization, one-hot, ordinal, label, polynomial, and missing-indicator workflows.
- Sequential pipelines fit preprocessing only on their training input, preserve schemas through ordered stages, and dispatch prediction, decision, and probability operations through an explicitly capable terminal estimator.
- Portable weighted ordinary least squares, ridge, lasso, elastic-net, binary and multinomial logistic regression, Poisson and Tweedie generalized linear models, plus binary and multiclass ridge classification keep immutable specifications separate from fitted coefficients and solver diagnostics.
- Deterministic K-fold, stratified K-fold, group K-fold, and expanding-window time-series splitters produce validated row views that can be explicitly materialized as aligned datasets.
- Weighted regression and binary classification metrics provide immutable higher-is-better scorers, plotting-neutral residual, ROC, and precision–recall data, stable score aggregation, and an explicit undefined-result policy.
- Cross-validation fits pipelines within deterministic folds and reports ordered train/test scores, CPU timings, optional fitted models and indices, and typed failures; the optional `modelkit-parallel` package adds bounded Domainslib fold execution.
- Typed finite grid search evaluates immutable pipeline configurations on shared deterministic splits, ranks candidates by a named scorer, records candidate failures, and refits the selected model on all training data.
- Versioned data-only artifacts save and load fitted built-in regression and binary-classification pipelines with feature-schema identity, bounded readers, and corruption detection.

## Motivation and Future Work

The library is built with a strong focus first on correctness, portability, and reproducibility; performance is a secondary goal to follow.

To this end, you will note that there is a significant amount of from-scratch implementation under ModelKit's hood. When implementation milestones are hit for being useful in real-world data science workflows, ModelKit will undergo benchmarking to gauge its performance against alternate implementations, such as `scikit-learn` itself.

Anticipating performance benefits from existing work such as using Owl for a numerical engine and Lacaml for acceleration, integration tasks will likely be brought above the line. Such changes will not be included in version 0.3.x, but in later versions. In that phase, users who have come to be familiar with the consistent contracts of ModelKit's public APIs will enjoy performance benefits without contract changes.

## Status

ModelKit 0.3.2 is the current evaluation release. It includes immutable dense dataset admission, explicit `Require_finite` and `Allow_nan` feature policies, aligned zero-copy row views, stable versioned schema fingerprints, and explicit copy/view reporting. `Allow_nan` treats NaN as a missing-value marker but still rejects positive and negative infinity.

Development toward 0.4.0 adds `Csr_matrix` for checked compressed sparse row storage. CSR admission copies caller-owned arrays and requires row offsets to span the stored values monotonically, with in-range columns in strictly increasing order within each row. `Csr_matrix.view` preserves row order and duplicates while sharing the source matrix; `Csr_matrix.materialize` is the explicit packing boundary. `Csr_matrix.memory` and `Csr_matrix.view_memory` report payload bytes separately from runtime object headers and allocator overhead.

Wrap a dense or sparse value with `Feature_matrix.dense` or `Feature_matrix.csr` to use the representation-dispatching `Reference_backend.feature_matrix_vector_product` and `Reference_backend.transposed_feature_matrix_vector_product` kernels:

```ocaml
let sparse =
  Csr_matrix.of_arrays ~rows:2 ~columns:3
    ~row_offsets:[| 0; 2; 3 |]
    ~column_indices:[| 0; 2; 1 |]
    ~values:[| 1.0; 3.0; 2.0 |]
  |> Result.get_ok

let product =
  Reference_backend.feature_matrix_vector_product
    (Feature_matrix.csr sparse)
    (Vector.of_array [| 2.0; 4.0; -1.0 |])
  |> Result.get_ok
```

The current estimators, datasets, pipelines, and model-selection workflows still accept dense `Matrix.t` feature inputs. `One_hot_encoder.transform_csr` can produce checked sparse output directly, but sparse estimator and workflow integration is scheduled in the remaining 0.4.0 work.

Dataset row views preserve ordering and duplicates without packing feature or metadata buffers. Use `Dataset.materialize` when an algorithm requires contiguous selected rows; its access report identifies the resulting copies.

Development toward 0.4.0 also introduces the optional `modelkit-nx` package for moving Raven Nx tensors into these portable data contracts. It accepts rank-two float64 features, rank-one float64 regression targets and weights, rank-one int64 classification targets and groups, Boolean feature null masks, and ordered feature names. Both contiguous tensors and strided views are read in logical order. Every returned ModelKit value owns its immutable storage, and `Conversion_report` makes the retained numeric payload and any full-size staging payload explicit. Install the adapter alongside the core with `opam install modelkit-nx`; code using it opens `Modelkit_nx` separately.

```ocaml
let admitted =
  Modelkit_nx.classification_dataset
    ~names:[| "temperature"; "pressure" |]
    ~feature_null_mask:nulls
    ~sample_weight:weights
    ~groups
    ~x
    ~y
    ()
  |> Result.get_ok

let dataset = admitted.dataset
let null_mask = admitted.feature_null_mask
let allocation = admitted.dataset_reports
```

Explicit feature nulls are written as NaN in the admitted matrix for compatibility with existing missing-value transforms, while the separately returned `Null_mask.t` preserves which positions were source nulls rather than genuine IEEE NaNs. Unmasked infinities, null-mask shape mismatches, invalid names, non-finite targets or weights, negative weights, all-zero weights, and int64 labels outside the current platform's OCaml `int` range return typed errors. Nx remains pinned to the tested `1.0.0~alpha3` release while Raven's API is alpha; Talon admission is not implemented yet.

The 0.3.2 API also provides `Simple_imputer`, `Standard_scaler`, and `Variance_threshold`. Imputation learns only from the supplied training matrix and treats NaN as the missing-value marker. Scaling uses population variance and maps constant centered features to zero with a scale of one. Variance filtering keeps columns whose variance is strictly greater than its threshold and preserves selected names in input order. These transformers reject infinities with typed errors rather than silently continuing.

Development toward 0.4.0 adds `Min_max_scaler`, `Max_abs_scaler`, and `Robust_scaler` for fitted per-feature scaling, plus the stateless `Normalizer` for L1, L2, or maximum-norm scaling of each sample. Constant features and zero-norm rows have defined finite behavior, learned statistics remain inspectable, and fitted stages verify their input schema before transforming new data.

`One_hot_encoder` and `Ordinal_encoder` learn deterministically sorted finite float64 categories and make unknown-category handling explicit. One-hot output is available as either a dense matrix through the common transformer protocol or checked CSR storage through `transform_csr`; an output-width limit prevents accidental allocation from unbounded cardinality. ModelKit does not yet own a heterogeneous string table type, so callers and future table adapters are responsible for mapping string categories to stable finite float values before using these core transforms. `Label_encoder` separately provides reversible sorted encoding for integer classification targets.

`Polynomial_features` creates a deterministic scikit-learn-compatible ordering of polynomial or interaction-only terms with configurable degree, bias inclusion, and output-width limit. `Missing_indicator` converts NaN markers into binary features, optionally selecting only columns that were missing during fitting and optionally rejecting new missing columns at transform time. Infinity remains invalid input.

All matrix transforms are immutable training specifications with distinct fitted values and can be installed in an in-memory pipeline with `Pipeline.transformer`, ensuring fitting occurs only on the matrix supplied to `Pipeline.fit`. Artifact-aware constructors and reviewed codecs for these new stages are not part of the current increment: pipelines that use them work normally in memory, while attempting to encode such a general extension stage returns a typed unsupported-component error.

`Pipeline` now packages these unsupervised transformers with any implementation of ModelKit's public `ESTIMATOR` protocol. Fitting learns every preprocessing stage exclusively from the supplied training matrix, then fits the terminal estimator on the transformed training output. The fitted pipeline reuses those exact stage values for `transform`, `predict`, `decision_function`, and `predict_proba`; unavailable terminal capabilities and named-stage failures are typed errors. Feature schemas are checked at the pipeline boundary and propagated after every transformation. Fixed root RNG state produces stage-local streams derived from stable logical names and positions.

`Linear_regression` fits weighted ordinary least squares with column-pivoted Householder QR and reports numerical rank, including for rank-deficient input. `Ridge_regression` solves an augmented least-squares system without forming normal equations and applies its non-negative `alpha` penalty only to coefficients. Both expose fitted coefficients, intercepts, and direct-solver reports.

Development toward 0.4.0 adds `Lasso_regression` and `Elastic_net_regression` for sparse-coefficient scalar regression. Their portable deterministic cyclic coordinate-descent solver minimizes a weighted objective normalized by total positive sample weight, leaves the optional intercept unpenalized, checks both coordinate updates and optimality residuals, and returns typed convergence failures. `Elastic_net_regression` mixes L1 and L2 coefficient penalties through `l1_ratio`; setting it to one gives lasso semantics.

`Lasso_path` and `Elastic_net_path` fit complete descending regularization paths, warm-starting each point from the preceding stronger penalty. Callers can provide explicit alphas or request a logarithmic path through `epsilon` and `count`. Path results expose alphas, one coefficient row and intercept per alpha, per-point solver reports, and checked access to an ordinary fitted estimator for direct prediction. To use a selected hyperparameter in a pipeline or cross-validation, create the corresponding immutable `Lasso_regression` or `Elastic_net_regression` specification with that alpha. Automatic elastic-net paths require a positive L1 ratio; pure L2 paths require explicit alphas because no finite penalty forces every coefficient to zero.

`Ridge_classifier` provides weighted binary and multiclass classification by fitting one ridge problem per ascending class against targets encoded as negative or positive one. Its decision function always returns a sample-by-class score matrix, including for binary problems, so coefficient rows, intercepts, reports, and score columns share one class order without a shape special case. Prediction selects the first maximum and therefore resolves exact ties to the lowest class label. Only classes represented by positive sample weight participate in fitting.

`Logistic_regression` supports exactly two integer classes, optional sample weights, an L2 coefficient penalty controlled by positive inverse strength `c`, and stable decision/probability calculations for extreme logits. Its deterministic damped Newton fit reports objective, iteration count, and stopping reason; iteration exhaustion and invalid training data are typed errors. These estimators accept only finite feature values, so missing values must be handled by an imputer or before fitting.

`Multinomial_logistic_regression` jointly fits three or more positively weighted classes using stable softmax cross-entropy and an L2 coefficient penalty. A sum-to-zero score constraint removes softmax's non-identifiable common direction while leaving intercepts unpenalized. Coefficient rows, intercepts, decision columns, probability columns, and ascending class labels remain aligned; probabilities stay finite and normalized even when score differences are extreme. Its deterministic damped Newton solver returns one report for the joint optimization.

`Poisson_regression` fits non-negative responses through a log link, while `Tweedie_regression` covers finite powers with `Auto`, `Identity`, and `Log` link selection. Automatic link selection uses identity for nonpositive powers and log for positive powers. Target validation follows the power's mathematical domain: real values for nonpositive powers, non-negative values below power two, and strictly positive values from power two onward. Both estimators normalize optional sample weights, apply `alpha` only to coefficients, expose fitted coefficients, intercepts, and solver reports, and use deterministic damped IRLS. Prediction returns a typed numerical error when the inverse link would overflow or produce an invalid mean.

```ocaml
let poisson = Poisson_regression.create ~alpha:0.1 () |> Result.get_ok

let fitted =
  Poisson_regression.fit poisson ~rng ~feature_schema ~x ~y ()
  |> Result.get_ok

let expected_counts =
  Poisson_regression.predict fitted ~feature_schema ~x:future_x
  |> Result.get_ok
```

`Sgd_regressor` adds portable incremental squared-error regression with no penalty, L1, L2, or elastic-net regularization and constant or inverse-scaling learning rates. `fit` processes one matrix for a configured epoch budget and implements the common regressor protocol. For streamed training, `start` creates an immutable zero-initialized checkpoint and each `partial_fit` call processes exactly one supplied batch, optionally using a deterministic shuffle from the checkpoint-owned RNG stream. The successor checkpoint records the coefficients, intercept, update count, completed batch count, objective, and RNG continuation; the input checkpoint remains reusable. Checkpoints are in-memory state and are not yet part of the artifact format.

```ocaml
let sgd =
  Sgd_regressor.create ~penalty:Sgd_regressor.Elastic_net ~alpha:0.0001 ~l1_ratio:0.15 ~learning_rate:(Sgd_regressor.Inverse_scaling { power_t = 0.25 }) ~eta0:0.01 ()
  |> Result.get_ok

let checkpoint = Sgd_regressor.start sgd ~rng ~feature_schema

let checkpoint =
  Sgd_regressor.partial_fit checkpoint ~sample_weight ~feature_schema ~x:batch_x ~y:batch_y ()
  |> Result.get_ok

let fitted = Sgd_regressor.to_fitted checkpoint |> Result.get_ok
```

`Sgd_classifier` trains binary and multiclass linear classifiers with the same immutable checkpoint contract, penalties, and learning-rate schedules, using either the `Hinge` margin loss or the stable `Log_loss`. A stream registers its complete class set when the checkpoint starts; later batches may omit classes but never introduce unregistered ones. Two classes train one model, more train one one-versus-rest model per ascending class, and every model shares the update counter and per-batch shuffle. `decision_function` returns one score column per model, `binary_decision_function` exposes the binary score as a vector for pipeline dispatch, and `predict_proba` is available for `Log_loss` only.

```ocaml
let classifier =
  Sgd_classifier.create ~loss:Sgd_classifier.Log_loss ~alpha:0.0001 ~eta0:0.01 ()
  |> Result.get_ok

let checkpoint =
  Sgd_classifier.start classifier ~rng ~feature_schema ~classes:[| 0; 1; 2 |]
  |> Result.get_ok

let checkpoint =
  Sgd_classifier.partial_fit checkpoint ~feature_schema ~x:batch_x ~y:batch_labels ()
  |> Result.get_ok

let probabilities =
  Sgd_classifier.to_fitted checkpoint
  |> Result.get_ok
  |> fun fitted -> Sgd_classifier.predict_proba fitted ~feature_schema ~x:future_x
```

### Linear workbench workflow coverage

| Component | Pipeline capabilities | Cross-validation and scoring | scikit-learn fixture | Current boundary |
| --- | --- | --- | --- | --- |
| `Lasso_regression`, `Elastic_net_regression` | Prediction | Regression scorers | Coefficients, intercepts, predictions | Dense input; in-memory only |
| `Lasso_path`, `Elastic_net_path` | Selected fitted models predict directly | Recreate an estimator specification with the selected alpha | Alpha order, coefficients, intercepts | Paths are fitting utilities, not estimator specifications |
| `Ridge_classifier` | Prediction and class metadata | Binary label scorers; multiclass label scorers and grid search | Binary/multiclass coefficients, scores, predictions | Matrix decision scores are available directly from the estimator |
| `Multinomial_logistic_regression` | Prediction, probabilities, class metadata | Multiclass label, log-loss, ROC AUC, and top-k scorers and grid search | Coefficients, intercepts, scores, probabilities, predictions | Matrix decision scores are available directly from the estimator |
| `Poisson_regression`, `Tweedie_regression` | Prediction | Regression scorers | Coefficients, intercepts, predictions | Dense input; in-memory only |
| `Sgd_regressor` | Prediction after `fit` or a checkpoint snapshot | Regression scorers and grid search | Full-fit and repeated-`partial_fit` coefficients, intercepts, predictions | Dense batches; in-memory checkpoints |
| `Sgd_classifier` | Prediction, binary decision scores, `Log_loss` probabilities, class metadata | Binary and multiclass label scorers for both losses, probability and ranking scorers for `Log_loss`, and grid search | Binary and multiclass hinge and log-loss coefficients, intercepts, scores, probabilities, predictions after `fit` and repeated `partial_fit` | Dense batches; in-memory checkpoints; multiclass scores are obtained directly from the estimator |

All estimator rows above use the common immutable estimator contract and have consolidated pipeline and scoring coverage. Lasso, elastic-net, Poisson, Tweedie, SGD regression, binary ridge classification, and binary SGD classification are exercised end to end through pipeline fitting, prediction, cross-validation, and the currently supported scorers; both SGD estimators are additionally exercised through fold-local standardization and finite grid search. A hinge-loss SGD classifier records a typed per-fold failure rather than a coerced score when a probability scorer is requested. Continuation-equivalence tests establish that ordered streams give identical parameters however batches are cut, that a snapshot resumed through its checkpoint matches the uninterrupted stream, that sibling continuations of one parent are identical while the parent is unchanged, and that rejected batches leave a checkpoint reusable. Multiclass ridge, multinomial logistic, and multiclass SGD pipelines are exercised through multiclass cross-validation and grid search, including balanced class weights and probability-based refit; multiclass outputs are never coerced through binary metrics. None of these additions yet has a built-in artifact constructor or codec, so their pipelines remain usable in memory while artifact encoding returns a typed unsupported-component error.

The current pipeline decision-function capability is vector-valued for binary estimators. Matrix-valued ridge and multinomial-logistic scores are therefore obtained directly from their respective `decision_function` functions, while pipeline prediction, probability, and class dispatch remain available as shown in the table.

Built-in logistic regression can be installed as a pipeline terminal with its optional capabilities:

```ocaml
let logistic = Logistic_regression.create () |> Result.get_ok

let terminal =
  Artifact.logistic_regression_estimator ~name:"logistic" logistic
  |> Result.get_ok

let specification =
  Pipeline.set_estimator Pipeline.empty terminal |> Result.get_ok
```

Use the corresponding `Artifact.simple_imputer_stage`, `Artifact.standard_scaler_stage`, and `Artifact.variance_threshold_stage` constructors for preprocessing that will be persisted. After fitting, encode and restore the complete typed pipeline without runtime-specific values:

```ocaml
let encoded =
  Artifact.encode_binary_classification fitted |> Result.get_ok

let restored =
  Artifact.decode_binary_classification encoded
  |> Result.get_ok |> Artifact.model
```

`Artifact.save_binary_classification` and `Artifact.load_binary_classification` provide file convenience functions; regression has matching APIs. Artifacts retain fitted values, feature schemas, solver reports, and optional non-secret training metadata, but never training observations, closures, commands, or `Marshal` data. The versioned binary format uses canonical big-endian integers and IEEE-754 values, a declared MD5 corruption checksum, and configurable byte, component, feature, string, and metadata limits. MD5 is used only to detect accidental corruption and does not authenticate or encrypt an artifact. The format is experimental during ModelKit 0.x, with a committed golden reader retained for each released schema. Pipelines assembled through the general extension constructors remain usable in memory; encoding returns a typed error when any component has no reviewed artifact codec.

Sample weights always route to the terminal estimator. A transformer stage receives them only when packaged with `~route_sample_weight:true`, which keeps stages that reject weights, such as the imputer, from failing on weighted datasets while letting `Standard_scaler` fit weighted means and population variances; the artifact-aware `Artifact.standard_scaler_stage` accepts the same flag. General metadata routing through nested consumers remains planned for a later milestone.

`Class_weight` turns a `Balanced` or `Explicit` class-weight specification into per-row sample weights. `Pipeline.classifier` and `Artifact.logistic_regression_estimator` accept `~class_weight` and resolve it on each fit's own labels and sample weights, so balanced weights under cross-validation and grid search are computed from the training fold alone. Balanced weights follow scikit-learn's `total / (classes * class_total)` rule over weighted class frequencies, explicit weights default unlisted labels to one, and zero-weight rows stay zero.

```ocaml
let balanced =
  Pipeline.classifier ~class_weight:Class_weight.balanced ~name:"model"
    ~predict_proba:Logistic_regression.predict_proba
    ~classes:Logistic_regression.classes
    (module Logistic_regression)
    (Logistic_regression.create ~c:1.0 () |> Result.get_ok)
  |> Result.get_ok
```

`K_fold`, `Stratified_k_fold`, `Group_k_fold`, and `Time_series_split` now provide the portable splitting primitives needed by evaluation workflows. K-fold variants balance test sizes; stratification balances each integer class; group splitting prevents a group from crossing train/test boundaries; and time-series splitting uses expanding chronological training prefixes with optional gaps. Shuffled variants use ModelKit’s immutable deterministic RNG and retain source-row order in emitted views.

`Split.create` and `Split.of_views` validate non-empty, unique, disjoint train/test selections over one source. `Split.materialize` is the explicit allocation boundary that copies those selections into independent datasets while retaining targets, feature names, sample weights, groups, and schema identity.

`Regression_metrics` provides weighted MAE, MSE, RMSE, R², and residual data. `Binary_classification_metrics` provides weighted accuracy, balanced accuracy, precision, recall, F1, log loss, ROC AUC, and deterministic ROC and precision–recall curve arrays. Binary probabilities are validated as finite values in `[0, 1]`, and binary label metrics accept a configurable positive label.

`Multiclass_classification_metrics` adds a weighted confusion matrix with explicit or ascending-union label order, accuracy and balanced accuracy for any label count, per-class precision, recall, F1, and support, and `Micro`, `Macro`, and `Weighted` averages that follow scikit-learn's definitions, including its treatment of a zero-division class as zero under `Use_fallback`. Multiclass log loss takes a probability matrix with its declared class order, clips to machine epsilon, and rejects rows that do not sum to one. `Multiclass_classification_scorer` names averaged scorers by mode, for example `f1_weighted`, and `Cross_validation.Multiclass_classification` and `Grid_search.Multiclass_classification` evaluate any pipeline whose terminal declares two or more classes, so binary pipelines can also be scored with multiclass scorers.

Ranking metrics complete the scorer surface. `Binary_classification_metrics.average_precision` sums precision over the recall steps of the precision–recall curve without interpolation. `Multiclass_ranking.roc_auc` offers one-versus-rest ROC AUC with `Macro`, `Weighted`, and `Micro` averaging and one-versus-one ROC AUC with `Macro` and `Weighted` averaging over the classes present, and `Multiclass_ranking.top_k_accuracy` counts a hit when fewer than `k` classes outrank the truth. `Ranking_metrics.dcg` and `ndcg` score per-row graded relevance against ranking scores with a logarithmic discount, an optional cutoff, and tie-averaged gains by default. Every one of these is available as a binary or multiclass scorer, so `roc_auc_ovo_weighted`, `top_2_accuracy`, and `average_precision` can drive cross-validation and grid-search refit.

```ocaml
let report =
  Cross_validation.Multiclass_classification.cross_validate
    ~splitter ~scorers:Multiclass_classification_scorer.[| accuracy; f1 ~average:Multiclass_classification_metrics.Macro (); neg_log_loss |]
    ~seed:(Seed.of_int 42) multinomial_pipeline dataset
  |> Result.get_ok
```

Undefined metrics are observable through `Undefined_metric_policy`: the default returns a typed error, while callers may explicitly request NaN or documented finite fallbacks. `Regression_scorer` and `Binary_classification_scorer` make every selection score higher-is-better by negating loss metrics, and `Score_aggregation` reports stable population summaries for fold scores.

`Cross_validation.Regression.cross_validate` and `Cross_validation.Binary_classification.cross_validate` run a complete evaluation in stable logical fold order. Splitters are adapted explicitly as target-independent or target-aware, each fold derives its RNG from a fit seed and logical fold index, and preprocessing is fitted only after the training partition is materialized. The fit seed defaults to the split seed but can vary independently for deterministic meta-estimators without changing split membership. Reports contain process CPU fit/score timings, multiple scorers in caller order, optional train scores, fitted models, and original row indices. The default `Abort` failure policy returns the lowest-index typed failure; `Record` retains structured prediction and scoring failures and continues evaluating later folds. Under parallel execution, per-fold CPU-time intervals can overlap and should not be summed as elapsed wall time.

Binary probability scorers require the pipeline terminal to declare its probability-column class order with `Pipeline.estimator ~classes`; ModelKit does not assume that a particular matrix column represents the configured positive label. Cross-validation uses the always-available sequential implementation unless an optional execution backend is supplied.

Install `modelkit-parallel` to evaluate independent folds with a bounded Domainslib pool. The requested `domains` count includes the calling domain, and `domains:1` takes the sequential path without creating a pool. Fixed seeds produce identical logical results across domain counts. `Modelkit_parallel.diagnostics` reports the requested fold concurrency, runtime-recommended domain count, detected or explicitly supplied inner numerical-library thread limit, estimated runnable threads, and typed oversubscription warnings; it never mutates process environment variables. Configure numerical libraries for one inner thread per fold, or explicitly choose sequential folds when an inner solver owns parallelism.

```ocaml
let parallel =
  Modelkit_parallel.create ~inner_threads:1 ~domains:4 () |> Result.get_ok

let execution = Modelkit_parallel.execution parallel
let diagnostics = Modelkit_parallel.diagnostics parallel
```

`Grid_search.axis` defines a non-empty, typed parameter axis by encoding report values and immutably updating a user-owned configuration record. `Grid_search.create` forms the finite Cartesian product in stable declaration order, and the regression and binary-classification search functions evaluate every candidate against identical split membership. Reports retain candidate parameters, mean CPU timings, aggregate train/test scores, ranks, underlying cross-validation reports, and typed build failures. The default `Record` policy excludes candidates with unavailable primary scores while continuing the search; `Abort` returns the first failure. The named `refit` scorer selects the winner, with the lowest candidate index resolving exact ties, and the winning specification is fitted once on the complete dataset. Passing an execution backend parallelizes each candidate's folds while retaining stable sequential candidate order.

## Architecture

The supported library API is the flat `Modelkit.*` namespace documented by `lib/modelkit.mli`. The physical `Modelkit_*` compilation units are private implementation details: consumers should depend on `modelkit` and use modules such as `Modelkit.Dataset`, `Modelkit.Pipeline`, and `Modelkit.Artifact`, rather than importing internal source units directly.

The portable implementation is organized by responsibility:

| Source unit | Responsibility |
| --- | --- |
| `modelkit_data` | Immutable vectors, dense and CSR matrices, row views, memory accounting, targets, schemas, datasets, and typed errors |
| `modelkit_protocols` | Extension contracts, deterministic random streams, execution, and dense/CSR reference numerical kernels |
| `modelkit_preprocessing` | Shared preprocessing validation, imputation, standardization, and variance filtering |
| `modelkit_transforms` | Numeric, categorical, target, polynomial, and missing-indicator transforms |
| `modelkit_pipeline` | Leakage-safe pipeline construction, fitting, and inference dispatch |
| `modelkit_linear_models` | Solver reports, shared numerical routines, and linear estimators |
| `modelkit_regularized_linear` | Coordinate-descent lasso and elastic-net estimators and regularization paths |
| `modelkit_sgd` | Immutable incremental SGD checkpoints and online linear estimators |
| `modelkit_linear_classifiers` | Binary and multiclass linear classifiers with matrix-valued class scores |
| `modelkit_glm` | Poisson and Tweedie generalized linear regression with stable link handling |
| `modelkit_splitting` | Validated splits and built-in cross-validation splitters |
| `modelkit_metrics` | Metrics, binary responses, scorers, and score aggregation |
| `modelkit_model_selection` | Cross-validation and finite grid search |
| `modelkit_artifact` | Versioned fitted-pipeline persistence and built-in component codecs |
| `modelkit.ml` | Public façade retaining the stable `Modelkit.*` namespace |

These units remain within the portable `modelkit` package under `lib/`; they are not separately installable packages or additional public namespaces. Dependencies flow from higher-level workflows toward data and protocol foundations. The optional `modelkit-parallel` package lives under `backends/parallel/` and the optional `modelkit-nx` package lives under `adapters/nx/`; both depend inward on the portable core. The Nx adapter exposes the separate `Modelkit_nx` namespace and does not add Raven to `modelkit` itself. Talon and accelerated numerical backends remain reserved under `adapters/` and `backends/` as separate future packages.

## Development

ModelKit requires OCaml 5.2 or newer. The platform locks currently use OCaml 5.3.0. The following set of commands will assume that you have installed and configured `git` and `opam`. The generated documentation will be available at `_build/default/_doc/_html/index.html`.

### Initial Setup

```commandline
opam update
opam switch create . 5.3.0 --deps-only --with-test --with-doc  # If running for the first time.
opam install ocamlformat.0.29.0

opam exec -- dune build @all @runtest @doc @fmt @opam @install --auto-promote
opam lint modelkit.opam
opam lint modelkit-parallel.opam
opam lint modelkit-nx.opam
```

### Windows

```commandline
opam lock ./modelkit.opam ./modelkit-parallel.opam --lock-suffix=locked.windows-x86_64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64
```

### macOS (arm64)

```sh
opam lock ./modelkit.opam ./modelkit-parallel.opam --lock-suffix=locked.macos-arm64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.macos-arm64
```

The ordinary Dune workspace uses the repository-local opam switch automatically. Reproducible locks are platform-specific because compiler and system dependency packages differ by host.

The full test suite combines named unit tests, deterministic generated properties, metamorphic invariants, executable documentation, a compiled end-to-end example, artifact golden-reader and adversarial-input tests, a compile-time public API consumer, and a reusable numerical-backend conformance suite. Run the current supervised workflow from a source checkout with `opam exec -- dune exec examples/evaluation.exe`.

GitHub Actions is configured to run the build, complete test suite, package build, and documentation generation on Linux x86-64, macOS arm64, and Windows x86-64 with OCaml 5.2, 5.3, and 5.5. These jobs use committed reference data and do not install or execute Python.

### Reference Fixtures and Benchmarks

Committed scikit-learn reference fixtures are ordinary test data, so the normal ModelKit build and test suite never require or execute Python. Maintainers only need the pinned development environment when regenerating those fixtures or collecting benchmark evidence. Python 3.14.3 is required, as recorded in `dev/python/PYTHON_VERSION`; the local virtual environment is stored in the ignored `env/` directory.

On Windows:

```commandline
env\Scripts\activate
python -m pip install --requirement dev\python\requirements.lock
python dev\fixtures\generate.py
python dev\benchmarks\run.py
```

On macOS/Linux:

```sh
source env/bin/activate
python -m pip install --requirement dev/python/requirements.lock
python dev/fixtures/generate.py
python dev/benchmarks/run.py
```

The committed smoke benchmark validates the measurement workflow only. The development preprocessing, dense-linear-model, regularized-linear, SGD-regression, SGD-classification, ridge-classifier, multinomial-logistic, generalized-linear-model, splitter, metrics, sequential and bounded-parallel cross-validation, and finite grid-search benchmarks compare ModelKit operations with pinned scikit-learn references on deterministic workloads. Build the corresponding OCaml worker and select a scenario under `dev/benchmarks/scenarios/`; the parallel cross-validation scenario records sequential and four-worker results for both runtimes so speedup, efficiency, wall time, and peak RSS can be compared. These reports are explicitly ineligible to support performance claims. See [the benchmark methodology](dev/benchmarks/README.md) for declared parity tolerances, scope, raw-result links, and limitations. Release comparisons will use the product plan's independent-CI benchmark contract.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
