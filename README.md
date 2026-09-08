# ModelKit

[![CI](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml/badge.svg)](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml)

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Feature Highlights

The full documentation is available via: [https://ocaml.org/p/modelkit/latest/doc/index.html](https://ocaml.org/p/modelkit/latest/doc/index.html)

- Reproducible foundations with deterministic random streams and stable reference numerical operations across supported platforms, OCaml versions, and execution schedules.
- Typed extension contracts separate immutable estimator specifications from fitted models and return actionable errors.
- Immutable, validated float64 data primitives catch shape, feature-order, and sample-alignment problems before model code runs.
- The optional `modelkit-nx` and `modelkit-talon` packages admit explicitly typed Nx tensors and explicitly selected Talon dataframe columns with checked shapes, names, null masks, groups, weights, and observable copy/allocation behavior without making Raven a core dependency.
- Checked immutable CSR matrices provide canonical sparse storage, zero-copy indexed row views, explicit materialization and payload-memory accounting, and portable dense/CSR numerical-kernel dispatch.
- Dense datasets admit aligned features, targets, weights, groups, and names under an explicit finiteness policy; stable schema fingerprints and copy/view reports make compatibility and allocation behavior observable.
- Immutable preprocessing specifications fit mean, median, or constant imputation, population standardization, and variance-based feature filtering without changing or losing feature identities.
- Dense univariate feature selection ranks columns with regression correlation F-scores or classification ANOVA F-scores, retains a checked count or percentile with deterministic tie handling, and fits only on each supervised pipeline's training rows.
- Dense model-based selection fits any estimator that implements the explicit feature-importance contract, supports mean, median, or numeric thresholds plus an optional feature cap, and provides checked coefficient helpers for scalar and multiclass linear models.
- Dense recursive feature elimination repeatedly refits an importance estimator over shrinking feature sets, supports integer or fractional elimination steps, reports deterministic rankings, and leaves a final estimator fitted on exactly the selected schema.
- Dense recursive feature elimination with cross-validation scores every fold-local elimination width, chooses the smallest width at an equal best mean score, enforces an optional pre-fit work bound, and refits the selected width on all training rows.
- Portable numeric, categorical, target, interaction, and missingness transforms cover min-max, max-absolute, robust, per-sample normalization, one-hot, ordinal, label, polynomial, and missing-indicator workflows.
- Sequential pipelines support unsupervised and target-aware preprocessing fitted only on training rows, with explicit sample-weight routing, preserved feature schemas, and terminal prediction, decision, and probability dispatch.
- Dense column-wise preprocessing combines independently fitted branches with checked index/name selectors, passthrough/drop, deterministic output names, and observable copy allocations.
- Typed metadata requests route weights, groups, and callbacks through nested fitting, inference, cross-validation, and search, with aligned rows and deterministic progress reporting.
- Opt-in content-addressed transform caching reuses fitted preprocessing state through pipelines, nested composition, cross-validation, and search. Complete feature, target, routed-weight, configuration, and seed identities prevent cross-fit reuse, while caller-scoped bounded memory and portable integrity-checked directory stores avoid global mutable state.
- Feature unions, column transformers, and nestable preprocessing chains combine supervised and unsupervised transformations with deterministic feature names and fold-local fitting.
- Transformed-target regression learns target mappings within each training fold, checks inverse transforms, and scores predictions in the original target space.
- Portable weighted ordinary least squares, ridge, lasso, elastic-net, binary and multinomial logistic regression, Poisson and Tweedie generalized linear models, binary and multiclass ridge classification, and incremental SGD estimators keep immutable specifications separate from fitted coefficients and solver diagnostics.
- Deterministic K-fold, stratified and repeated K-fold, shuffle and stratified-shuffle, holdout, group and stratified-group K-fold, predefined, leave-one-out/group-out, and expanding-window time-series splitters produce validated row views; train/test helpers materialize aligned features, targets, weights, and groups.
- Weighted regression, binary, multiclass, and ranking metrics provide immutable higher-is-better scorers, plotting-neutral residual, ROC, and precision–recall data, stable score aggregation, and an explicit undefined-result policy.
- Cross-validation fits pipelines within deterministic folds and reports ordered train/test scores, CPU timings, optional fitted models and indices, and typed failures; out-of-fold prediction restores regression values, classification labels, or globally aligned class probabilities to source row order and rejects splitters without exact test coverage; learning curves evaluate nested training-fold prefixes, validation curves evaluate one typed immutable parameter sequence over shared fixed folds, and permutation tests estimate corrected significance globally or within groups; the optional `modelkit-parallel` package adds bounded Domainslib execution to these evaluation workflows.
- Typed grid, randomized, and successive-halving search evaluate immutable configurations on shared deterministic splits, support named-score or custom multi-metric selection and optional refitting, retain candidate failures, and resume from validated checkpoints; halving adds bounded training-row budgets and deterministic promotion.
- Versioned data-only artifacts save and load fitted built-in regression and binary-classification pipelines with feature-schema identity, bounded readers, and corruption detection.

## Motivation and Future Work

The library is built with a strong focus first on correctness, portability, and reproducibility; performance is a secondary goal to follow.

To this end, you will note that there is a significant amount of from-scratch implementation under ModelKit's hood. When implementation milestones are hit for being useful in real-world data science workflows, ModelKit will undergo benchmarking to gauge its performance against alternate implementations, such as `scikit-learn` itself.

Anticipating performance benefits from existing work such as using Owl for a numerical engine and Lacaml for acceleration, integration tasks will likely be brought above the line. Such changes are not included in version 0.4.x, but in later versions. In that phase, users who have come to be familiar with the consistent contracts of ModelKit's public APIs will enjoy performance benefits without contract changes.

## Status

This branch builds ModelKit 0.5.0-dev. The supported API is the flat `Modelkit.*` namespace documented in the [manual](https://ocaml.org/p/modelkit/latest/doc/index.html); the physical `Modelkit_*` source units are private. Optional integrations ship as separate packages that depend inward on the core: `modelkit-parallel` for bounded Domainslib fold execution, and `modelkit-nx` and `modelkit-talon` for checked admission of Raven tensors and dataframe columns.

Compared with 0.3.2, this release adds:

- **Sparse storage.** Checked immutable CSR matrices with indexed row views, explicit materialization, payload-memory accounting, and dense/CSR kernel dispatch, plus direct CSR output from one-hot encoding.
- **A fuller preprocessing set.** Min-max, max-absolute, and robust scaling, per-sample normalization, one-hot, ordinal, and label encoding, polynomial features, and missing indicators, all as immutable specifications with distinct fitted states and feature-name propagation.
- **Dense univariate feature selection.** Regression and classification selectors provide correlation or ANOVA F-score ranking, checked count and percentile policies, deterministic original-column ties, named output schemas, and fold-local target-aware pipeline and cross-validation integration.
- **Dense model-based feature selection.** `Select_from_model.Make` accepts an explicitly adapted importance estimator, fits it within each training partition, validates one finite non-negative importance per input feature, and applies mean, median, or numeric thresholds with an optional deterministic feature cap. Coefficient helpers cover absolute scalar coefficients and L1, L2, or maximum reduction across multiclass coefficient rows.
- **Dense recursive feature elimination.** `Recursive_feature_elimination.Make` repeatedly fits an explicitly adapted importance estimator on shrinking dense feature sets until a checked target width is reached. Integer and fractional steps, deterministic weakest-feature ties and rankings, named output schemas, routed sample weights, logical-round seeds, and fold-local supervised pipeline/CV use are supported.
- **Cross-validated recursive feature elimination.** The task-typed `Recursive_feature_elimination_cv` functors score complete fold-local elimination paths, deterministically choose a feature width, and perform a fresh full-data refit. Configured splitters receive groups, estimators and scorers receive aligned sample weights, fold execution can use the optional bounded parallel backend, and `max_fits` rejects excessive work before the first estimator fit.
- **More linear estimators.** Lasso and elastic-net regression with regularization paths, binary and multiclass ridge classification, multinomial logistic regression, Poisson and Tweedie generalized linear models, and SGD regression and classification with an explicit incremental-training and checkpoint contract. Every estimator exposes coefficients, intercepts, and solver diagnostics.
- **Weights and multiclass evaluation.** Fold-local class weights, opt-in sample-weight routing to transformers, confusion-matrix and multiclass metrics with micro, macro, and weighted averaging, average precision, one-versus-rest and one-versus-one ROC AUC, top-k accuracy, DCG and NDCG, and multiclass cross-validation and grid search.
- **Ecosystem adapters.** `modelkit-nx` and `modelkit-talon` admit explicitly typed features, targets, null masks, groups, names, and weights with conversion and allocation reports and a shared conformance suite. Both are pinned to Raven `1.0.0~alpha3` and build on Linux and macOS only.
- **Third-party protocol tooling.** Published capability descriptions distinguish optional estimator, transformer, and scorer behavior from mandatory protocol invariants. Framework-neutral conformance reports exercise external implementations, while first-class custom scorers can be combined with built-in scorers in cross-validation, grid search, and randomized search without changing the built-in scorer parameter contract.
- **Transform caching.** `Simple_imputer` and `Standard_scaler` provide stable fitted-state codecs and can be packaged with `Pipeline.cacheable_transformer`. `Pipeline.with_cache` threads the explicit store through ordinary and supervised pipelines, transformer chains, feature unions, column transformers, cross-validation, and search; cloned specifications retain the same caller-owned store. Keys include the complete input schema and values, supervised targets, routed sample weights, transformer configuration, and the logical stage seed. Unsupported ordinary or metadata-aware transformer leaves fail preflight before any stage fits. Persistent readers reject oversized or corrupt entries before reuse, while temporary-file publication and atomic rename keep concurrent readers from observing partial writes.

Persistent cache entries are plaintext fitted state: their checksums detect accidental corruption but do not authenticate content or provide encryption. Consumers caching data derived from secrets must protect the cache root, backups, and retention lifecycle with their environment's access controls and encryption.

Caching is disabled unless a store is attached to an immutable pipeline specification:

```ocaml
let memory = Modelkit.Transform_cache.Memory.create () in
let store = Modelkit.Transform_cache.Store.memory memory in
let scale =
  Modelkit.Pipeline.cacheable_transformer ~name:"scale"
    (module Modelkit.Standard_scaler)
    (Modelkit.Standard_scaler.create ())
  |> Result.get_ok
in
let builder = Modelkit.Pipeline.add_transformer Modelkit.Pipeline.empty scale |> Result.get_ok in
let estimator =
  Modelkit.Pipeline.estimator ~name:"linear"
    (module Modelkit.Linear_regression)
    (Modelkit.Linear_regression.create ())
  |> Result.get_ok
in
let pipeline = Modelkit.Pipeline.set_estimator builder estimator |> Result.get_ok in
let cached_pipeline = Modelkit.Pipeline.with_cache pipeline store
```

For reuse across processes, create a `Transform_cache.Persistent.t` with an application-managed root and wrap it with `Transform_cache.Store.persistent`. `Pipeline.without_cache` returns an otherwise identical specification with caching disabled. A warm hit skips transformer fitting but still decodes the fitted state and transforms the current training matrix; terminal estimators are always refitted.

Univariate selection is packaged as an ordinary supervised pipeline stage, so every cross-validation or search fold learns its scores from training rows alone:

```ocaml
let select =
  Modelkit.Univariate_selection.Regression.create
    (Modelkit.Univariate_selection.Count 12)
  |> Result.get_ok
  |> Modelkit.Pipeline.Supervised.transformer ~name:"select"
       (module Modelkit.Univariate_selection.Regression)
  |> Result.get_ok
```

`Percentile p` retains `floor (input_width * p / 100)` columns. Both modes rank higher scores first, prefer the lower original column index at a tie, preserve selected columns in input order, and propagate named schemas. The current statistical scope is finite, unweighted dense input using regression correlation F-scores or classification one-way ANOVA F-scores. These scores are for ranking: p-values, multiple-testing corrections, mutual-information and chi-squared scores, sample-weighted statistics, sparse inputs, and artifact/cache codecs are not yet included.

Model-based selection uses a structural module contract instead of inspecting estimator attributes at runtime. For a scalar linear model, the adapter and reusable selector module are:

```ocaml
module Ridge_importance = struct
  include Modelkit.Ridge_regression

  let feature_importances fitted =
    Modelkit.Feature_importance.absolute_coefficients (coefficients fitted)
end

module Ridge_selector = Modelkit.Select_from_model.Make (Ridge_importance)

let select =
  Ridge_selector.create
    ~threshold:Modelkit.Select_from_model.Mean
    ~max_features:12
    (Modelkit.Ridge_regression.create ~alpha:1.0 () |> Result.get_ok)
  |> Result.get_ok
  |> Modelkit.Pipeline.Supervised.transformer ~name:"select"
       (module Ridge_selector)
  |> Result.get_ok
```

For multiclass coefficient matrices, use `Feature_importance.coefficient_norms`; its default L1 reduction matches scikit-learn's model-selection convention, while L2 and maximum reductions are explicit alternatives. Model-based selectors accept finite dense features and can pass sample weights to their estimator when the stage is packaged with `~route_sample_weight:true`. The fitted selector exposes its resolved threshold, validated importances, selected indices, and underlying fitted estimator. Sparse input and artifact/cache codecs remain deferred.

Recursive feature elimination uses the same explicit importance contract but refits the estimator after each elimination step:

```ocaml
module Ridge_rfe = Modelkit.Recursive_feature_elimination.Make (Ridge_importance)

let recursive_select =
  Ridge_rfe.create
    ~step:(Modelkit.Recursive_feature_elimination.Count 2)
    ~feature_count:12
    (Modelkit.Ridge_regression.create ~alpha:1.0 () |> Result.get_ok)
  |> Result.get_ok
  |> Modelkit.Pipeline.Supervised.transformer ~name:"recursive_select"
       (module Ridge_rfe)
  |> Result.get_ok
```

A fractional step is resolved once against the original input width; the fraction must be strictly between zero and one. Weakest features are removed first, equal importances remove the lower original column index first, and selected output columns retain input order. Ranking `1` denotes a selected feature, while larger values denote earlier elimination. Every round receives a logical child seed and any explicitly routed sample weights. The fitted selector exposes its selected indices, ranking, final-estimator importances, and final estimator.

Cross-validated recursive elimination reuses the same adapted estimator and elimination steps, but learns the output width from untouched validation rows:

```ocaml
module Ridge_rfecv = Modelkit.Recursive_feature_elimination_cv.Regression.Make (Ridge_importance)

let splitter =
  Modelkit.K_fold.create ~folds:5 ~shuffle:true ()
  |> Result.get_ok
  |> Modelkit.Cross_validation.target_independent_splitter (module Modelkit.K_fold)

let recursive_select_cv =
  Ridge_rfecv.create
    ~min_feature_count:4
    ~step:(Modelkit.Recursive_feature_elimination.Count 2)
    ~max_fits:150
    ~splitter
    ~scorer:Modelkit.Regression_scorer.neg_mean_squared_error
    (Modelkit.Ridge_regression.create ~alpha:1.0 () |> Result.get_ok)
  |> Result.get_ok
  |> Modelkit.Pipeline.Supervised.metadata_transformer ~name:"recursive_select_cv"
       (module Ridge_rfecv)
  |> Result.get_ok
```

Each validation fold fits its elimination path only on that fold's training rows and scores each visited width on its test rows. Fold paths can run through a supplied `Execution.t`, while estimator fits within one path remain sequential to avoid nested oversubscription. Feature counts are reported in ascending order; the smaller width wins an exact mean-score tie. The conservative `max_fits` check reserves every fold path plus the longest possible final refit before fitting begins. Groups and sample weights are routed through the metadata-aware pipeline stage. Classification variants currently accept label-response scorers; probability-response scoring needs a future importance-estimator response protocol. Inputs are finite dense matrices; sequential selection, sparse input, and artifact/cache codecs remain planned work.

Every new estimator runs through pipelines, cross-validation, scoring, and grid search, and every metric and solver is checked against committed scikit-learn reference fixtures. Dense univariate and model-based selectors check scores or coefficient importances, thresholds, selected indices, and transformed matrices against `sklearn.feature_selection`; recursive elimination checks elimination rankings and final-estimator importances against `sklearn.feature_selection.RFE`, while its cross-validated variant additionally checks every fold score, mean, standard deviation, and selected width against `sklearn.feature_selection.RFECV`; learning-curve training sizes and scores are checked against `sklearn.model_selection.learning_curve`; fold-local scaled ridge validation-curve scores are checked against `sklearn.model_selection.validation_curve`; grouped permutation scores and corrected p-values are checked against `sklearn.model_selection.permutation_test_score`. The comparative benchmarks under `dev/benchmarks/` are development evidence only; they record convergence parity across data shapes together with a throughput gap on wide designs that later releases will address.

Source checkouts additionally support target-aware pipelines, dense column transformation, feature unions, nested preprocessing chains, expanded splitters, out-of-fold prediction, leakage-safe learning and validation curves, cross-validated permutation significance tests and recursive feature elimination, resumable randomized and successive-halving search, and a runnable nested-CV recipe. Learning-curve schedules accept absolute counts or fractions of the smallest base training fold and optionally shuffle nested prefixes deterministically. Validation curves preserve caller-typed values while applying an immutable setter and pipeline builder, evaluate every value on one shared split, and report per-fold plus aggregate train/test scores without selecting or refitting a winner. Permutation tests evaluate one higher-is-better scorer on shared folds, shuffle targets globally or strictly within dataset groups, and report the observed score, ordered null scores, and corrected upper-tail p-value. Curves, permutation tests, and cross-validated recursive elimination can reject an excessive fit plan before fitting. The [nested-CV example](examples/nested_cv.ml) keeps every inner search inside its corresponding outer training fold before evaluating the selected model on untouched outer rows, then demonstrates a separately reserved final holdout. Planned for later versions: sparse feature input to estimators, artifact codecs for the estimators added since 0.3.2, tree and ensemble models, and accelerated numerical backends. The artifact format remains experimental during 0.x, with a committed golden reader for each released schema.

## Development

ModelKit requires OCaml 5.2 or newer. The platform locks currently use OCaml 5.3.0. The following set of commands will assume that you have installed and configured `git` and `opam`. The generated documentation will be available at `_build/default/_doc/_html/index.html`.

The repository holds four packages. `modelkit` and `modelkit-parallel` are portable. `modelkit-nx` and `modelkit-talon` depend on Raven's `nx` and `talon`, which need OpenBLAS headers, zlib, and pkg-config on Linux and are not buildable on Windows; opam installs those system packages through its depext prompt when the adapter dependencies are resolved. A workspace-wide `dune build @all` includes the adapter libraries and their tests, so it needs `nx` and `talon` in the switch. Use `--only-packages modelkit,modelkit-parallel` to build and test the portable packages on a switch without them.

### Initial Setup (Linux and macOS)

```commandline
opam update
opam switch create . 5.3.0 --deps-only --with-test --with-doc  # If running for the first time.
opam install ocamlformat.0.29.0

opam exec -- dune build @all @runtest @doc @fmt @opam @install --auto-promote
opam lint modelkit.opam
opam lint modelkit-parallel.opam
opam lint modelkit-nx.opam
opam lint modelkit-talon.opam
```

### Windows

Create the switch without installing anything, then install and build only the portable packages:

```commandline
opam update
opam switch create . 5.3.0 --no-install  # If running for the first time.
opam install ocamlformat.0.29.0
opam install ./modelkit.opam ./modelkit-parallel.opam --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64

opam exec -- dune build --only-packages modelkit,modelkit-parallel @all @runtest @doc @fmt @opam @install --auto-promote
opam lint modelkit.opam
opam lint modelkit-parallel.opam
```

To refresh the Windows lockfiles:

```commandline
opam lock ./modelkit.opam ./modelkit-parallel.opam --lock-suffix=locked.windows-x86_64
```

The Raven adapter packages declare themselves unavailable on Windows in their opam metadata and are not locked, installed, or built there; see `adapters/README.md`.

### macOS (arm64)

```sh
opam lock ./modelkit.opam ./modelkit-parallel.opam ./modelkit-nx.opam ./modelkit-talon.opam --lock-suffix=locked.macos-arm64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.macos-arm64
```

The four opam files must be locked together so that the in-tree `modelkit` dependency of the optional packages resolves.

The ordinary Dune workspace uses the repository-local opam switch automatically. Reproducible locks are platform-specific because compiler and system dependency packages differ by host.

The full test suite combines named unit tests, deterministic generated properties, metamorphic invariants, executable documentation, runnable end-to-end and nested-CV examples, artifact golden-reader and adversarial-input tests, a compile-time public API consumer, public estimator/transformer/scorer conformance reports with deliberately invalid external examples, a reusable numerical-backend conformance suite, and a source-neutral adapter conformance suite shared by every adapter package. Run the current supervised workflow from a source checkout with `opam exec -- dune exec examples/evaluation.exe`, or run the complete model-selection recipe with `opam exec -- dune exec examples/nested_cv.exe`.

GitHub Actions is configured to run the build, complete test suite, package build, and documentation generation on Linux x86-64, macOS arm64, and Windows x86-64 with OCaml 5.2, 5.3, and 5.5. The Linux and macOS jobs build and test all four packages; the Windows jobs build and test only the portable `modelkit` and `modelkit-parallel` packages because the Raven adapters cannot be built there at the current pin. These jobs use committed reference data and do not install or execute Python.

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

The committed smoke benchmark validates the measurement workflow only. The development preprocessing, transform-cache, dense-linear-model, regularized-linear, SGD-regression, SGD-classification, ridge-classifier, multinomial-logistic, generalized-linear-model, splitter, metrics, sequential and bounded-parallel cross-validation, finite grid-search, adapter-admission, sparse-kernel, and solver-shape benchmarks compare ModelKit operations with pinned scikit-learn and SciPy references on deterministic workloads. Build the corresponding OCaml worker and select a scenario under `dev/benchmarks/scenarios/`; the parallel cross-validation scenario records sequential and four-worker results for both runtimes so speedup, efficiency, wall time, and peak RSS can be compared. These reports are explicitly ineligible to support performance claims. See [the benchmark methodology](dev/benchmarks/README.md) for declared parity tolerances, scope, raw-result links, and limitations. Any published comparison will first be reproduced on independent CI targets.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
