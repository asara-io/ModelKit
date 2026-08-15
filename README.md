# ModelKit

[![CI](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml/badge.svg)](https://github.com/asara-io/ModelKit/actions/workflows/ci.yml)

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Feature Highlights

- Reproducible foundations with deterministic random streams and stable reference numerical operations across supported platforms, OCaml versions, and execution schedules.
- Typed extension contracts separate immutable estimator specifications from fitted models and return actionable errors.
- Immutable, validated float64 data primitives catch shape, feature-order, and sample-alignment problems before model code runs.
- Dense datasets admit aligned features, targets, weights, groups, and names under an explicit finiteness policy; stable schema fingerprints and copy/view reports make compatibility and allocation behavior observable.
- Immutable preprocessing specifications fit mean, median, or constant imputation, population standardization, and variance-based feature filtering without changing or losing feature identities.
- Sequential pipelines fit preprocessing only on their training input, preserve schemas through ordered stages, and dispatch prediction, decision, and probability operations through an explicitly capable terminal estimator.
- Portable weighted ordinary least squares, ridge regression, and binary logistic regression keep immutable specifications separate from fitted coefficients and solver diagnostics.
- Deterministic K-fold, stratified K-fold, group K-fold, and expanding-window time-series splitters produce validated row views that can be explicitly materialized as aligned datasets.
- Weighted regression and binary classification metrics provide immutable higher-is-better scorers, plotting-neutral residual, ROC, and precision–recall data, stable score aggregation, and an explicit undefined-result policy.
- Portable sequential cross-validation fits pipelines within deterministic folds and reports ordered train/test scores, CPU timings, optional fitted models and indices, and typed failures.

## Motivation and Future Work

The library is built with a strong focus first on correctness, portability, and reproducibility; performance is a secondary goal to follow.

To this end, you will note that there is a significant amount of from-scratch implementation under ModelKit's hood. When implementation milestones are hit for being useful in real-world data science workflows, ModelKit will undergo benchmarking to gauge its performance against alternate implementations, such as `scikit-learn` itself.

Anticipating performance benefits from existing work such as using Owl for a numerical engine and Lacaml for acceleration, integration tasks will likely be brought above the line. In that phase, users who have come to be familiar with the consistent contracts of ModelKit's public APIs will enjoy performance benefits without contract changes.

## Status

ModelKit 0.2.1 is the current foundation release and adds OCaml 5.5 build support. The `0.3.x` development branch now includes immutable dense dataset admission, explicit `Require_finite` and `Allow_nan` feature policies, aligned zero-copy row views, stable versioned schema fingerprints, and explicit copy/view reporting. `Allow_nan` treats NaN as a missing-value marker but still rejects positive and negative infinity.

Dataset row views preserve ordering and duplicates without packing feature or metadata buffers. Use `Dataset.materialize` when an algorithm requires contiguous selected rows; its access report identifies the resulting copies.

The development API also provides `Simple_imputer`, `Standard_scaler`, and `Variance_threshold`. Imputation learns only from the supplied training matrix and treats NaN as the missing-value marker. Scaling uses population variance and maps constant centered features to zero with a scale of one. Variance filtering keeps columns whose variance is strictly greater than its threshold and preserves selected names in input order. These transformers reject infinities with typed errors rather than silently continuing.

`Pipeline` now packages these unsupervised transformers with any implementation of ModelKit's public `ESTIMATOR` protocol. Fitting learns every preprocessing stage exclusively from the supplied training matrix, then fits the terminal estimator on the transformed training output. The fitted pipeline reuses those exact stage values for `transform`, `predict`, `decision_function`, and `predict_proba`; unavailable terminal capabilities and named-stage failures are typed errors. Feature schemas are checked at the pipeline boundary and propagated after every transformation. Fixed root RNG state produces stage-local streams derived from stable logical names and positions.

`Linear_regression` fits weighted ordinary least squares with column-pivoted Householder QR and reports numerical rank, including for rank-deficient input. `Ridge_regression` solves an augmented least-squares system without forming normal equations and applies its non-negative `alpha` penalty only to coefficients. Both expose fitted coefficients, intercepts, and direct-solver reports.

`Logistic_regression` supports exactly two integer classes, optional sample weights, an L2 coefficient penalty controlled by positive inverse strength `c`, and stable decision/probability calculations for extreme logits. Its deterministic damped Newton fit reports objective, iteration count, and stopping reason; iteration exhaustion and invalid training data are typed errors. All three estimators accept only finite feature values, so missing values must be handled by an imputer or before fitting.

Built-in logistic regression can be installed as a pipeline terminal with its optional capabilities:

```ocaml
let logistic = Logistic_regression.create () |> Result.get_ok

let terminal =
  Pipeline.estimator ~name:"logistic"
    (module Logistic_regression)
    ~decision_function:Logistic_regression.decision_function
    ~predict_proba:Logistic_regression.predict_proba
    ~classes:Logistic_regression.classes logistic
  |> Result.get_ok

let specification =
  Pipeline.set_estimator Pipeline.empty terminal |> Result.get_ok
```

In the current pipeline contract, sample weights route to the terminal estimator and are not passed to the current unsupervised transformers. General transformer metadata routing remains planned for a later milestone. Fitted artifacts and the finished end-to-end workflow are not yet implemented.

`K_fold`, `Stratified_k_fold`, `Group_k_fold`, and `Time_series_split` now provide the portable splitting primitives needed by evaluation workflows. K-fold variants balance test sizes; stratification balances each integer class; group splitting prevents a group from crossing train/test boundaries; and time-series splitting uses expanding chronological training prefixes with optional gaps. Shuffled variants use ModelKit’s immutable deterministic RNG and retain source-row order in emitted views.

`Split.create` and `Split.of_views` validate non-empty, unique, disjoint train/test selections over one source. `Split.materialize` is the explicit allocation boundary that copies those selections into independent datasets while retaining targets, feature names, sample weights, groups, and schema identity.

`Regression_metrics` provides weighted MAE, MSE, RMSE, R², and residual data. `Binary_classification_metrics` provides weighted accuracy, balanced accuracy, precision, recall, F1, log loss, ROC AUC, and deterministic ROC and precision–recall curve arrays. Binary probabilities are validated as finite values in `[0, 1]`, and binary label metrics accept a configurable positive label.

Undefined metrics are observable through `Undefined_metric_policy`: the default returns a typed error, while callers may explicitly request NaN or documented finite fallbacks. `Regression_scorer` and `Binary_classification_scorer` make every selection score higher-is-better by negating loss metrics, and `Score_aggregation` reports stable population summaries for fold scores. Finite grid search is not yet implemented.

`Cross_validation.Regression.cross_validate` and `Cross_validation.Binary_classification.cross_validate` run a complete evaluation sequentially in stable fold order. Splitters are adapted explicitly as target-independent or target-aware, each fold derives its RNG from the root seed and logical fold index, and preprocessing is fitted only after the training partition is materialized. Reports contain process CPU fit/score timings, multiple scorers in caller order, optional train scores, fitted models, and original row indices. The default `Abort` failure policy returns the first typed failure; `Record` retains structured prediction and scoring failures and continues evaluating later folds.

Binary probability scorers require the pipeline terminal to declare its probability-column class order with `Pipeline.estimator ~classes`; ModelKit does not assume that a particular matrix column represents the configured positive label. Cross-validation currently uses the always-available sequential implementation. Bounded Domainslib fold execution and finite grid search remain planned work.

The portable package lives under `lib/`. Optional ecosystem adapters and accelerated backends are reserved under `adapters/` and `backends/`; they will remain separate packages that depend on the portable core when implemented.

## Development

ModelKit requires OCaml 5.2 or newer. The platform locks currently use OCaml 5.3.0. The following set of commands will assume that you have installed and configured `git` and `opam`. The generated documentation will be available at `_build/default/_doc/_html/index.html`.

### Initial Setup

```commandline
opam update
opam switch create . 5.3.0 --deps-only --with-test --with-doc  # If running for the first time.
opam install ocamlformat.0.29.0

opam exec -- dune build @all @runtest @doc @fmt @opam @install --auto-promote
opam lint modelkit.opam
```

### Windows

```commandline
opam lock ./modelkit.opam --lock-suffix=locked.windows-x86_64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64
```

### macOS (arm64)

```sh
opam lock ./modelkit.opam --lock-suffix=locked.macos-arm64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.macos-arm64
```

The ordinary Dune workspace uses the repository-local opam switch automatically. Reproducible locks are platform-specific because compiler and system dependency packages differ by host.

The full test suite combines named unit tests, deterministic generated properties, metamorphic invariants, executable documentation, a compile-time public API consumer, and a reusable numerical-backend conformance suite.

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

The committed smoke benchmark validates the measurement workflow only. The development preprocessing, dense-linear-model, splitter, and metrics benchmarks compare portable ModelKit operations with pinned scikit-learn references on deterministic workloads. Build the corresponding OCaml worker and select `dev/benchmarks/scenarios/preprocessing_dense.json`, `dev/benchmarks/scenarios/linear_models_dense.json`, `dev/benchmarks/scenarios/splitters_dense.json`, or `dev/benchmarks/scenarios/metrics_dense.json`. These reports are explicitly ineligible to support performance claims. See [the benchmark methodology](dev/benchmarks/README.md) for scope, raw-result links, and limitations. Release comparisons will use the product plan's independent-CI benchmark contract.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
