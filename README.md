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
- Portable numeric, categorical, target, interaction, and missingness transforms cover min-max, max-absolute, robust, per-sample normalization, one-hot, ordinal, label, polynomial, and missing-indicator workflows.
- Sequential pipelines support unsupervised and target-aware preprocessing fitted only on training rows, with explicit sample-weight routing, preserved feature schemas, and terminal prediction, decision, and probability dispatch.
- Dense column-wise preprocessing combines independently fitted branches with checked index/name selectors, passthrough/drop, deterministic output names, and observable copy allocations.
- Typed metadata requests route weights, groups, and callbacks through nested fitting, inference, cross-validation, and search, with aligned rows and deterministic progress reporting.
- Feature unions, column transformers, and nestable preprocessing chains combine supervised and unsupervised transformations with deterministic feature names and fold-local fitting.
- Transformed-target regression learns target mappings within each training fold, checks inverse transforms, and scores predictions in the original target space.
- Portable weighted ordinary least squares, ridge, lasso, elastic-net, binary and multinomial logistic regression, Poisson and Tweedie generalized linear models, binary and multiclass ridge classification, and incremental SGD estimators keep immutable specifications separate from fitted coefficients and solver diagnostics.
- Deterministic K-fold, stratified and repeated K-fold, shuffle and stratified-shuffle, holdout, group K-fold, and expanding-window time-series splitters produce validated row views; train/test helpers materialize aligned features, targets, weights, and groups.
- Weighted regression, binary, multiclass, and ranking metrics provide immutable higher-is-better scorers, plotting-neutral residual, ROC, and precision–recall data, stable score aggregation, and an explicit undefined-result policy.
- Cross-validation fits pipelines within deterministic folds and reports ordered train/test scores, CPU timings, optional fitted models and indices, and typed failures; the optional `modelkit-parallel` package adds bounded Domainslib fold execution.
- Typed finite grid search evaluates immutable pipeline configurations on shared deterministic splits, ranks candidates by a named scorer, records candidate failures, and refits the selected model on all training data.
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
- **More linear estimators.** Lasso and elastic-net regression with regularization paths, binary and multiclass ridge classification, multinomial logistic regression, Poisson and Tweedie generalized linear models, and SGD regression and classification with an explicit incremental-training and checkpoint contract. Every estimator exposes coefficients, intercepts, and solver diagnostics.
- **Weights and multiclass evaluation.** Fold-local class weights, opt-in sample-weight routing to transformers, confusion-matrix and multiclass metrics with micro, macro, and weighted averaging, average precision, one-versus-rest and one-versus-one ROC AUC, top-k accuracy, DCG and NDCG, and multiclass cross-validation and grid search.
- **Ecosystem adapters.** `modelkit-nx` and `modelkit-talon` admit explicitly typed features, targets, null masks, groups, names, and weights with conversion and allocation reports and a shared conformance suite. Both are pinned to Raven `1.0.0~alpha3` and build on Linux and macOS only.

Every new estimator runs through pipelines, cross-validation, scoring, and grid search, and every metric and solver is checked against committed scikit-learn reference fixtures. The comparative benchmarks under `dev/benchmarks/` are development evidence only; they record convergence parity across data shapes together with a throughput gap on wide designs that later releases will address.

Source checkouts additionally support target-aware pipelines, dense column transformation, feature unions, and nested preprocessing chains. Planned for later versions: further splitters and randomized search, sparse feature input to estimators, artifact codecs for the estimators added since 0.3.2, tree and ensemble models, and accelerated numerical backends. The artifact format remains experimental during 0.x, with a committed golden reader for each released schema.

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

The full test suite combines named unit tests, deterministic generated properties, metamorphic invariants, executable documentation, a compiled end-to-end example, artifact golden-reader and adversarial-input tests, a compile-time public API consumer, a reusable numerical-backend conformance suite, and a source-neutral adapter conformance suite shared by every adapter package. Run the current supervised workflow from a source checkout with `opam exec -- dune exec examples/evaluation.exe`.

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

The committed smoke benchmark validates the measurement workflow only. The development preprocessing, dense-linear-model, regularized-linear, SGD-regression, SGD-classification, ridge-classifier, multinomial-logistic, generalized-linear-model, splitter, metrics, sequential and bounded-parallel cross-validation, finite grid-search, adapter-admission, sparse-kernel, and solver-shape benchmarks compare ModelKit operations with pinned scikit-learn and SciPy references on deterministic workloads. Build the corresponding OCaml worker and select a scenario under `dev/benchmarks/scenarios/`; the parallel cross-validation scenario records sequential and four-worker results for both runtimes so speedup, efficiency, wall time, and peak RSS can be compared. These reports are explicitly ineligible to support performance claims. See [the benchmark methodology](dev/benchmarks/README.md) for declared parity tolerances, scope, raw-result links, and limitations. Any published comparison will first be reproduced on independent CI targets.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
