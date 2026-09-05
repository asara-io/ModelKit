# 0.4.1 (2026-09-04)

- Add checked immutable CSR matrices with canonical structure validation,
  zero-copy indexed row views, explicit materialization, payload-memory
  accounting, and dense/CSR numerical-kernel dispatch through
  `Feature_matrix`.
- Add min-max, max-absolute, and robust scalers, per-sample normalization,
  one-hot and ordinal encoders with explicit unknown-category policies and
  direct CSR one-hot output, reversible label encoding, polynomial features,
  and missing indicators as immutable transforms with feature-name
  propagation.
- Add lasso and elastic-net regression with deterministic weighted coordinate
  descent and warm-started descending regularization paths, binary and
  multiclass ridge classification, multinomial logistic regression with stable
  softmax probabilities and matrix-valued decision scores, and Poisson and
  Tweedie generalized linear models with target-domain validation and damped
  IRLS, each with solver diagnostics and parity fixtures.
- Add SGD regression and binary/multiclass SGD classification with hinge and
  logistic losses under an immutable incremental-training contract: explicit
  initial class registration, ordered streams, optional shuffling, checkpoints,
  and continuation equivalence.
- Finish sample-weight and class-weight propagation with fold-local balanced
  and explicit class weights for classifier terminals, opt-in sample-weight
  routing to transformers, and a weighted standard scaler.
- Add confusion-matrix and multiclass classification metrics with micro,
  macro, and weighted averaging, average precision, one-versus-rest and
  one-versus-one ROC AUC, top-k accuracy, DCG and NDCG ranking metrics, and
  multiclass scorers with multiclass cross-validation and grid search.
- Add the separately installable `modelkit-nx` and `modelkit-talon` adapter
  packages for checked admission of explicitly typed Nx tensors and explicitly
  selected Talon dataframe columns, with shared `Admission` result records,
  conversion and allocation reports, a source-neutral adapter conformance
  suite, and macOS arm64 lockfiles; the adapters build on Linux and macOS
  only at the pinned Raven `1.0.0~alpha3` release.
- Rewrite the reference numerical kernels to read immutable Bigarray storage
  directly with unboxed compensated sums, verified bit for bit against an
  independent Neumaier fold.
- Add deterministic comparative benchmark reports for regularized linear
  models, ridge classification, multinomial logistic regression, generalized
  linear models, SGD, adapter admission, sparse kernel dispatch, and solver
  convergence and scale across tall, square, wide, and rank-deficient shapes;
  every report remains development evidence with no performance claim.
- Defer the optional Lacaml numerical backend to a later release, where
  LAPACK factorizations first have a consumer.
- Artifacts written by this release record producer version 0.4.1; the
  artifact schema is unchanged and 0.3.x artifacts continue to load.

# 0.3.2 (2026-08-17)

- Add immutable dense dataset admission with explicit finiteness policies,
  aligned zero-copy row views, stable schema fingerprints, and observable
  copy/view reporting.
- Add immutable mean, median, and constant imputation, population
  standardization, and variance-based feature filtering while preserving
  feature identities.
- Add leakage-safe sequential pipelines that fit preprocessing only on training
  input, propagate schemas, derive stage-local random streams, and dispatch
  prediction, decision, and probability operations.
- Add portable weighted ordinary least squares, ridge regression, and binary
  logistic regression with stable numerical methods, fitted coefficients, and
  solver diagnostics.
- Add deterministic K-fold, stratified K-fold, group K-fold, and
  expanding-window time-series splitters with validated row views and explicit
  aligned dataset materialization.
- Add weighted regression and binary classification metrics, plotting-neutral
  residual, ROC, and precision-recall data, higher-is-better scorers, stable
  score aggregation, and explicit undefined-result policies.
- Add deterministic cross-validation with ordered train and test scores, CPU
  timings, optional fitted models and row indices, structured failures, and
  independent split and fit seeds.
- Add the separately installable `modelkit-parallel` package for bounded
  Domainslib fold execution with sequential fallback, stable logical results,
  and oversubscription diagnostics.
- Add typed finite grid search over immutable configurations with shared splits,
  failure-aware candidate reports, named-scorer ranking, deterministic tie
  handling, and best-model refitting on all training data.
- Add versioned data-only artifacts for fitted built-in regression and binary
  classification pipelines with feature-schema identity, bounded readers,
  accidental-corruption detection, file conveniences, and golden compatibility
  coverage.
- Add committed scikit-learn parity fixtures and deterministic comparative
  benchmark reports for preprocessing, linear models, splitters, metrics,
  cross-validation, parallel execution, and grid search without adding a Python
  runtime dependency or making performance claims.
- Expand executable documentation and the end-to-end evaluation example to cover
  the complete supervised workflow.
- Decompose the portable implementation into private cohesive source units while
  retaining the existing flat `Modelkit.*` public API and optional-package
  dependency boundaries.

# 0.2.1 (2026-08-11)

- Restore package builds on OCaml 5.5 by making typed target accessor specializations explicit.
- Test OCaml 5.5 on Linux, macOS, and Windows in addition to the existing OCaml 5.2 and 5.3 matrix.

# 0.2.0 (2026-08-10)

- Add immutable float64 vectors and matrices, row views, typed targets, feature schemas, sample weights, groups, and validation errors over portable Bigarray storage.
- Define common contracts for immutable specifications, fitted estimators, classifiers, regressors, transformers, scorers, splitters, execution, random-number generation, and numerical backends.
- Add deterministic seeds and random streams, stable sequential execution, and a portable reference backend with cancellation-safe reductions and matrix-vector kernels.
- Establish unit, property, metamorphic, mdx, public API compatibility, sklearn fixture, and reusable numerical-backend conformance tests.
- Add pinned development-only Python tooling and committed metadata for reproducible sklearn reference fixtures and comparative benchmark collection without introducing a runtime Python dependency.
- Verify build, tests, package installation targets, and odoc generation on Linux x86-64, macOS arm64, and Windows x86-64 with OCaml 5.2 and 5.3 in GitHub Actions.

# 0.1.0 (2026-08-04)

- Establish the initial ModelKit package with an OCaml 5.2 compiler floor and Apache-2.0 licensing.
- Add the portable library skeleton and reserved package boundaries for optional adapters and accelerated backends.
- Add the Dune workspace, generated odoc documentation, formatting checks, and platform-specific opam lock workflow.
- Document project governance, support, and maintenance policies.
