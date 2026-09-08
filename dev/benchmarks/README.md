# ModelKit development benchmarks

These benchmarks collect reproducible development evidence. They do not run as
part of a normal build or test, and every scenario states whether it may support
a public performance claim.

## Dense preprocessing v1

`preprocessing_dense_v1` applies mean, median, and constant imputation,
population standardization, and variance-threshold selection to the same
deterministic 50,000 by 40 float64 matrix in ModelKit and scikit-learn. Five
percent of non-constant feature values are NaN. Both workers are sequential,
and their output signatures must agree within `1e-12` absolute and relative
tolerance before a report is written.

The harness performs one warmup and three interleaved measured runs. Each run
uses a fresh process; elapsed time therefore includes runtime startup, data
generation, fitting, and transformation. Peak RSS is sampled every millisecond.
The ModelKit worker also reports OCaml heap allocation words. Python allocation
words are unavailable, so peak RSS is the cross-runtime memory comparison.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 1.108 s | 106,364,928 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.990 s | 241,467,392 bytes |

This scenario is `claim_eligible: false`. It is a development regression record,
not the release benchmark workload, and it has not run on independent Linux
x86-64 and arm64 CI targets. No comparative statement in product documentation
may be based on this report.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/preprocessing_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/preprocessing_dense.json
```

The raw report is
`results/preprocessing_dense_v1.darwin-arm64.json`. The scenario, toolchain
versions, raw measurements, thread configuration, checksums, output signatures,
and methodology are embedded in that file.

## Dense linear models v1

`linear_models_dense_v1` fits weighted ordinary least squares, weighted ridge,
and weighted binary logistic regression, then predicts on the same deterministic
10,000 by 12 float64 matrix in ModelKit and scikit-learn. ModelKit uses its
portable column-pivoted Householder QR and damped Newton solvers; scikit-learn
uses its default OLS implementation, SVD ridge solver, and Newton-Cholesky
logistic solver. Both workers are sequential. Selected regression predictions,
binary probabilities near the decision boundary, and predicted classes must
agree within `1e-7` absolute and relative tolerance before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes, so timings include runtime startup, deterministic data generation,
all three fits, and predictions. Peak RSS is sampled every millisecond, and the
ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 0.286 s | 12,271,616 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.737 s | 131,121,152 bytes |

This scenario is `claim_eligible: false`. It combines three algorithms and
includes process startup and data generation, so it does not isolate solver
throughput. It is local development evidence for correctness, deterministic
output, allocations, and gross regressions—not support for a comparative
performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/linear_models_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/linear_models_dense.json
```

The raw report is
`results/linear_models_dense_v1.darwin-arm64.json`; it records every raw run,
toolchain versions, thread limits, output signatures, allocations, and the full
scenario.

## Dense regularized linear models v1

`regularized_linear_dense_v1` fits weighted lasso and elastic-net regressors
and three-point descending paths on the same deterministic 3,000 by 12 float64
matrix in ModelKit and scikit-learn. Both implementations use cyclic coordinate
descent and warm-start each path from its preceding stronger penalty. Selected
coefficients, intercepts, and predictions must agree within `1e-6` absolute and
relative tolerance before a report is written.

The harness performs one warmup and five interleaved measured runs in fresh
processes, so timings include runtime startup, deterministic data generation,
two ordinary fits, two path fits, and prediction. Peak RSS is sampled every
millisecond, and the ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 0.045 s | 7,897,088 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.728 s | 126,124,032 bytes |

This scenario is `claim_eligible: false`. It includes process startup and data
generation, uses a small fixed path, and has not run on independent CI targets.
It records parity, deterministic output, allocations, and gross regressions,
not support for a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/regularized_linear_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/regularized_linear_dense.json
```

The raw report is
`results/regularized_linear_dense_v1.darwin-arm64.json`; it records every raw
run, toolchain versions, thread limits, output signatures, allocations, and the
full scenario.

## Dense SGD regression v1

`sgd_regression_dense_v1` runs both a 20-epoch weighted squared-error fit and
the equivalent chain of 20 `partial_fit` calls on the same deterministic 5,000
by 16 float64 matrix in ModelKit and scikit-learn. Both use input row order, no
penalty, a constant learning rate, and unaveraged parameters. Selected update
counts, coefficients, intercepts, and predictions must agree within `1e-7`
absolute and relative tolerance before a report is written.

The harness performs one warmup and five interleaved measured runs in fresh
processes. Timings include runtime startup, deterministic data generation,
both training workflows, and prediction. Peak RSS is sampled every millisecond,
and the ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 0.261 s | 8,142,848 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.738 s | 126,418,944 bytes |

The ModelKit worker allocated 94,345,776 OCaml words in each measured run.
This scenario is `claim_eligible: false`: it includes process startup and data
generation, tests one aligned learning configuration, and has not run on the
independent CI targets required for a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/sgd_regressor_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/sgd_regression_dense.json
```

The raw report is `results/sgd_regression_dense_v1.darwin-arm64.json`; it
records every raw run, toolchain versions, thread limits, output signatures,
allocations, and the full scenario.

## Dense SGD classification v1

`sgd_classification_dense_v1` trains three weighted stochastic-gradient
workflows on the same deterministic 5,000 by 16 float64 matrix in ModelKit and
scikit-learn: a 20-epoch binary hinge fit with decision scores and predictions,
a 20-epoch three-class log-loss fit with one-versus-rest probabilities and
predictions, and the equivalent chain of 20 three-class log-loss `partial_fit`
calls with the classes registered up front. All use input row order, no
penalty, a constant learning rate, and unaveraged parameters. Selected update
counts, coefficients, intercepts, decision scores, probabilities, and
predictions must agree within `1e-7` absolute and relative tolerance before a
report is written.

The harness performs one warmup and five interleaved measured runs in fresh
processes. Timings include runtime startup, deterministic data generation, all
three training workflows, and inference. Peak RSS is sampled every
millisecond, and the ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 0.896 s | 11,976,704 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.777 s | 127,516,672 bytes |

The ModelKit worker allocated 333,347,875 OCaml words in each measured run.
This scenario is `claim_eligible: false`: it includes process startup and data
generation, tests one aligned learning configuration per loss, and has not run
on the independent CI targets required for a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/sgd_classifier_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/sgd_classification_dense.json
```

The raw report is `results/sgd_classification_dense_v1.darwin-arm64.json`; it
records every raw run, toolchain versions, thread limits, output signatures,
allocations, and the full scenario.

## Dense ridge classification v1

`ridge_classifier_dense_v1` fits weighted binary and three-class ridge
classifiers and predicts scores and classes on the same deterministic 10,000 by
12 float64 matrix in ModelKit and scikit-learn. ModelKit uses one portable
column-pivoted Householder QR ridge solve per class; scikit-learn uses its SVD
solver. Both workers are sequential. Selected binary and multiclass decision
scores and predictions must agree within `1e-7` absolute and relative tolerance
before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes, so timings include runtime startup, deterministic data generation,
both fits, and prediction. Peak RSS is sampled every millisecond, and the
ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 0.125 s | 16,187,392 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.731 s | 132,202,496 bytes |

This scenario is `claim_eligible: false`. It combines binary and multiclass
fits and includes process startup and data generation, so it does not isolate
solver throughput. It is local development evidence for parity, deterministic
output, allocations, and gross regressions, not support for a comparative
performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/ridge_classifier_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/ridge_classifier_dense.json
```

The raw report is
`results/ridge_classifier_dense_v1.darwin-arm64.json`; it records every raw
run, toolchain versions, thread limits, output signatures, allocations, and the
full scenario.

## Dense multinomial logistic regression v1

`multinomial_logistic_dense_v1` fits a weighted three-class logistic model and
computes decision scores, softmax probabilities, and predictions on the same
deterministic 10,000 by 12 float64 matrix in ModelKit and scikit-learn. ModelKit
uses its portable sum-to-zero damped Newton solver with QR linear solves;
scikit-learn uses its Newton-Cholesky solver. Both workers are sequential.
Selected scores, probabilities, and classes must agree within `1e-7` absolute
and relative tolerance before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes, so timings include runtime startup, deterministic data generation,
fitting, and inference. Peak RSS is sampled every millisecond, and the ModelKit
worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 1.057 s | 9,650,176 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.749 s | 129,236,992 bytes |

This scenario is `claim_eligible: false`. It includes process startup and data
generation, compares different Newton linear-system solvers, and has not run on
independent CI targets. It records parity, deterministic output, allocations,
and gross regressions, not support for a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/multinomial_logistic_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/multinomial_logistic_dense.json
```

The raw report is
`results/multinomial_logistic_dense_v1.darwin-arm64.json`; it records every raw
run, toolchain versions, thread limits, output signatures, allocations, and the
full scenario.

## Dense generalized linear models v1

`glm_dense_v1` fits and predicts with weighted Poisson and power-1.5 Tweedie
regressors on the same deterministic 3,000 by 12 float64 matrix in ModelKit and
scikit-learn. ModelKit uses its portable damped IRLS solver with QR linear
solves; scikit-learn uses L-BFGS. Both workers are sequential. Selected
coefficients, intercepts, and predictions must agree within `1e-6` absolute and
relative tolerance before a report is written.

The harness performs one warmup and five interleaved measured runs in fresh
processes, so timings include runtime startup, deterministic data generation,
both fits, and both prediction passes. Peak RSS is sampled every millisecond,
and the ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 0.069 s | 6,389,760 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.723 s | 125,009,920 bytes |

This scenario is `claim_eligible: false`. It includes process startup and data
generation, compares different solvers, and has not run on independent CI
targets. It records parity, deterministic output, allocations, and gross
regressions, not support for a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/glm_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/glm_dense.json
```

The raw report is `results/glm_dense_v1.darwin-arm64.json`; it records every
raw run, toolchain versions, thread limits, output signatures, allocations, and
the full scenario.

## Dense splitters v1

`splitters_dense_v1` generates five folds from the same deterministic 100,000
row input using K-fold, stratified K-fold, group K-fold, and expanding-window
time-series splitting in ModelKit and scikit-learn. Both workers are sequential.
Fold counts, total train/test membership, and minimum/maximum test sizes must
agree exactly before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Timings include runtime startup, input and metadata allocation, and
all four splitter calls. Peak RSS is sampled every millisecond, and the ModelKit
worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 0.057 s | 38,633,472 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.708 s | 136,773,632 bytes |

This scenario is `claim_eligible: false`. It combines four splitters and
includes process startup and input generation. It records correctness,
determinism, allocation, and gross regression evidence only; it cannot support
a comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/splitters_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/splitters_dense.json
```

The raw report is
`results/splitters_dense_v1.darwin-arm64.json`; it includes every raw run,
toolchain versions, the exact signature, thread limits, allocations, and the
full scenario.

## Dense metrics v1

`metrics_dense_v1` evaluates weighted regression and binary classification
metrics, full ROC and precision-recall curves, and scalar score aggregation on
100,000 deterministic samples in ModelKit and scikit-learn. Both workers are
sequential. Eleven scalar metrics, curve lengths, and aggregation results must
agree within `1e-7` absolute and relative tolerance before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Timings include runtime startup, deterministic input allocation, all
metrics, both curve sorts, and aggregation. Peak RSS is sampled every
millisecond, and the ModelKit worker reports OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 0.157 s | 49,283,072 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.738 s | 136,904,704 bytes |

This scenario is `claim_eligible: false`. It combines multiple metric families,
process startup, data generation, and sorting-based curves. It records parity,
allocation, and gross regression evidence only and cannot support a comparative
performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/metrics_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/metrics_dense.json
```

The raw report is
`results/metrics_dense_v1.darwin-arm64.json`; it records every raw run,
toolchain versions, thread limits, output signatures, allocations, and the full
scenario.

## Dense cross-validation v1

`cross_validation_dense_v1` runs five deterministic stratified folds over a
20,000 by 20 binary dataset with deterministic missing values. Both runtimes
fit mean imputation, population standardization, and logistic regression within
each training fold, then compute accuracy, balanced accuracy, negative log
loss, and ROC AUC for the train and test partitions. Fitted models and original
row indices are also retained.

All 40 train/test scores, 20 index count/sum values, and five model-retention
markers must agree within `1e-7` absolute and relative tolerance before a report
is written. ModelKit also rejects negative per-fold timings. Both workers are
sequential and every numerical thread pool is limited to one thread. ModelKit
uses `Cross_validation.Binary_classification.cross_validate`; scikit-learn uses
`sklearn.model_selection.cross_validate` with equivalent options.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Timings therefore include runtime startup, deterministic data
generation, all fold-local preprocessing and fitting, scoring, and report
construction. Peak RSS is sampled every millisecond. ModelKit also reports
cumulative OCaml heap allocation words, which are allocation traffic rather
than retained memory and exclude Bigarray storage.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS | OCaml allocation words |
| --- | ---: | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 4.243 s | 62,914,560 bytes | 750,193,782 |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.974 s | 170,606,592 bytes | unavailable |

This scenario is `claim_eligible: false`. It establishes correctness,
determinism, allocation, and sequential performance-regression evidence for the
current implementation. It does not exercise Domainslib, the release-scale
million-row workload, eight-core scaling, independent CI, confidence intervals,
or the release benchmark's wall-time and RSS gates. No comparative product
claim may be based on this report.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/cross_validation_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/cross_validation_dense.json
```

The raw report is
`results/cross_validation_dense_v1.darwin-arm64.json`; it records every raw run,
toolchain versions, thread limits, per-fold output signatures, allocations, and
the full scenario.

## Bounded parallel cross-validation v1

`parallel_cross_validation_dense_v1` runs eight deterministic stratified folds
over the same 20,000 by 20 binary workflow as the sequential cross-validation
scenario. It records four implementations: ModelKit and scikit-learn each with
one fold worker and with four fold workers. Every inner numerical thread pool
is limited to one thread. ModelKit uses a bounded Domainslib pool whose domain
count includes the caller; scikit-learn uses its pinned version's default
joblib process backend for `n_jobs=4`.

All 104 scores, index statistics, and model-retention markers from every
implementation must agree with sequential scikit-learn within `1e-7` absolute
and relative tolerance before a report is written. The harness also requires a
stable checksum across all measured runs and verifies the configured inner
thread limit. ModelKit reports its effective fold-domain count, estimated
runnable threads, and diagnostic-warning count.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Timings therefore include runtime and worker startup, deterministic
data generation, preprocessing, fitting, scoring, and report construction.
Peak RSS includes recursive child processes and is sampled every millisecond.
OCaml allocation words are omitted because `Gc.allocated_bytes` does not
provide the aggregate cross-domain allocation traffic needed for a comparable
parallel measurement.

The committed macOS arm64 report recorded these medians:

| Implementation | Fold workers | Wall time | Peak RSS |
| --- | ---: | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 1 | 7.522 s | 75,071,488 bytes |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 4 | 2.045 s | 103,366,656 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 1 | 1.109 s | 175,210,496 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 4 | 2.008 s | 849,149,952 bytes |

For these local measurements, ModelKit's four-domain path achieved 3.68x
speedup over its sequential path, or 91.9% four-worker efficiency. Its parallel
median was 1.8% longer than the parallel scikit-learn median. Scikit-learn's
parallel median was 1.81x longer than its sequential median because process
startup outweighed fold-level speedup at this workload size. These observations
characterize this scenario only; notably, Domainslib threads and joblib
processes have different startup and memory behavior.

This scenario is `claim_eligible: false`. It is a local development record, not
the release-scale workload, and it lacks independent Linux x86-64 and arm64 CI,
confidence intervals, sustained worker-pool measurements, and the release
benchmark's wall-time and RSS gates. It cannot support a comparative product
claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/parallel_cross_validation_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/parallel_cross_validation_dense.json
```

The raw report is
`results/parallel_cross_validation_dense_v1.darwin-arm64.json`; it records all
raw timings, peak RSS samples, toolchain versions, thread limits, output
signatures, execution diagnostics, and the full scenario.

## Dense finite grid search v1

`grid_search_dense_v1` evaluates six ridge-regression configurations formed by
the Cartesian product of three regularization strengths and two intercept
choices. Each candidate receives the same five deterministic K-fold splits over
a 10,000 by 12 float64 regression dataset. Both runtimes aggregate train and
test negative mean-squared error and R², rank candidates by mean test R², and
refit the winner on the complete dataset.

All parameter encodings, 24 aggregate scores, six ranks, the selected candidate
index, and two predictions from the refitted model must agree within `1e-7`
absolute and relative tolerance before a report is written. ModelKit uses
`Grid_search.Regression.search`; scikit-learn uses `GridSearchCV` with its SVD
ridge solver. Both workers and numerical thread pools are sequential.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Timings include runtime startup, deterministic data generation, 30
fold fits, scoring, candidate-report construction, and the full-data refit. Peak
RSS is sampled every millisecond. ModelKit also reports cumulative OCaml heap
allocation words, which exclude Bigarray storage and measure allocation traffic
rather than retained memory.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS | OCaml allocation words |
| --- | ---: | ---: | ---: |
| ModelKit 0.3.0-dev / OCaml 5.3.0 | 0.546 s | 25,690,112 bytes | 98,941,046 |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.807 s | 134,299,648 bytes | unavailable |

This scenario is `claim_eligible: false`. It is local correctness, determinism,
allocation, and gross performance-regression evidence. It does not isolate
search orchestration from ridge fitting, exercise failed candidates or parallel
execution, run the release-scale workload, or satisfy the independent-CI and
confidence-interval requirements for a comparative product claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/grid_search_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/grid_search_dense.json
```

The raw report is
`results/grid_search_dense_v1.darwin-arm64.json`; it records every raw run,
toolchain versions, thread limits, candidate output signatures, allocations,
and the full scenario.

## Dense grouped permutation test v1

`permutation_test_dense_v1` evaluates a fold-local standard-scaling and ridge
pipeline on one observed regression target and 20 within-group permutations
over a deterministic 2,000 by 10 float64 dataset. Five shared K-fold partitions
produce 105 complete pipeline fits. Each two-row group has one constant target,
so shuffling within groups leaves the target unchanged and lets the harness
compare all observed and null scores plus the corrected p-value across runtimes
without requiring ModelKit to reproduce NumPy's random stream.

The output signatures must agree within `1e-7` absolute and relative tolerance
before a report is written. The harness performs one warmup and three interleaved
measured runs in fresh sequential processes. Timings include runtime startup,
deterministic data generation, target permutations, all fits and scores, and
report construction. Peak RSS is sampled every millisecond; ModelKit also
reports cumulative OCaml heap allocation words.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS | OCaml allocation words |
| --- | ---: | ---: | ---: |
| ModelKit 0.5.0-dev / OCaml 5.3.0 | 0.455 s | 13,615,104 bytes | 80,898,368 |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.926 s | 128,925,696 bytes | unavailable |

This scenario is `claim_eligible: false`. Its invariant grouped targets isolate
orchestration and repeated fitting but do not represent a useful significance
test, and it has not run on independent CI targets. It provides parity,
allocation, and gross regression evidence rather than support for a comparative
product claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/permutation_test_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/permutation_test_dense.json
```

The raw report is
`results/permutation_test_dense_v1.darwin-arm64.json`; it records every raw
run, toolchain versions, thread limits, output signatures, allocations, and the
full scenario.

## Adapter admission v1

`adapter_admission_dense_v1` measures the copy and allocation cost of moving a
deterministic 100,000 by 40 float64 feature matrix with a five-percent null
mask, an int64 classification target, float64 sample weights, and int64 groups
into a complete ModelKit dataset through both Raven adapters. The ModelKit
worker builds an Nx tensor set and an equivalent Talon dataframe from the same
generator, then times `Modelkit_nx.classification_dataset` and
`Modelkit_talon.classification_dataset` separately inside one process. The
scikit-learn worker validates the same data through `check_array` and
`_check_sample_weight` from a NumPy array and from a column-stacked dictionary
of NumPy columns. Row and column counts, the NaN-skipping feature sum, the null
count, and the label, weight, and group sums must agree within `1e-7` absolute
and relative tolerance for both adapters before a report is written.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Process wall time includes runtime startup and source construction,
which for ModelKit means allocating the Nx tensors and Talon columns in OCaml.
The per-adapter admission timings and allocations inside the ModelKit worker
isolate the adapter cost. Each adapter also reports the payload bytes recorded
by its `Conversion_report` values, so the allocation ratio compares the OCaml
heap allocated during admission with the numeric payload ModelKit retains.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 1.168 s | 283,869,184 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.748 s | 255,672,320 bytes |

| Adapter | Admission time | Allocated words | Retained payload | Staging payload | Allocated bytes per retained byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| `modelkit-nx` | 0.247 s | 47,521,815 | 66,400,000 bytes | 1,600,000 bytes | 5.73 |
| `modelkit-talon` | 0.228 s | 39,526,023 | 66,400,000 bytes | 1,600,000 bytes | 4.76 |

The retained payload is the 32,000,000-byte feature matrix, the
32,000,000-byte Boolean null mask, and 800,000 bytes each for the target,
weights, and groups. The staging payload is the OCaml-int array each adapter
fills before constructing the target and groups. The remaining allocation is
boxed floats crossing the `Matrix.init` and `Vector.init` closures plus the
per-row arrays of the null mask; it is proportional to the payload and is not
a copy of the source.

This scenario replaced an earlier implementation that read every element
through `Nx.item` with a freshly allocated index list. On the same machine that
version needed 7.4 s and 5,106,271,511 words for the Nx admission and 2.7 s and
1,977,084,649 words for Talon, more than 600 and 230 allocated bytes per
retained byte. Both adapters now read through the tensor's flat buffer with its
offset and element strides, which also preserves logical-order reads of strided
views without materializing them.

This scenario is `claim_eligible: false`: it includes process startup and
source construction, compares against validation utilities rather than a
complete scikit-learn workflow, and has not run on the independent CI targets
required for a comparative performance claim. The worker is unavailable on
Windows, where the Raven adapters are unsupported.

Build and run it from the repository root:

```sh
MODELKIT_ADAPTER_BENCH=1 opam exec -- dune build bench/ocaml/adapter_admission_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/adapter_admission_dense.json
```

The worker is opted in through `MODELKIT_ADAPTER_BENCH` because it depends on
the optional Raven adapters, which are not installed on Windows or on switches
that build only the portable packages.

The raw report is `results/adapter_admission_dense_v1.darwin-arm64.json`; it
records every raw run, per-adapter timings and allocations, payload accounting,
toolchain versions, thread limits, output signatures, and the full scenario.

## Sparse kernels v1

`sparse_kernels_v1` measures ModelKit's dense-versus-CSR kernel dispatch,
one-hot output formats, and row-view materialization against SciPy sparse and
NumPy. A deterministic 50,000 by 200 float64 matrix is generated at one, five,
and twenty-five percent density. For each density both workers time twenty
repetitions of the matrix-vector product and the transposed product through
CSR storage and through the dense equivalent. The same workers then encode a
20,000 by 10 categorical matrix with fifty categories per feature into dense
and CSR one-hot output, and materialize every other row of the five-percent
matrix through a row view. Stored-entry counts, product sums, one-hot output
shapes and column-weighted sums, and the materialized row count, stored-entry
count, and value sum must agree within `1e-7` absolute and relative tolerance
before a report is written. ModelKit's Neumaier-compensated reductions and
NumPy's pairwise summation differ in the last bits, which the tolerance
absorbs.

The harness performs one warmup and three interleaved measured runs in fresh
processes. Process wall time includes runtime startup and generating the three
matrices, which dominates for ModelKit because every density is built through
checked CSR admission and then densified. The per-kernel timings inside the
report isolate the operations of interest; the ModelKit worker also reports
OCaml heap allocation words per operation.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 8.734 s | 293,568,512 bytes |
| SciPy 1.18.0 / NumPy 2.5.2 / Python 3.14.3 | 1.578 s | 623,738,880 bytes |

Twenty repetitions of each kernel:

| Density | Stored entries | ModelKit CSR product | ModelKit dense product | SciPy CSR product | NumPy dense product | ModelKit CSR transposed | ModelKit dense transposed | SciPy CSR transposed | NumPy dense transposed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 % | 100,000 | 15.6 ms | 1,057 ms | 1.7 ms | 56 ms | 13.4 ms | 1,346 ms | 1.9 ms | 116 ms |
| 5 % | 500,000 | 70.8 ms | 1,078 ms | 5.1 ms | 54 ms | 46.0 ms | 1,349 ms | 6.1 ms | 116 ms |
| 25 % | 2,500,000 | 329.3 ms | 1,133 ms | 29.4 ms | 54 ms | 185.3 ms | 1,367 ms | 29.2 ms | 116 ms |

CSR storage held 2,000,008, 8,400,008, and 40,400,008 bytes against an
80,000,000-byte dense equivalent, so CSR dispatch pays off in both time and
memory below roughly half density, and the crossover for the transposed
product is later still because the dense transposed kernel walks columns
against row-major storage. One-hot CSR output held 3,360,008 bytes and took
70 ms against 80,000,000 bytes and 172 ms for the dense output, allocating
13,872,100 words against 24,344,075. Materializing 25,000 selected rows of the
five-percent matrix took 4.9 ms, allocated 200,000 bytes of row indices in the
view, and copied 4,200,008 payload bytes out of 8,400,008 shared bytes;
SciPy's fancy row indexing took 0.7 ms.

ModelKit's kernels are roughly ten to twenty times slower than SciPy and NumPy
at every density. The remaining gap is the cost of the compensated,
non-finite-tracking reduction that ModelKit performs on every product, run as
portable OCaml without SIMD or a BLAS call, against SciPy's C loops and
NumPy's BLAS-backed dense product. The relative ordering, which is the
solver-selection evidence this scenario exists for, is the same in both
runtimes: CSR wins by the density ratio, and the dense product never wins
below full density.

This scenario replaced an earlier implementation of the reference kernels
that folded through a heap-allocated accumulator record and boxed every
float crossing the `Vector.get`, `Matrix.get`, and `Csr_matrix.iter_row`
closures. On the same machine that version needed 48.1 s of wall time and
13,325,295,082 allocated words for this worker; the dense product took over
five seconds and the CSR product 71 ms at one-percent density. The kernels
now read the immutable Bigarray storage directly and keep the compensated
sum in unboxed locals, which brought the worker to 155,271,903 words. A
property test checks that every kernel still matches an independently written
Neumaier fold bit for bit, including NaN, infinity, and overflow inputs.

This scenario is `claim_eligible: false`: it includes process startup and
data generation, measures kernels rather than an end-to-end workflow, and has
not run on the independent CI targets required for a comparative performance
claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/sparse_kernels_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/sparse_kernels.json
```

The raw report is `results/sparse_kernels_v1.darwin-arm64.json`; it records
every raw run, per-kernel timings and allocations, CSR and dense memory
accounting, toolchain versions, thread limits, output signatures, and the full
scenario.

## Solver shapes v1

`solver_shapes_v1` fits weighted ordinary least squares, weighted ridge,
weighted binary logistic regression, and weighted multinomial logistic
regression on four deterministic float64 designs in ModelKit and scikit-learn:
a tall 40,000 by 8 matrix, a square 2,000 by 200 matrix, a wide 300 by 400
matrix with more features than samples, and a rank-deficient 5,000 by 24
matrix whose last eight columns duplicate its first eight. ModelKit uses its
portable column-pivoted Householder QR for the direct fits and for every
damped Newton step; scikit-learn uses its LAPACK least-squares OLS, its
shape-selected `auto` ridge solver, and Newton-Cholesky logistic solvers with
the same regularization, tolerance, and iteration limit. Training-set
predictions for the direct fits, the reported numerical rank of the OLS
design, and selected binary and multiclass probabilities must agree within
`1e-7` absolute and relative tolerance for every shape before a report is
written. Predictions rather than coefficients are compared because the wide
and rank-deficient least-squares problems have many exact solutions; both
libraries report rank 299 for the wide design and rank 16 for the
rank-deficient design.

The harness performs one warmup and three interleaved measured runs in fresh
processes; the per-fit timings, Newton iteration counts, convergence flags,
and reported ranks inside the report isolate each solver. The ModelKit worker
also reports OCaml heap allocation words per fit.

The committed macOS arm64 report recorded these medians:

| Implementation | Wall time | Peak RSS |
| --- | ---: | ---: |
| ModelKit 0.4.0-dev / OCaml 5.3.0 | 58.945 s | 76,103,680 bytes |
| scikit-learn 1.9.0 / Python 3.14.3 | 1.288 s | 191,447,040 bytes |

Per-fit medians, with Newton iteration counts in parentheses:

| Shape | ModelKit OLS | scikit-learn OLS | ModelKit ridge | scikit-learn ridge | ModelKit logistic | scikit-learn logistic | ModelKit multinomial | scikit-learn multinomial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tall 40,000 × 8 | 36 ms | 5 ms | 36 ms | 2 ms | 511 ms (11) | 20 ms (11) | 2,262 ms (12) | 68 ms (11) |
| square 2,000 × 200 | 255 ms | 14 ms | 280 ms | 3 ms | 4,808 ms (9) | 15 ms (8) | 24,369 ms (9) | 105 ms (9) |
| wide 300 × 400 | 71 ms | 11 ms | 273 ms | 2 ms | 3,596 ms (8) | 23 ms (8) | 20,781 ms (8) | 235 ms (8) |
| rank-deficient 5,000 × 24 | 17 ms | 2 ms | 19 ms | 1 ms | 288 ms (10) | 4 ms (10) | 1,182 ms (10) | 15 ms (10) |

Convergence behaviour is the same in both libraries. Every Newton fit
converged on every shape within at most one iteration of scikit-learn's count,
and both direct solvers detected the rank deficiency. The scale behaviour is
not. ModelKit's direct fits are five to one hundred times slower than LAPACK,
and its Newton fits are twenty-five to three hundred times slower, with the
gap widest on the square and wide shapes where each iteration assembles and
factors a 200- or 400-dimensional Hessian, or a 600- or 1,200-dimensional
joint Hessian for the multinomial fit. The allocation counts explain most of
it: the square multinomial fit allocated 3,338,053,496 words over nine
iterations, about one word per floating-point operation, because the Hessian
assembly and the QR solve work on boxed `float array array` rows read through
`Matrix.get`. This is the same defect class that the adapter-admission and
sparse-kernel scenarios exposed and that the reference kernels have since been
rewritten to avoid. It is recorded here as the standing performance debt of
the 0.4.0 linear estimators: correctness and convergence parity are proven
across shapes, throughput on wider designs is not competitive, and both the
unboxed rewrite of the Newton solvers and the deferred Lacaml backend are
the candidate remedies.

This scenario is `claim_eligible: false`: it includes process startup and
data generation, compares single fits rather than an end-to-end workflow, and
has not run on the independent CI targets required for a comparative
performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/solver_shapes_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/solver_shapes.json
```

The raw report is `results/solver_shapes_v1.darwin-arm64.json`; it records
every raw run, per-fit timings, iteration counts, convergence flags, ranks,
allocations, toolchain versions, thread limits, output signatures, and the
full scenario.

## Dense transform cache v1

`transform_cache_dense_v1` performs one cold and three warm fits of a
standard-scaler and ridge pipeline on the same deterministic 20,000 by 20
float64 regression matrix. ModelKit uses an explicitly scoped in-memory
fitted-state cache. scikit-learn uses `Pipeline(memory=...)` with a fresh
joblib directory per worker process. Both terminal ridge estimators are
refitted on every pass. Selected predictions must agree within `1e-7` absolute
and relative tolerance before a report is written.

The harness performs one warmup and five interleaved measured runs in fresh
processes. Worker wall time includes runtime startup, data generation, one cold
fit, and three warm fits. The workers also report isolated cold and mean warm
fit CPU/wall timings; ModelKit reports process CPU time while Python reports
monotonic elapsed time, so cold-to-warm ratios are meaningful within each
runtime but the isolated timings are not a controlled cross-runtime clock
comparison.

The committed macOS arm64 report recorded these medians:

| Implementation | Worker wall time | Peak RSS | Cold fit | Warm fit | Cold / warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| ModelKit 0.5.0-dev / OCaml 5.3.0 | 0.788 s | 60,719,104 bytes | 0.197 s | 0.182 s | 1.08× |
| scikit-learn 1.9.0 / Python 3.14.3 | 0.808 s | 159,760,384 bytes | 0.024 s | 0.015 s | 1.58× |

The result confirms deterministic reuse and lower observed process memory, but
also shows the current scope clearly. A ModelKit hit avoids scaler fitting while
still hashing the complete matrix, decoding fitted state, transforming the
training matrix, and refitting ridge. scikit-learn's joblib cache retains the
fitted transformer's transformed training output, so its warm-path reduction is
larger. ModelKit's portable ridge kernel also dominates its isolated fit time.
These measurements establish a regression baseline; they do not justify a
comparative performance claim.

Build and run it from the repository root:

```sh
opam exec -- dune build bench/ocaml/transform_cache_worker.exe
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/transform_cache_dense.json
```

The raw report is
`results/transform_cache_dense_v1.darwin-arm64.json`; it records all raw runs,
cache hit/miss counts, internal cold and warm timings, allocations, toolchain
versions, thread limits, output signatures, and methodology. This scenario is
`claim_eligible: false` and has not run on the independent CI targets required
by the release benchmark contract.

## Benchmark record for 0.4.0

| Scenario | Evidence | Parity tolerance | ModelKit versus reference |
| --- | --- | ---: | --- |
| `preprocessing_dense_v1` | imputation, scaling, variance filtering | 1e-12 | comparable wall time, lower RSS |
| `linear_models_dense_v1` | OLS, ridge, logistic on 10,000 × 12 | 1e-7 | faster wall time, lower RSS |
| `regularized_linear_dense_v1` | lasso, elastic net, paths | 1e-6 | see section |
| `ridge_classifier_dense_v1` | binary and multiclass ridge classification | 1e-7 | see section |
| `multinomial_logistic_dense_v1` | multinomial fit on 10,000 × 12 | 1e-7 | see section |
| `glm_dense_v1` | Poisson and Tweedie IRLS | 1e-6 | see section |
| `sgd_regression_dense_v1`, `sgd_classification_dense_v1` | SGD fits and `partial_fit` chains | 1e-7 | comparable wall time, lower RSS |
| `splitters_dense_v1`, `metrics_dense_v1` | split materialization, scoring | exact / 1e-7 | see sections |
| `cross_validation_dense_v1`, `parallel_cross_validation_dense_v1`, `grid_search_dense_v1` | evaluation workflows | 1e-7 | see sections |
| `adapter_admission_dense_v1` | Nx and Talon copy and allocation | 1e-7 | about five allocated bytes per retained byte |
| `sparse_kernels_v1` | CSR versus dense dispatch, one-hot output, materialization | 1e-7 | same dispatch ordering, ten to twenty times slower per kernel |
| `solver_shapes_v1` | direct and Newton solvers across tall, wide, square, and rank-deficient shapes | 1e-7 | convergence parity, five to three hundred times slower on wider designs |

Every scenario is `claim_eligible: false`. None has run on independent CI
targets, none isolates a workload that has been designated for a release
claim, and the wider-shape solver gap above rules out any comparative
throughput statement for 0.4.0.
