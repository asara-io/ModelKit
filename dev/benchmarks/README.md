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
