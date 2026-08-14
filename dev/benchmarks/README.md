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
