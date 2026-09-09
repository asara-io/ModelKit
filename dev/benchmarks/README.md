# ModelKit development benchmarks

These benchmarks collect reproducible development evidence. They do not run as
part of a normal build or test, and every scenario states whether it may support
a public performance claim.

## 0.5.0 checkpoint

The committed macOS arm64 reports were refreshed together on 8 September 2026
for the 0.5.0 development checkpoint. They use:

- ModelKit 0.5.0 and OCaml 5.3.0;
- Python 3.14.3;
- scikit-learn 1.9.0, NumPy 2.5.2, and SciPy 1.18.0; and
- an eight-core arm64 host running macOS 15.5.

Unless a scenario says otherwise, the harness performs one warmup followed by
three or five interleaved measured runs. Every measurement launches a fresh
worker process. Wall time therefore includes runtime startup, deterministic
data generation, fitting or transformation, and result production. Peak
resident set size (RSS) is sampled every millisecond. ModelKit workers also
record OCaml heap allocation where applicable.

All correctness signatures, deterministic-output checks, thread-limit checks,
and RSS checks passed before the reports below were written. Wall time and RSS
are medians of the measured runs.

| Scenario | ModelKit wall | Reference wall | ModelKit RSS | Reference RSS |
| --- | ---: | ---: | ---: | ---: |
| Adapter admission | 1.176 s | 0.737 s | 266.55 MiB | 243.97 MiB |
| Cross-validation | 4.614 s | 1.039 s | 56.22 MiB | 164.11 MiB |
| Generalized linear models | 0.070 s | 0.764 s | 6.75 MiB | 122.02 MiB |
| Grid search | 0.559 s | 0.862 s | 26.64 MiB | 132.89 MiB |
| Linear models | 0.305 s | 0.768 s | 12.73 MiB | 128.16 MiB |
| Metrics | 0.152 s | 0.760 s | 43.88 MiB | 132.59 MiB |
| Multinomial logistic | 1.022 s | 0.763 s | 10.22 MiB | 124.17 MiB |
| Permutation test | 0.458 s | 0.936 s | 12.70 MiB | 123.16 MiB |
| Preprocessing | 1.148 s | 0.993 s | 103.94 MiB | 230.08 MiB |
| Regularized linear | 0.026 s | 0.740 s | 8.06 MiB | 120.88 MiB |
| Ridge classification | 0.121 s | 0.757 s | 14.73 MiB | 129.39 MiB |
| SGD classification | 0.787 s | 0.809 s | 11.97 MiB | 121.94 MiB |
| SGD regression | 0.232 s | 0.774 s | 8.73 MiB | 121.77 MiB |
| Solver shapes | 59.172 s | 1.649 s | 88.70 MiB | 193.80 MiB |
| Sparse kernels | 8.796 s | 1.603 s | 276.81 MiB | 590.78 MiB |
| Splitters | 0.060 s | 0.717 s | 40.14 MiB | 133.41 MiB |
| Transform cache | 0.794 s | 0.825 s | 58.62 MiB | 161.31 MiB |

The parallel cross-validation scenario records both sequential and four-worker
execution because its purpose is backend scaling rather than a single
cross-runtime result:

| Implementation | Wall time | Peak RSS | Speedup from sequential |
| --- | ---: | ---: | ---: |
| ModelKit sequential | 7.996 s | 72.81 MiB | 1.00x |
| ModelKit four-worker | 2.310 s | 99.92 MiB | 3.46x |
| scikit-learn sequential | 1.129 s | 156.86 MiB | 1.00x |
| scikit-learn four-worker | 2.065 s | 819.66 MiB | 0.55x |

The small reference-only smoke scenario completed in 0.722 s with 120.34 MiB
peak RSS. It validates the Python measurement path and is not a ModelKit
comparison.

These measurements are intentionally interpreted as regression evidence, not
as a product claim. ModelKit used less median RSS in all comparative scenarios
except adapter admission, but elapsed time varied significantly by workload.
Fresh-process startup dominates several short scenarios. Solver shapes, sparse
kernels, cross-validation, and multinomial logistic regression remain the most
visible throughput opportunities. The parallel backend substantially improves
ModelKit cross-validation elapsed time on this workload, with a corresponding
increase in RSS.

## Focused measurements

The aggregate worker measurements above are useful for regression detection,
but several scenarios also record operation-level timings. These medians expose
the intended performance boundary without treating runtime startup or data
generation as kernel work.

### Adapter admission

The adapter scenario admits the same 100,000 by 40 feature data, null mask,
target, weights, and groups through both optional Raven adapters. The retained
payload is 66,400,000 bytes and temporary staging is 1,600,000 bytes.

| Adapter | Admission time | Allocated words | Bytes per retained byte |
| --- | ---: | ---: | ---: |
| `modelkit-nx` | 0.247 s | 47,521,815 | 5.73 |
| `modelkit-talon` | 0.229 s | 39,526,023 | 4.76 |

The process-level Python reference validates equivalent NumPy sources with
scikit-learn utilities. It is useful for signature and gross resource checks,
but is not an equivalent complete adapter workflow.

### Sparse kernels

Each product measurement below covers twenty repetitions over a deterministic
50,000 by 200 matrix. Times are medians in milliseconds.

| Density | ModelKit CSR | ModelKit dense | SciPy CSR | NumPy dense |
| ---: | ---: | ---: | ---: | ---: |
| 1% | 16.010 | 1,063.866 | 1.713 | 55.338 |
| 5% | 68.717 | 1,094.283 | 5.169 | 54.755 |
| 25% | 324.809 | 1,141.242 | 29.452 | 54.206 |

Transposed products over the same inputs:

| Density | ModelKit CSR | ModelKit dense | SciPy CSR | NumPy dense |
| ---: | ---: | ---: | ---: | ---: |
| 1% | 13.295 | 1,340.586 | 1.922 | 116.253 |
| 5% | 45.990 | 1,352.252 | 6.108 | 116.567 |
| 25% | 185.663 | 1,364.532 | 28.793 | 117.110 |

ModelKit's CSR one-hot transformation took 70.635 ms versus 173.748 ms for its
dense output; the SciPy and NumPy references took 5.879 ms and 11.884 ms,
respectively. Materializing the selected CSR row view took 5.183 ms in ModelKit
and 0.774 ms in SciPy. CSR retains its expected storage and dispatch advantage
at low density, while the portable kernels remain a clear throughput target.

### Solver shapes

The solver scenario records fit-only medians over tall, square, wide, and
rank-deficient designs. Times are milliseconds; every checked fit converged and
the prediction signatures and reported numerical ranks passed parity checks.

| Shape | Fit | ModelKit | scikit-learn |
| --- | --- | ---: | ---: |
| Tall, 40,000 x 8 | OLS | 36.652 | 6.460 |
|  | Ridge | 35.795 | 2.564 |
|  | Binary logistic | 508.830 | 20.523 |
|  | Multinomial logistic | 2,272.824 | 68.930 |
| Square, 2,000 x 200 | OLS | 238.896 | 14.246 |
|  | Ridge | 264.854 | 2.411 |
|  | Binary logistic | 4,847.672 | 14.928 |
|  | Multinomial logistic | 24,577.642 | 107.352 |
| Wide, 300 x 400 | OLS | 69.253 | 10.706 |
|  | Ridge | 258.063 | 1.645 |
|  | Binary logistic | 3,599.026 | 23.543 |
|  | Multinomial logistic | 20,750.444 | 239.924 |
| Rank-deficient, 5,000 x 24 | OLS | 17.032 | 2.570 |
|  | Ridge | 19.639 | 0.858 |
|  | Binary logistic | 295.059 | 4.266 |
|  | Multinomial logistic | 1,201.114 | 14.630 |

The wider Newton systems dominate this scenario. These measurements continue
to motivate unboxed solver kernels and the planned optional accelerated linear
algebra backend.

### Transform cache

The transform-cache scenario performs one cold and three warm fits of a
standard-scaler and ridge pipeline. ModelKit caches fitted transformer state;
scikit-learn's joblib cache retains the transformed training output, so the
warm paths do not cache identical values.

| Implementation | Cold fit | Mean warm fit | Cold / warm |
| --- | ---: | ---: | ---: |
| ModelKit | 0.198 s | 0.184 s | 1.08x |
| scikit-learn | 0.030 s | 0.015 s | 1.97x |

Both workers produced identical results across the cold and warm paths within
the declared tolerance. The comparison records each cache contract's current
behavior rather than asserting equivalent cache semantics.

## Scenario coverage

| Scenario | Workload | Parity tolerance |
| --- | --- | ---: |
| `adapter_admission_dense_v1` | Nx and Talon dataset admission | 1e-7 |
| `cross_validation_dense_v1` | fold-local preprocessing and scoring | 1e-7 |
| `glm_dense_v1` | Poisson and Tweedie regression | 1e-6 |
| `grid_search_dense_v1` | finite parameter search | 1e-7 |
| `linear_models_dense_v1` | OLS, ridge, and binary logistic | 1e-7 |
| `metrics_dense_v1` | regression, classification, and curves | 1e-7 |
| `multinomial_logistic_dense_v1` | three-class logistic regression | 1e-7 |
| `parallel_cross_validation_dense_v1` | sequential and bounded parallel CV | 1e-7 |
| `permutation_test_dense_v1` | grouped permutation significance | 1e-7 |
| `preprocessing_dense_v1` | imputation, scaling, and selection | 1e-12 |
| `regularized_linear_dense_v1` | lasso, elastic net, and paths | 1e-6 |
| `ridge_classifier_dense_v1` | binary and multiclass classification | 1e-7 |
| `sgd_classification_dense_v1` | hinge and log-loss incremental fits | 1e-7 |
| `sgd_regression_dense_v1` | squared-error incremental fits | 1e-7 |
| `sklearn_dummy_cv_smoke_v1` | reference harness smoke test | exact checksum |
| `solver_shapes_v1` | solvers on tall, square, wide, and rank-deficient data | 1e-7 |
| `sparse_kernels_v1` | CSR dispatch, one-hot output, and row views | 1e-7 |
| `splitters_dense_v1` | K-fold, stratified, grouped, and time-series splits | exact |
| `transform_cache_dense_v1` | cold and cached pipeline fits | 1e-7 |

Each scenario uses deterministic generated input and compares a scenario-specific
signature before its report is written. The signatures cover outputs relevant
to the workload, including predictions, probabilities, scores, coefficients,
fold membership, convergence state, selected features, or sparse structure.
The complete scenario dimensions and checked signatures are embedded in the raw
reports.

## Reproducing the checkpoint

Create the pinned Python development environment as described in
`dev/python/README.md`, then build the portable benchmark workers:

```sh
opam exec -- dune build @all
```

The adapter admission worker is optional because it requires the Raven Nx and
Talon adapters. On a supported switch with those dependencies installed, build
it explicitly:

```sh
MODELKIT_ADAPTER_BENCH=1 opam exec -- \
  dune build bench/ocaml/adapter_admission_worker.exe
```

Run one scenario by passing its JSON definition:

```sh
env/bin/python dev/benchmarks/run.py \
  --scenario dev/benchmarks/scenarios/preprocessing_dense.json
```

Run every registered scenario on macOS or Linux after all workers are built:

```sh
for scenario_path in dev/benchmarks/scenarios/*.json; do
  env/bin/python dev/benchmarks/run.py --scenario "$scenario_path" || exit 1
done
```

The adapter worker and its scenario are unavailable on Windows. The portable
scenarios remain usable there. A report filename includes its operating-system
and architecture suffix so results from different targets do not overwrite one
another.

## Raw reports

The committed JSON reports contain the scenario definition, raw measurements,
environment and package versions, thread-pool observations, checksums, output
signatures, allocation data, and methodology:

- `results/adapter_admission_dense_v1.darwin-arm64.json`
- `results/cross_validation_dense_v1.darwin-arm64.json`
- `results/glm_dense_v1.darwin-arm64.json`
- `results/grid_search_dense_v1.darwin-arm64.json`
- `results/linear_models_dense_v1.darwin-arm64.json`
- `results/metrics_dense_v1.darwin-arm64.json`
- `results/multinomial_logistic_dense_v1.darwin-arm64.json`
- `results/parallel_cross_validation_dense_v1.darwin-arm64.json`
- `results/permutation_test_dense_v1.darwin-arm64.json`
- `results/preprocessing_dense_v1.darwin-arm64.json`
- `results/regularized_linear_dense_v1.darwin-arm64.json`
- `results/ridge_classifier_dense_v1.darwin-arm64.json`
- `results/sgd_classification_dense_v1.darwin-arm64.json`
- `results/sgd_regression_dense_v1.darwin-arm64.json`
- `results/sklearn_dummy_cv_smoke_v1.darwin-arm64.json`
- `results/solver_shapes_v1.darwin-arm64.json`
- `results/sparse_kernels_v1.darwin-arm64.json`
- `results/splitters_dense_v1.darwin-arm64.json`
- `results/transform_cache_dense_v1.darwin-arm64.json`

## Claim status

Every current scenario is `claim_eligible: false`. The reports have not been
reproduced on the independent Linux x86-64 and arm64 CI targets required by the
release benchmark contract, and the workloads have not been designated as
release-claim workloads. No comparative performance statement in product or
release documentation may be based on these local reports.
