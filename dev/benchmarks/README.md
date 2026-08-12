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
