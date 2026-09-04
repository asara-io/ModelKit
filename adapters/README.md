# Optional adapters

This directory contains optional packages that convert ecosystem data into ModelKit's portable representations. Adapters depend inward on `modelkit`; the portable package never depends on an adapter.

## Nx

`modelkit-nx` admits explicitly typed Nx tensors without sharing Nx's mutable storage. The adapter reads logical indices, so both contiguous tensors and strided views have the same result. Numeric payload accounting excludes OCaml headers, allocator metadata, feature-name storage, and small result records.

| Input | Required type and shape | Null and numeric policy | ModelKit payload allocation |
| --- | --- | --- | --- |
| Features | float64, rank 2 | Optional same-shape Boolean mask; unmasked NaN retained; unmasked infinity rejected | One retained float64 matrix; no full-size staging payload |
| Feature null mask | bool, rank 2 matching features | `true` remains distinguishable from a genuine NaN | One retained OCaml Boolean payload; no full-size staging payload |
| Regression target | float64, rank 1 | NaN and infinity rejected | One retained float64 vector; no full-size staging payload |
| Classification target | int64, rank 1 | Every label must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained target payload |
| Sample weights | float64, rank 1 | Finite, non-negative, and not all zero | One retained float64 vector; no full-size staging payload |
| Groups | int64, rank 1 | Every group must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained group payload |
| Feature names | OCaml string array matching feature width | Non-empty, unique, ordered exactly as columns | Name-array allocation is outside numeric payload reports |

`Modelkit_nx.features` returns the matrix, schema, optional explicit mask, and per-source reports. `regression_dataset` and `classification_dataset` additionally validate row alignment and return a complete `Modelkit.Dataset.t`. Feature nulls are represented as NaN in that dataset under `Dataset.Allow_nan`, so an imputer or an estimator that declares missing-value support is still required before fitting.

The adapter pins Nx `1.0.0~alpha3` while Raven remains alpha. A Talon adapter and consolidated cross-adapter conformance, allocation benchmarks, compatibility guidance, and platform lockfiles are still pending.
