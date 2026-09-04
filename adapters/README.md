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

The adapter pins Nx `1.0.0~alpha3` while Raven remains alpha.

## Talon

`modelkit-talon` admits explicitly selected Talon dataframe columns by name and role. It reads each column's typed tensor directly and never uses Talon's `to_nx` convenience conversion, which casts numeric columns to float32 and folds nulls into NaN. Column selection order becomes the feature order, and column names become the feature names. Every role in a dataset must name a distinct column.

| Input | Required column type | Null and numeric policy | ModelKit payload allocation |
| --- | --- | --- | --- |
| Features | float64, one or more named columns | Column null masks merged into one explicit mask and written as NaN; unmasked NaN retained; unmasked infinity rejected | One retained float64 matrix; no full-size staging payload |
| Feature null mask | Derived from the selected columns' Talon masks | Returned only when at least one selected column carries a mask; `true` remains distinguishable from a genuine NaN | One retained OCaml Boolean payload; no full-size staging payload |
| Regression target | float64 column | Nulls rejected; NaN and infinity rejected | One retained float64 vector; no full-size staging payload |
| Classification target | int64 column | Nulls rejected; every label must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained target payload |
| Sample weights | float64 column | Nulls rejected; finite, non-negative, and not all zero | One retained float64 vector; no full-size staging payload |
| Groups | int64 column | Nulls rejected; every group must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained group payload |
| Feature names | The selected column names, in selection order | Non-empty and unique, which Talon already guarantees for column names | Name-array allocation is outside numeric payload reports |

A column with the wrong type is a validation error that names the column and its actual type; the remediation is an explicit cast in Talon before admission. Feature reports carry no single-source contiguity flag because a dataframe stores its columns as separate tensors, while target, weight, and group reports record the contiguity of the column tensor they read. `Modelkit_talon.features`, `regression_dataset`, and `classification_dataset` return the same record shapes as the Nx adapter.

The adapter pins Talon `1.0.0~alpha3` alongside Nx. Consolidated cross-adapter conformance, allocation benchmarks, compatibility guidance, and platform lockfiles are still pending.
