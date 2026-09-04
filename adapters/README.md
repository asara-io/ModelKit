# Optional adapters

This directory contains optional packages that convert ecosystem data into ModelKit's portable representations. Adapters depend inward on `modelkit`; the portable package never depends on an adapter.

## Shared contract

Every adapter returns the records defined by `Modelkit.Admission`: a `conversion` pairs one admitted value with its `Conversion_report`, `features` carries the matrix, schema, optional explicit null mask, and feature reports, and `dataset` carries a complete `Modelkit.Dataset.t` with the feature null mask and every report produced during admission. `Admission.retained_payload_bytes`, `temporary_payload_bytes`, and `allocated_payload_bytes` total a report list.

The shared semantics are:

- Admitted values always own immutable storage. No adapter shares mutable source memory, so the current zero-copy criterion is that no adapter qualifies; every report states the retained copy explicitly.
- Features are float64 and are read in logical row-major order. Explicit source nulls are written as NaN in the matrix and recorded in the returned `Null_mask.t`; unmasked NaN remains data; unmasked infinity is rejected with a `Data` error.
- Regression targets and sample weights are float64 and must be finite; weights must be non-negative with at least one positive value. Classification targets and groups are int64 and every value must fit OCaml `int` on the current platform, otherwise a `Validation` error names the offending value.
- Feature names are non-empty, unique, and ordered exactly as the admitted columns.
- Payload accounting counts numeric payload only. Float payloads copy directly into their retained storage with no full-size staging; integer payloads pass through one OCaml-int staging array before the retained target or group storage, and that staging array is reported as temporary payload.

These semantics are enforced by the source-neutral conformance suite in `test/adapter_conformance.ml`. Each adapter test executable instantiates `Adapter_conformance.Make` with a module that builds the adapter's own source values from plain OCaml arrays, so the same eight conformance cases run against every adapter alongside its source-specific tests.

## Nx

`modelkit-nx` admits explicitly typed Nx tensors without sharing Nx's mutable storage. The adapter reads through each tensor's flat buffer using its offset and element strides, so both contiguous tensors and strided views are read in logical order without materialization. A view whose strides are not computable is materialized first and that full-size copy is reported as temporary payload.

| Input | Required type and shape | Null and numeric policy | ModelKit payload allocation |
| --- | --- | --- | --- |
| Features | float64, rank 2 | Optional same-shape Boolean mask; unmasked NaN retained; unmasked infinity rejected | One retained float64 matrix; no full-size staging payload |
| Feature null mask | bool, rank 2 matching features | `true` remains distinguishable from a genuine NaN; an all-false mask is still returned | One retained OCaml Boolean payload; no full-size staging payload |
| Regression target | float64, rank 1 | NaN and infinity rejected | One retained float64 vector; no full-size staging payload |
| Classification target | int64, rank 1 | Every label must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained target payload |
| Sample weights | float64, rank 1 | Finite, non-negative, and not all zero | One retained float64 vector; no full-size staging payload |
| Groups | int64, rank 1 | Every group must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained group payload |
| Feature names | OCaml string array matching feature width | Non-empty, unique, ordered exactly as columns | Name-array allocation is outside numeric payload reports |

`Modelkit_nx.features` returns the matrix, schema, optional explicit mask, and per-source reports. `regression_dataset` and `classification_dataset` additionally validate row alignment and return a complete `Modelkit.Dataset.t`. Feature nulls are represented as NaN in that dataset under `Dataset.Allow_nan`, so an imputer or an estimator that declares missing-value support is still required before fitting.

## Talon

`modelkit-talon` admits explicitly selected Talon dataframe columns by name and role. It reads each column's typed tensor directly through its buffer and never uses Talon's `to_nx` convenience conversion, which casts numeric columns to float32 and folds nulls into NaN. Column selection order becomes the feature order, and column names become the feature names. Every role in a dataset must name a distinct column.

| Input | Required column type | Null and numeric policy | ModelKit payload allocation |
| --- | --- | --- | --- |
| Features | float64, one or more named columns | Column null masks merged into one explicit mask and written as NaN; unmasked NaN retained; unmasked infinity rejected | One retained float64 matrix; no full-size staging payload |
| Feature null mask | Derived from the selected columns' Talon masks | Returned only when at least one selected column carries a mask; `true` remains distinguishable from a genuine NaN | One retained OCaml Boolean payload; no full-size staging payload |
| Regression target | float64 column | Nulls rejected; NaN and infinity rejected | One retained float64 vector; no full-size staging payload |
| Classification target | int64 column | Nulls rejected; every label must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained target payload |
| Sample weights | float64 column | Nulls rejected; finite, non-negative, and not all zero | One retained float64 vector; no full-size staging payload |
| Groups | int64 column | Nulls rejected; every group must fit OCaml `int` on the current platform | One OCaml-int staging payload and one retained group payload |
| Feature names | The selected column names, in selection order | Non-empty and unique, which Talon already guarantees for column names | Name-array allocation is outside numeric payload reports |

A column with the wrong type is a validation error that names the column and its actual type; the remediation is an explicit cast in Talon before admission. Feature reports carry no single-source contiguity flag because a dataframe stores its columns as separate tensors, while target, weight, and group reports record the contiguity of the column tensor they read. Talon normalizes away a nullable column's mask when it contains no nulls, so unlike the Nx adapter a Talon feature set without any null returns no mask even when the source columns were built with the nullable constructors.

## Compatibility

| Package | Raven pin | OCaml | Platforms verified in CI | Lockfiles |
| --- | --- | --- | --- | --- |
| `modelkit-nx` | `nx = 1.0.0~alpha3` | 5.2, 5.3, 5.5 | Linux x86-64, macOS arm64 | `modelkit-nx.opam.locked.macos-arm64` |
| `modelkit-talon` | `talon = 1.0.0~alpha3`, `nx = 1.0.0~alpha3` | 5.2, 5.3, 5.5 | Linux x86-64, macOS arm64 | `modelkit-talon.opam.locked.macos-arm64` |

Windows x86-64 is not supported for either adapter at this pin. Nx 1.0.0~alpha3 requires OpenBLAS headers and a C++ toolchain during its build, and the opam environment on the Windows CI runners provides neither, so the Windows jobs verify only the portable `modelkit` and `modelkit-parallel` packages and no Windows lockfiles exist for the adapters. Both adapters are pinned to exact Raven alpha releases because the Raven API is still changing; the pins move together with a conformance run, never independently.

The copy and allocation cost of both adapters is measured by the `adapter_admission_dense_v1` development benchmark described in `dev/benchmarks/README.md`. Owl adapters, zero-copy admission criteria for sources that can guarantee immutability, and broader compatibility matrices remain scheduled for the later adapter milestone.
