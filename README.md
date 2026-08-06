# ModelKit

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Status

ModelKit 0.1.0 is the initial foundation release. Development after that release now includes the first public data contracts and protocol module types. Concrete estimators and end-to-end machine learning workflows are not yet implemented.

The portable package lives under `lib/`. Optional ecosystem adapters and accelerated backends are reserved under `adapters/` and `backends/`; they will remain separate packages that depend on the portable core when implemented.

## Available Data Foundation

The `Modelkit` namespace provides opaque, immutable `Vector` and `Matrix` values backed by float64 C-layout Bigarrays; bounded immutable `Row_view` selections; regression and classification `Target` values; unique aligned `Feature_names`; validated `Sample_weight` values; aligned integer `Groups`; and structured `Data_error` failures.

Bigarrays and arrays are copied when admitted or exported so caller mutation cannot change accepted data. Matrix rows and row-index selections avoid copying feature buffers. Regression targets must be finite, sample weights must be finite and non-negative with at least one positive value, and feature names must be non-empty and unique.

## Available Protocol Foundation

Public `ESTIMATOR`, `CLASSIFIER`, `REGRESSOR`, `TRANSFORMER`, `SCORER`, and `SPLITTER` module types define the boundaries that future implementations and optional adapters will share. Estimator specifications and fitted values have separate abstract types, classifiers and regressors carry their target kind at compile time, and operations that can fail return implementation-defined typed errors.

The `EXECUTION`, `RNG`, and `NUMERICAL_BACKEND` module types establish substitutable boundaries for stable-order bounded work, functional random-number state with logically derived child seeds, and portable or accelerated numerical primitives. ModelKit does not yet provide concrete implementations of these protocols.

## Development

ModelKit requires OCaml 5.2 or newer. The platform locks currently use OCaml 5.3.0. The following set of commands will assume that you have installed and configured `git` and `opam`. The generated documentation will be available at `_build/default/_doc/_html/index.html`.

### Initial Setup

```commandline
opam update
opam switch create . 5.3.0 --deps-only --with-test --with-doc  # If running for the first time.
opam install ocamlformat.0.29.0

opam exec -- dune build @opam
opam exec -- dune promote
opam lint modelkit.opam
```

### Windows

```commandline
opam lock ./modelkit.opam --lock-suffix=locked.windows-x86_64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64

opam exec -- dune build @all @runtest @doc @fmt @opam @install
opam lint modelkit.opam
```

### macOS (arm64)

```sh
opam lock ./modelkit.opam --lock-suffix=locked.macos-arm64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.macos-arm64

opam exec -- dune build @all @runtest @doc @fmt @opam @install
opam lint modelkit.opam
```

The ordinary Dune workspace uses the repository-local opam switch automatically. Reproducible locks are platform-specific because compiler and system dependency packages differ by host.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
