# ModelKit

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Status

ModelKit 0.1.0 is a foundation release from the starting engineering milestone. Development after that release now includes the first public data contracts. Estimators and end-to-end machine learning workflows are not yet implemented.

The portable package lives under `lib/`. Optional ecosystem adapters and accelerated backends are reserved under `adapters/` and `backends/`; they will remain separate packages that depend on the portable core when implemented.

## Available Data Foundation

The `Modelkit` namespace provides opaque, immutable `Vector` and `Matrix` values backed by float64 C-layout Bigarrays; bounded immutable `Row_view` selections; regression and classification `Target` values; unique aligned `Feature_names`; validated `Sample_weight` values; aligned integer `Groups`; and structured `Data_error` failures.

Bigarrays and arrays are copied when admitted or exported so caller mutation cannot change accepted data. Matrix rows and row-index selections avoid copying feature buffers. Regression targets must be finite, sample weights must be finite and non-negative with at least one positive value, and feature names must be non-empty and unique.

## Development

ModelKit requires OCaml 5.2 or newer. Install development dependencies in an opam switch, then run the standard checks:

```console
opam install ocamlformat.0.29.0
opam install . --deps-only --with-test --with-doc
opam exec -- dune build @all @runtest @doc
opam exec -- dune build @fmt
```

`@fmt` reports formatting differences without changing source files. Apply a reviewed formatting change with `opam exec -- dune promote`.

The ordinary Dune workspace uses the active opam switch so compiler-matrix builds remain straightforward. Reproducible opam locks are platform-specific because compiler and system dependency packages differ by host. Refresh and consume the checked-in Windows x86-64 solution with:

```console
opam lock ./modelkit.opam --lock-suffix=locked.windows-x86_64
opam install ocamlformat.0.29.0
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64
opam exec -- dune build @all @runtest @doc @fmt
```

Each supported platform uses its own `modelkit.opam.locked.<platform>` file. Regenerating a lock is an intentional dependency update and its diff should be reviewed.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
