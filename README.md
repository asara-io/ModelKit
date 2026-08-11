# ModelKit

ModelKit (`modelkit`) is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.

Python users of `scikit-learn` will find this library familiar in serving the same needs.

## Feature Highlights

- Reproducible by design, with stable numerical results and random behaviour across supported platforms, OCaml versions, and execution schedules.
- Model development with immutable configurations, distinct fitted states, and actionable errors.
- Immutable, validated tabular data types to catch shape, feature-order, and sample-alignment problems.
- 

## Motivation and Future Work

The library is built with a strong focus first on correctness, portability, and reproducibility; performance is a secondary goal to follow. 

To this end, you will note that there is a significant amount of from-scratch implementation under ModelKit's hood. When implementation milestones are hit for being useful in real-world data science workflows, ModelKit will undergo benchmarking to gauge its performance against alternate implementations, such as `scikit-learn` itself.

Anticipating performance benefits from existing work such as using Owl for a numerical engine and Lacaml for acceleration, integration tasks will likely be brought above the line. In that phase, users who have come to be familiar with the consistent contracts of ModelKit's public APIs will enjoy performance benefits without contract changes.

## Status

ModelKit 0.1.0 is the initial foundation release. Development after that release now includes public data contracts, feature schemas, structured errors, protocol module types, and portable reference implementations for foundational numerical and execution operations. Concrete estimators and end-to-end machine learning workflows are not yet implemented.

The portable package lives under `lib/`. Optional ecosystem adapters and accelerated backends are reserved under `adapters/` and `backends/`; they will remain separate packages that depend on the portable core when implemented.

## Development

ModelKit requires OCaml 5.2 or newer. The platform locks currently use OCaml 5.3.0. The following set of commands will assume that you have installed and configured `git` and `opam`. The generated documentation will be available at `_build/default/_doc/_html/index.html`.

### Initial Setup

```commandline
opam update
opam switch create . 5.3.0 --deps-only --with-test --with-doc  # If running for the first time.
opam install ocamlformat.0.29.0

opam exec -- dune build @all @runtest @doc @fmt @opam @install --auto-promote
opam lint modelkit.opam
```

### Windows

```commandline
opam lock ./modelkit.opam --lock-suffix=locked.windows-x86_64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.windows-x86_64
```

### macOS (arm64)

```sh
opam lock ./modelkit.opam --lock-suffix=locked.macos-arm64
opam install . --deps-only --with-test --with-doc --locked --lock-suffix=locked.macos-arm64
```

The ordinary Dune workspace uses the repository-local opam switch automatically. Reproducible locks are platform-specific because compiler and system dependency packages differ by host.

The full test suite combines named unit tests, deterministic generated properties, metamorphic invariants, executable documentation, a compile-time public API consumer, and a reusable numerical-backend conformance suite.

### Reference Fixtures and Benchmarks

Committed scikit-learn reference fixtures are ordinary test data, so the normal ModelKit build and test suite never require or execute Python. Maintainers only need the pinned development environment when regenerating those fixtures or collecting benchmark evidence. Python 3.14.3 is required, as recorded in `dev/python/PYTHON_VERSION`; the local virtual environment is stored in the ignored `env/` directory.

On Windows:

```commandline
env\Scripts\activate
python -m pip install --requirement dev\python\requirements.lock
python dev\fixtures\generate.py
python dev\benchmarks\run.py
```

On macOS/Linux:

```sh
source env/bin/activate
python -m pip install --requirement dev/python/requirements.lock
python dev/fixtures/generate.py
python dev/benchmarks/run.py
```

The committed smoke benchmark validates the measurement workflow only and is explicitly ineligible to support performance claims in the current build. Algorithm-specific comparative reports will use the release benchmark contract once equivalent ModelKit estimators exist.

## Project Policies

- [Changes](CHANGES.md) records the contents of each published release.
- [Governance](GOVERNANCE.md) describes roles and how project decisions are made.
- [Support](SUPPORT.md) defines version, compiler, and platform support.
- [License](LICENSE) contains the Apache License 2.0 terms.

Development happens at [asara-io/ModelKit](https://github.com/asara-io/ModelKit). Please use the [issue tracker](https://github.com/asara-io/ModelKit/issues) for bug reports and support requests.

## License

ModelKit is licensed under the Apache License, Version 2.0.
