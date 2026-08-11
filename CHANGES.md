# 0.2.0 (2026-08-10)

- Add immutable float64 vectors and matrices, row views, typed targets, feature schemas, sample weights, groups, and validation errors over portable Bigarray storage.
- Define common contracts for immutable specifications, fitted estimators, classifiers, regressors, transformers, scorers, splitters, execution, random-number generation, and numerical backends.
- Add deterministic seeds and random streams, stable sequential execution, and a portable reference backend with cancellation-safe reductions and matrix-vector kernels.
- Establish unit, property, metamorphic, mdx, public API compatibility, sklearn fixture, and reusable numerical-backend conformance tests.
- Add pinned development-only Python tooling and committed metadata for reproducible sklearn reference fixtures and comparative benchmark collection without introducing a runtime Python dependency.
- Verify build, tests, package installation targets, and odoc generation on Linux x86-64, macOS arm64, and Windows x86-64 with OCaml 5.2 and 5.3 in GitHub Actions.

# 0.1.0 (2026-08-04)

- Establish the initial ModelKit package with an OCaml 5.2 compiler floor and Apache-2.0 licensing.
- Add the portable library skeleton and reserved package boundaries for optional adapters and accelerated backends.
- Add the Dune workspace, generated odoc documentation, formatting checks, and platform-specific opam lock workflow.
- Document project governance, support, and maintenance policies.
