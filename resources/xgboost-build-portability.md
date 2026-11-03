# XGBoost binding build portability: patch and release handoff

Prepared against the **opam xgboost 0.2.6 source archive**, on 2026-11-02.
Nothing has been submitted, pushed, published, or committed on your behalf.

## What is ready and what still blocks release

ModelKit's training-reference, documentation-threading, deployment-worker-path,
and CI duplication fixes are implemented in this checkout. The sibling
[xgboost-build-portability.patch](xgboost-build-portability.patch) contains the
remaining dependency changes. It applies with `patch -p1` to xgboost 0.2.6.
The binding's public API, native XGBoost source version (3.0.4), archive naming,
and supported architecture filters stay unchanged.

A package cannot repair a dependency whose own installation has already failed.
Until a patched binding is available through opam (a new release or an approved
opam-repository patch revision), **the published binding still has the
CMake/toolchain/resource gaps described below**. The normal CI matrix deliberately tests the released binding without
patching it. The separate `Companion patch / Clang libc++ / cmake3` job tests the
proposed fix. Its success alone would not establish that the released dependency
is fixed.

Docker's daemon is unavailable in this workstation session. The new Linux
matrix must complete successfully before release. No local Linux ABI, musl,
Clang-on-Linux, or s390x result is claimed.

## Suggested upstream PR title

Honor native build limits and CMake-discovered toolchains in vendored builds

## Suggested upstream PR description

The vendored fallback currently discovers only `cmake`, even though
`conf-cmake` accepts and prefers `cmake3`, and does not check the native source's
CMake 3.18 minimum before starting configuration. It also links GNU C++/OpenMP
runtimes on every non-macOS platform regardless of the CMake-selected compiler,
and selects up to six C++ compilation jobs without consulting caller or memory
budgets. These assumptions can make an otherwise supported installation fail
before a downstream package can build.

This change:

- Honors an explicit `CMAKE` command, otherwise selects a sufficiently recent
  `cmake3` or `cmake`, requiring version 3.18 or newer. An invalid explicit
  override produces a configuration error instead of silently selecting another
  tool.
- Obtains C++ runtime libraries/search directories and OpenMP libraries from
  CMake's compiler/OpenMP discovery. Installed consumers therefore link the
  selected toolchain's runtimes, including Clang/libc++, instead of unconditionally
  requesting `libstdc++` and `libgomp`.
- Bounds native jobs by online CPUs (maximum six), `OPAMJOBS`,
  `CMAKE_BUILD_PARALLEL_LEVEL`, `XGBOOST_BUILD_JOBS`, Linux `MemAvailable`, and
  cgroup v1/v2 memory limits. The opam build exports its actual job setting to
  `OPAMJOBS`. Explicit limits must be positive integers. Memory budgeting reserves
  one GiB and budgets one GiB per compiler, with at least one job; this is a
  conservative heuristic, not a guarantee that an undersized machine can compile
  XGBoost.
- Reads virtual memory-control files as streams rather than relying on their
  often-zero file size, and checks both root and process-membership cgroup paths.
- Tracks the relevant CMake/compiler/pkg-config/job environment inputs in Dune.
  The installed-linking script's `REUSE_BUILD=1` path consumes the completed
  installation manifest directly, so a different temporary pkg-config directory
  does not trigger a second native compilation.
- Tests supported/old/missing CMake commands, command precedence and explicit
  overrides, CPU/memory/caller job limits, and invalid job settings.

No estimator behavior or persistence format is changed. Native C++17/OpenMP
capabilities remain checked by CMake; an empty C++ compiler probe is not treated
as evidence of those capabilities.

Local validation: OCaml 5.4.1, CMake 4.4.4, Apple Clang/libc++, Homebrew libomp,
forced vendored static native 3.0.4. Configuration policy tests, compilation, and the binding's own native/property
suites pass. Native, ordinary bytecode, and self-contained bytecode installed consumers
pass training/prediction/restore after the source build is hidden and the install
prefix relocated. ModelKit's complete adapter suite also passes against the
staged patched binding. The Linux matrix is prepared but has not run locally.

## Patch contents

| File | Change |
| --- | --- |
| `config/build_policy.ml` | Testable CMake selection and job-budget policies |
| `config/build_policy_test.ml` | Policy regression cases |
| `config/discover.ml` | Tool selection, virtual-file memory probes, runtime metadata, bounded jobs |
| `config/dune` | Explicit modules, test stanza, environment dependencies |
| `vendor/xgboost/CMakeLists.txt` | Export CMake's discovered C++/OpenMP runtime metadata |
| `vendor/README.md` | Preserve runtime-metadata exports on vendor refreshes |
| `scripts/check-installed-linking.sh` | Reuse a completed install build without recompilation |
| `xgboost.opam` | Export opam's selected native-build job budget |
| `README.md` | Explain requirements, controls, and memory-budget limits |

`CMAKE_CXX_IMPLICIT_LINK_LIBRARIES`,
`CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES`, and `OpenMP_CXX_LIBRARIES` describe the
selected compiler and OpenMP runtime. Their values are written during CMake
configuration and read by the OCaml configurator. Native source/build archive
paths are still excluded from installed link metadata; installed toolchain
runtime paths are allowed, since those runtimes remain external dependencies.

## Numbered implementation-to-release steps

1. Create your working branch in your fork of
   [tarides/xgboost-ocaml](https://github.com/tarides/xgboost-ocaml), based on
   `v0.2.6` or its equivalent released source. If upstream has advanced, inspect
   the diff and rebase these changes rather than replacing newer fixes.

2. Set `modelkit_checkout` to the absolute path of this ModelKit checkout. From
   the XGBoost checkout, preview and apply the patch:

   ```sh
   patch --dry-run -p1 < "$modelkit_checkout/resources/xgboost-build-portability.patch"
   patch -p1 < "$modelkit_checkout/resources/xgboost-build-portability.patch"
   ```

3. Install the binding's build/test dependencies in your selected switch and
   run the configuration policy tests and metadata lint:

   ```sh
   opam install ./xgboost.opam --deps-only --with-test --yes
   opam exec -- dune runtest config
   opam lint xgboost.opam
   ```

4. Force one vendored build, masking pkg-config only after dependencies such as
   libffi have been installed. Use the same checkout for the following step:

   ```sh
   empty_pkgconfig=$(mktemp -d)
   PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR="$empty_pkgconfig" \
     XGBOOST_BUILD_JOBS=2 DUNE_CACHE=disabled \
     opam exec -- dune build -p xgboost -j 2 @install @runtest
   ```

   Inspect `_build/default/config/c_library_flags.sexp`: it must contain
   `-lxgboost_vendored`, `-ldmlc_vendored`, and the selected toolchain's runtime
   requirements. It must not contain a source/build path to either static
   archive. On Clang/libc++ Linux, verify libc++/libomp rather than assuming GNU
   runtime names. CMake's configuration output must confirm C++17 and OpenMP.

5. Verify installed linking without compiling XGBoost again:

   ```sh
   REUSE_BUILD=1 XGBOOST_BUILD_JOBS=2 \
     bash scripts/check-installed-linking.sh "$PWD" vendored
   rmdir "$empty_pkgconfig"
   ```

   The script hides the build directory, relocates the installation, and tests
   native, bytecode, and self-contained bytecode consumers. Retain its final
   success message and the configuration/runtime output for the PR.

6. Run the released-dependency and companion jobs from ModelKit's expanded
   `.github/workflows/xgboost.yml`. For a direct container reproduction, from
   ModelKit's checkout:

   ```sh
   docker run --rm --platform linux/amd64 \
     --volume "$PWD:/source:ro" \
     --env CI_PROVIDER=vendored --env CI_MINIMUM=0 --env CI_COMPANION=1 \
     --entrypoint /bin/bash \
     ocaml/opam:debian-13-ocaml-5.5@sha256:5f3c1eddb76096c9bbdf877998ac7255c5d33fb9f7d8d691a288b70ef530ef8b \
     /source/dev/ci/check_xgboost_linux.sh
   ```

   This explicitly applies the companion patch to a temporary version-pinned
   binding, selects Clang/libc++, invokes `cmake3`, builds the vendored provider,
   removes its build directory, and runs ModelKit's tests/docs/installed
   consumers. The ordinary matrix separately covers unmodified released binding
   0.2.6 on Debian system libraries, CentOS and Alpine vendored libraries,
   arm64 and emulated s390x, and minimum-compatible dependencies. System-library
   jobs also exercise checksum-pinned native 3.0.0 and 3.1.0 libraries on x86-64
   and arm64; they verify the actual loaded version to prevent silent fallback.

7. Before merging upstream, require successful Linux GCC and Clang/libc++
   installed-consumer results, plus Alpine/musl and the resource-policy tests.
   Do not describe the new Linux configurations as validated merely because the
   workflow file parses. For a CMake-name regression reproduction, supply only
   `cmake3` (or an old `cmake` beside a new `cmake3`) in a disposable build
   environment; policy tests already cover both selections. Test an explicit
   `XGBOOST_BUILD_JOBS=1` on the smallest supported builder.

8. Open the upstream PR yourself using the title/body above, the patch, and the
   actual Linux CI evidence. The patch also records the vendored CMake modification in
   `vendor/README.md`, so it survives a vendor refresh. Have the
   maintainer review the one-GiB-per-compiler heuristic and the standard
   CMake-reported runtime fields. No additional architecture ban is needed for
   these fixes.

9. After merge, coordinate a new binding release with the maintainer. The
   maintainer chooses the version; do not assume that `0.2.7` is available or
   unused. Ensure the source archive includes the new policy/test files and
   patched vendored CMake file. Its opam metadata must include `build-env`, not
   just the repository's source tree.

10. Publish that release to `ocaml/opam-repository` through the maintainer's
    normal release flow. Verify source checksums, package lint, and opam CI
    results for system/vendored Linux providers and architectures. If an opam
    metadata revision or upstream follow-up is needed, resolve it before
    describing the dependency blocker as closed.

    A possible faster packaging route is an opam-repository revision to
    `xgboost.0.2.6`: carry this source patch using `patches` and an `extra-source`
    at an immutable URL with its SHA-256 checksum. That revision must **also**
    add `build-env: [ [OPAMJOBS = "%{jobs}%"] ]` to the repository's package
    metadata: changing the source archive's `xgboost.opam` after extraction does
    not change the solver/build metadata opam already selected. Coordinate this
    route with the upstream maintainer and opam reviewers. It would not require
    a ModelKit version-pin change, but still requires fresh dependency installs
    and the same successful CI evidence. This checkout has not submitted either
    route.

11. If a new corrected binding version is published, update ModelKit's exact
    XGBoost pin in `dune-project`, regenerate `modelkit-xgboost.opam`, and update
    the CI version selectors, installation/compatibility documentation,
    `CHANGES.md`, the temporary companion job's patch base/selectors, and
    current-fit binding provenance in
    `adapters/xgboost/modelkit_xgboost.ml`. Find every current occurrence with
    `rg -n '0\\.2\\.6' dune-project modelkit-xgboost.opam adapters dev .github`.
    Preserve historical artifact/fixture provenance rather than rewriting it.
    No XGBoost platform lockfile currently exists; refresh any newly added
    relevant lockfile if applicable.

12. Run the complete relevant build/test/doc/install checks, every native release
    check, and `dev/check_xgboost_deployment.sh` against the corrected **published,
    unmodified** binding. Run the new Linux matrix and opam-repository CI on the
    actual ModelKit release archive. Only then remove or convert the temporary
    companion-patch CI job, retire this patch, and mark the dependency blockers
    resolved. You retain control of commits, pushes, and releases throughout.

## Primary references

- [Released binding configuration](https://github.com/tarides/xgboost-ocaml/blob/v0.2.6/config/discover.ml)
- [conf-cmake's command selection](https://github.com/ocaml/opam-source-archives/blob/main/patches/conf-cmake/configure.sh)
- [CMake implicit runtime libraries](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_IMPLICIT_LINK_LIBRARIES.html)
- [CMake implicit runtime directories](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_IMPLICIT_LINK_DIRECTORIES.html)
- [CMake FindOpenMP result variables](https://cmake.org/cmake/help/latest/module/FindOpenMP.html)
- [XGBoost's official CPU wheel distribution](https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only)
