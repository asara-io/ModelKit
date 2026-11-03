# ModelKit XGBoost Linux build audit

Date: 2026-11-02. Source: commit `e5e9562`, ModelKit 0.6.3.

The original audit below covers commit `e5e9562`. Its reproduction experiments
used temporary copies. The subsequent implementation addresses the ModelKit
findings and supplies a companion dependency patch; the original findings are
retained below as evidence.

## Implementation status

- **Corrected in ModelKit:** every regression/classification fixture now runs
  independently parameterized native comparisons, including unsampled cases,
  zero-weight classification filtering, and row sampling. Frozen Python
  comparisons are additional checks for native 3.0.0/3.0.4/3.4.1/3.4.2; sampled
  frozen comparisons additionally run only on macOS. Tolerances
  remain absolute-plus-relative 2e-6. Native 3.1.0 now passes the complete suite.
- **Corrected in ModelKit:** directory-scoped OpenMP limits cover MDX and nested
  tests. A test-only MDX prelude asserts both limits before executing examples.
- **Corrected in ModelKit CI:** retain existing vendored archives instead of
  unconditionally recompiling; add system, vendored, musl, arm64, s390x,
  OCaml 5.2/5.5, and compatible minimum dependency configurations. Image tags
  and multi-architecture digests were verified against Docker Hub.
- **Additional minimum-build issue corrected:** deployment tests now resolve
  their worker path relative to the test directory before spawning it. Dune
  3.23's bare executable paths otherwise searched PATH and caused seven
  reproducible deployment failures. The same tests now pass without PATH edits.
- **Dependency bounds clarified:** ctypes 0.20 has the required API but its
  available releases exclude OCaml 5.2. ctypes/ctypes-foreign 0.22 are the first
  compatible pair, so the package now expresses that effective floor directly.
- **Companion patch prepared, publication pending:**
  [xgboost-build-portability.patch](xgboost-build-portability.patch) fixes CMake
  discovery/version checks, obtains runtime link requirements from CMake, tracks
  relevant environment inputs, and bounds build jobs by caller/CPU/memory
  limits. Its reusable installed-consumer check does not rebuild the dependency.
  [Handoff and release gates](xgboost-build-portability.md) explain the changes
  and the upstream-to-opam-to-ModelKit sequence.

The ordinary Linux CI jobs use the released, unmodified binding 0.2.6. Only the
explicitly named companion-patch job applies the proposed dependency patch.
**Adding jobs is not evidence that they passed:** Docker's daemon is unavailable
locally. Linux/glibc/musl and emulated s390x results remain CI release gates.
The dependency fixes cannot protect installations of the unmodified 0.2.6
release on affected toolchains until a corrected binding release (with an
updated ModelKit pin) or an approved opam patch revision is available. No new platform exclusions or persistence relaxations
were introduced.

## Implementation validation

- Released binding 0.2.6, OCaml 5.4.1: complete relevant package build, tests,
  executable documentation, installation, generated metadata, and isolated
  native/complete-bytecode deployment consumers pass.
- Adapter suites pass on checksum-verified native 3.0.0, 3.0.4, 3.1.0, and
  Homebrew 3.4.2. The new release-harness script passes end to end for macOS
  arm64 3.0.0/3.1.0 and asserts that each requested version was actually loaded.
- OCaml 5.2.1, Dune/dune-configurator 3.23.0, ctypes/ctypes-foreign 0.22.0,
  Yojson 2.2.2, MDX 2.4.0, Alcotest 1.9.0: the complete relevant build, tests,
  docs, installation, metadata, and isolated deployment consumers pass.
- The MDX prelude passes with an uncapped parent environment. A deliberate
  negative control fails before documentation examples run; the correct prelude
  is restored and the documentation checks pass.
- Companion binding: CMake policy and resource-budget tests pass, native
  configuration compiles with warnings treated as errors in OCaml, and the
  forced vendored CMake 4.4.4/Clang/libc++ build passes. Installed native,
  bytecode, and complete-bytecode consumers pass with the build hidden and the
  prefix relocated. The complete ModelKit adapter suite passes against the
  staged patched binding with static native 3.0.4. The binding's own native and
  property-based test suites also pass.
- Workflow expressions/YAML validate with actionlint 1.7.12; shell syntax,
  Python compilation, opam lint, formatting, generated opam metadata, patch
  application, and whitespace checks pass. All four Linux CPU wheel fixtures
  (3.0.0/3.1.0 on x86-64/arm64) pass checksum, archive-layout, ELF, and loader-alias
  checks; actual Linux execution remains pending.

## Original audit summary

There is an additional reproducible test failure with native XGBoost 3.1.0.
The suite selects 3.4.1 Python references for every native minor release after
3.0, although training behavior is not identical across those releases.
There is also a confirmed gap in the OpenMP environment applied to documentation
tests. Neither problem calls for excluding more Linux distributions or
architectures, or weakening fitted-model integrity checks.

| Priority | Finding | Evidence | Owner |
| --- | --- | --- | --- |
| P1 | Frozen training references are selected too broadly | Reproduced failures on native 3.1.0, including seven with Linux reference-selection behavior | ModelKit |
| P2 | MDX tests have no package-scoped OpenMP limits | An executed MDX diagnostic observed both variables unset | ModelKit |
| P2 | CMake command and minimum-version requirements are incompletely checked | Upstream configuration sources disagree on accepted command names; native CMake requires 3.18 | XGBoost binding / opam configuration |
| P2 | Vendored Linux linking assumes GNU C++/OpenMP runtimes | Upstream emits `-lstdc++ -lgomp` regardless of CMake's selected Linux compiler | XGBoost binding |
| P2 | Native build parallelism is not memory-aware; GitHub CI can build it twice | Upstream caps host CPU count at six; our workflow installs then unconditionally reinstalls | Upstream and ModelKit CI |
| P2 | Linux provider/libc/architecture coverage is narrow | Our native job has one Ubuntu runner and one OCaml version; older native coverage runs on macOS | ModelKit CI |

## 1. Training references can block an otherwise working adapter

Relevant code:

- `adapters/xgboost/regression_test.ml:230`: minor zero selects the 3.0.4
  reference; every other minor selects the newer reference.
- `adapters/xgboost/classification_test.ml:428`: the same selection rule.
- `adapters/xgboost/regression_test.ml:325` and
  `adapters/xgboost/classification_test.ml:528`: matched-native checks are used
  only for column-sampled cases. Unsampled cases always check frozen references.
- `adapters/xgboost/native_reference.ml:54`: platform selection controls the
  frozen sampled reference, without also checking the native release.

The OCaml dependency pin is 0.2.6, but the selected native library can vary by
distribution. ModelKit admits native major three, including 3.1.0.

Using the official, checksum-verified native 3.1.0 macOS arm64 wheel reproduced
ten parity failures in the unmodified suite. Forcing the Linux reference branch
in the temporary copy removed the sampled-reference failures but left seven
unsampled multiclass failures: all-missing, dense, imbalanced, missing,
regularized, weighted, and zero-weight-class fixtures.

As controls, the sampled regression and dense multiclass fixtures were compared
with independently parameterized native calls on that same 3.1.0 runtime.
Both passed. This establishes reference drift for those cases; it does not
establish matched-native equivalence for every remaining fixture.

Recommended change:

1. Retain frozen Python checks for explicitly verified native/reference pairs.
2. Run independent matched-native checks across the admitted runtime family,
   including unsampled cases, rather than only sampled cases.
3. Preserve ModelKit's classification semantics when constructing the reference:
   exclude zero-weight training rows and encode the effective classes correctly.
4. Check row subsampling as well as column sampling when selecting tests affected
   by native random-number and C++ standard-library behavior.
5. Keep the numerical tolerances and protocol tests; do not simply loosen
   tolerances or skip native 3.1.

This concerns newly trained models and their reference outputs. It is separate
from the saved-model intercept-preservation check added for 0.6.3.

Evidence logs:

- `/tmp/modelkit-linux-audit-native310.log`
- `/tmp/modelkit-linux-audit-native310-linux-branch.log`
- `/tmp/modelkit-linux-audit-native310-matched.log`

## 2. Documentation tests escape the OpenMP limits

The ordinary tests in `adapters/xgboost/dune` explicitly set
`OMP_NUM_THREADS=1` and `OMP_THREAD_LIMIT=1`. The MDX stanza at line 200 does not.
GitHub's job-wide settings mask this difference. The supplied opam CI setup
does not provide those job-wide settings.

A temporary MDX diagnostic observed `AUDIT_OMP=unset/unset`. A separate deliberate
exception confirmed that the documentation blocks are executed. No timeout or
out-of-memory failure was reproduced in this audit: the confirmed finding is
the missing environment, and oversubscription is its resource risk.

This matters because the binding's ordinary dense DMatrix constructor does not
supply a constructor thread limit; limiting booster threads alone does not
establish the documented process-start contract.

Recommended change: apply the limits through a directory-scoped Dune test/build
environment that also covers MDX. Keep the child-process startup-profile tests,
which deliberately remove or replace the inherited limits. This should not add
runtime environment mutation to the installed library.

Diagnostic artifact:
`/tmp/modelkit-linux-audit-docs/_build/default/adapters/xgboost/.mdx/index.mld.corrected`.

## 3. CMake discovery has two upstream portability gaps

The [conf-cmake configuration probe](https://github.com/ocaml/opam-source-archives/blob/main/patches/conf-cmake/configure.sh)
accepts `cmake3` first, then `cmake`, and records the selected command. It does
not test a minimum version.

The binding's [config/discover.ml](https://github.com/tarides/xgboost-ocaml/blob/v0.2.6/config/discover.ml)
instead searches only for `cmake`, falling back to the literal command `cmake`.
Its bundled [CMakeLists.txt](https://github.com/tarides/xgboost-ocaml/blob/v0.2.6/vendor/xgboost/CMakeLists.txt)
requires CMake 3.18 or newer.

Consequently, dependency configuration can succeed while the vendored native
build fails on a `cmake3`-only host or a host with an older `cmake`. This is a
source-confirmed conditional blocker, not a newly reproduced Linux failure.
It can also choose an old `cmake` when a newer `cmake3` is installed and selected
by the opam probe.
The supplied CentOS log also reports a stale/missing `cmake3` system package,
although that warning was not its fatal error.

Recommended upstream fix: honor the configured CMake command, or discover both
names, and verify the required version before starting the native build.

## 4. Compiler availability does not establish the required native toolchain

The bundled native build requires C++17 and OpenMP. The
[conf-c++ package](https://github.com/ocaml/opam-repository/blob/master/packages/conf-c%2B%2B/conf-c%2B%2B.1.0/opam)
checks compilation of an empty C++ file; it does not check those capabilities.

For non-macOS builds, the binding emits `-lstdc++ -lgomp`. A Linux configuration
using Clang/libomp or libc++ can therefore differ from the runtime libraries
assumed by the installed link flags. Stock GCC installations generally provide
the expected libraries, but the package's platform filter does not express this
toolchain assumption. No Clang-only Linux link failure was reproduced locally.

Recommended upstream fix: derive runtime link requirements from the selected
CMake toolchain, or clearly validate the GNU runtime requirement. Keep a small
installed-consumer check for that provider; successful library compilation alone
does not prove downstream linking.

## 5. Resource limits and duplicate native builds

Upstream chooses `min(host CPU count, 6)` for its CMake build. This is an
improvement over using every host CPU, but it does not use the container memory
budget or the caller's opam/Dune job limit. Six concurrent C++ compilations can
still exceed a constrained worker's memory, especially under emulation.

Our GitHub workflow first installs runtime dependencies, then unconditionally
reinstalls XGBoost with its vendored provider and Dune caching disabled. On a
cold Ubuntu runner that already selected the fallback, that can compile the
native dependency twice. This duplicate work is specific to our workflow; it is
not an assertion that opam-repository always builds the dependency twice.

Recommended changes: let upstream accept an explicit native-build job limit;
in our workflow, avoid a second rebuild when the installed binding already has
the required vendored archives, while retaining cleanup and consumer checks.

## 6. Coverage needed for Linux variability

The current native workflow covers Ubuntu 24.04 with a forced vendored provider
and macOS with a system provider, both on OCaml 5.3. The native 3.0.0 regression
check also runs on macOS. It does not establish Linux system-provider, musl,
older-toolchain, arm64, or big-endian behavior.

A practical expansion, rather than a full Cartesian product, is:

- Debian with its packaged system libxgboost, plus installed consumers.
- Ubuntu/CentOS with vendored XGBoost and a constrained native-build budget.
- Alpine/musl with vendored XGBoost.
- At least one Linux arm64 job and opam CI confirmation on a big-endian target.
- OCaml 5.2 and the newest supported compiler on selected native jobs.
- A minimum-compatible dependency job for the declared ctypes/Yojson/MDX bounds.

The minimum ctypes API was inspected: 0.20.0 already provides
`dynamic_funptr` and `with_fun`. There is no evidence from this audit that this
API requires raising the current ctypes lower bound. The full minimum dependency
combination was not built.

The bundled dmlc subproject requires CMake 3.15, rather than the old pre-3.5
compatibility modes removed by recent CMake releases. No incompatibility with
CMake 4 was established by this audit.

## Validation limits and implementation order

Experiments ran on macOS arm64, OCaml 5.4.1, binding 0.2.6, using an isolated
temporary switch. The Linux reference-selection control changed only the test
branch in a temporary source copy; it was not a Linux ABI or libc test.
Docker's daemon was unavailable, so no fresh Linux-container run is claimed.

Recommended order: fix reference selection/matched-native coverage first, apply
the MDX OpenMP limits next, then expand Linux provider/libc coverage and pursue
the upstream build-tool fixes. No additional architecture exclusion is proposed.
