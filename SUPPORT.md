# ModelKit Support Policy

ModelKit is pre-release software under active development. This policy states
the intended support contract.

## Compiler Support

ModelKit requires OCaml 5.2 or newer. OCaml 4 is not supported, although the 
project may in the future add compatibility code for it, depending if developer
interest substantially requires it. The package metadata enforces this 
compiler floor.

The continuous-integration matrix tests OCaml 5.2 and 5.3 on every supported 
platform. A release may raise the minimum compiler version only in a
documented compatibility change.

## Platform Support

The portable core targets native OCaml, Stdlib, and Bigarray on Linux, macOS,
and Windows, with an always-available sequential implementation. A platform is
considered verified for a release only when that release's CI matrix passes on
it.

The portable suite covers x86-64 Linux and Windows alongside arm64 macOS.
Optional adapters and accelerated backends publish their own dependency and
platform matrices.

## Version Support

Before 1.0, public APIs and artifact formats may change between minor
releases. Such changes must be documented. Within a published 0.x minor line,
patch releases do not intentionally break documented public APIs.

Once 1.0 is released, breaking public API changes require a documented
deprecation cycle unless a security or correctness issue makes continued
support unsafe. The project supports the latest released minor line. Fixes for
older lines are considered according to severity, user impact, and maintainer
capacity.

## Getting Help and Reporting Defects

Use the [Issues tracker](https://github.com/asara-io/ModelKit/issues) for bug
reports, portability problems, usage questions, and more. Include the 
ModelKit revision or version, OCaml and opam versions, operating system and
architecture, the smallest reproducer possible (code snippets greatly 
appreciated), and the complete error message.

Support is provided on a best-effort basis; the open-source project offers no
response-time or resolution-time service-level agreement. Please do not put
credentials, proprietary datasets, personal data, or unannounced security
details in public issues.

Questions about unsupported compiler versions, modified forks, unreleased
optional dependencies, or downstream application code may receive guidance,
but they are outside the support contract.
