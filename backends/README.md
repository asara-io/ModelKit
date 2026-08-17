# Optional backends

This directory contains optional numerical and execution backends. Backends
depend inward on `modelkit` and must conform to the portable sequential
implementation; the portable package never requires an accelerated backend.

`parallel/` provides the separately installable `modelkit-parallel` package.
It uses Domainslib for bounded fold execution and retains sequential fallback,
stable result ordering, deterministic logical seeds, and oversubscription
diagnostics. Its requested domain count includes the calling domain; requesting
one domain avoids pool creation. Diagnostics observe common BLAS and OpenMP
thread-limit variables without changing them.
