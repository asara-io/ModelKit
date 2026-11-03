#!/usr/bin/env bash
set -euo pipefail

provider=${CI_PROVIDER:?Select system or vendored}
case "$provider" in system|vendored) ;; *) exit 2 ;; esac
minimum=${CI_MINIMUM:-0}
companion=${CI_COMPANION:-0}
work=$(mktemp -d /tmp/modelkit-linux.XXXXXX)
trap 'rm -rf "$work"' EXIT
mkdir "$work/modelkit" "$work/empty-pkgconfig"
(cd /source && tar --exclude=_build --exclude=__pycache__ -cf - \
  dune dune-project dune-workspace .ocamlformat *.opam *.opam.template *.md \
  LICENSE adapters backends bench dev doc examples lib resources test) \
  | tar -C "$work/modelkit" -xf -
cd "$work/modelkit"
eval "$(opam env)"
export OPAMJOBS=2
export OPAMERRLOGLEN=0
export CI=true
export DUNE_CACHE=disabled
opam update --yes

if [[ "$companion" == 1 ]]; then
  sudo apt-get update
  sudo apt-get install -y clang libc++-dev libc++abi-dev libomp-dev
  opam source xgboost.0.2.6 --dir="$work/xgboost"
  patch -d "$work/xgboost" -p1 < resources/xgboost-build-portability.patch
  opam pin add --no-action --yes xgboost.0.2.6 "$work/xgboost"
  export CC=clang
  export CXX="$work/clang-libcxx"
  cat > "$CXX" <<'CXX'
#!/bin/sh
exec clang++ -stdlib=libc++ "$@"
CXX
  chmod +x "$CXX"
fi

minimum_packages=(dune.3.23.0 dune-configurator.3.23.0 ctypes.0.22.0
  ctypes-foreign.0.22.0 yojson.2.2.2 mdx.2.4.0 alcotest.1.9.0)
if [[ "$minimum" == 1 ]]; then
  for package in "${minimum_packages[@]}"; do
    opam pin add --kind=version --no-action --yes "${package%%.*}" "${package#*.}"
  done
  opam install --yes "${minimum_packages[@]}"
fi

# Limit pkg-config masking to the binding build; libffi still needs discovery.
opam install --yes xgboost.0.2.6 --deps-only
if [[ "$companion" == 1 ]]; then
  cmake_path=$(command -v cmake)
  mkdir "$work/cmake-bin"
  ln -s "$cmake_path" "$work/cmake-bin/cmake3"
  export PATH="$work/cmake-bin:$PATH"
  export CMAKE=cmake3
  export XGBOOST_BUILD_JOBS=2
fi
if [[ "$provider" == vendored ]]; then
  PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR="$work/empty-pkgconfig" \
    opam install --yes xgboost.0.2.6
else
  pkg-config --atleast-version=3.0 xgboost
  pkg-config --modversion xgboost
  opam install --yes xgboost.0.2.6
fi
bindings="$(opam var lib)/xgboost/bindings"
if [[ "$provider" == vendored ]]; then
  test -s "$bindings/libxgboost_vendored.a"
  test -s "$bindings/libdmlc_vendored.a"
else
  test ! -e "$bindings/libxgboost_vendored.a"
fi
# Pinned development builds may be retained; consumers must survive removal.
if [[ "$companion" == 1 ]]; then
  rm -rf "$(opam var prefix)/.opam-switch/build/xgboost.0.2.6"
fi
test ! -d "$(opam var prefix)/.opam-switch/build/xgboost.0.2.6"

opam install --yes ./modelkit.opam ./modelkit-parallel.opam ./modelkit-xgboost.opam \
  --deps-only --with-test --with-doc
if [[ "$minimum" == 1 ]]; then
  for package in "${minimum_packages[@]}"; do
    opam list --installed --short "$package" | grep -q .
  done
fi
opam exec -- cc -std=c11 -D_POSIX_C_SOURCE=200809L -U_DEFAULT_SOURCE \
  -I "$(opam var lib)/ocaml" -fsyntax-only adapters/xgboost/workflow_probe.c
# An uncapped parent environment checks the package-scoped MDX startup contract.
env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT opam exec -- dune build \
  --only-packages modelkit,modelkit-parallel,modelkit-xgboost -j 2 \
  @all @runtest @doc @install
opam exec -- dune build -p modelkit,modelkit-parallel,modelkit-xgboost -j 2 \
  @install @runtest @doc
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 bash dev/check_xgboost_deployment.sh

if [[ "$provider" == system && "$(uname -m)" != s390x ]]; then
  sudo apt-get install -y python3 curl
  python3 dev/ci/check_xgboost_native_releases.py
fi
