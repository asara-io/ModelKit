#!/usr/bin/env bash
set -euo pipefail

deployment_repo=$(cd "$(dirname "$0")/.." && pwd)
deployment_tmp=$(mktemp -d "${TMPDIR:-/tmp}/modelkit-deployment.XXXXXX")
deployment_build=".deployment-build.$(basename "$deployment_tmp")"
trap 'rm -rf "$deployment_tmp" "$deployment_repo/$deployment_build"' EXIT

cd "$deployment_repo"
opam exec -- dune build --build-dir "$deployment_build" -p modelkit,modelkit-xgboost @install
for deployment_package in modelkit modelkit-xgboost; do
  opam-installer --prefix "$deployment_tmp/install" "$deployment_build/default/$deployment_package.install"
done

mkdir "$deployment_tmp/consumer"
cp adapters/xgboost/deployment_worker.ml "$deployment_tmp/consumer/deployment_worker.ml"
cat > "$deployment_tmp/consumer/dune-project" <<'PROJECT'
(lang dune 3.23)
(name modelkit_deployment_consumer)
PROJECT
cat > "$deployment_tmp/consumer/dune" <<'BUILD'
(executable
 (name deployment_worker)
 (libraries modelkit modelkit-xgboost yojson))
BUILD

deployment_path="$deployment_tmp/install/lib${OCAMLPATH:+:$OCAMLPATH}"
for deployment_package in modelkit modelkit-xgboost; do
  deployment_location=$(opam exec -- env OCAMLPATH="$deployment_path" ocamlfind query "$deployment_package")
  if [[ "$deployment_location" != "$deployment_tmp/install/lib/"* ]]; then
    echo "Consumer resolved $deployment_package outside the staged install" >&2
    exit 1
  fi
done
opam exec -- env OCAMLPATH="$deployment_path" dune build --root "$deployment_tmp/consumer" deployment_worker.exe

for deployment_metadata in "$deployment_repo"/test/fixtures/xgboost/artifacts/*/*.metadata.json; do
  "$deployment_tmp/consumer/_build/default/deployment_worker.exe" \
    "$(dirname "$deployment_metadata")" "$(basename "$deployment_metadata")" --auto-compat \
    > "$deployment_tmp/response.json"
  printf 'Verified %s/%s\n' "$(basename "$(dirname "$deployment_metadata")")" "$(basename "$deployment_metadata")"
done
