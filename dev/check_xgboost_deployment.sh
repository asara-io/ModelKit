#!/usr/bin/env bash
set -euo pipefail

deployment_repo=$(cd "$(dirname "$0")/.." && pwd)
deployment_tmp=$(mktemp -d "${TMPDIR:-/tmp}/modelkit-deployment.XXXXXX")
deployment_build=".deployment-build.$(basename "$deployment_tmp")"
trap 'rm -rf "$deployment_tmp" "$deployment_repo/$deployment_build"' EXIT

cd "$deployment_repo"
opam exec -- dune build --build-dir "$deployment_build" -p modelkit,modelkit-xgboost @install
opam exec -- dune install --build-dir "$deployment_build" --prefix "$deployment_tmp/install" modelkit modelkit-xgboost

mkdir "$deployment_tmp/consumer"
cp adapters/xgboost/deployment_worker.ml "$deployment_tmp/consumer/deployment_worker.ml"
cp dev/xgboost_streaming_consumer.ml "$deployment_tmp/consumer/streaming_consumer.ml"
cat > "$deployment_tmp/consumer/dune-project" <<'PROJECT'
(lang dune 3.23)
(name modelkit_deployment_consumer)
PROJECT
cat > "$deployment_tmp/consumer/dune" <<'BUILD'
(executable
 (name deployment_worker)
 (modules deployment_worker)
 (libraries modelkit modelkit-xgboost yojson))
(executable
 (name streaming_consumer)
 (modules streaming_consumer)
 (libraries modelkit modelkit-xgboost))
BUILD

deployment_path="$deployment_tmp/install/lib${OCAMLPATH:+:$OCAMLPATH}"
for deployment_package in modelkit modelkit-xgboost; do
  deployment_location=$(opam exec -- env OCAMLPATH="$deployment_path" ocamlfind query "$deployment_package")
  if [[ "$deployment_location" != "$deployment_tmp/install/lib/"* ]]; then
    echo "Consumer resolved $deployment_package outside the staged install" >&2
    exit 1
  fi
done
opam exec -- env OCAMLPATH="$deployment_path" dune build --root "$deployment_tmp/consumer" deployment_worker.exe streaming_consumer.exe

mkdir "$deployment_tmp/streaming-cache"
"$deployment_tmp/consumer/_build/default/streaming_consumer.exe" "$deployment_tmp/streaming-cache"
"$deployment_tmp/consumer/_build/default/streaming_consumer.exe" "$deployment_tmp/streaming-cache" --load-pipeline

for deployment_metadata in "$deployment_repo"/test/fixtures/xgboost/artifacts/*/*.metadata.json; do
  "$deployment_tmp/consumer/_build/default/deployment_worker.exe" \
    "$(dirname "$deployment_metadata")" "$(basename "$deployment_metadata")" --auto-compat \
    > "$deployment_tmp/response.json"
  printf 'Verified %s/%s\n' "$(basename "$(dirname "$deployment_metadata")")" "$(basename "$deployment_metadata")"
done
