# XGBoost persistence goldens

Each native-version directory contains regression, binary, and multiclass native
model and fitted-pipeline artifacts. The pipelines include mean imputation,
weighted standard scaling, and variance filtering. Metadata contains new inference
probe rows and expected responses, not training observations or targets.

Artifacts are trusted fixtures produced by ModelKit's supported writers. Checksums
detect corruption, not authenticity. Native 3.0.4 artifacts are checked on 3.0.4
and 3.4.2; 3.4.2 artifacts are checked on 3.4.2 and rejected by 3.0.4 as forward
version loads. Prediction/probability tolerances are absolute plus relative 2e-6.

Regenerate deliberately with the intended linked runtime:

```console
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 opam exec -- dune exec adapters/xgboost/pipeline_artifact_test.exe -- --write-fixtures test/fixtures/xgboost/artifacts
```

Ordinary tests read these files without rewriting them. Fresh-process deployment
uses `deployment_worker.exe`, linked only to ModelKit, the adapter, and declared
runtime dependencies; that worker has no training or refitting code.
