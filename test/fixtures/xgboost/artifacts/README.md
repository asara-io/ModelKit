# XGBoost persistence goldens

Each native-version directory contains regression, binary, and multiclass native
model and fitted-pipeline artifacts. The pipelines include mean imputation,
weighted standard scaling, and variance filtering. Metadata contains new inference
probe rows and expected responses, not training observations or targets.

Artifacts are trusted fixtures produced by ModelKit's supported writers. Checksums
detect corruption, not authenticity. Every artifact is attempted on the linked
supported runtime, including older patch versions. A successful load must match
the stored predictions/probabilities at absolute plus relative tolerance 2e-6.
For artifacts from a newer minor release, the deployment worker may report a
typed native-loader rejection or intercept-preservation failure after attempting
both standalone and pipeline loads. This accommodates actual representation changes without asserting that
every newer version is incompatible. Same-minor loading failures, envelope or
schema errors, and disagreement between standalone/pipeline loading fail tests.

Regenerate deliberately with the intended linked runtime:

```console
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 opam exec -- dune exec adapters/xgboost/pipeline_artifact_test.exe -- --write-fixtures test/fixtures/xgboost/artifacts
```

Ordinary tests read these files without rewriting them. Fresh-process deployment
uses `deployment_worker.exe`, linked only to ModelKit, the adapter, and declared
runtime dependencies; that worker has no training or refitting code.
