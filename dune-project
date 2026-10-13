(lang dune 3.23)
(using mdx 0.4)
(name modelkit)
(version 0.6.1)
(generate_opam_files true)
(implicit_transitive_deps false)

(source
 (github asara-io/ModelKit))

(authors Asara)
(maintainers "Asara developers <devs@asara.io>")
(license Apache-2.0)
(maintenance_intent "(latest)")

(package
 (name modelkit)
 (synopsis "Portable classical machine learning workflows for OCaml")
 (description
  "ModelKit is a native OCaml library for cohesive classical machine learning workflows. It is designed around immutable estimator specifications, leakage-safe pipelines, deterministic evaluation, and portable fitted artifacts.")
 (tags
  (data-science machine-learning))
 (depends
  (ocaml (>= 5.2))
  (alcotest (and :with-test (>= 1.9.0)))
  (mdx (and :with-test (>= 2.4.0)))
  (qcheck-core (and :with-test (>= 0.90)))))

(package
 (name modelkit-parallel)
 (synopsis "Bounded parallel execution for ModelKit")
 (description
  "ModelKit Parallel provides an optional Domainslib execution backend for deterministic bounded cross-validation.")
 (tags
  (data-science machine-learning parallelism))
 (depends
  (ocaml (>= 5.2))
  (modelkit (= :version))
  (domainslib (>= 0.5.2))
  (alcotest (and :with-test (>= 1.9.0)))))

(package
 (name modelkit-nx)
 (synopsis "Checked Nx tensor admission for ModelKit")
 (description
  "ModelKit Nx converts explicitly typed Nx tensors into immutable ModelKit features, targets, masks, groups, names, and weights with typed validation and allocation reports.")
 (tags
  (data-science machine-learning adapters tensors))
 (depends
  (ocaml (>= 5.2))
  (modelkit (= :version))
  (nx (= 1.0.0~alpha3))
  (alcotest (and :with-test (>= 1.9.0)))))

(package
 (name modelkit-xgboost)
 (synopsis "CPU histogram-tree estimators with XGBoost for ModelKit")
 (description
  "ModelKit XGBoost provides weighted CPU histogram-tree regression and binary/multiclass classification, immutable specifications, checked float32 input conversion, native missing-value routing, allocation reports, schema-checked prediction, and trusted fitted-model persistence without adding XGBoost dependencies to ModelKit's portable core.")
 (tags
  (data-science machine-learning adapters xgboost))
 (depends
  (ocaml (>= 5.2))
  (modelkit (= :version))
  (xgboost (= 0.2.1))
  (yojson (and (>= 2.2.2) (< 4.0.0)))
  (modelkit-parallel (and :with-test (= :version)))
  (mdx (and :with-test (>= 2.4.0)))
  (alcotest (and :with-test (>= 1.9.0)))))

(package
 (name modelkit-talon)
 (synopsis "Checked Talon dataframe column admission for ModelKit")
 (description
  "ModelKit Talon converts explicitly selected, typed Talon dataframe columns into immutable ModelKit features, targets, masks, groups, names, and weights with typed validation and allocation reports.")
 (tags
  (data-science machine-learning adapters dataframes))
 (depends
  (ocaml (>= 5.2))
  (modelkit (= :version))
  (talon (= 1.0.0~alpha3))
  (nx (= 1.0.0~alpha3))
  (alcotest (and :with-test (>= 1.9.0)))))
