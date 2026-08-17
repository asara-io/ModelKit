(lang dune 3.23)
(using mdx 0.4)
(name modelkit)
(version 0.3.0)
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
