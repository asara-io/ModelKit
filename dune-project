(lang dune 3.23)
(name modelkit)
(version 0.1.0)
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
  (ocaml (>= 5.2))))
