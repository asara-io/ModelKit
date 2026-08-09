open Modelkit

let vector_values = QCheck.(array small_int)

let vector_ownership =
  QCheck.Test.make ~count:500
    ~name:"vector admission and export preserve ownership" vector_values
    (fun values ->
      let admitted = Array.map Float.of_int values in
      let source = Array.copy admitted in
      let vector = Vector.of_array source in
      if Array.length source > 0 then source.(0) <- source.(0) +. 1.0;
      let exported = Vector.to_array vector in
      if Array.length exported > 0 then exported.(0) <- exported.(0) +. 1.0;
      Vector.to_array vector = admitted)

let sequential_order =
  QCheck.Test.make ~count:500
    ~name:"sequential execution preserves logical order" vector_values
    (fun values ->
      match
        Sequential_execution.map Sequential_execution.default
          ~f:(fun ~index value -> Ok (index, value))
          values
      with
      | Error _ -> false
      | Ok observed ->
          Array.to_list observed
          = List.mapi (fun index value -> (index, value)) (Array.to_list values))

let seed_derivation =
  QCheck.Test.make ~count:500
    ~name:"logical seed derivation is a pure function of its inputs"
    QCheck.(pair int64 small_int)
    (fun (root, index) ->
      let root = Seed.of_int64 root in
      Seed.equal
        (Seed.derive root ~operation:"property" ~index)
        (Seed.derive root ~operation:"property" ~index))

let rng_purity =
  QCheck.Test.make ~count:500
    ~name:"random generation does not mutate its input" QCheck.int64
    (fun seed ->
      let state = Rng.create (Seed.of_int64 seed) in
      let first, successor = Rng.next_int64 state in
      let repeated_first, repeated_successor = Rng.next_int64 state in
      let second, _ = Rng.next_int64 successor in
      let repeated_second, _ = Rng.next_int64 repeated_successor in
      Int64.equal first repeated_first && Int64.equal second repeated_second)

let () =
  let random = Random.State.make [| 0x4d4f4445; 0x4c4b4954 |] in
  let failures =
    QCheck_base_runner.run_tests ~verbose:true ~rand:random
      [ vector_ownership; sequential_order; seed_derivation; rng_purity ]
  in
  if failures <> 0 then exit failures
