open Modelkit
open Evaluation_metadata_support

let matrix n = Matrix.init ~rows:n ~columns:1 (fun _ _ -> 0.) |> data

let test_fixture () =
  let fields =
    In_channel.with_open_text (Sys.getenv "MODELKIT_PARTITIONING_FIXTURE")
      (fun channel ->
        In_channel.input_lines channel
        |> List.filter_map (fun line ->
            match String.split_on_char '\t' line with
            | [ key; values ] ->
                Some
                  ( key,
                    String.split_on_char ',' values
                    |> List.map int_of_string |> Array.of_list )
            | _ -> None))
  in
  let values name = List.assoc name fields in
  let check name expected_count result =
    Alcotest.(check int)
      "reference fold count" expected_count (Array.length result);
    Array.iteri
      (fun i (train, test) ->
        let check suffix view =
          let key = name ^ "_" ^ string_of_int i ^ suffix in
          Alcotest.(check (array int)) key (values key) (Row_view.indices view)
        in
        check "_train" train;
        check "_test" test)
      result
  in
  let rng = Rng.create (Seed.of_int 1729) in
  let assignments = values "assignments" in
  Predefined_split.split
    (Predefined_split.create ~test_folds:assignments () |> get)
    ~rng
    ~x:(matrix (Array.length assignments))
    ~y:None ()
  |> get |> check "predefined" 3;
  Leave_one_out.split (Leave_one_out.create ()) ~rng ~x:(matrix 4) ~y:None ()
  |> get |> check "leave_one_out" 4;
  let groups = values "leave_groups" in
  Leave_one_group_out.split
    (Leave_one_group_out.create ())
    ~rng
    ~x:(matrix (Array.length groups))
    ~groups:(Groups.create ~expected_length:(Array.length groups) groups |> data)
    ~y:None ()
  |> get
  |> check "leave_one_group_out" 3;
  let groups = values "groups" and labels = values "labels" in
  Stratified_group_k_fold.split
    (Stratified_group_k_fold.create ~folds:3 () |> get)
    ~rng
    ~x:(matrix (Array.length groups))
    ~groups:(Groups.create ~expected_length:(Array.length groups) groups |> data)
    ~y:(Some (Target.classification labels))
    ()
  |> get |> check "stratified_group" 3

let () =
  Alcotest.run "Partitioning reference"
    [ ("sklearn", [ ("exact unshuffled row parity", `Quick, test_fixture) ]) ]
