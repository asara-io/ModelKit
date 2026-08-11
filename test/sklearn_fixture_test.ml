type fold = { index : int; train : int array; test : int array }

let parse_indices value =
  if String.equal value "" then [||]
  else
    value |> String.split_on_char ',' |> List.map int_of_string
    |> Array.of_list

let read_fixture path =
  let target = ref None in
  let folds = Hashtbl.create 3 in
  let add_fold index kind indices =
    let existing =
      Option.value (Hashtbl.find_opt folds index)
        ~default:{ index; train = [||]; test = [||] }
    in
    let fold =
      match kind with
      | "train" -> { existing with train = indices }
      | "test" -> { existing with test = indices }
      | _ -> Alcotest.failf "unknown split kind %S" kind
    in
    Hashtbl.replace folds index fold
  in
  In_channel.with_open_text path (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
             if String.length line > 0 && line.[0] <> '#' then
               match String.split_on_char '\t' line with
               | [ "target"; values ] -> target := Some (parse_indices values)
               | [ "fold"; index; kind; values ] ->
                   add_fold (int_of_string index) kind (parse_indices values)
               | fields ->
                   Alcotest.failf "invalid fixture row with %d fields"
                     (List.length fields)));
  let target =
    Option.value !target ~default:[||]
  in
  let folds = Hashtbl.to_seq_values folds |> Array.of_seq in
  Array.sort (fun left right -> Int.compare left.index right.index) folds;
  (target, folds)

let check_partition sample_count fold =
  let seen = Array.make sample_count 0 in
  Array.iter
    (fun index ->
      Alcotest.(check bool)
        "train index is in bounds" true
        (index >= 0 && index < sample_count);
      seen.(index) <- seen.(index) + 1)
    fold.train;
  Array.iter
    (fun index ->
      Alcotest.(check bool)
        "test index is in bounds" true
        (index >= 0 && index < sample_count);
      seen.(index) <- seen.(index) + 1)
    fold.test;
  Array.iter
    (Alcotest.(check int) "sample occurs once in a partition" 1)
    seen

let test_fixture path () =
  let target, folds = read_fixture path in
  Alcotest.(check int) "sample count" 12 (Array.length target);
  Alcotest.(check int) "fold count" 3 (Array.length folds);
  let test_occurrences = Array.make (Array.length target) 0 in
  Array.iter
    (fun fold ->
      check_partition (Array.length target) fold;
      Array.iter
        (fun index -> test_occurrences.(index) <- test_occurrences.(index) + 1)
        fold.test)
    folds;
  Array.iter
    (Alcotest.(check int) "sample appears in one test fold" 1)
    test_occurrences

let () =
  if Array.length Sys.argv <> 2 then
    Alcotest.fail "expected the fixture path as the only argument";
  Alcotest.run "sklearn fixtures"
    [ ("stratified k-fold", [ Alcotest.test_case "v1" `Quick (test_fixture Sys.argv.(1)) ]) ]
