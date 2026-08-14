open Modelkit

type fixture = {
  vectors : (string, int array) Hashtbl.t;
  matrices : (string, (int, int array) Hashtbl.t) Hashtbl.t;
}

let parse_indices value =
  if String.equal value "" then [||]
  else
    value |> String.split_on_char ',' |> List.map int_of_string |> Array.of_list

let read_fixture path =
  let fixture = { vectors = Hashtbl.create 8; matrices = Hashtbl.create 8 } in
  In_channel.with_open_text path (fun input ->
      In_channel.input_lines input
      |> List.iter (fun line ->
          if String.length line > 0 && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ name; values ] ->
                Hashtbl.replace fixture.vectors name (parse_indices values)
            | [ name; fold; values ] ->
                let folds =
                  match Hashtbl.find_opt fixture.matrices name with
                  | Some folds -> folds
                  | None ->
                      let folds = Hashtbl.create 4 in
                      Hashtbl.add fixture.matrices name folds;
                      folds
                in
                Hashtbl.replace folds (int_of_string fold)
                  (parse_indices values)
            | fields ->
                Alcotest.failf "invalid fixture row with %d fields"
                  (List.length fields)));
  fixture

let vector fixture name =
  match Hashtbl.find_opt fixture.vectors name with
  | Some values -> values
  | None -> Alcotest.failf "fixture vector %S is missing" name

let fold_rows fixture name fold =
  match Hashtbl.find_opt fixture.matrices name with
  | None -> Alcotest.failf "fixture matrix %S is missing" name
  | Some folds -> (
      match Hashtbl.find_opt folds fold with
      | Some values -> values
      | None -> Alcotest.failf "fixture matrix %S fold %d is missing" name fold)

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let matrix rows = Matrix.init ~rows ~columns:1 (fun _ _ -> 0.0) |> get_data
let rng () = Rng.create (Seed.of_int 1729)

let check_splits fixture name observed =
  Alcotest.(check int) (name ^ " fold count") 3 (Array.length observed);
  Array.iteri
    (fun fold (train, test) ->
      Alcotest.(check (array int))
        (name ^ " train")
        (fold_rows fixture (name ^ "_train") fold)
        (Row_view.indices train);
      Alcotest.(check (array int))
        (name ^ " test")
        (fold_rows fixture (name ^ "_test") fold)
        (Row_view.indices test))
    observed

let test_k_fold fixture () =
  let samples = (vector fixture "k_fold_sample_count").(0) in
  let specification = K_fold.create ~folds:3 () |> get in
  let observed =
    K_fold.split specification ~rng:(rng ()) ~x:(matrix samples) ~y:None ()
    |> get
  in
  check_splits fixture "k_fold" observed

let test_stratified fixture () =
  let target = vector fixture "stratified_target" in
  let specification = Stratified_k_fold.create ~folds:3 () |> get in
  let observed =
    Stratified_k_fold.split specification ~rng:(rng ())
      ~x:(matrix (Array.length target))
      ~y:(Some (Target.classification target))
      ()
    |> get
  in
  check_splits fixture "stratified" observed

let test_group fixture () =
  let values = vector fixture "group_values" in
  let groups =
    Groups.create ~expected_length:(Array.length values) values |> get_data
  in
  let specification = Group_k_fold.create ~folds:3 () |> get in
  let observed =
    Group_k_fold.split specification ~rng:(rng ()) ~groups
      ~x:(matrix (Array.length values))
      ~y:None ()
    |> get
  in
  check_splits fixture "group" observed

let test_time fixture () =
  let samples = (vector fixture "time_sample_count").(0) in
  let specification =
    Time_series_split.create ~folds:3 ~test_size:2 ~gap:1 () |> get
  in
  let observed =
    Time_series_split.split specification ~rng:(rng ()) ~x:(matrix samples)
      ~y:None ()
    |> get
  in
  check_splits fixture "time" observed

let () =
  let fixture_path =
    match Sys.getenv_opt "MODELKIT_SKLEARN_SPLITTER_FIXTURE" with
    | Some path -> path
    | None -> Alcotest.fail "MODELKIT_SKLEARN_SPLITTER_FIXTURE is not set"
  in
  let fixture = read_fixture fixture_path in
  Alcotest.run "sklearn splitter fixtures"
    [
      ( "exact membership",
        [
          Alcotest.test_case "K-fold" `Quick (test_k_fold fixture);
          Alcotest.test_case "stratified K-fold" `Quick
            (test_stratified fixture);
          Alcotest.test_case "group K-fold" `Quick (test_group fixture);
          Alcotest.test_case "time-series" `Quick (test_time fixture);
        ] );
    ]
