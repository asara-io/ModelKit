open Modelkit
open Evaluation_metadata_support

let fixture () =
  In_channel.with_open_text (Sys.getenv "MODELKIT_RESAMPLING_FIXTURE")
    (fun channel ->
      In_channel.input_lines channel
      |> List.filter_map (fun line ->
          match String.split_on_char '\t' line with
          | [ name; values ] ->
              Some
                ( name,
                  String.split_on_char ',' values
                  |> List.map int_of_string |> Array.of_list )
          | _ -> None))

let test_fixture () =
  let values = fixture () in
  let read name = List.assoc name values in
  let check name observed =
    Alcotest.(check (array int)) name (read name) observed
  in
  let labels = read "labels" in
  let x =
    Matrix.init ~rows:(Array.length labels) ~columns:1 (fun i _ ->
        float_of_int i)
    |> data
  in
  let rng = Rng.create (Seed.of_int 1729) in
  let small =
    Matrix.init ~rows:11 ~columns:1 (fun i _ -> float_of_int i) |> data
  in
  let holdout =
    Holdout.split
      (Holdout.create ~shuffle:false ~test_size:(Split_size.Fraction 0.3) ()
      |> get)
      ~rng ~x:small ~y:None ()
    |> get
  in
  check "holdout_train" (fst holdout.(0) |> Row_view.indices);
  check "holdout_test" (snd holdout.(0) |> Row_view.indices);
  let sizes name result =
    Array.iteri
      (fun i (train, test) ->
        check
          (name ^ string_of_int i)
          [| Row_view.length train; Row_view.length test |])
      result
  in
  let counts view =
    let counts = Array.make 3 0 in
    Array.iter
      (fun row -> counts.(labels.(row)) <- counts.(labels.(row)) + 1)
      (Row_view.indices view);
    counts
  in
  Shuffle_split.split
    (Shuffle_split.create ~splits:3 ~test_size:(Split_size.Fraction 0.3) ()
    |> get)
    ~rng ~x ~y:None ()
  |> get |> sizes "shuffle_sizes_";
  Stratified_shuffle_split.split
    (Stratified_shuffle_split.create ~splits:3 ~train_size:(Split_size.Count 20)
       ~test_size:(Split_size.Count 10) ()
    |> get)
    ~rng ~x
    ~y:(Some (Target.classification labels))
    ()
  |> get
  |> Array.iteri (fun i (train, test) ->
      check ("stratified_train_" ^ string_of_int i) (counts train);
      check ("stratified_test_" ^ string_of_int i) (counts test));
  Repeated_k_fold.split
    (Repeated_k_fold.create ~folds:4 ~repeats:2 () |> get)
    ~rng ~x ~y:None ()
  |> get |> sizes "repeated_sizes_";
  Repeated_stratified_k_fold.split
    (Repeated_stratified_k_fold.create ~folds:4 ~repeats:2 () |> get)
    ~rng ~x
    ~y:(Some (Target.classification labels))
    ()
  |> get
  |> Array.iteri (fun i (_, test) ->
      check ("repeated_stratified_test_" ^ string_of_int i) (counts test))

let () =
  Alcotest.run "Resampling reference"
    [
      ("sklearn", [ ("partition and allocation parity", `Quick, test_fixture) ]);
    ]
