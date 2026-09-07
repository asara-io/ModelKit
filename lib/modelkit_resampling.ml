open Modelkit_data
open Modelkit_protocols
open Modelkit_splitting

module Split_size = struct
  type t = Count of int | Fraction of float
end

module Internal = struct
  let ( let* ) = Result.bind

  let invalid reason =
    Error
      (Error.make
         (Error.Validation { name = "resampling"; reason })
         ~remediation:
           "use positive feasible partition sizes and repetition counts with \
            aligned labels")

  let validate_size = function
    | Split_size.Count n ->
        if n > 0 then Ok () else invalid "row counts must be positive"
    | Split_size.Fraction f ->
        if Float.is_finite f && f > 0. && f < 1. then Ok ()
        else
          invalid "fractions must be finite and strictly between zero and one"

  let[@warning "-4"] validate_sizes train test =
    let* () = Option.fold ~none:(Ok ()) ~some:validate_size train in
    let* () = Option.fold ~none:(Ok ()) ~some:validate_size test in
    match (train, test) with
    | Some (Split_size.Fraction a), Some (Split_size.Fraction b)
      when a +. b > 1. ->
        invalid "training and test fractions exceed one"
    | _ -> Ok ()

  let sizes ~default_test ~train_size ~test_size n =
    let* () = validate_sizes train_size test_size in
    let test_size =
      if train_size = None && test_size = None then
        Some (Split_size.Fraction default_test)
      else test_size
    in
    let count round = function
      | Split_size.Count n -> n
      | Split_size.Fraction f -> int_of_float (round (f *. float_of_int n))
    in
    let train = Option.map (count floor) train_size in
    let test = Option.map (count ceil) test_size in
    let train, test =
      match (train, test) with
      | Some a, Some b -> (a, b)
      | Some a, None -> (a, n - a)
      | None, Some b -> (n - b, b)
      | None, None -> assert false
    in
    if train <= 0 || test <= 0 || train > n || test > n - train then
      invalid "partition sizes leave empty or overlapping partitions"
    else Ok (train, test)

  let positive n =
    if n > 0 && n <= Sys.max_array_length then Ok ()
    else
      invalid "split/repetition count exceeds array limits or is not positive"

  let child rng operation index =
    Rng.create (Seed.derive (Rng.to_seed rng) ~operation ~index)

  let shuffle_in_place rng a =
    let state = ref rng in
    let rec bounded bound =
      let bits, next = Rng.next_int64 !state in
      state := next;
      let draw = Int64.shift_right_logical bits 1 in
      let bound = Int64.of_int bound in
      let limit = Int64.sub Int64.max_int (Int64.rem Int64.max_int bound) in
      if draw >= limit then bounded (Int64.to_int bound)
      else Int64.to_int (Int64.rem draw bound)
    in
    for i = Array.length a - 1 downto 1 do
      let j = bounded (i + 1) in
      let v = a.(i) in
      a.(i) <- a.(j);
      a.(j) <- v
    done

  let pair n train test =
    let* split = Split.create ~source_size:n ~train ~test in
    Ok (Split.train split, Split.test split)

  let generate count f =
    let rec loop index acc =
      if index = count then Ok (Array.of_list (List.rev acc))
      else
        let* result = f index in
        loop (index + 1) (result :: acc)
    in
    loop 0 []

  let plain ~shuffle:random ~rng ~n ~train ~test =
    let order = Array.init n Fun.id in
    if random then shuffle_in_place rng order;
    if random then pair n (Array.sub order test train) (Array.sub order 0 test)
    else pair n (Array.sub order 0 train) (Array.sub order train test)

  let classes ~n ~y =
    match y with
    | None -> invalid "stratification requires classification labels"
    | Some y when Target.length y <> n ->
        invalid "stratification labels must match feature rows"
    | Some y ->
        let labels = Target.classification_values y in
        let table = Hashtbl.create n in
        let order = ref [] in
        Array.iteri
          (fun row label ->
            match Hashtbl.find_opt table label with
            | None ->
                order := label :: !order;
                Hashtbl.add table label [ row ]
            | Some rows -> Hashtbl.replace table label (row :: rows))
          labels;
        let rows =
          List.rev !order
          |> List.map (fun label ->
              Hashtbl.find table label |> List.rev |> Array.of_list)
          |> Array.of_list
        in
        if Array.exists (fun rows -> Array.length rows < 2) rows then
          invalid "each stratification class needs at least two rows"
        else Ok rows

  let allocation ~rng counts draws =
    let total = Array.fold_left ( + ) 0 counts in
    let quotas =
      Array.map
        (fun count ->
          float_of_int count /. float_of_int total *. float_of_int draws)
        counts
    in
    let allocated =
      Array.map (fun quota -> int_of_float (floor quota)) quotas
    in
    let order = Array.init (Array.length counts) Fun.id in
    shuffle_in_place rng order;
    Array.stable_sort
      (fun a b ->
        Float.compare
          (quotas.(b) -. float_of_int allocated.(b))
          (quotas.(a) -. float_of_int allocated.(a)))
      order;
    let remaining = draws - Array.fold_left ( + ) 0 allocated in
    for i = 0 to remaining - 1 do
      allocated.(order.(i)) <- allocated.(order.(i)) + 1
    done;
    allocated

  let stratified ~rng ~n ~train ~test rows =
    let counts = Array.map Array.length rows in
    let train_counts =
      allocation
        ~rng:(child rng "stratified-shuffle-train-allocation" 0)
        counts train
    in
    let remaining =
      Array.mapi (fun i count -> count - train_counts.(i)) counts
    in
    let test_counts =
      allocation
        ~rng:(child rng "stratified-shuffle-test-allocation" 0)
        remaining test
    in
    let train_rows = Array.make train 0 and test_rows = Array.make test 0 in
    let train_pos = ref 0 and test_pos = ref 0 in
    Array.iteri
      (fun i original ->
        let order = Array.copy original in
        shuffle_in_place (child rng "stratified-shuffle-class" i) order;
        Array.blit order 0 train_rows !train_pos train_counts.(i);
        Array.blit order train_counts.(i) test_rows !test_pos test_counts.(i);
        train_pos := !train_pos + train_counts.(i);
        test_pos := !test_pos + test_counts.(i))
      rows;
    shuffle_in_place (child rng "stratified-shuffle-train-order" 0) train_rows;
    shuffle_in_place (child rng "stratified-shuffle-test-order" 0) test_rows;
    pair n train_rows test_rows

  let feasible_classes rows train test =
    if train < Array.length rows || test < Array.length rows then
      invalid "each partition must have at least as many rows as classes"
    else Ok ()
end

module Shuffle_split = struct
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(splits = 10) ?train_size ?test_size () =
    let open Internal in
    let* () = positive splits in
    let* () = validate_sizes train_size test_size in
    Ok { splits; train_size; test_size }

  let clone t = t
  let params t = t

  let split t ~rng ?groups:_ ~x ~y:_ () =
    let open Internal in
    let n = Matrix.rows x in
    let* train, test =
      sizes ~default_test:0.1 ~train_size:t.train_size ~test_size:t.test_size n
    in
    generate t.splits (fun index ->
        plain ~shuffle:true
          ~rng:(child rng "shuffle-split" index)
          ~n ~train ~test)
end

module Stratified_shuffle_split = struct
  type params = {
    splits : int;
    train_size : Split_size.t option;
    test_size : Split_size.t option;
  }

  type t = params
  type target = Target.classification Target.t
  type rng = Rng.t

  let create ?(splits = 10) ?train_size ?test_size () =
    let open Internal in
    let* () = positive splits in
    let* () = validate_sizes train_size test_size in
    Ok { splits; train_size; test_size }

  let clone t = t
  let params t = t

  let split t ~rng ?groups:_ ~x ~y () =
    let open Internal in
    let n = Matrix.rows x in
    let* train, test =
      sizes ~default_test:0.1 ~train_size:t.train_size ~test_size:t.test_size n
    in
    let* rows = classes ~n ~y in
    let* () = feasible_classes rows train test in
    generate t.splits (fun index ->
        stratified
          ~rng:(child rng "stratified-shuffle-split" index)
          ~n ~train ~test rows)
end

module Holdout = struct
  type params = {
    train_size : Split_size.t option;
    test_size : Split_size.t option;
    shuffle : bool;
  }

  type t = params
  type target = unit
  type rng = Rng.t

  let create ?train_size ?test_size ?(shuffle = true) () =
    Result.map
      (fun () -> { train_size; test_size; shuffle })
      (Internal.validate_sizes train_size test_size)

  let clone t = t
  let params t = t

  let split t ~rng ?groups:_ ~x ~y:_ () =
    let open Internal in
    let n = Matrix.rows x in
    let* train, test =
      sizes ~default_test:0.25 ~train_size:t.train_size ~test_size:t.test_size n
    in
    let* result = plain ~shuffle:t.shuffle ~rng ~n ~train ~test in
    Ok [| result |]
end

module Repeated_internal = struct
  type params = { folds : int; repeats : int }

  let create ~folds ~repeats =
    let open Internal in
    let* () = positive repeats in
    if folds < 2 || folds > Sys.max_array_length / repeats then
      invalid
        "fold count must be at least two and total splits must fit an array"
    else Ok { folds; repeats }

  let run t ~rng ~operation f =
    let open Internal in
    let* batches =
      generate t.repeats (fun index -> f (child rng operation index))
    in
    Ok (Array.concat (Array.to_list batches))
end

module Repeated_k_fold = struct
  type params = Repeated_internal.params = { folds : int; repeats : int }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) ?(repeats = 10) () =
    Repeated_internal.create ~folds ~repeats

  let clone t = t
  let params t = t

  let split t ~rng ?groups ~x ~y () =
    let open Internal in
    let* base = K_fold.create ~folds:t.folds ~shuffle:true () in
    Repeated_internal.run t ~rng ~operation:"repeated-k-fold" (fun rng ->
        K_fold.split base ~rng ?groups ~x ~y ())
end

module Repeated_stratified_k_fold = struct
  type params = Repeated_internal.params = { folds : int; repeats : int }
  type t = params
  type target = Target.classification Target.t
  type rng = Rng.t

  let create ?(folds = 5) ?(repeats = 10) () =
    Repeated_internal.create ~folds ~repeats

  let clone t = t
  let params t = t

  let split t ~rng ?groups ~x ~y () =
    let open Internal in
    let* base = Stratified_k_fold.create ~folds:t.folds ~shuffle:true () in
    Repeated_internal.run t ~rng ~operation:"repeated-stratified-k-fold"
      (fun rng -> Stratified_k_fold.split base ~rng ?groups ~x ~y ())
end

module Train_test_split = struct
  let split ?train_size ?test_size ?(shuffle = true) ?stratify ~rng dataset () =
    let open Internal in
    let n = Dataset.sample_count dataset in
    let* train, test = sizes ~default_test:0.25 ~train_size ~test_size n in
    let* train, test =
      match stratify with
      | None -> plain ~shuffle ~rng ~n ~train ~test
      | Some _ when not shuffle ->
          invalid "stratification requires shuffled splitting"
      | Some _ ->
          let* rows = classes ~n ~y:stratify in
          let* () = feasible_classes rows train test in
          stratified ~rng ~n ~train ~test rows
    in
    let* split = Split.of_views ~train ~test in
    Split.materialize dataset split
end
