open Modelkit_data
open Modelkit_protocols

module Split = struct
  type t = {
    split_source_size : int;
    split_train : Row_view.t;
    split_test : Row_view.t;
  }

  let validation ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name = "split"; reason })

  let validate_partition ~source_size ~name ~seen view =
    if Row_view.source_size view <> source_size then
      Error
        (validation
           ~reason:
             (Format.sprintf "%s rows refer to source size %d instead of %d"
                name
                (Row_view.source_size view)
                source_size)
           ~remediation:"construct both partitions from the same aligned source")
    else if Row_view.length view = 0 then
      Error
        (validation ~reason:(name ^ " rows are empty")
           ~remediation:"provide at least one row in each split partition")
    else
      let rec loop position =
        if position = Row_view.length view then Ok ()
        else
          let row = Row_view.get view position in
          if seen.(row) then
            Error
              (validation
                 ~reason:
                   (Format.sprintf "%s row %d occurs more than once" name row)
                 ~remediation:"remove duplicate rows from each split partition")
          else (
            seen.(row) <- true;
            loop (position + 1))
      in
      loop 0

  let of_views ~train ~test =
    let source_size = Row_view.source_size train in
    let train_seen = Array.make source_size false in
    let test_seen = Array.make source_size false in
    let ( let* ) = Result.bind in
    let* () =
      validate_partition ~source_size ~name:"training" ~seen:train_seen train
    in
    let* () =
      validate_partition ~source_size ~name:"test" ~seen:test_seen test
    in
    let rec disjoint row =
      if row = source_size then Ok ()
      else if train_seen.(row) && test_seen.(row) then
        Error
          (validation
             ~reason:
               (Format.sprintf "row %d occurs in both training and test data"
                  row)
             ~remediation:"make the training and test partitions disjoint")
      else disjoint (row + 1)
    in
    let* () = disjoint 0 in
    Ok
      {
        split_source_size = source_size;
        split_train = train;
        split_test = test;
      }

  let create ~source_size ~train ~test =
    let ( let* ) = Result.bind in
    let* train =
      Row_view.create ~source_size train
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide valid training row indices"
            error)
    in
    let* test =
      Row_view.create ~source_size test
      |> Result.map_error (fun error ->
          Error.of_data_error ~remediation:"provide valid test row indices"
            error)
    in
    of_views ~train ~test

  let train split = split.split_train
  let test split = split.split_test
  let pair split = (split.split_train, split.split_test)

  let materialize dataset split =
    let expected = Dataset.sample_count dataset in
    if split.split_source_size <> expected then
      Error
        (validation
           ~reason:
             (Format.sprintf
                "split source size is %d but the dataset has %d rows"
                split.split_source_size expected)
           ~remediation:"use a split generated for this dataset")
    else
      let ( let* ) = Result.bind in
      let materialize_partition rows =
        let* view =
          Dataset.view dataset rows
          |> Result.map_error (fun error ->
              Error.of_data_error
                ~remediation:"use rows aligned with the source dataset" error)
        in
        Dataset.materialize view
        |> Result.map_error (fun error ->
            Error.of_data_error
              ~remediation:
                "ensure selected targets, weights, and groups remain valid"
              error)
      in
      let* train = materialize_partition split.split_train in
      let* test = materialize_partition split.split_test in
      Ok (train, test)
end

module Splitter_internal = struct
  let ( let* ) = Result.bind

  let validation ~name ~reason ~remediation =
    Error.make ~remediation (Error.Validation { name; reason })

  let validate_folds ~name folds =
    if folds >= 2 then Ok ()
    else
      Error
        (validation ~name ~reason:"fold count must be at least two"
           ~remediation:"choose two or more folds")

  let validate_sample_count ~name ~folds sample_count =
    if folds <= sample_count then Ok ()
    else
      Error
        (validation ~name
           ~reason:
             (Format.sprintf "%d folds exceed the %d available samples" folds
                sample_count)
           ~remediation:"reduce the fold count or provide more samples")

  let validate_aligned_length ~name ~expected observed =
    if expected = observed then Ok ()
    else
      Error
        (Error.make
           ~remediation:("provide " ^ name ^ " aligned to feature rows")
           (Error.Shape_mismatch
              { name; expected = [ expected ]; observed = [ observed ] }))

  let shuffle rng values =
    let state = ref rng in
    for upper = Array.length values - 1 downto 1 do
      let random, successor = Rng.next_int64 !state in
      state := successor;
      let nonnegative = Int64.shift_right_logical random 1 in
      let selected =
        Int64.rem nonnegative (Int64.of_int (upper + 1)) |> Int64.to_int
      in
      let value = values.(upper) in
      values.(upper) <- values.(selected);
      values.(selected) <- value
    done

  let split_from_assignments ~source_size ~folds assignments =
    Array.init folds (fun fold ->
        let train_count = ref 0 in
        let test_count = ref 0 in
        Array.iter
          (fun assigned ->
            if assigned = fold then incr test_count else incr train_count)
          assignments;
        let train = Array.make !train_count 0 in
        let test = Array.make !test_count 0 in
        let train_position = ref 0 in
        let test_position = ref 0 in
        Array.iteri
          (fun row assigned ->
            if assigned = fold then (
              test.(!test_position) <- row;
              incr test_position)
            else (
              train.(!train_position) <- row;
              incr train_position))
          assignments;
        Split.create ~source_size ~train ~test)
    |> Array.fold_left
         (fun accumulated split ->
           let* values = accumulated in
           let* value = split in
           Ok (value :: values))
         (Ok [])
    |> Result.map (fun reversed ->
        reversed |> List.rev_map Split.pair |> Array.of_list)

  let balanced_assignments ~sample_count ~folds ~shuffle:rng =
    let order = Array.init sample_count Fun.id in
    Option.iter (fun state -> shuffle state order) rng;
    let assignments = Array.make sample_count 0 in
    let base_size = sample_count / folds in
    let remainder = sample_count mod folds in
    let offset = ref 0 in
    for fold = 0 to folds - 1 do
      let size = base_size + if fold < remainder then 1 else 0 in
      for position = !offset to !offset + size - 1 do
        assignments.(order.(position)) <- fold
      done;
      offset := !offset + size
    done;
    assignments
end

module K_fold = struct
  type params = { folds : int; shuffle : bool }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) ?(shuffle = false) () =
    Result.map
      (fun () -> { folds; shuffle })
      (Splitter_internal.validate_folds ~name:"K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_sample_count ~name:"K-fold"
        ~folds:specification.folds sample_count
    in
    let shuffle = if specification.shuffle then Some rng else None in
    let assignments =
      Splitter_internal.balanced_assignments ~sample_count
        ~folds:specification.folds ~shuffle
    in
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Stratified_k_fold = struct
  type params = { folds : int; shuffle : bool }
  type t = params
  type target = Target.classification Target.t
  type rng = Rng.t

  let create ?(folds = 5) ?(shuffle = false) () =
    Result.map
      (fun () -> { folds; shuffle })
      (Splitter_internal.validate_folds ~name:"stratified K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng ?groups:_ ~x ~y () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_sample_count ~name:"stratified K-fold"
        ~folds:specification.folds sample_count
    in
    let* target =
      match y with
      | Some target -> Ok target
      | None ->
          Error
            (Splitter_internal.validation ~name:"stratified K-fold target"
               ~reason:"a classification target is required"
               ~remediation:"provide aligned integer class labels")
    in
    let labels = Target.classification_values target in
    let* () =
      Splitter_internal.validate_aligned_length ~name:"stratified K-fold target"
        ~expected:sample_count (Array.length labels)
    in
    let class_by_label = Hashtbl.create sample_count in
    let encoded = Array.make sample_count 0 in
    let counts = Array.make sample_count 0 in
    let class_count = ref 0 in
    for row = 0 to sample_count - 1 do
      let class_index =
        match Hashtbl.find_opt class_by_label labels.(row) with
        | Some index -> index
        | None ->
            let index = !class_count in
            incr class_count;
            Hashtbl.add class_by_label labels.(row) index;
            index
      in
      encoded.(row) <- class_index;
      counts.(class_index) <- counts.(class_index) + 1
    done;
    let assignments = Array.make sample_count 0 in
    let class_offset = ref 0 in
    for class_index = 0 to !class_count - 1 do
      let allocation = Array.make specification.folds 0 in
      for position = 0 to counts.(class_index) - 1 do
        let fold = (!class_offset + position) mod specification.folds in
        allocation.(fold) <- allocation.(fold) + 1
      done;
      let class_assignments = Array.make counts.(class_index) 0 in
      let position = ref 0 in
      for fold = 0 to specification.folds - 1 do
        for _ = 1 to allocation.(fold) do
          class_assignments.(!position) <- fold;
          incr position
        done
      done;
      (if specification.shuffle then
         let class_rng =
           Seed.derive (Rng.to_seed rng) ~operation:"stratified-k-fold-class"
             ~index:class_index
           |> Rng.create
         in
         Splitter_internal.shuffle class_rng class_assignments);
      let position = ref 0 in
      for row = 0 to sample_count - 1 do
        if encoded.(row) = class_index then (
          assignments.(row) <- class_assignments.(!position);
          incr position)
      done;
      class_offset := !class_offset + counts.(class_index)
    done;
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Group_k_fold = struct
  type params = { folds : int }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) () =
    Result.map
      (fun () -> { folds })
      (Splitter_internal.validate_folds ~name:"group K-fold" folds)

  let clone specification = specification
  let params specification = specification

  let split specification ~rng:_ ?groups ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let ( let* ) = Result.bind in
    let* groups =
      match groups with
      | Some groups -> Ok groups
      | None ->
          Error
            (Splitter_internal.validation ~name:"group K-fold groups"
               ~reason:"group labels are required"
               ~remediation:"provide one integer group per feature row")
    in
    let* () =
      Splitter_internal.validate_aligned_length ~name:"group K-fold groups"
        ~expected:sample_count (Groups.length groups)
    in
    let counts_by_group = Hashtbl.create sample_count in
    for row = 0 to sample_count - 1 do
      let group = Groups.get groups row in
      let count =
        Option.value (Hashtbl.find_opt counts_by_group group) ~default:0
      in
      Hashtbl.replace counts_by_group group (count + 1)
    done;
    let group_ids = Hashtbl.to_seq_keys counts_by_group |> Array.of_seq in
    let* () =
      Splitter_internal.validate_sample_count
        ~name:"group K-fold distinct groups" ~folds:specification.folds
        (Array.length group_ids)
    in
    Array.sort
      (fun left right ->
        let by_count =
          Int.compare
            (Hashtbl.find counts_by_group right)
            (Hashtbl.find counts_by_group left)
        in
        if by_count <> 0 then by_count else Int.compare right left)
      group_ids;
    let fold_sizes = Array.make specification.folds 0 in
    let fold_by_group = Hashtbl.create (Array.length group_ids) in
    Array.iter
      (fun group ->
        let selected = ref 0 in
        for fold = 1 to specification.folds - 1 do
          if fold_sizes.(fold) < fold_sizes.(!selected) then selected := fold
        done;
        Hashtbl.add fold_by_group group !selected;
        fold_sizes.(!selected) <-
          fold_sizes.(!selected) + Hashtbl.find counts_by_group group)
      group_ids;
    let assignments =
      Array.init sample_count (fun row ->
          Hashtbl.find fold_by_group (Groups.get groups row))
    in
    Splitter_internal.split_from_assignments ~source_size:sample_count
      ~folds:specification.folds assignments
end

module Time_series_split = struct
  type params = { folds : int; test_size : int option; gap : int }
  type t = params
  type target = unit
  type rng = Rng.t

  let create ?(folds = 5) ?test_size ?(gap = 0) () =
    let ( let* ) = Result.bind in
    let* () =
      Splitter_internal.validate_folds ~name:"time-series split" folds
    in
    let* () =
      match test_size with
      | None -> Ok ()
      | Some size when size > 0 -> Ok ()
      | Some _ ->
          Error
            (Splitter_internal.validation ~name:"time-series test size"
               ~reason:"test size must be positive"
               ~remediation:"choose at least one test row")
    in
    if gap >= 0 then Ok { folds; test_size; gap }
    else
      Error
        (Splitter_internal.validation ~name:"time-series gap"
           ~reason:"gap must be non-negative"
           ~remediation:"choose zero or more excluded rows")

  let clone specification = specification
  let params specification = specification

  let split specification ~rng:_ ?groups:_ ~x ~y:_ () =
    let sample_count = Matrix.rows x in
    let test_size =
      Option.value specification.test_size
        ~default:(sample_count / (specification.folds + 1))
    in
    if test_size <= 0 then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:"the default test window is empty"
           ~remediation:"provide more samples or an explicit positive test size")
    else if specification.gap >= sample_count then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:"the gap leaves no initial training rows"
           ~remediation:"reduce the gap or provide more samples")
    else if
      test_size > (sample_count - specification.gap - 1) / specification.folds
    then
      Error
        (Splitter_internal.validation ~name:"time-series split"
           ~reason:
             "fold count, test size, and gap leave no initial training rows"
           ~remediation:
             "reduce folds, test size, or gap, or provide more samples")
    else
      Array.init specification.folds (fun fold ->
          let test_start =
            sample_count - (specification.folds * test_size) + (fold * test_size)
          in
          let train_size = test_start - specification.gap in
          let train = Array.init train_size Fun.id in
          let test = Array.init test_size (fun offset -> test_start + offset) in
          Split.create ~source_size:sample_count ~train ~test)
      |> Array.fold_left
           (fun accumulated split ->
             let ( let* ) = Result.bind in
             let* values = accumulated in
             let* value = split in
             Ok (value :: values))
           (Ok [])
      |> Result.map (fun reversed ->
          reversed |> List.rev_map Split.pair |> Array.of_list)
end
