open Modelkit_data
open Modelkit_protocols
open Modelkit_splitting

module Internal = struct
  let ( let* ) = Result.bind

  let invalid name reason =
    Error
      (Splitter_internal.validation ~name ~reason
         ~remediation:
           "provide aligned labels and enough distinct nonempty partitions for \
            training and testing")

  let sorted_unique values =
    Array.to_list values |> List.sort_uniq Int.compare |> Array.of_list

  let encode values =
    let unique = sorted_unique values in
    let indices = Hashtbl.create (Array.length unique) in
    Array.iteri (fun index value -> Hashtbl.add indices value index) unique;
    (unique, Array.map (Hashtbl.find indices) values)

  let groups ~name ~rows = function
    | None -> invalid name "group labels are required"
    | Some groups ->
        let* () =
          Splitter_internal.validate_aligned_length ~name ~expected:rows
            (Groups.length groups)
        in
        Ok (encode (Groups.to_array groups))
end

module Predefined_split = struct
  type params = { test_folds : int array }
  type t = { assignment : int array; ids : int array }
  type target = unit
  type rng = Rng.t

  let create ~test_folds () =
    let open Internal in
    if Array.exists (fun id -> id < -1) test_folds then
      invalid "predefined split" "fold IDs must be -1 or nonnegative"
    else
      let ids =
        sorted_unique test_folds |> Array.to_list
        |> List.filter (fun id -> id >= 0)
        |> Array.of_list
      in
      if Array.length ids = 0 then
        invalid "predefined split" "at least one test fold is required"
      else if Array.length ids = 1 && not (Array.mem (-1) test_folds) then
        invalid "predefined split" "the sole test fold leaves no training rows"
      else Ok { assignment = Array.copy test_folds; ids }

  let clone t = t
  let params t = { test_folds = Array.copy t.assignment }
  let fold_ids t = Array.copy t.ids

  let split t ~rng:_ ?groups:_ ~x ~y:_ () =
    let open Internal in
    let n = Matrix.rows x in
    let* () =
      Splitter_internal.validate_aligned_length
        ~name:"predefined fold assignments" ~expected:n
        (Array.length t.assignment)
    in
    let lookup = Hashtbl.create (Array.length t.ids) in
    Array.iteri (fun i id -> Hashtbl.add lookup id i) t.ids;
    let assignment =
      Array.map
        (fun id -> if id = -1 then -1 else Hashtbl.find lookup id)
        t.assignment
    in
    Splitter_internal.split_from_assignments ~source_size:n
      ~folds:(Array.length t.ids) assignment
end

module Leave_one_out = struct
  type t = unit
  type params = unit
  type target = unit
  type rng = Rng.t

  let create () = ()
  let clone () = ()
  let params () = ()

  let split () ~rng:_ ?groups:_ ~x ~y:_ () =
    let n = Matrix.rows x in
    if n < 2 then
      Internal.invalid "leave-one-out" "at least two rows are required"
    else
      Splitter_internal.split_from_assignments ~source_size:n ~folds:n
        (Array.init n Fun.id)
end

module Leave_one_group_out = struct
  type t = unit
  type params = unit
  type target = unit
  type rng = Rng.t

  let create () = ()
  let clone () = ()
  let params () = ()

  let split () ~rng:_ ?groups:provided_groups ~x ~y:_ () =
    let open Internal in
    let n = Matrix.rows x in
    let* ids, assignment =
      groups ~name:"leave-one-group-out groups" ~rows:n provided_groups
    in
    if Array.length ids < 2 then
      invalid "leave-one-group-out" "at least two distinct groups are required"
    else
      Splitter_internal.split_from_assignments ~source_size:n
        ~folds:(Array.length ids) assignment
end

module Stratified_group_k_fold = struct
  type params = { folds : int; shuffle : bool }
  type t = params
  type target = Target.classification Target.t
  type rng = Rng.t

  type group = {
    index : int;
    size : int;
    counts : (int * int) array;
    dispersion : float;
  }

  let create ?(folds = 5) ?(shuffle = false) () =
    Result.map
      (fun () -> { folds; shuffle })
      (Splitter_internal.validate_folds ~name:"stratified-group K-fold" folds)

  let clone t = t
  let params t = t

  let summarize ~class_count ~group_count encoded_groups encoded_labels =
    let histograms = Array.init group_count (fun _ -> Hashtbl.create 4) in
    let sizes = Array.make group_count 0
    and totals = Array.make class_count 0 in
    Array.iteri
      (fun row group ->
        let label = encoded_labels.(row) in
        let histogram = histograms.(group) in
        Hashtbl.replace histogram label
          (1 + Option.value (Hashtbl.find_opt histogram label) ~default:0);
        sizes.(group) <- sizes.(group) + 1;
        totals.(label) <- totals.(label) + 1)
      encoded_groups;
    let groups =
      Array.mapi
        (fun index histogram ->
          let counts = Hashtbl.to_seq histogram |> Array.of_seq in
          Array.sort (fun (a, _) (b, _) -> Int.compare a b) counts;
          let size = sizes.(index) in
          let mean = float_of_int size /. float_of_int class_count in
          let squares =
            Array.fold_left
              (fun sum (_, count) ->
                let difference = float_of_int count -. mean in
                sum +. (difference *. difference))
              (float_of_int (class_count - Array.length counts) *. mean *. mean)
              counts
          in
          {
            index;
            size;
            counts;
            dispersion = sqrt (squares /. float_of_int class_count);
          })
        histograms
    in
    (totals, groups)

  let objective totals counts =
    let folds = Array.length counts in
    let sum = ref 0. in
    Array.iteri
      (fun label total ->
        let fraction fold =
          float_of_int counts.(fold).(label) /. float_of_int total
        in
        let mean = ref 0. in
        for fold = 0 to folds - 1 do
          mean := !mean +. fraction fold
        done;
        mean := !mean /. float_of_int folds;
        let variance = ref 0. in
        for fold = 0 to folds - 1 do
          let delta = fraction fold -. !mean in
          variance := !variance +. (delta *. delta)
        done;
        sum := !sum +. sqrt (!variance /. float_of_int folds))
      totals;
    !sum /. float_of_int (Array.length totals)

  let assign ~folds totals groups =
    let counts =
      Array.init folds (fun _ -> Array.make (Array.length totals) 0)
    in
    let sizes = Array.make folds 0 in
    let assignments = Array.make (Array.length groups) 0 in
    let empty = ref folds in
    Array.iteri
      (fun position group ->
        let remaining = Array.length groups - position in
        let best = ref (-1) and best_score = ref infinity in
        for fold = 0 to folds - 1 do
          if remaining > !empty || sizes.(fold) = 0 then (
            Array.iter
              (fun (label, n) ->
                counts.(fold).(label) <- counts.(fold).(label) + n)
              group.counts;
            let score = objective totals counts in
            Array.iter
              (fun (label, n) ->
                counts.(fold).(label) <- counts.(fold).(label) - n)
              group.counts;
            let tied = abs_float (score -. !best_score) <= 1e-12 in
            if
              !best = -1
              || (tied && sizes.(fold) < sizes.(!best))
              || ((not tied) && score < !best_score)
            then (
              best := fold;
              best_score := score))
        done;
        let selected = !best in
        if sizes.(selected) = 0 then decr empty;
        sizes.(selected) <- sizes.(selected) + group.size;
        assignments.(group.index) <- selected;
        Array.iter
          (fun (label, n) ->
            counts.(selected).(label) <- counts.(selected).(label) + n)
          group.counts)
      groups;
    assignments

  let split t ~rng ?groups:provided_groups ~x ~y () =
    let open Internal in
    let n = Matrix.rows x in
    let* group_ids, encoded_groups =
      groups ~name:"stratified-group K-fold groups" ~rows:n provided_groups
    in
    let group_count = Array.length group_ids in
    if group_count < t.folds then
      invalid "stratified-group K-fold"
        "fold count exceeds distinct group count"
    else
      let* labels =
        match y with
        | None ->
            invalid "stratified-group K-fold"
              "classification labels are required"
        | Some labels -> Ok labels
      in
      let* () =
        Splitter_internal.validate_aligned_length
          ~name:"stratified-group K-fold target" ~expected:n
          (Target.length labels)
      in
      let classes, encoded_labels =
        encode (Target.classification_values labels)
      in
      let totals, group_data =
        summarize ~class_count:(Array.length classes) ~group_count
          encoded_groups encoded_labels
      in
      if t.shuffle then Splitter_internal.shuffle rng group_data;
      Array.stable_sort
        (fun a b -> Float.compare b.dispersion a.dispersion)
        group_data;
      let assignment = assign ~folds:t.folds totals group_data in
      Splitter_internal.split_from_assignments ~source_size:n ~folds:t.folds
        (Array.map (Array.get assignment) encoded_groups)
end
