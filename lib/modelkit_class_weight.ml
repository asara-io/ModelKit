open Modelkit_data

module Class_weight = struct
  let ( let* ) = Result.bind

  type t = Balanced | Explicit of (int * float) array

  let balanced = Balanced

  let validation ~reason ~remediation =
    Error.make ~remediation
      (Error.Validation { name = "class weights"; reason })

  let explicit weights =
    let weights = Array.of_list weights in
    let sorted = Array.copy weights in
    Array.sort (fun (left, _) (right, _) -> Int.compare left right) sorted;
    let rec distinct index =
      index >= Array.length sorted
      || (fst sorted.(index - 1) <> fst sorted.(index) && distinct (index + 1))
    in
    if Array.length weights = 0 then
      Error
        (validation
           ~reason:"explicit class weights must name at least one class"
           ~remediation:"list every class whose weight differs from one")
    else if not (distinct 1) then
      Error
        (validation ~reason:"explicit class weights repeat a class label"
           ~remediation:"give each class label at most one weight")
    else if
      Array.exists
        (fun (_, weight) -> (not (Float.is_finite weight)) || weight < 0.0)
        weights
    then
      Error
        (validation
           ~reason:"explicit class weights must be finite and non-negative"
           ~remediation:"choose finite non-negative class weights")
    else Ok (Explicit sorted)

  let validate_length sample_weight target =
    match sample_weight with
    | None -> Ok ()
    | Some weights ->
        let expected = Target.length target in
        let observed = Sample_weight.length weights in
        if expected = observed then Ok ()
        else
          Error
            (Error.of_data_error
               ~remediation:"provide one sample weight per training row"
               (Data_error.Length_mismatch
                  { name = "class-weight sample weights"; expected; observed }))

  let row_weight sample_weight row =
    match sample_weight with
    | None -> 1.0
    | Some weights -> Sample_weight.get weights row

  (* Weighted class frequencies: classes carrying no positive weight receive no
     entry because their rows contribute nothing to fitting. *)
  let weighted_counts sample_weight labels =
    let totals = Hashtbl.create 8 in
    Array.iteri
      (fun row label ->
        let weight = row_weight sample_weight row in
        if weight > 0.0 then
          let current =
            Option.value ~default:0.0 (Hashtbl.find_opt totals label)
          in
          Hashtbl.replace totals label (current +. weight))
      labels;
    let counts = Hashtbl.to_seq totals |> Array.of_seq in
    Array.sort (fun (left, _) (right, _) -> Int.compare left right) counts;
    counts

  let class_weights specification ?sample_weight target =
    let* () = validate_length sample_weight target in
    let labels = Target.classification_values target in
    let counts = weighted_counts sample_weight labels in
    if Array.length counts = 0 then
      Error
        (validation ~reason:"no training row carries positive weight"
           ~remediation:"provide at least one positively weighted row")
    else
      match specification with
      | Balanced ->
          let total =
            Array.fold_left (fun sum (_, count) -> sum +. count) 0.0 counts
          in
          let classes = Float.of_int (Array.length counts) in
          Ok
            (Array.map
               (fun (label, count) -> (label, total /. (classes *. count)))
               counts)
      | Explicit weights ->
          Ok
            (Array.map
               (fun (label, _) ->
                 let weight =
                   Array.fold_left
                     (fun current (candidate, weight) ->
                       if candidate = label then weight else current)
                     1.0 weights
                 in
                 (label, weight))
               counts)

  let resolve specification ?sample_weight target =
    let* weights = class_weights specification ?sample_weight target in
    let lookup = Hashtbl.create (Array.length weights) in
    Array.iter
      (fun (label, weight) -> Hashtbl.replace lookup label weight)
      weights;
    let labels = Target.classification_values target in
    let values =
      Array.mapi
        (fun row label ->
          let base = row_weight sample_weight row in
          if base > 0.0 then
            base *. Option.value ~default:1.0 (Hashtbl.find_opt lookup label)
          else 0.0)
        labels
    in
    match
      Sample_weight.of_array ~expected_length:(Array.length values) values
    with
    | Ok weights -> Ok weights
    | Error error ->
        Error
          (Error.of_data_error
             ~remediation:"report invalid resolved class weights" error)
end
