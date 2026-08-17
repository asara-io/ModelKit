open Modelkit

type thread_limit = { variable : string; threads : int }

type thread_limit_source =
  | Assumed_sequential
  | Explicit
  | Environment of thread_limit array

type warning =
  | Invalid_environment_limit of { variable : string; value : string }
  | Conflicting_environment_limits of thread_limit array
  | Domains_exceed_recommended of {
      requested_domains : int;
      recommended_domains : int;
    }
  | Nested_parallelism of { fold_domains : int; inner_threads : int }
  | Potential_oversubscription of {
      estimated_runnable_threads : int;
      recommended_domains : int;
    }

type diagnostics = {
  fold_domains : int;
  recommended_domains : int;
  inner_threads : int;
  inner_thread_source : thread_limit_source;
  estimated_runnable_threads : int;
  warnings : warning array;
}

type t = { domains : int; diagnostics : diagnostics }

let thread_variables =
  [|
    "BLIS_NUM_THREADS";
    "MKL_NUM_THREADS";
    "OMP_NUM_THREADS";
    "OPENBLAS_NUM_THREADS";
    "VECLIB_MAXIMUM_THREADS";
  |]

let validation ~name ~reason ~remediation =
  Error.make ~remediation (Error.Validation { name; reason })

let environment_limits () =
  Array.fold_left
    (fun (limits, invalid) variable ->
      match Sys.getenv_opt variable with
      | None -> (limits, invalid)
      | Some value -> (
          match int_of_string_opt value with
          | Some threads when threads > 0 ->
              ({ variable; threads } :: limits, invalid)
          | None | Some _ ->
              (limits, Invalid_environment_limit { variable; value } :: invalid)
          ))
    ([], []) thread_variables
  |> fun (limits, invalid) -> (Array.of_list (List.rev limits), List.rev invalid)

let distinct_thread_counts limits =
  Array.fold_left
    (fun counts limit ->
      if List.mem limit.threads counts then counts else limit.threads :: counts)
    [] limits

let saturated_product left right =
  if right > max_int / left then max_int else left * right

let warning_to_string = function
  | Invalid_environment_limit { variable; value } ->
      Format.sprintf "%s=%S is not a positive thread count" variable value
  | Conflicting_environment_limits limits ->
      limits |> Array.to_list
      |> List.map (fun limit ->
          Format.sprintf "%s=%d" limit.variable limit.threads)
      |> String.concat ", "
      |> Format.sprintf "thread-limit environment variables conflict: %s"
  | Domains_exceed_recommended { requested_domains; recommended_domains } ->
      Format.sprintf
        "%d fold domains exceed the runtime recommendation of %d domains"
        requested_domains recommended_domains
  | Nested_parallelism { fold_domains; inner_threads } ->
      Format.sprintf
        "%d fold domains each permit %d inner threads; prefer one inner thread \
         per fold"
        fold_domains inner_threads
  | Potential_oversubscription
      { estimated_runnable_threads; recommended_domains } ->
      Format.sprintf
        "up to %d runnable threads may exceed the runtime recommendation of %d"
        estimated_runnable_threads recommended_domains

let create ?inner_threads ~domains () =
  if domains < 1 then
    Error
      (validation ~name:"parallel fold domains" ~reason:"must be at least one"
         ~remediation:"request one domain for sequential fallback or more")
  else
    match inner_threads with
    | Some threads when threads < 1 ->
        Error
          (validation ~name:"inner thread limit" ~reason:"must be at least one"
             ~remediation:"use one numerical-library thread per fold domain")
    | Some inner_threads ->
        let recommended_domains = max 1 (Domain.recommended_domain_count ()) in
        let estimated_runnable_threads =
          saturated_product domains inner_threads
        in
        let warnings =
          [
            (if domains > recommended_domains then
               Some
                 (Domains_exceed_recommended
                    { requested_domains = domains; recommended_domains })
             else None);
            (if domains > 1 && inner_threads > 1 then
               Some
                 (Nested_parallelism { fold_domains = domains; inner_threads })
             else None);
            (if estimated_runnable_threads > recommended_domains then
               Some
                 (Potential_oversubscription
                    { estimated_runnable_threads; recommended_domains })
             else None);
          ]
          |> List.filter_map Fun.id |> Array.of_list
        in
        Ok
          {
            domains;
            diagnostics =
              {
                fold_domains = domains;
                recommended_domains;
                inner_threads;
                inner_thread_source = Explicit;
                estimated_runnable_threads;
                warnings;
              };
          }
    | None ->
        let limits, invalid = environment_limits () in
        let inner_threads =
          Array.fold_left
            (fun maximum limit -> max maximum limit.threads)
            1 limits
        in
        let recommended_domains = max 1 (Domain.recommended_domain_count ()) in
        let estimated_runnable_threads =
          saturated_product domains inner_threads
        in
        let conflicting = List.length (distinct_thread_counts limits) > 1 in
        let warnings =
          invalid
          @ ([
               (if conflicting then Some (Conflicting_environment_limits limits)
                else None);
               (if domains > recommended_domains then
                  Some
                    (Domains_exceed_recommended
                       { requested_domains = domains; recommended_domains })
                else None);
               (if domains > 1 && inner_threads > 1 then
                  Some
                    (Nested_parallelism
                       { fold_domains = domains; inner_threads })
                else None);
               (if estimated_runnable_threads > recommended_domains then
                  Some
                    (Potential_oversubscription
                       { estimated_runnable_threads; recommended_domains })
                else None);
             ]
            |> List.filter_map Fun.id)
        in
        Ok
          {
            domains;
            diagnostics =
              {
                fold_domains = domains;
                recommended_domains;
                inner_threads;
                inner_thread_source =
                  (if Array.length limits = 0 then Assumed_sequential
                   else Environment limits);
                estimated_runnable_threads;
                warnings = Array.of_list warnings;
              };
          }

let diagnostics configuration = configuration.diagnostics
let concurrency configuration = configuration.domains

let lower_failure lowest index =
  let rec update () =
    let observed = Atomic.get lowest in
    if index >= observed then ()
    else if not (Atomic.compare_and_set lowest observed index) then update ()
  in
  update ()

let map configuration ~f inputs =
  if configuration.domains = 1 then
    Sequential_execution.map Sequential_execution.default ~f inputs
  else
    let count = Array.length inputs in
    if count = 0 then Ok [||]
    else
      let outputs = Array.init count (fun _ -> Atomic.make None) in
      let errors = Array.init count (fun _ -> Atomic.make None) in
      let lowest_failure = Atomic.make count in
      let pool =
        Domainslib.Task.setup_pool ~num_domains:(configuration.domains - 1) ()
      in
      Fun.protect
        ~finally:(fun () -> Domainslib.Task.teardown_pool pool)
        (fun () ->
          Domainslib.Task.run pool (fun () ->
              Domainslib.Task.parallel_for ~chunk_size:1 ~start:0
                ~finish:(count - 1)
                ~body:(fun index ->
                  if index <= Atomic.get lowest_failure then
                    match f ~index inputs.(index) with
                    | Ok output -> Atomic.set outputs.(index) (Some output)
                    | Error error ->
                        Atomic.set errors.(index) (Some error);
                        lower_failure lowest_failure index)
                pool);
          let failed = Atomic.get lowest_failure in
          if failed < count then
            match Atomic.get errors.(failed) with
            | Some error -> Error error
            | None -> assert false
          else
            Ok
              (Array.map
                 (fun output ->
                   match Atomic.get output with
                   | Some value -> value
                   | None -> assert false)
                 outputs))

let execution configuration =
  Execution.of_backend
    (module struct
      type nonrec t = t

      let concurrency = concurrency
      let map = map
    end)
    configuration
