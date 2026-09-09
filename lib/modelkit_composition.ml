open Modelkit_data
open Modelkit_metadata
open Modelkit_protocols
open Modelkit_pipeline

module Internal = struct
  let ( let* ) = Result.bind

  let invalid name reason =
    Error
      (Error.make
         ~remediation:
           "use valid, unique names and selectors within the input schema"
         (Error.Validation { name; reason }))

  let data result =
    Result.map_error
      (fun error ->
        Error.of_data_error
          ~remediation:"provide valid composition dimensions and feature names"
          error)
      result

  let with_stage name = Result.map_error (Error.with_context (Error.Stage name))

  let validate_name name =
    if String.trim name = "" || name = "remainder" then
      invalid "column branch name"
        "must be non-blank and different from remainder"
    else Ok ()

  let unique ~name values =
    let seen = Hashtbl.create (Array.length values) in
    let rec loop index =
      if index = Array.length values then Ok ()
      else if Hashtbl.mem seen values.(index) then
        invalid name "duplicate selection"
      else (
        Hashtbl.add seen values.(index) ();
        loop (index + 1))
    in
    loop 0

  let schema names =
    let* names =
      Feature_names.create ~expected_count:(Array.length names) names |> data
    in
    Ok (Feature_schema.named names)

  let input_names schema =
    match Feature_schema.names schema with
    | Some names -> Feature_names.to_array names
    | None ->
        Array.init (Feature_schema.feature_count schema) (fun column ->
            "x" ^ string_of_int column)

  let payload_bytes ~rows ~columns =
    let rows = Int64.of_int rows and columns = Int64.of_int columns in
    if columns <> 0L && rows > Int64.div (Int64.div Int64.max_int 8L) columns
    then
      invalid "column allocation"
        "dense payload size exceeds the reporting limit"
    else Ok (Int64.mul 8L (Int64.mul rows columns))

  let add_bytes left right =
    if left > Int64.sub Int64.max_int right then
      invalid "column allocation"
        "cumulative selection copies exceed the reporting limit"
    else Ok (Int64.add left right)
end

module Column_selector = struct
  open Internal

  type t = All | Indices of int array | Names of string array

  let all = All

  let indices values =
    let values = Array.copy values in
    let* () = unique ~name:"column indices" values in
    if Array.exists (fun index -> index < 0) values then
      invalid "column indices" "negative indices are not supported"
    else Ok (Indices values)

  let names values =
    let values = Array.copy values in
    let* _ = schema values in
    Ok (Names values)

  let resolve selector feature_schema =
    let width = Feature_schema.feature_count feature_schema in
    match selector with
    | All ->
        if width > Sys.max_array_length then
          invalid "column selection" "schema width exceeds the array size limit"
        else Ok (Array.init width Fun.id)
    | Indices indices ->
        if Array.exists (fun index -> index >= width) indices then
          invalid "column indices" "an index is outside the input schema"
        else Ok (Array.copy indices)
    | Names selected -> (
        if Array.length selected = 0 then Ok [||]
        else
          match Feature_schema.names feature_schema with
          | None ->
              invalid "column names"
                "name selection requires a named input schema"
          | Some names ->
              let positions = Hashtbl.create width in
              Array.iteri
                (fun index name -> Hashtbl.add positions name index)
                (Feature_names.to_array names);
              let result = Array.make (Array.length selected) 0 in
              let rec loop index =
                if index = Array.length selected then Ok result
                else
                  match Hashtbl.find_opt positions selected.(index) with
                  | None ->
                      invalid "column names"
                        (Format.sprintf "feature %S is absent" selected.(index))
                  | Some position ->
                      result.(index) <- position;
                      loop (index + 1)
              in
              loop 0)
end

module Stage = struct
  open Internal

  let unsupervised (transformer : Pipeline.transformer) =
    {
      Pipeline.name = transformer.Pipeline.transformer_name;
      stage_cache_check = transformer.Pipeline.transformer_cache_check;
      stage_fit_metadata_check =
        transformer.Pipeline.transformer_fit_metadata_check;
      stage_transform_metadata_check =
        transformer.Pipeline.transformer_transform_metadata_check;
      validate_target = (fun ~x:_ ~y:_ -> Ok ());
      fit_stage =
        (fun ~cache ~metadata ~rng ~feature_schema ~x ~y:_ ->
          transformer.Pipeline.fit_transform ~cache ~metadata ~rng
            ~feature_schema ~x);
    }

  let validate_cache stages () =
    Array.fold_left
      (fun result stage ->
        let* () = result in
        with_stage stage.Pipeline.name (stage.Pipeline.stage_cache_check ()))
      (Ok ()) stages

  let validate stages ~x ~y =
    Array.fold_left
      (fun result stage ->
        let* () = result in
        with_stage stage.Pipeline.name (stage.Pipeline.validate_target ~x ~y))
      (Ok ()) stages

  let validate_fit stages metadata =
    Array.fold_left
      (fun result stage ->
        let* () = result in
        with_stage stage.Pipeline.name
          (stage.Pipeline.stage_fit_metadata_check metadata))
      (Ok ()) stages

  let validate_transform stages metadata =
    Array.fold_left
      (fun result stage ->
        let* () = result in
        with_stage stage.Pipeline.name
          (stage.Pipeline.stage_transform_metadata_check metadata))
      (Ok ()) stages

  let validate_fitted stages metadata =
    Array.fold_left
      (fun result stage ->
        let* () = result in
        with_stage stage.Pipeline.stage_name
          (stage.Pipeline.fitted_transform_metadata_check metadata))
      (Ok ()) stages

  let package ~name ~validate_cache ~validate_target ~validate_fit_metadata
      ~validate_transform_metadata ~fit_transform ~transform ~output_schema
      specification =
    if String.trim name = "" then
      invalid "pipeline stage name" "must not be blank"
    else
      let fit_stage ~cache ~metadata ~rng ~feature_schema ~x ~y =
        let metadata = Metadata.scope (Error.Stage name) metadata in
        let* fitted, output =
          fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x
            ~y ()
        in
        let schema = output_schema fitted in
        let packaged =
          {
            Pipeline.stage_name = name;
            transform_input_schema = feature_schema;
            transform_output_schema = schema;
            apply_transform =
              (fun ~metadata ~feature_schema ~x ->
                transform fitted
                  ~metadata:(Metadata.scope (Error.Stage name) metadata)
                  ~feature_schema ~x);
            fitted_transform_metadata_check = validate_transform_metadata;
            transformer_provenance = None;
            encode_transformer = None;
          }
        in
        Ok (packaged, output, schema)
      in
      Ok
        Pipeline.
          {
            name;
            stage_cache_check = validate_cache;
            validate_target;
            stage_fit_metadata_check = validate_fit_metadata;
            stage_transform_metadata_check = validate_transform_metadata;
            fit_stage;
          }

  let supervised stage =
    let validate_target ~x ~y =
      let expected = Matrix.rows x and observed = Target.length y in
      if expected <> observed then
        Error
          (Error.of_data_error
             ~remediation:"provide one target per training row"
             (Data_error.Length_mismatch
                { name = "composition targets"; expected; observed }))
      else stage.Pipeline.validate_target ~x ~y
    in
    { stage with Pipeline.validate_target }

  let erase (stage : unit Pipeline.stage) =
    {
      Pipeline.transformer_name = stage.Pipeline.name;
      transformer_cache_check = stage.Pipeline.stage_cache_check;
      transformer_fit_metadata_check = stage.Pipeline.stage_fit_metadata_check;
      transformer_transform_metadata_check =
        stage.Pipeline.stage_transform_metadata_check;
      fit_transform =
        (fun ~cache ~metadata ~rng ~feature_schema ~x ->
          stage.Pipeline.fit_stage ~cache ~metadata ~rng ~feature_schema ~x
            ~y:());
    }
end

module Column_transformer_core = struct
  open Internal

  type remainder = Drop | Passthrough
  type 'target action = Transform of 'target Pipeline.stage | Pass | Omit

  type 'target branch = {
    branch_name : string;
    columns : Column_selector.t;
    action : 'target action;
  }

  type 'target t = {
    branches : 'target branch array;
    remainder : remainder;
    max_output_features : int;
  }

  type branch_info = {
    name : string;
    input_indices : int array;
    output_start : int;
    output_count : int;
  }

  type allocation = { selected_input_bytes : int64; output_bytes : int64 }

  type fitted_branch = {
    info : branch_info;
    selected_schema : Feature_schema.t;
    transformer : Pipeline.fitted_transformer option;
  }

  type 'target fitted = {
    specification : 'target t;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
    fitted_branches : fitted_branch array;
  }

  type block = Selected of int array | Transformed of Matrix.t

  let transformer ~columns transformer =
    {
      branch_name = transformer.Pipeline.name;
      columns;
      action = Transform transformer;
    }

  let passthrough ~name ~columns =
    let* () = validate_name name in
    Ok { branch_name = name; columns; action = Pass }

  let drop ~name ~columns =
    let* () = validate_name name in
    Ok { branch_name = name; columns; action = Omit }

  let create ?(remainder = Drop) ?(max_output_features = 100_000) branches =
    let branches = Array.copy branches in
    let* () =
      unique ~name:"column branch names"
        (Array.map (fun (branch : _ branch) -> branch.branch_name) branches)
    in
    let* () =
      Array.fold_left
        (fun result (branch : _ branch) ->
          let* () = result in
          validate_name branch.branch_name)
        (Ok ()) branches
    in
    if max_output_features < 0 || max_output_features > Sys.max_array_length
    then
      invalid "column output limit"
        "must be between zero and Sys.max_array_length"
    else Ok { branches; remainder; max_output_features }

  let clone specification = specification
  let params specification = specification
  let fitted_params fitted = fitted.specification
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema

  let branches fitted =
    Array.map
      (fun branch ->
        {
          branch.info with
          input_indices = Array.copy branch.info.input_indices;
        })
      fitted.fitted_branches

  let resolve specification feature_schema =
    let width = Feature_schema.feature_count feature_schema in
    let* () =
      if width > Sys.max_array_length then
        invalid "column input schema" "width exceeds the array size limit"
      else Ok ()
    in
    let consumed = Array.make width false in
    let rec loop index reversed =
      if index = Array.length specification.branches then (
        let result = List.rev reversed in
        match specification.remainder with
        | Drop -> Ok (Array.of_list result)
        | Passthrough ->
            let remaining = ref [] in
            for column = width - 1 downto 0 do
              if not consumed.(column) then remaining := column :: !remaining
            done;
            let* remainder =
              Column_selector.indices (Array.of_list !remaining)
            in
            let* indices = Column_selector.resolve remainder feature_schema in
            Ok
              (Array.of_list
                 (result
                 @ [
                     ( {
                         branch_name = "remainder";
                         columns = remainder;
                         action = Pass;
                       },
                       indices );
                   ])))
      else
        let branch = specification.branches.(index) in
        let* indices =
          with_stage branch.branch_name
            (Column_selector.resolve branch.columns feature_schema)
        in
        Array.iter (fun column -> consumed.(column) <- true) indices;
        loop (index + 1) ((branch, indices) :: reversed)
    in
    loop 0 []

  let select x indices =
    let* _ =
      payload_bytes ~rows:(Matrix.rows x) ~columns:(Array.length indices)
    in
    Matrix.init ~rows:(Matrix.rows x) ~columns:(Array.length indices)
      (fun row column -> Matrix.get x row indices.(column))
    |> data

  let validate_output ~rows ~schema x =
    if Matrix.rows x <> rows then
      Error
        (Error.make ~remediation:"preserve the input row count and order"
           (Error.Shape_mismatch
              {
                name = "column transformer output rows";
                expected = [ rows ];
                observed = [ Matrix.rows x ];
              }))
    else Feature_schema.validate_matrix schema x |> data

  let stages specification =
    Array.to_list specification.branches
    |> List.filter_map (fun branch ->
        match branch.action with
        | Transform stage -> Some stage
        | Pass | Omit -> None)
    |> Array.of_list

  let fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y =
    let* () = Feature_schema.validate_matrix feature_schema x |> data in
    let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
    let* () = Stage.validate_fit (stages specification) metadata in
    let* resolved = resolve specification feature_schema in
    let source_names = input_names feature_schema in
    let rec loop index width copies reversed_branches reversed_blocks
        reversed_names =
      if index = Array.length resolved then
        let* output_schema = schema (Array.concat (List.rev reversed_names)) in
        Ok
          ( {
              specification;
              input_schema = feature_schema;
              output_schema;
              fitted_branches = Array.of_list (List.rev reversed_branches);
            },
            Array.of_list (List.rev reversed_blocks),
            copies )
      else
        let branch, indices = resolved.(index) in
        let check_width count =
          if count > specification.max_output_features - width then
            invalid "column output limit"
              "combined branch width exceeds max_output_features"
          else Ok ()
        in
        let* selected_schema =
          schema (Array.map (Array.get source_names) indices)
        in
        let* fitted_transformer, block, names, copied =
          with_stage branch.branch_name
            (if Array.length indices = 0 then Ok (None, Selected [||], [||], 0L)
             else
               match branch.action with
               | Omit -> Ok (None, Selected [||], [||], 0L)
               | Pass ->
                   let* () = check_width (Array.length indices) in
                   Ok (None, Selected indices, input_names selected_schema, 0L)
               | Transform transformer ->
                   let* selected = select x indices in
                   let child_rng =
                     Rng.create
                       (Seed.derive (Rng.to_seed rng)
                          ~operation:("column-transformer:" ^ branch.branch_name)
                          ~index)
                   in
                   let* fitted, output, output_schema =
                     transformer.Pipeline.fit_stage ~cache ~y ~metadata
                       ~rng:child_rng ~feature_schema:selected_schema
                       ~x:selected
                   in
                   let* () =
                     validate_output ~rows:(Matrix.rows x) ~schema:output_schema
                       output
                   in
                   let* () =
                     check_width (Feature_schema.feature_count output_schema)
                   in
                   let* copied =
                     payload_bytes ~rows:(Matrix.rows x)
                       ~columns:(Array.length indices)
                   in
                   Ok
                     ( Some fitted,
                       Transformed output,
                       input_names output_schema,
                       copied ))
        in
        let count = Array.length names in
        let* copies = add_bytes copies copied in
        let info =
          {
            name = branch.branch_name;
            input_indices = indices;
            output_start = width;
            output_count = count;
          }
        in
        let fitted_branch =
          { info; selected_schema; transformer = fitted_transformer }
        in
        let names =
          Array.map (fun name -> branch.branch_name ^ "__" ^ name) names
        in
        loop (index + 1) (width + count) copies
          (fitted_branch :: reversed_branches)
          (block :: reversed_blocks) (names :: reversed_names)
    in
    loop 0 0 0L [] [] []

  let concatenate fitted ~x blocks selected_input_bytes =
    let rows = Matrix.rows x in
    let columns = Feature_schema.feature_count fitted.output_schema in
    let* output_bytes = payload_bytes ~rows ~columns in
    let sources = Array.make columns (0, 0) in
    Array.iteri
      (fun index branch ->
        for column = 0 to branch.info.output_count - 1 do
          sources.(branch.info.output_start + column) <- (index, column)
        done)
      fitted.fitted_branches;
    let* output =
      Matrix.init ~rows ~columns (fun row column ->
          let index, column = sources.(column) in
          match blocks.(index) with
          | Selected indices -> Matrix.get x row indices.(column)
          | Transformed output -> Matrix.get output row column)
      |> data
    in
    Ok (output, { selected_input_bytes; output_bytes })

  let fit specification ~cache ~metadata ~rng ~feature_schema ~x ~y () =
    let* fitted, _, _ =
      fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y
    in
    Ok fitted

  let fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y ()
      =
    let* fitted, blocks, copies =
      fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y
    in
    let* output, allocation = concatenate fitted ~x blocks copies in
    Ok (fitted, output, allocation)

  let transform_with_report fitted ~metadata ~feature_schema ~x =
    let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
    let* () = Stage.validate_transform (stages fitted.specification) metadata in
    let* () =
      if Feature_schema.equal fitted.input_schema feature_schema then
        Feature_schema.validate_matrix feature_schema x |> data
      else
        Error
          (Error.make
             ~remediation:
               "provide the same ordered input feature schema as during fitting"
             (Error.Feature_schema_mismatch
                { expected = fitted.input_schema; observed = feature_schema }))
    in
    let rec loop index copies reversed =
      if index = Array.length fitted.fitted_branches then
        concatenate fitted ~x (Array.of_list (List.rev reversed)) copies
      else
        let branch = fitted.fitted_branches.(index) in
        let* block, copied =
          with_stage branch.info.name
            (match branch.transformer with
            | None ->
                Ok
                  ( Selected
                      (if branch.info.output_count = 0 then [||]
                       else branch.info.input_indices),
                    0L )
            | Some transformer ->
                let* selected = select x branch.info.input_indices in
                let* output =
                  transformer.Pipeline.apply_transform ~metadata
                    ~feature_schema:branch.selected_schema ~x:selected
                in
                let* () =
                  validate_output ~rows:(Matrix.rows x)
                    ~schema:transformer.Pipeline.transform_output_schema output
                in
                let* copied =
                  payload_bytes ~rows:(Matrix.rows x)
                    ~columns:(Array.length branch.info.input_indices)
                in
                Ok (Transformed output, copied))
        in
        let* copies = add_bytes copies copied in
        loop (index + 1) copies (block :: reversed)
    in
    loop 0 0L []

  let transform fitted ~metadata ~feature_schema ~x =
    let* output, _ =
      transform_with_report fitted ~metadata ~feature_schema ~x
    in
    Ok output

  let stage ~name specification =
    let fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y
        () =
      let* fitted, output, _ =
        fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y
          ()
      in
      Ok (fitted, output)
    in
    Stage.package ~name
      ~validate_cache:(Stage.validate_cache (stages specification))
      ~validate_target:(Stage.validate (stages specification))
      ~validate_fit_metadata:(Stage.validate_fit (stages specification))
      ~validate_transform_metadata:
        (Stage.validate_transform (stages specification))
      ~fit_transform ~transform ~output_schema specification
end

module Composition_internal = struct
  include Internal

  let validate_names component names =
    let* () = unique ~name:component names in
    if Array.exists (fun name -> String.trim name = "") names then
      invalid component "stage names must not be blank"
    else Ok ()

  let validate_fit ~metadata ~feature_schema ~x =
    let* () = Feature_schema.validate_matrix feature_schema x |> data in
    Metadata.validate ~rows:(Matrix.rows x) metadata

  let validate_input expected observed x =
    if Feature_schema.equal expected observed then
      Feature_schema.validate_matrix observed x |> data
    else
      Error
        (Error.make
           ~remediation:
             "provide the same ordered feature schema used during fitting"
           (Error.Feature_schema_mismatch { expected; observed }))

  let apply fitted ~metadata ~feature_schema ~x =
    let* () =
      validate_input fitted.Pipeline.transform_input_schema feature_schema x
    in
    let* output =
      fitted.Pipeline.apply_transform ~metadata ~feature_schema ~x
    in
    let* () =
      Pipeline.validate_transform_output ~input:x
        ~output_schema:fitted.Pipeline.transform_output_schema output
    in
    Ok output
end

module Transformer_pipeline_core = struct
  open Composition_internal

  type 'target t = 'target Pipeline.stage array

  type 'target fitted = {
    specification : 'target t;
    steps : Pipeline.fitted_transformer array;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
  }

  let create stages =
    let stages = Array.copy stages in
    let* () =
      validate_names "transformer pipeline stage names"
        (Array.map (fun stage -> stage.Pipeline.name) stages)
    in
    Ok stages

  let clone specification = specification
  let params specification = specification
  let fitted_params fitted = fitted.specification
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema

  let stage_names specification =
    Array.map (fun stage -> stage.Pipeline.name) specification

  let fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y ()
      =
    let* () = validate_fit ~metadata ~feature_schema ~x in
    let* () = Stage.validate_fit specification metadata in
    let rec loop index current_schema current_x reversed =
      if index = Array.length specification then
        Ok
          ( {
              specification;
              steps = Array.of_list (List.rev reversed);
              input_schema = feature_schema;
              output_schema = current_schema;
            },
            current_x )
      else
        let step = specification.(index) in
        let child_rng =
          Rng.create
            (Seed.derive (Rng.to_seed rng)
               ~operation:("transformer-pipeline:" ^ step.Pipeline.name)
               ~index)
        in
        let* fitted, output, output_schema =
          with_stage step.Pipeline.name
            (step.Pipeline.fit_stage ~cache ~y ~metadata ~rng:child_rng
               ~feature_schema:current_schema ~x:current_x)
        in
        loop (index + 1) output_schema output (fitted :: reversed)
    in
    loop 0 feature_schema x []

  let fit specification ~cache ~metadata ~rng ~feature_schema ~x ~y () =
    let* fitted, _ =
      fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y ()
    in
    Ok fitted

  let transform fitted ~metadata ~feature_schema ~x =
    let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
    let* () = Stage.validate_fitted fitted.steps metadata in
    let* () = validate_input fitted.input_schema feature_schema x in
    let rec loop index schema x =
      if index = Array.length fitted.steps then Ok x
      else
        let step = fitted.steps.(index) in
        let* output =
          with_stage step.Pipeline.stage_name
            (apply step ~metadata ~feature_schema:schema ~x)
        in
        loop (index + 1) step.Pipeline.transform_output_schema output
    in
    loop 0 feature_schema x

  let stage ~name specification =
    Stage.package ~name
      ~validate_cache:(Stage.validate_cache specification)
      ~validate_target:(Stage.validate specification)
      ~validate_fit_metadata:(Stage.validate_fit specification)
      ~validate_transform_metadata:(Stage.validate_transform specification)
      ~fit_transform ~transform ~output_schema specification
end

module Feature_union_core = struct
  open Composition_internal

  type 'target action = Transform of 'target Pipeline.stage | Pass | Omit
  type 'target branch = { branch_name : string; action : 'target action }

  type 'target t = {
    branches : 'target branch array;
    max_output_features : int;
  }

  type branch_info = { name : string; output_start : int; output_count : int }

  type fitted_branch = {
    info : branch_info;
    transformer : Pipeline.fitted_transformer option;
  }

  type 'target fitted = {
    specification : 'target t;
    input_schema : Feature_schema.t;
    output_schema : Feature_schema.t;
    fitted_branches : fitted_branch array;
  }

  type allocation = { output_bytes : int64 }

  let transformer transformer =
    { branch_name = transformer.Pipeline.name; action = Transform transformer }

  let passthrough ~name =
    let* () = validate_names "feature union branch name" [| name |] in
    Ok { branch_name = name; action = Pass }

  let drop ~name =
    let* () = validate_names "feature union branch name" [| name |] in
    Ok { branch_name = name; action = Omit }

  let create ?(max_output_features = 100_000) branches =
    let branches = Array.copy branches in
    let* () =
      validate_names "feature union branch names"
        (Array.map (fun branch -> branch.branch_name) branches)
    in
    if max_output_features < 0 || max_output_features > Sys.max_array_length
    then
      invalid "feature union output limit"
        "must be between zero and Sys.max_array_length"
    else Ok { branches; max_output_features }

  let clone specification = specification
  let params specification = specification
  let fitted_params fitted = fitted.specification
  let input_schema fitted = fitted.input_schema
  let output_schema fitted = fitted.output_schema

  let branches fitted =
    Array.map (fun branch -> branch.info) fitted.fitted_branches

  let stages specification =
    Array.to_list specification.branches
    |> List.filter_map (fun branch ->
        match branch.action with
        | Transform stage -> Some stage
        | Pass | Omit -> None)
    |> Array.of_list

  let fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y =
    let* () = validate_fit ~metadata ~feature_schema ~x in
    let* () = Stage.validate_fit (stages specification) metadata in
    let rec loop index width reversed_branches reversed_outputs reversed_names =
      if index = Array.length specification.branches then
        let* output_schema = schema (Array.concat (List.rev reversed_names)) in
        Ok
          ( {
              specification;
              input_schema = feature_schema;
              output_schema;
              fitted_branches = Array.of_list (List.rev reversed_branches);
            },
            Array.of_list (List.rev reversed_outputs) )
      else
        let branch = specification.branches.(index) in
        let* fitted_transformer, output, output_schema =
          with_stage branch.branch_name
            (match branch.action with
            | Omit -> Ok (None, x, None)
            | Pass -> Ok (None, x, Some feature_schema)
            | Transform transformer ->
                let child_rng =
                  Rng.create
                    (Seed.derive (Rng.to_seed rng)
                       ~operation:("feature-union:" ^ branch.branch_name)
                       ~index)
                in
                let* fitted, output, output_schema =
                  transformer.Pipeline.fit_stage ~cache ~y ~metadata
                    ~rng:child_rng ~feature_schema ~x
                in
                Ok (Some fitted, output, Some output_schema))
        in
        let count =
          match output_schema with
          | None -> 0
          | Some schema -> Feature_schema.feature_count schema
        in
        let* () =
          with_stage branch.branch_name
            (if count > specification.max_output_features - width then
               invalid "feature union output limit"
                 "combined branch width exceeds max_output_features"
             else Ok ())
        in
        let names =
          match output_schema with
          | None -> [||]
          | Some schema ->
              Array.map
                (fun name -> branch.branch_name ^ "__" ^ name)
                (input_names schema)
        in
        let info =
          {
            name = branch.branch_name;
            output_start = width;
            output_count = count;
          }
        in
        loop (index + 1) (width + count)
          ({ info; transformer = fitted_transformer } :: reversed_branches)
          (output :: reversed_outputs)
          (names :: reversed_names)
    in
    loop 0 0 [] [] []

  let concatenate fitted ~rows outputs =
    let columns = Feature_schema.feature_count fitted.output_schema in
    let* output_bytes = payload_bytes ~rows ~columns in
    let sources = Array.make columns (0, 0) in
    Array.iteri
      (fun index branch ->
        for column = 0 to branch.info.output_count - 1 do
          sources.(branch.info.output_start + column) <- (index, column)
        done)
      fitted.fitted_branches;
    let* output =
      Matrix.init ~rows ~columns (fun row column ->
          let index, column = sources.(column) in
          Matrix.get outputs.(index) row column)
      |> data
    in
    Ok (output, { output_bytes })

  let fit specification ~cache ~metadata ~rng ~feature_schema ~x ~y () =
    let* fitted, _ =
      fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y
    in
    Ok fitted

  let fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y ()
      =
    let* fitted, outputs =
      fit_branches specification ~cache ~metadata ~rng ~feature_schema ~x ~y
    in
    let* output, allocation =
      concatenate fitted ~rows:(Matrix.rows x) outputs
    in
    Ok (fitted, output, allocation)

  let transform_with_report fitted ~metadata ~feature_schema ~x =
    let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
    let* () = Stage.validate_transform (stages fitted.specification) metadata in
    let* () = validate_input fitted.input_schema feature_schema x in
    let rec loop index reversed =
      if index = Array.length fitted.fitted_branches then
        concatenate fitted ~rows:(Matrix.rows x)
          (Array.of_list (List.rev reversed))
      else
        let branch = fitted.fitted_branches.(index) in
        let* output =
          with_stage branch.info.name
            (match branch.transformer with
            | None -> Ok x
            | Some transformer -> apply transformer ~metadata ~feature_schema ~x)
        in
        loop (index + 1) (output :: reversed)
    in
    loop 0 []

  let transform fitted ~metadata ~feature_schema ~x =
    let* output, _ =
      transform_with_report fitted ~metadata ~feature_schema ~x
    in
    Ok output

  let stage ~name specification =
    let fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y
        () =
      let* fitted, output, _ =
        fit_transform specification ~cache ~metadata ~rng ~feature_schema ~x ~y
          ()
      in
      Ok (fitted, output)
    in
    let stages =
      Array.to_list specification.branches
      |> List.filter_map (fun branch ->
          match branch.action with
          | Transform stage -> Some stage
          | Pass | Omit -> None)
      |> Array.of_list
    in
    Stage.package ~name
      ~validate_cache:(Stage.validate_cache stages)
      ~validate_target:(Stage.validate stages)
      ~validate_fit_metadata:(Stage.validate_fit stages)
      ~validate_transform_metadata:(Stage.validate_transform stages)
      ~fit_transform ~transform ~output_schema specification
end

module Column_transformer = struct
  include Column_transformer_core

  type nonrec t = unit t
  type params = t
  type target = unit
  type rng = Rng.t
  type nonrec fitted = unit fitted
  type nonrec branch = unit branch

  let transformer ~columns stage =
    transformer ~columns (Stage.unsupervised stage)

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y:_ () =
    fit specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_transform specification ?sample_weight ~rng ~feature_schema ~x ~y:_ ()
      =
    fit_transform specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_with_metadata specification ~metadata ~rng ~feature_schema ~x ~y:_ ()
      =
    Column_transformer_core.fit specification ~cache:None ~metadata ~rng
      ~feature_schema ~x ~y:() ()

  let fit_transform_with_metadata specification ~metadata ~rng ~feature_schema
      ~x ~y:_ () =
    Column_transformer_core.fit_transform specification ~cache:None ~metadata
      ~rng ~feature_schema ~x ~y:() ()

  let transform_with_metadata = Column_transformer_core.transform

  let transform fitted ~feature_schema ~x =
    transform_with_metadata fitted ~metadata:Metadata.empty ~feature_schema ~x

  let transform_with_report_with_metadata =
    Column_transformer_core.transform_with_report

  let transform_with_report fitted ~feature_schema ~x =
    transform_with_report_with_metadata fitted ~metadata:Metadata.empty
      ~feature_schema ~x

  let stage ~name specification =
    Result.map Stage.erase (stage ~name specification)

  module Supervised = struct
    type 'kind t = 'kind Target.t Column_transformer_core.t
    type 'kind branch = 'kind Target.t Column_transformer_core.branch

    let transformer = Column_transformer_core.transformer
    let passthrough = passthrough
    let drop = drop
    let create = Column_transformer_core.create

    let stage ~name specification =
      Result.map Stage.supervised
        (Column_transformer_core.stage ~name specification)
  end
end

module Transformer_pipeline = struct
  include Transformer_pipeline_core

  type nonrec t = unit t
  type params = t
  type target = unit
  type rng = Rng.t
  type nonrec fitted = unit fitted

  let create stages = create (Array.map Stage.unsupervised stages)

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y:_ () =
    fit specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_transform specification ?sample_weight ~rng ~feature_schema ~x ~y:_ ()
      =
    fit_transform specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_with_metadata specification ~metadata ~rng ~feature_schema ~x ~y:_ ()
      =
    Transformer_pipeline_core.fit specification ~cache:None ~metadata ~rng
      ~feature_schema ~x ~y:() ()

  let fit_transform_with_metadata specification ~metadata ~rng ~feature_schema
      ~x ~y:_ () =
    Transformer_pipeline_core.fit_transform specification ~cache:None ~metadata
      ~rng ~feature_schema ~x ~y:() ()

  let transform_with_metadata = Transformer_pipeline_core.transform

  let transform fitted ~feature_schema ~x =
    transform_with_metadata fitted ~metadata:Metadata.empty ~feature_schema ~x

  let stage ~name specification =
    Result.map Stage.erase (stage ~name specification)

  module Supervised = struct
    type 'kind t = 'kind Target.t Transformer_pipeline_core.t

    let create = Transformer_pipeline_core.create

    let stage ~name specification =
      Result.map Stage.supervised
        (Transformer_pipeline_core.stage ~name specification)
  end
end

module Feature_union = struct
  include Feature_union_core

  type nonrec t = unit t
  type params = t
  type target = unit
  type rng = Rng.t
  type nonrec fitted = unit fitted
  type nonrec branch = unit branch

  let transformer stage = transformer (Stage.unsupervised stage)

  let fit specification ?sample_weight ~rng ~feature_schema ~x ~y:_ () =
    fit specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_transform specification ?sample_weight ~rng ~feature_schema ~x ~y:_ ()
      =
    fit_transform specification ~cache:None
      ~metadata:(Metadata.create ?sample_weight ())
      ~rng ~feature_schema ~x ~y:() ()

  let fit_with_metadata specification ~metadata ~rng ~feature_schema ~x ~y:_ ()
      =
    Feature_union_core.fit specification ~cache:None ~metadata ~rng
      ~feature_schema ~x ~y:() ()

  let fit_transform_with_metadata specification ~metadata ~rng ~feature_schema
      ~x ~y:_ () =
    Feature_union_core.fit_transform specification ~cache:None ~metadata ~rng
      ~feature_schema ~x ~y:() ()

  let transform_with_metadata = Feature_union_core.transform

  let transform fitted ~feature_schema ~x =
    transform_with_metadata fitted ~metadata:Metadata.empty ~feature_schema ~x

  let transform_with_report_with_metadata =
    Feature_union_core.transform_with_report

  let transform_with_report fitted ~feature_schema ~x =
    transform_with_report_with_metadata fitted ~metadata:Metadata.empty
      ~feature_schema ~x

  let stage ~name specification =
    Result.map Stage.erase (stage ~name specification)

  module Supervised = struct
    type 'kind t = 'kind Target.t Feature_union_core.t
    type 'kind branch = 'kind Target.t Feature_union_core.branch

    let transformer = Feature_union_core.transformer
    let passthrough = passthrough
    let drop = drop
    let create = Feature_union_core.create

    let stage ~name specification =
      Result.map Stage.supervised (Feature_union_core.stage ~name specification)
  end
end
