open Modelkit_data
open Modelkit_metadata
open Modelkit_protocols

module Conformance = struct
  type issue =
    | Protocol_error of Error.t
    | Violation of string
    | Raised of string

  type outcome = Passed | Failed of issue
  type check = { name : string; outcome : outcome }
  type report = { report_checks : check array }

  let issue_to_string = function
    | Protocol_error error -> Error.to_string error
    | Violation reason | Raised reason -> reason

  let checks report = Array.copy report.report_checks

  let passed report =
    Array.for_all
      (fun check ->
        match check.outcome with Passed -> true | Failed _ -> false)
      report.report_checks

  let failures report =
    Array.to_list report.report_checks
    |> List.filter (fun check ->
        match check.outcome with Passed -> false | Failed _ -> true)
    |> Array.of_list

  let run name check =
    let outcome =
      try match check () with Ok () -> Passed | Error issue -> Failed issue
      with exception_value ->
        Failed (Raised (Printexc.to_string exception_value))
    in
    { name; outcome }

  let protocol (type value) (result : (value, Error.t) result) =
    Result.map_error (fun error -> Protocol_error error) result

  let require condition reason =
    if condition then Ok () else Error (Violation reason)

  let matrix_equal left right =
    Matrix.shape left = Matrix.shape right
    &&
    let rows, columns = Matrix.shape left in
    let equal = ref true in
    let row = ref 0 in
    while !equal && !row < rows do
      let column = ref 0 in
      while !equal && !column < columns do
        equal :=
          Int64.bits_of_float (Matrix.get left !row !column)
          = Int64.bits_of_float (Matrix.get right !row !column);
        incr column
      done;
      incr row
    done;
    !equal

  module Estimator = struct
    type ('specification, 'params, 'target, 'prediction, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
      prediction_length : 'prediction -> int;
      equal_prediction : 'prediction -> 'prediction -> bool;
    }

    let check (type specification params target prediction fitted rng)
        (module Implementation : ESTIMATOR
          with type t = specification
           and type params = params
           and type target = target
           and type prediction = prediction
           and type fitted = fitted
           and type rng = rng)
        (fixture :
          (specification, params, target, prediction, fitted, rng) fixture) =
      let fit () =
        Implementation.fit fixture.specification
          ?sample_weight:fixture.sample_weight ~rng:(fixture.rng ())
          ~feature_schema:fixture.feature_schema ~x:fixture.x ~y:fixture.y ()
      in
      let fitted = lazy (fit ()) in
      let predict () =
        let ( let* ) = Result.bind in
        let* fitted = Lazy.force fitted in
        Implementation.predict fitted ~feature_schema:fixture.feature_schema
          ~x:fixture.x
      in
      {
        report_checks =
          [|
            run "clone preserves parameters" (fun () ->
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.params
                        (Implementation.clone fixture.specification)))
                  "clone changed the training parameters");
            run "fit accepts the valid fixture" (fun () ->
                protocol (Lazy.force fitted) |> Result.map (fun _ -> ()));
            run "fitted parameters match the specification" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.fitted_params fitted))
                  "fitted_params differs from the admitted specification");
            run "fitted schema matches the training schema" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (Feature_schema.equal fixture.feature_schema
                     (Implementation.feature_schema fitted))
                  "feature_schema differs from the training schema");
            run "prediction preserves row count" (fun () ->
                let ( let* ) = Result.bind in
                let* prediction = protocol (predict ()) in
                require
                  (fixture.prediction_length prediction = Matrix.rows fixture.x)
                  "predict returned a different number of rows");
            run "repeated prediction is deterministic" (fun () ->
                let ( let* ) = Result.bind in
                let* first = protocol (predict ()) in
                let* second = protocol (predict ()) in
                require
                  (fixture.equal_prediction first second)
                  "repeated prediction returned different values");
          |];
      }
  end

  module Metadata_estimator = struct
    type ('specification, 'params, 'target, 'prediction, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target;
      metadata : Metadata.t;
      equal_params : 'params -> 'params -> bool;
      prediction_length : 'prediction -> int;
      equal_prediction : 'prediction -> 'prediction -> bool;
    }

    let check (type specification params target prediction fitted rng)
        (module Implementation : METADATA_ESTIMATOR
          with type t = specification
           and type params = params
           and type target = target
           and type prediction = prediction
           and type fitted = fitted
           and type rng = rng)
        (fixture :
          (specification, params, target, prediction, fitted, rng) fixture) =
      let request = Implementation.fit_request fixture.specification in
      let validate_metadata () =
        let ( let* ) = Result.bind in
        let* () =
          Metadata.validate ~rows:(Matrix.rows fixture.x) fixture.metadata
        in
        Metadata.validate_request request fixture.metadata
      in
      let fit () =
        let ( let* ) = Result.bind in
        let* () = validate_metadata () in
        Metadata.consume ~name:"conformance estimator"
          ~operation:Modelkit_callback.Callback.Fit request fixture.metadata
          (fun metadata ->
            Implementation.fit fixture.specification ~metadata
              ~rng:(fixture.rng ()) ~feature_schema:fixture.feature_schema
              ~x:fixture.x ~y:fixture.y ())
      in
      let fitted = lazy (fit ()) in
      let predict () =
        let ( let* ) = Result.bind in
        let* fitted = Lazy.force fitted in
        Implementation.predict fitted ~feature_schema:fixture.feature_schema
          ~x:fixture.x
      in
      {
        report_checks =
          [|
            run "clone preserves parameters" (fun () ->
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.params
                        (Implementation.clone fixture.specification)))
                  "clone changed the training parameters");
            run "fit request accepts the valid fixture" (fun () ->
                protocol (validate_metadata ()));
            run "fit accepts routed metadata" (fun () ->
                protocol (Lazy.force fitted) |> Result.map (fun _ -> ()));
            run "fitted parameters match the specification" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.fitted_params fitted))
                  "fitted_params differs from the admitted specification");
            run "fitted schema matches the training schema" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (Feature_schema.equal fixture.feature_schema
                     (Implementation.feature_schema fitted))
                  "feature_schema differs from the training schema");
            run "prediction preserves row count" (fun () ->
                let ( let* ) = Result.bind in
                let* prediction = protocol (predict ()) in
                require
                  (fixture.prediction_length prediction = Matrix.rows fixture.x)
                  "predict returned a different number of rows");
            run "repeated prediction is deterministic" (fun () ->
                let ( let* ) = Result.bind in
                let* first = protocol (predict ()) in
                let* second = protocol (predict ()) in
                require
                  (fixture.equal_prediction first second)
                  "repeated prediction returned different values");
          |];
      }
  end

  module Transformer = struct
    type ('specification, 'params, 'target, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target option;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    let check (type specification params target fitted rng)
        (module Implementation : TRANSFORMER
          with type t = specification
           and type params = params
           and type target = target
           and type fitted = fitted
           and type rng = rng)
        (fixture : (specification, params, target, fitted, rng) fixture) =
      let fit () =
        Implementation.fit fixture.specification
          ?sample_weight:fixture.sample_weight ~rng:(fixture.rng ())
          ~feature_schema:fixture.feature_schema ~x:fixture.x ~y:fixture.y ()
      in
      let fitted = lazy (fit ()) in
      let transform () =
        let ( let* ) = Result.bind in
        let* fitted = Lazy.force fitted in
        Implementation.transform fitted ~feature_schema:fixture.feature_schema
          ~x:fixture.x
      in
      {
        report_checks =
          [|
            run "clone preserves parameters" (fun () ->
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.params
                        (Implementation.clone fixture.specification)))
                  "clone changed the training parameters");
            run "fit accepts the valid fixture" (fun () ->
                protocol (Lazy.force fitted) |> Result.map (fun _ -> ()));
            run "fitted parameters match the specification" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.fitted_params fitted))
                  "fitted_params differs from the admitted specification");
            run "input schema matches the training schema" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                require
                  (Feature_schema.equal fixture.feature_schema
                     (Implementation.input_schema fitted))
                  "input_schema differs from the training schema");
            run "output matches its declared schema" (fun () ->
                let ( let* ) = Result.bind in
                let* fitted = protocol (Lazy.force fitted) in
                let* transformed = protocol (transform ()) in
                Feature_schema.validate_matrix
                  (Implementation.output_schema fitted)
                  transformed
                |> Result.map_error (fun error ->
                    Violation (Data_error.to_string error)));
            run "transform preserves row count" (fun () ->
                let ( let* ) = Result.bind in
                let* transformed = protocol (transform ()) in
                require
                  (Matrix.rows transformed = Matrix.rows fixture.x)
                  "transform returned a different number of rows");
            run "repeated transform is deterministic" (fun () ->
                let ( let* ) = Result.bind in
                let* first = protocol (transform ()) in
                let* second = protocol (transform ()) in
                require
                  (matrix_equal first second)
                  "repeated transform returned different values");
          |];
      }
  end

  module Scorer = struct
    type ('specification, 'params, 'truth, 'prediction) fixture = {
      specification : 'specification;
      capabilities : Capability.scorer;
      truth : 'truth;
      prediction : 'prediction;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    let check (type specification params truth prediction)
        (module Implementation : SCORER
          with type t = specification
           and type params = params
           and type truth = truth
           and type prediction = prediction)
        (fixture : (specification, params, truth, prediction) fixture) =
      let scorer =
        Modelkit_protocols.Scorer.of_module ~capabilities:fixture.capabilities
          (module Implementation)
          fixture.specification
      in
      let score () =
        Modelkit_protocols.Scorer.score scorer
          ?sample_weight:fixture.sample_weight ~truth:fixture.truth
          ~prediction:fixture.prediction ()
      in
      {
        report_checks =
          [|
            run "clone preserves parameters" (fun () ->
                require
                  (fixture.equal_params
                     (Implementation.params fixture.specification)
                     (Implementation.params
                        (Implementation.clone fixture.specification)))
                  "clone changed the scoring parameters");
            run "name is non-blank" (fun () ->
                require
                  (String.trim (Modelkit_protocols.Scorer.name scorer) <> "")
                  "name returned a blank report field");
            run "score is finite" (fun () ->
                let ( let* ) = Result.bind in
                let* value = protocol (score ()) in
                require (Float.is_finite value)
                  "score returned a non-finite value");
            run "repeated scoring is deterministic" (fun () ->
                let ( let* ) = Result.bind in
                let* first = protocol (score ()) in
                let* second = protocol (score ()) in
                require
                  (Int64.bits_of_float first = Int64.bits_of_float second)
                  "repeated scoring returned different values");
          |];
      }
  end
end
