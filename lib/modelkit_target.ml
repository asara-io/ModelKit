open Modelkit_data
open Modelkit_metadata
open Modelkit_protocols
open Modelkit_pipeline

module Transformed_target_regressor = struct
  module type TRANSFORMER = sig
    include SPECIFICATION

    type fitted

    val fit_request : t -> Metadata.Request.t

    val fit :
      t ->
      metadata:Metadata.t ->
      rng:Rng.t ->
      y:Target.regression Target.t ->
      (fitted, Error.t) result

    val transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result

    val inverse_transform :
      fitted ->
      Target.regression Target.t ->
      (Target.regression Target.t, Error.t) result
  end

  type target = Target.regression Target.t

  type learned = {
    forward : target -> (target, Error.t) result;
    inverse : target -> (target, Error.t) result;
  }

  type transformer = {
    request : Metadata.Request.t;
    learn :
      metadata:Metadata.t -> rng:Rng.t -> y:target -> (learned, Error.t) result;
  }

  let ( let* ) = Result.bind

  let transformer (type specification fitted)
      (module T : TRANSFORMER
        with type t = specification
         and type fitted = fitted) specification =
    let request = T.fit_request specification in
    let learn ~metadata ~rng ~y =
      let* fitted = T.fit (T.clone specification) ~metadata ~rng ~y in
      Ok { forward = T.transform fitted; inverse = T.inverse_transform fitted }
    in
    { request; learn }

  let functions ~transform ~inverse_transform =
    let apply f y =
      Target.regression_values y |> Vector.to_array |> Array.map f
      |> Vector.of_array |> Target.regression
      |> Result.map_error (fun error ->
          Error.of_data_error error
            ~remediation:
              "use target functions that return finite values on this domain")
    in
    {
      request = Metadata.Request.none;
      learn =
        (fun ~metadata:_ ~rng:_ ~y:_ ->
          Ok { forward = apply transform; inverse = apply inverse_transform });
    }

  let within name result =
    Result.map_error (Error.with_context (Error.Stage name)) result

  let validation ~remediation name reason =
    Error (Error.make (Error.Validation { name; reason }) ~remediation)

  let check_length name expected y =
    let observed = Target.length y in
    if expected = observed then Ok ()
    else
      Error
        (Error.make
           (Error.Shape_mismatch
              { name; expected = [ expected ]; observed = [ observed ] })
           ~remediation:"preserve one target or prediction per input row")

  let check_inverse ~rtol ~atol y restored =
    let original = Target.regression_values y in
    let restored = Target.regression_values restored in
    let rec loop i =
      if i = Vector.length original then Ok ()
      else
        let a = Vector.get original i and b = Vector.get restored i in
        (* Scaling keeps the inverse check meaningful near float limits. *)
        let scale = max 1. (max (abs_float a) (abs_float b)) in
        let distance =
          if Float.sign_bit a = Float.sign_bit b then
            abs_float (a -. b) /. scale
          else abs_float ((a /. scale) -. (b /. scale))
        in
        if distance -. (atol /. scale) <= rtol *. (abs_float a /. scale) then
          loop (i + 1)
        else
          validation
            ~remediation:
              "supply mutually inverse target mappings or explicitly adjust \
               the tolerances"
            "target inverse"
            (Printf.sprintf "round-trip mismatch at training row %d" i)
    in
    loop 0

  let create ?(rtol = 1e-7) ?(atol = 1e-9) ~name ~transformer ~regressor () =
    let* () =
      if String.trim name = "" then
        validation ~remediation:"choose a non-empty pipeline stage name"
          "target regressor name" "must not be blank"
      else Ok ()
    in
    let* () =
      if
        Float.is_finite rtol && Float.is_finite atol && rtol >= 0. && atol >= 0.
      then Ok ()
      else
        validation
          ~remediation:"use finite nonnegative relative and absolute tolerances"
          "target inverse tolerances" "must be finite and nonnegative"
    in
    let check metadata =
      let* () =
        within "target_transform"
          (Metadata.validate_request transformer.request metadata)
      in
      within regressor.Pipeline.estimator_name
        (regressor.Pipeline.estimator_fit_metadata_check metadata)
    in
    let fit_estimator ~metadata ~rng ~feature_schema ~x ~y () =
      let* () = check_length "regression target" (Matrix.rows x) y in
      let* () = Metadata.validate ~rows:(Matrix.rows x) metadata in
      let* () = check metadata in
      let metadata = Metadata.scope (Error.Stage name) metadata in
      let seed = Rng.to_seed rng in
      let child index =
        Rng.create
          (Seed.derive seed ~operation:"transformed_target_regressor" ~index)
      in
      let* learned, transformed =
        within "target_transform"
          (Metadata.consume ~name:"target_transform"
             ~operation:Modelkit_callback.Callback.Fit transformer.request
             metadata (fun metadata ->
               let* learned = transformer.learn ~metadata ~rng:(child 0) ~y in
               let* transformed = learned.forward y in
               let* () =
                 check_length "transformed target" (Target.length y) transformed
               in
               let* restored = learned.inverse transformed in
               let* () =
                 check_length "inverse target" (Target.length y) restored
               in
               let* () = check_inverse ~rtol ~atol y restored in
               Ok (learned, transformed)))
      in
      let* fitted =
        within regressor.Pipeline.estimator_name
          (regressor.Pipeline.fit_estimator ~metadata ~rng:(child 1)
             ~feature_schema ~x ~y:transformed ())
      in
      let terminal_predict ~feature_schema ~x =
        let* transformed =
          within regressor.Pipeline.estimator_name
            (fitted.Pipeline.terminal_predict ~feature_schema ~x)
        in
        let* () =
          check_length "regressor prediction" (Matrix.rows x) transformed
        in
        within "target_inverse"
          (let* prediction = learned.inverse transformed in
           let* () =
             check_length "inverse prediction" (Matrix.rows x) prediction
           in
           Ok prediction)
      in
      Ok
        {
          Pipeline.terminal_name = name;
          terminal_predict;
          terminal_decision_function = None;
          terminal_predict_proba = None;
          terminal_classes = None;
          encode_estimator = None;
        }
    in
    Ok
      {
        Pipeline.estimator_name = name;
        estimator_fit_metadata_check = check;
        estimator_capabilities =
          { Pipeline.decision_function = false; predict_proba = false };
        fit_estimator;
      }
end
