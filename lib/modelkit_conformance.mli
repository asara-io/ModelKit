open Modelkit_data
open Modelkit_protocols

(** Framework-neutral protocol checks for third-party components.

    Reports are ordinary values so package authors can use them from Alcotest,
    OUnit, expect tests, or their own build tooling without adding a ModelKit
    test-framework dependency. *)
module Conformance : sig
  type issue =
    | Protocol_error of Error.t
    | Violation of string
    | Raised of string

  type outcome = Passed | Failed of issue
  type check = { name : string; outcome : outcome }
  type report

  val issue_to_string : issue -> string
  val checks : report -> check array
  val passed : report -> bool
  val failures : report -> check array

  module Estimator : sig
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

    val check :
      (module ESTIMATOR
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type prediction = 'prediction
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'params, 'target, 'prediction, 'fitted, 'rng) fixture ->
      report
  end

  module Transformer : sig
    type ('specification, 'params, 'target, 'fitted, 'rng) fixture = {
      specification : 'specification;
      rng : unit -> 'rng;
      feature_schema : Feature_schema.t;
      x : Matrix.t;
      y : 'target option;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    val check :
      (module TRANSFORMER
         with type t = 'specification
          and type params = 'params
          and type target = 'target
          and type fitted = 'fitted
          and type rng = 'rng) ->
      ('specification, 'params, 'target, 'fitted, 'rng) fixture ->
      report
  end

  module Scorer : sig
    type ('specification, 'params, 'truth, 'prediction) fixture = {
      specification : 'specification;
      capabilities : Capability.scorer;
      truth : 'truth;
      prediction : 'prediction;
      sample_weight : Sample_weight.t option;
      equal_params : 'params -> 'params -> bool;
    }

    val check :
      (module SCORER
         with type t = 'specification
          and type params = 'params
          and type truth = 'truth
          and type prediction = 'prediction) ->
      ('specification, 'params, 'truth, 'prediction) fixture ->
      report
  end
end
