open Modelkit

let get_data = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Data_error.to_string error)

let get = function
  | Ok value -> value
  | Error error -> Alcotest.fail (Error.to_string error)

let expect_error predicate = function
  | Error error when predicate (Error.kind error) -> ()
  | Error error -> Alcotest.fail ("unexpected error: " ^ Error.to_string error)
  | Ok _ -> Alcotest.fail "expected an error"

let[@warning "-4"] is_validation = function
  | Error.Validation _ -> true
  | _ -> false

let[@warning "-4"] is_shape = function
  | Error.Shape_mismatch _ -> true
  | _ -> false

let check_float message expected observed =
  Alcotest.check (Alcotest.float 1e-12) message expected observed

let matrix values = Matrix.of_arrays values |> get_data
let classification values = Target.classification values

let weights values =
  Sample_weight.of_array ~expected_length:(Array.length values) values
  |> get_data

let test_average_precision () =
  let truth = classification [| 1; 0; 1; 0 |] in
  let perfect = Vector.of_array [| 0.9; 0.1; 0.8; 0.2 |] in
  check_float "perfect ranking" 1.0
    (Binary_classification_metrics.average_precision ~truth
       ~positive_probabilities:perfect ()
    |> get);
  let reversed = Vector.of_array [| 0.1; 0.9; 0.2; 0.8 |] in
  check_float "reversed ranking" (5.0 /. 12.0)
    (Binary_classification_metrics.average_precision ~truth
       ~positive_probabilities:reversed ()
    |> get);
  let single = classification [| 0; 0 |] in
  Binary_classification_metrics.average_precision ~truth:single
    ~positive_probabilities:(Vector.of_array [| 0.5; 0.5 |])
    ()
  |> expect_error is_validation;
  check_float "fallback without positives" 0.0
    (Binary_classification_metrics.average_precision
       ~undefined:Undefined_metric_policy.Use_fallback ~truth:single
       ~positive_probabilities:(Vector.of_array [| 0.5; 0.5 |])
       ()
    |> get);
  let scorer = Binary_classification_scorer.average_precision () in
  Alcotest.(check string)
    "scorer name" "average_precision"
    (Binary_classification_scorer.name scorer);
  check_float "scorer dispatch" 1.0
    (Binary_classification_scorer.score scorer ~truth
       ~prediction:
         (Binary_prediction.create ~positive_probabilities:perfect () |> get)
       ()
    |> get)

let truth = classification [| 0; 1; 2; 1 |]
let classes = [| 0; 1; 2 |]

let probabilities =
  matrix
    [|
      [| 0.6; 0.3; 0.1 |];
      [| 0.2; 0.5; 0.3 |];
      [| 0.1; 0.2; 0.7 |];
      [| 0.4; 0.4; 0.2 |];
    |]

let test_multiclass_auc () =
  let open Multiclass_ranking in
  let separable =
    matrix
      [|
        [| 0.8; 0.1; 0.1 |];
        [| 0.1; 0.8; 0.1 |];
        [| 0.1; 0.1; 0.8 |];
        [| 0.2; 0.6; 0.2 |];
      |]
  in
  List.iter
    (fun strategy ->
      List.iter
        (fun average ->
          check_float "separable scores give one" 1.0
            (roc_auc ~strategy ~average ~truth ~classes ~probabilities:separable
               ()
            |> get))
        Multiclass_classification_metrics.[ Macro; Weighted ])
    [ One_vs_rest; One_vs_one ];
  check_float "micro one-versus-rest is one when separable" 1.0
    (roc_auc ~average:Multiclass_classification_metrics.Micro ~truth ~classes
       ~probabilities:separable ()
    |> get);
  roc_auc ~strategy:One_vs_one ~average:Multiclass_classification_metrics.Micro
    ~truth ~classes ~probabilities ()
  |> expect_error is_validation;
  let missing = classification [| 0; 1; 1; 0 |] in
  roc_auc ~truth:missing ~classes ~probabilities ()
  |> expect_error is_validation;
  check_float "absent class falls back to one half under OvR macro"
    ((1.0 +. 1.0 +. 0.5) /. 3.0)
    (roc_auc ~undefined:Undefined_metric_policy.Use_fallback ~truth:missing
       ~classes
       ~probabilities:
         (matrix
            [|
              [| 0.7; 0.2; 0.1 |];
              [| 0.2; 0.7; 0.1 |];
              [| 0.3; 0.6; 0.1 |];
              [| 0.6; 0.3; 0.1 |];
            |])
       ()
    |> get);
  check_float "OvO ignores absent classes" 1.0
    (roc_auc ~strategy:One_vs_one ~truth:missing ~classes
       ~probabilities:
         (matrix
            [|
              [| 0.7; 0.2; 0.1 |];
              [| 0.2; 0.7; 0.1 |];
              [| 0.3; 0.6; 0.1 |];
              [| 0.6; 0.3; 0.1 |];
            |])
       ()
    |> get);
  let duplicated_truth = classification [| 0; 1; 2; 1; 0; 1 |] in
  let duplicated =
    matrix
      [|
        [| 0.6; 0.3; 0.1 |];
        [| 0.2; 0.5; 0.3 |];
        [| 0.1; 0.2; 0.7 |];
        [| 0.4; 0.4; 0.2 |];
        [| 0.6; 0.3; 0.1 |];
        [| 0.2; 0.5; 0.3 |];
      |]
  in
  let sample_weight = weights [| 2.0; 2.0; 1.0; 1.0 |] in
  List.iter
    (fun strategy ->
      check_float "integer weights match row replication"
        (roc_auc ~strategy ~average:Multiclass_classification_metrics.Weighted
           ~truth:duplicated_truth ~classes ~probabilities:duplicated ()
        |> get)
        (roc_auc ~strategy ~average:Multiclass_classification_metrics.Weighted
           ~sample_weight ~truth ~classes ~probabilities ()
        |> get))
    [ One_vs_rest; One_vs_one ];
  roc_auc ~truth ~classes:[| 0; 1 |] ~probabilities () |> expect_error is_shape

let test_top_k () =
  let open Multiclass_ranking in
  check_float "top-1 breaks the row-3 tie toward the higher column" 1.0
    (top_k_accuracy ~k:1 ~truth ~classes ~probabilities () |> get);
  let lower_tie = classification [| 0; 1; 2; 0 |] in
  check_float "the lower tied column loses" 0.75
    (top_k_accuracy ~k:1 ~truth:lower_tie ~classes ~probabilities () |> get);
  check_float "top-2" 1.0
    (top_k_accuracy ~k:2 ~truth ~classes ~probabilities () |> get);
  top_k_accuracy ~k:0 ~truth ~classes ~probabilities ()
  |> expect_error is_validation;
  top_k_accuracy ~k:3 ~truth ~classes ~probabilities ()
  |> expect_error is_validation;
  let scorer = Multiclass_classification_scorer.top_k_accuracy ~k:2 in
  Alcotest.(check string)
    "top-k scorer name" "top_2_accuracy"
    (Multiclass_classification_scorer.name scorer);
  Alcotest.(check string)
    "ovo weighted scorer name" "roc_auc_ovo_weighted"
    (Multiclass_classification_scorer.name
       (Multiclass_classification_scorer.roc_auc ~strategy:One_vs_one
          ~average:Multiclass_classification_metrics.Weighted ()));
  Alcotest.(check string)
    "ovr macro scorer name" "roc_auc_ovr"
    (Multiclass_classification_scorer.name
       (Multiclass_classification_scorer.roc_auc ()));
  let prediction =
    Multiclass_prediction.create ~classes ~probabilities () |> get
  in
  check_float "scorer dispatch" 1.0
    (Multiclass_classification_scorer.score scorer ~truth ~prediction () |> get)

let test_gains () =
  let open Ranking_metrics in
  let relevance = matrix [| [| 3.0; 2.0; 0.0 |] |] in
  let ideal = matrix [| [| 0.9; 0.5; 0.1 |] |] in
  let expected_dcg = 3.0 +. (2.0 /. Float.log2 3.0) in
  check_float "ideal dcg" expected_dcg (dcg ~relevance ~scores:ideal () |> get);
  check_float "ideal ndcg" 1.0 (ndcg ~relevance ~scores:ideal () |> get);
  let reversed = matrix [| [| 0.1; 0.5; 0.9 |] |] in
  check_float "reversed dcg"
    ((2.0 /. Float.log2 3.0) +. (3.0 /. Float.log2 4.0))
    (dcg ~relevance ~scores:reversed () |> get);
  let tied = matrix [| [| 0.5; 0.5; 0.1 |] |] in
  check_float "tie averaging shares the top two discounts"
    (2.5 *. (1.0 +. (1.0 /. Float.log2 3.0)))
    (dcg ~relevance ~scores:tied () |> get);
  check_float "ignored ties rank the higher column first"
    (2.0 +. (3.0 /. Float.log2 3.0))
    (dcg ~ignore_ties:true ~relevance ~scores:tied () |> get);
  check_float "cutoff keeps the first rank only" 3.0
    (dcg ~k:1 ~relevance ~scores:ideal () |> get);
  check_float "all-zero relevance scores zero" 0.0
    (ndcg
       ~relevance:(matrix [| [| 0.0; 0.0 |] |])
       ~scores:(matrix [| [| 0.2; 0.1 |] |])
       ()
    |> get);
  let two_rows = matrix [| [| 3.0; 2.0; 0.0 |]; [| 0.0; 0.0; 1.0 |] |] in
  let two_scores = matrix [| [| 0.9; 0.5; 0.1 |]; [| 0.9; 0.5; 0.1 |] |] in
  check_float "weighted mean over rows"
    (((2.0 *. 1.0) +. (1.0 /. Float.log2 4.0)) /. 3.0)
    (ndcg
       ~sample_weight:(weights [| 2.0; 1.0 |])
       ~relevance:two_rows ~scores:two_scores ()
    |> get);
  ndcg ~relevance:(matrix [| [| 1.0 |] |]) ~scores:(matrix [| [| 1.0 |] |]) ()
  |> expect_error is_validation;
  ndcg ~relevance ~scores:(matrix [| [| 1.0; 2.0 |] |]) ()
  |> expect_error is_shape;
  ndcg ~relevance:(matrix [| [| -1.0; 0.0; 1.0 |] |]) ~scores:ideal ()
  |> expect_error is_validation;
  ndcg ~k:0 ~relevance ~scores:ideal () |> expect_error is_validation

let () =
  Alcotest.run "ranking metrics"
    [
      ( "metrics",
        [
          Alcotest.test_case "average precision" `Quick test_average_precision;
          Alcotest.test_case "multiclass ROC AUC" `Quick test_multiclass_auc;
          Alcotest.test_case "top-k accuracy" `Quick test_top_k;
          Alcotest.test_case "discounted cumulative gain" `Quick test_gains;
        ] );
    ]
