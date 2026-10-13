open Bigarray

let vector values = Array1.of_array float32 c_layout values
let matrix values = Array2.of_array float32 c_layout values

let parameters ~value ~objective ~classes =
  [
    ("objective", objective);
    ("booster", "gbtree");
    ("device", "cpu");
    ("tree_method", "hist");
    ("verbosity", "0");
    ("nthread", "1");
    ("seed", value "native_seed");
  ]
  @ List.map
      (fun (native, field) -> (native, value field))
      [
        ("max_depth", "max_depth");
        ("max_bin", "max_bin");
        ("eta", "learning_rate");
        ("min_child_weight", "min_child_weight");
        ("gamma", "min_split_loss");
        ("alpha", "reg_alpha");
        ("lambda", "reg_lambda");
        ("subsample", "subsample");
        ("colsample_bytree", "colsample_bytree");
      ]
  @ if classes > 2 then [ ("num_class", string_of_int classes) ] else []

let predict ~parameters ~rounds ~x ~y ~sample_weight ~test =
  let training = Xgboost.DMatrix.of_bigarray2 (matrix x) in
  Fun.protect
    ~finally:(fun () -> Xgboost.DMatrix.free training)
    (fun () ->
      Xgboost.DMatrix.set_label training (vector y);
      Option.iter
        (fun weights -> Xgboost.DMatrix.set_weight training (vector weights))
        sample_weight;
      let booster = Xgboost.Booster.create ~cache:[ training ] () in
      Fun.protect
        ~finally:(fun () -> Xgboost.Booster.free booster)
        (fun () ->
          Xgboost.Booster.set_params booster parameters;
          for iter = 0 to rounds - 1 do
            Xgboost.Booster.update_one_iter booster ~iter ~dtrain:training
          done;
          let predictions =
            Xgboost.Booster.predict_dense booster (matrix test)
          in
          Array.init (Array1.dim predictions) (Array1.get predictions)))

(* Column sampling uses std::shuffle, whose sequence depends on the C++ runtime. *)
let frozen_sampling_reference () = Native_reference_platform.system = "macosx"
