from importlib.metadata import version
from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "python"))
import environment


def native_seed(seed: int) -> int:
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & 0x7FFFFFFF


def cases():
    import numpy as np

    for name, rows, columns in [
        ("dense", 32, 3),
        ("weighted", 24, 3),
        ("missing", 24, 3),
        ("sampled", 64, 4),
        ("regularized", 32, 3),
        ("large_scale", 24, 3),
        ("constant_features", 13, 3),
        ("constant_target", 17, 2),
        ("single_row", 1, 2),
        ("all_missing", 16, 2),
        ("zero_learning_rate", 24, 3),
    ]:
        row = np.arange(rows)[:, None]
        column = np.arange(columns)[None, :]
        x = ((row * (3 + column * 5) + column * 7) % 31 - 15) / 4.0
        y = 1.5 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + (row[:, 0] % 3) * 0.1
        weight = None if name in {"dense", "sampled", "all_missing"} else (
            np.where(row[:, 0] % 5 == 0, 0.0, 1.0 + (row[:, 0] % 3) / 2.0)
        )
        if name == "single_row":
            weight = np.array([2.0])
        if name == "constant_features":
            x[:] = 2.0
        if name == "constant_target":
            y[:] = -2.5
        if name == "all_missing":
            x[:] = np.nan
        if name == "missing":
            x[(row + column) % 5 == 0] = np.nan
        if name == "large_scale":
            x *= 1e10
            y *= 1e8
        test_row = np.arange(8)[:, None]
        test = ((test_row * (11 + column * 3) + column * 13) % 29 - 14) / 3.0
        if name in {"missing", "all_missing"}:
            test[::2, 0] = np.nan
        if name == "large_scale":
            test *= 1e10
        config = {
            "rounds": 12,
            "max_depth": 3,
            "max_bin": 32,
            "learning_rate": 0.0 if name == "zero_learning_rate" else 0.2,
            "min_child_weight": 2.0 if name == "regularized" else 1.0,
            "min_split_loss": 0.1 if name == "regularized" else 0.0,
            "reg_alpha": 0.2 if name == "regularized" else 0.0,
            "reg_lambda": 3.0 if name == "regularized" else 1.0,
            "subsample": 0.75 if name == "sampled" else 1.0,
            "colsample_bytree": 0.75 if name == "sampled" else 1.0,
            "threads": 1,
            "seed": 41,
        }
        yield name, config, x, y, weight, test


def generate_regression(reference_version: str) -> None:
    environment.validate()
    observed = version("xgboost")
    if observed != reference_version:
        raise RuntimeError(f"expected xgboost {reference_version}, observed {observed}")
    import numpy as np
    import xgboost as xgb

    directory = ROOT / "test" / "fixtures" / "xgboost"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# ModelKit XGBoost regression reference v1"]
    configurations = {}

    def numbers(values) -> str:
        return ",".join(format(float(value), ".17g") for value in values)

    for name, config, x, y, weight, test in cases():
        parameters = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "device": "cpu",
            "tree_method": "hist",
            "validate_parameters": True,
            "verbosity": 0,
            "seed": native_seed(config["seed"]),
            "nthread": config["threads"],
            "max_depth": config["max_depth"],
            "max_bin": config["max_bin"],
            "eta": config["learning_rate"],
            "min_child_weight": config["min_child_weight"],
            "gamma": config["min_split_loss"],
            "alpha": config["reg_alpha"],
            "lambda": config["reg_lambda"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
        }
        training = xgb.DMatrix(
            x.astype(np.float32), label=y.astype(np.float32),
            weight=None if weight is None else weight.astype(np.float32),
            missing=np.nan, nthread=1,
        )
        model = xgb.train(parameters, training, num_boost_round=config["rounds"])
        expected = model.inplace_predict(test.astype(np.float32))
        for field, value in config.items():
            lines.append(f"{name}\t{field}\t{value}")
        lines.append(f"{name}\tnative_seed\t{parameters['seed']}")
        for field, matrix in [("x_train", x), ("x_predict", test)]:
            for index, values in enumerate(matrix):
                lines.append(f"{name}\t{field}\t{index}\t{numbers(values)}")
        for field, values in [("target", y), ("expected", expected)]:
            lines.append(f"{name}\t{field}\t{numbers(values)}")
        if weight is not None:
            lines.append(f"{name}\tweight\t{numbers(weight)}")
        configurations[name] = config

    stem = "regression_v1" if reference_version == "3.4.1" else "regression_v1_3_0_4"
    (directory / f"{stem}.tsv").write_text("\n".join(lines) + "\n")
    metadata = {
        "fixture": f"xgboost_{stem}",
        "generator": "dev/fixtures/generate_xgboost.py",
        "schema_version": 1,
        "reference": "xgboost.train / Booster.inplace_predict",
        "environment": {**environment.metadata(), "xgboost": observed},
        "configuration": configurations,
        "precision": "float32 features, labels, weights, and native predictions",
        "tolerance": {"relative": 2e-6, "absolute": 2e-6},
        "license": "Apache-2.0",
    }
    (directory / f"{stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def classification_cases():
    import numpy as np

    for kind, labels in [
        ("binary", np.array([16777219, -16777219], dtype=np.int64)),
        ("multiclass", np.array([20000003, -19, 7], dtype=np.int64)),
    ]:
        for variant in ["dense", "weighted", "missing", "sampled", "regularized",
                        "constant_features", "all_missing", "imbalanced", "zero_weight_class"]:
            rows, columns = (80, 4) if variant in {"sampled", "imbalanced"} else (48, 3)
            row = np.arange(rows)[:, None]
            column = np.arange(columns)[None, :]
            x = ((row * (3 + column * 5) + column * 7) % 31 - 15) / 4.0
            if kind == "binary":
                encoded = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(np.int64)
            else:
                scores = np.stack([x[:, 0] - 0.2 * x[:, 1], x[:, 1] + 0.5 * x[:, 2], -x[:, 0] - 0.7 * x[:, 1]], axis=1)
                encoded = np.argmax(scores, axis=1)
            if variant == "imbalanced":
                index = row[:, 0]
                encoded = (index % 13 == 0).astype(np.int64) if kind == "binary" else np.where(index % 17 == 0, 0, np.where(index % 19 == 0, 2, 1))
            if variant == "constant_features":
                x[:] = 1.0
                encoded = row[:, 0] % len(labels)
            y = labels[encoded]
            weight = None
            if variant in {"weighted", "missing", "regularized"}:
                weight = np.where(row[:, 0] % 11 == 0, 0.0, 1.0 + (row[:, 0] % 3) / 2.0)
            if variant == "missing":
                x[(row + column) % 5 == 0] = np.nan
            if variant == "all_missing":
                x[:] = np.nan
            if variant == "zero_weight_class":
                x = np.vstack([x, np.full((1, columns), 1e6)])
                y = np.append(y, 99999999)
                weight = np.append(np.ones(rows), 0.0)
            test_row = np.arange(8)[:, None]
            test = ((test_row * (11 + column * 3) + column * 13) % 29 - 14) / 3.0
            if variant in {"missing", "all_missing"}:
                test[::2, 0] = np.nan
            config = {
                "rounds": 12, "max_depth": 3, "max_bin": 32,
                "learning_rate": 0.2,
                "min_child_weight": 2.0 if variant == "regularized" else 1.0,
                "min_split_loss": 0.1 if variant == "regularized" else 0.0,
                "reg_alpha": 0.2 if variant == "regularized" else 0.0,
                "reg_lambda": 3.0 if variant == "regularized" else 1.0,
                "subsample": 0.75 if variant == "sampled" else 1.0,
                "colsample_bytree": 0.75 if variant == "sampled" else 1.0,
                "threads": 1, "seed": 41,
            }
            yield f"{kind}_{variant}", config, x, y, weight, test


def generate_classification(reference_version: str) -> None:
    environment.validate()
    observed = version("xgboost")
    if observed != reference_version:
        raise RuntimeError(f"expected xgboost {reference_version}, observed {observed}")
    import numpy as np
    import xgboost as xgb

    directory = ROOT / "test" / "fixtures" / "xgboost"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# ModelKit XGBoost classification reference v1"]
    configurations = {}

    def numbers(values) -> str:
        return ",".join(format(float(value), ".17g") for value in values)

    def integers(values) -> str:
        return ",".join(str(int(value)) for value in values)

    for name, config, x, y, weight, test in classification_cases():
        active = np.ones(len(y), dtype=bool) if weight is None else weight > 0
        classes = np.unique(y[active])
        encoded = np.searchsorted(classes, y[active]).astype(np.float32)
        parameters = {
            "objective": "binary:logistic" if len(classes) == 2 else "multi:softprob",
            "booster": "gbtree", "device": "cpu", "tree_method": "hist",
            "validate_parameters": True, "verbosity": 0,
            "seed": native_seed(config["seed"]), "nthread": 1,
            "max_depth": config["max_depth"], "max_bin": config["max_bin"],
            "eta": config["learning_rate"], "min_child_weight": config["min_child_weight"],
            "gamma": config["min_split_loss"], "alpha": config["reg_alpha"],
            "lambda": config["reg_lambda"], "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
        }
        if len(classes) > 2:
            parameters["num_class"] = len(classes)
        training = xgb.DMatrix(
            x[active].astype(np.float32), label=encoded,
            weight=None if weight is None else weight[active].astype(np.float32),
            missing=np.nan, nthread=1,
        )
        model = xgb.train(parameters, training, num_boost_round=config["rounds"])
        raw = model.inplace_predict(test.astype(np.float32)).astype(np.float64)
        probability = np.stack([1.0 - raw, raw], axis=1) if len(classes) == 2 else raw / raw.sum(axis=1, keepdims=True)
        prediction = classes[np.argmax(probability, axis=1)]
        for field, value in config.items():
            lines.append(f"{name}\t{field}\t{value}")
        lines.append(f"{name}\tnative_seed\t{parameters['seed']}")
        lines.append(f"{name}\teffective_rows\t{int(active.sum())}")
        for field, matrix in [("x_train", x), ("x_predict", test), ("probabilities", probability)]:
            for index, values in enumerate(matrix):
                lines.append(f"{name}\t{field}\t{index}\t{numbers(values)}")
        for field, values in [("target", y), ("classes", classes), ("prediction", prediction)]:
            lines.append(f"{name}\t{field}\t{integers(values)}")
        if weight is not None:
            lines.append(f"{name}\tweight\t{numbers(weight)}")
        configurations[name] = {**config, "objective": parameters["objective"], "classes": classes.tolist()}

    stem = "classification_v1" if reference_version == "3.4.1" else "classification_v1_3_0_4"
    (directory / f"{stem}.tsv").write_text("\n".join(lines) + "\n")
    metadata = {
        "fixture": f"xgboost_{stem}", "generator": "dev/fixtures/generate_xgboost.py",
        "schema_version": 1, "reference": "xgboost.train / Booster.inplace_predict",
        "environment": {**environment.metadata(), "xgboost": observed},
        "configuration": configurations,
        "precision": "float32 native input/output; float64 normalized probability rows",
        "label_policy": "sorted positively weighted original integer labels; zero-weight rows excluded",
        "prediction_ties": "lowest original label",
        "tolerance": {"relative": 2e-6, "absolute": 2e-6}, "license": "Apache-2.0",
    }
    (directory / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-version", choices=("3.4.1", "3.0.4"), default="3.4.1")
    parser.add_argument("--task", choices=("regression", "classification", "all"), default="regression")
    arguments = parser.parse_args()
    if arguments.task in {"regression", "all"}:
        generate_regression(arguments.reference_version)
    if arguments.task in {"classification", "all"}:
        generate_classification(arguments.reference_version)
