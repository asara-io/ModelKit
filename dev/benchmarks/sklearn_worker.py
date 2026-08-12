from pathlib import Path
import argparse
import hashlib
import json


def threadpools() -> list[dict[str, object]]:
    from threadpoolctl import threadpool_info

    return [
        {
            key: pool.get(key)
            for key in (
                "architecture",
                "internal_api",
                "num_threads",
                "prefix",
                "user_api",
                "version",
            )
        }
        for pool in threadpool_info()
    ]


def dummy_cv(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    dataset = scenario["dataset"]
    rng = np.random.default_rng(dataset["seed"])
    x = rng.standard_normal((dataset["samples"], dataset["features"]))
    y = (x[:, 0] + 0.25 * x[:, 1] > 0.0).astype(np.int64)
    splitter = StratifiedKFold(
        n_splits=scenario["splitter"]["folds"],
        shuffle=True,
        random_state=scenario["splitter"]["seed"],
    )
    result = cross_validate(
        DummyClassifier(strategy="prior"),
        x,
        y,
        cv=splitter,
        scoring=("accuracy", "neg_log_loss"),
        n_jobs=1,
    )
    scores = np.concatenate((result["test_accuracy"], result["test_neg_log_loss"]))
    return {
        "checksum": hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest(),
        "folds": len(result["test_accuracy"]),
        "threadpools": threadpools(),
    }


def preprocessing(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = ((rows * 17 + columns * 31 + dataset["seed"]) % 1000).astype(np.float64)
    x /= 100.0
    x[:, 0] = 1.0
    missing = (
        (columns > 0)
        & ((rows * 101 + columns * 53 + dataset["seed"]) % dataset["missing_modulus"] == 0)
    )
    x[missing] = np.nan
    complete = SimpleImputer(strategy="mean").fit_transform(x)
    median_complete = SimpleImputer(strategy="median").fit_transform(x)
    constant_complete = SimpleImputer(
        strategy="constant", fill_value=scenario["imputation_constant"]
    ).fit_transform(x)
    scaled = StandardScaler().fit_transform(complete)
    selected = VarianceThreshold(threshold=scenario["variance_threshold"]).fit_transform(
        scaled
    )
    signature = np.array(
        [
            selected.shape[0],
            selected.shape[1],
            selected[0, 0],
            selected[-1, -1],
            median_complete[0, 1],
            constant_complete[0, 1],
        ],
        dtype="<f8",
    )
    return {
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "allocated_words": None,
        "features_out": selected.shape[1],
        "operations": [
            "constant_imputation",
            "mean_imputation",
            "median_imputation",
            "standard_scaling",
            "variance_threshold",
        ],
        "samples": selected.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    workload = scenario.get("workload", "dummy_cv")
    if workload == "dummy_cv":
        result = dummy_cv(scenario)
    elif workload == "preprocessing":
        result = preprocessing(scenario)
    else:
        raise ValueError(f"unknown workload {workload!r}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
