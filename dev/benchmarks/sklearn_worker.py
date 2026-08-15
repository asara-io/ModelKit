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


def linear_models(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    coefficients = ((columns[0] % 5) - 2).astype(np.float64) * 0.2
    noise = (((rows[:, 0] * 13 + 1729) % 11) - 5).astype(np.float64) * 0.01
    regression_target = 1.25 + x @ coefficients + noise
    score = x[:, 0] + 0.25 * x[:, 1] - 0.1 * x[:, 2]
    classification_target = np.where(score > 0.0, 7, -3)
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25

    linear = LinearRegression().fit(
        x, regression_target, sample_weight=sample_weight
    )
    linear_prediction = linear.predict(x)
    ridge = Ridge(alpha=scenario["ridge_alpha"], solver="svd").fit(
        x, regression_target, sample_weight=sample_weight
    )
    ridge_prediction = ridge.predict(x)
    logistic = LogisticRegression(
        C=scenario["logistic_c"],
        solver=scenario["logistic_solver"],
        tol=scenario["logistic_tolerance"],
        max_iter=scenario["logistic_max_iterations"],
    ).fit(x, classification_target, sample_weight=sample_weight)
    probabilities = logistic.predict_proba(x)
    prediction = logistic.predict(x)
    boundary = int(np.argmin(np.abs(score)))
    next_boundary = min(x.shape[0] - 1, boundary + 1)
    signature = np.array(
        [
            linear_prediction[0],
            linear_prediction[-1],
            ridge_prediction[0],
            ridge_prediction[-1],
            probabilities[boundary, 1],
            probabilities[next_boundary, 1],
            prediction[boundary],
            prediction[next_boundary],
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "ordinary_least_squares",
            "ridge_regression",
            "binary_logistic_regression",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def splitters(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    dataset = scenario["dataset"]
    samples = dataset["samples"]
    folds = scenario["folds"]
    x = np.zeros((samples, 1), dtype=np.float64)
    target = np.arange(samples, dtype=np.int64) % dataset["classes"]
    groups = np.arange(samples, dtype=np.int64) // dataset["group_size"]
    split_values = [
        list(KFold(n_splits=folds, shuffle=False).split(x)),
        list(StratifiedKFold(n_splits=folds, shuffle=False).split(x, target)),
        list(GroupKFold(n_splits=folds).split(x, groups=groups)),
        list(
            TimeSeriesSplit(
                n_splits=folds,
                test_size=scenario["time_test_size"],
                gap=scenario["time_gap"],
            ).split(x)
        ),
    ]

    def statistics(splits) -> list[int]:
        train_sizes = [len(train) for train, _ in splits]
        test_sizes = [len(test) for _, test in splits]
        return [
            len(splits),
            sum(train_sizes),
            sum(test_sizes),
            min(test_sizes),
            max(test_sizes),
        ]

    signature = [value for splits in split_values for value in statistics(splits)]
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(
            np.asarray(signature, dtype="<i8").tobytes()
        ).hexdigest(),
        "folds": folds,
        "operations": [
            "k_fold",
            "stratified_k_fold",
            "group_k_fold",
            "time_series_split",
        ],
        "samples": samples,
        "signature": signature,
        "threadpools": threadpools(),
    }


def metrics(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_curve,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        root_mean_squared_error,
    )

    samples = scenario["dataset"]["samples"]
    index = np.arange(samples, dtype=np.int64)
    regression_truth = (index % 1000).astype(np.float64) / 10.0
    regression_prediction = regression_truth + ((index % 7) - 3) * 0.01
    classification_truth = index % 2
    classification_prediction = np.where(
        index % 11 == 0, 1 - classification_truth, classification_truth
    )
    adjustment = (index % 17) * 0.02
    positive_probability = np.where(
        classification_truth == 1, 0.55 + adjustment, 0.45 - adjustment
    )
    sample_weight = 1.0 + ((index % 5) * 0.25)
    false_positive_rate, _, roc_thresholds = roc_curve(
        classification_truth,
        positive_probability,
        sample_weight=sample_weight,
        drop_intermediate=False,
    )
    _, _, precision_recall_thresholds = precision_recall_curve(
        classification_truth,
        positive_probability,
        sample_weight=sample_weight,
    )
    scalar = np.array(
        [
            mean_absolute_error(
                regression_truth,
                regression_prediction,
                sample_weight=sample_weight,
            ),
            mean_squared_error(
                regression_truth,
                regression_prediction,
                sample_weight=sample_weight,
            ),
            root_mean_squared_error(
                regression_truth,
                regression_prediction,
                sample_weight=sample_weight,
            ),
            r2_score(
                regression_truth,
                regression_prediction,
                sample_weight=sample_weight,
            ),
            accuracy_score(
                classification_truth,
                classification_prediction,
                sample_weight=sample_weight,
            ),
            balanced_accuracy_score(
                classification_truth,
                classification_prediction,
                sample_weight=sample_weight,
            ),
            precision_score(
                classification_truth,
                classification_prediction,
                sample_weight=sample_weight,
            ),
            recall_score(
                classification_truth,
                classification_prediction,
                sample_weight=sample_weight,
            ),
            f1_score(
                classification_truth,
                classification_prediction,
                sample_weight=sample_weight,
            ),
            log_loss(
                classification_truth,
                positive_probability,
                sample_weight=sample_weight,
                labels=[0, 1],
            ),
            roc_auc_score(
                classification_truth,
                positive_probability,
                sample_weight=sample_weight,
            ),
        ],
        dtype=np.float64,
    )
    signature = [
        *scalar.tolist(),
        float(len(roc_thresholds)),
        float(len(precision_recall_thresholds)),
        float(np.mean(scalar)),
        float(np.std(scalar)),
    ]
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(
            np.asarray(signature, dtype="<f8").tobytes()
        ).hexdigest(),
        "operations": [
            "regression_metrics",
            "classification_metrics",
            "ranking_curves",
            "score_aggregation",
        ],
        "samples": samples,
        "signature": signature,
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
    elif workload == "linear_models":
        result = linear_models(scenario)
    elif workload == "splitters":
        result = splitters(scenario)
    elif workload == "metrics":
        result = metrics(scenario)
    else:
        raise ValueError(f"unknown workload {workload!r}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
