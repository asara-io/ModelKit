from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "python"))

import environment


def comma_separated(values) -> str:
    return ",".join(str(int(value)) for value in values)


def float_value(value) -> str:
    value = float(value)
    if value != value:
        return "nan"
    return format(value, ".17g")


def float_values(values) -> str:
    return ",".join(float_value(value) for value in values)


def generate_split_fixture(fixture_dir: Path) -> None:
    from sklearn.model_selection import StratifiedKFold

    target = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=1729)
    splits = list(splitter.split([[0.0]] * len(target), target))

    data_path = fixture_dir / "stratified_kfold_v1.tsv"
    metadata_path = fixture_dir / "stratified_kfold_v1.metadata.json"
    rows = [
        "# ModelKit sklearn reference fixture v1",
        "target\t" + comma_separated(target),
    ]
    for index, (train, test) in enumerate(splits):
        rows.append(f"fold\t{index}\ttrain\t{comma_separated(train)}")
        rows.append(f"fold\t{index}\ttest\t{comma_separated(test)}")
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    metadata = {
        "configuration": {
            "n_splits": 3,
            "random_state": 1729,
            "shuffle": True,
        },
        "environment": environment.metadata(),
        "fixture": "stratified_kfold_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "reference": "sklearn.model_selection.StratifiedKFold",
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_splitter_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    k_fold_samples = 11
    stratified_target = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        dtype=np.int64,
    )
    group_values = np.array(
        [10] * 6 + [20] * 5 + [30] * 4 + [40] * 3 + [50] * 2 + [60] * 2,
        dtype=np.int64,
    )
    time_samples = 12
    folds = 3

    k_fold = list(KFold(n_splits=folds, shuffle=False).split(np.zeros(k_fold_samples)))
    stratified = list(
        StratifiedKFold(n_splits=folds, shuffle=False).split(
            np.zeros(len(stratified_target)), stratified_target
        )
    )
    grouped = list(
        GroupKFold(n_splits=folds).split(
            np.zeros(len(group_values)), groups=group_values
        )
    )
    time_series = list(
        TimeSeriesSplit(n_splits=folds, test_size=2, gap=1).split(
            np.zeros(time_samples)
        )
    )

    rows = ["# ModelKit sklearn splitter reference fixture v1"]

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{comma_separated(values)}")

    def add_splits(name: str, splits) -> None:
        for index, (train, test) in enumerate(splits):
            rows.append(f"{name}_train\t{index}\t{comma_separated(train)}")
            rows.append(f"{name}_test\t{index}\t{comma_separated(test)}")

    add_vector("k_fold_sample_count", [k_fold_samples])
    add_splits("k_fold", k_fold)
    add_vector("stratified_target", stratified_target)
    add_splits("stratified", stratified)
    add_vector("group_values", group_values)
    add_splits("group", grouped)
    add_vector("time_sample_count", [time_samples])
    add_splits("time", time_series)

    data_path = fixture_dir / "splitters_v1.tsv"
    metadata_path = fixture_dir / "splitters_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "folds": folds,
            "group_shuffle": False,
            "k_fold_shuffle": False,
            "stratified_shuffle": False,
            "time_gap": 1,
            "time_test_size": 2,
        },
        "environment": environment.metadata(),
        "fixture": "splitters_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.model_selection.KFold",
            "sklearn.model_selection.StratifiedKFold",
            "sklearn.model_selection.GroupKFold",
            "sklearn.model_selection.TimeSeriesSplit",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_preprocessing_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    x = np.array(
        [
            [1.0, np.nan, 5.0, 0.0],
            [2.0, 4.0, 5.0, 1.0],
            [np.nan, 8.0, 5.0, 2.0],
            [4.0, 6.0, 5.0, 3.0],
            [5.0, 10.0, 5.0, 4.0],
        ],
        dtype=np.float64,
    )
    mean_imputer = SimpleImputer(strategy="mean")
    median_imputer = SimpleImputer(strategy="median")
    constant_imputer = SimpleImputer(strategy="constant", fill_value=-2.0)
    mean_output = mean_imputer.fit_transform(x)
    median_output = median_imputer.fit_transform(x)
    constant_output = constant_imputer.fit_transform(x)
    scaler = StandardScaler().fit(mean_output)
    scaled_output = scaler.transform(mean_output)
    threshold = 1.5
    selector = VarianceThreshold(threshold=threshold).fit(mean_output)
    selected_output = selector.transform(mean_output)

    rows = ["# ModelKit sklearn preprocessing reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("input", x)
    add_vector("mean_statistics", mean_imputer.statistics_)
    add_matrix("mean_output", mean_output)
    add_vector("median_statistics", median_imputer.statistics_)
    add_matrix("median_output", median_output)
    add_vector("constant_statistics", constant_imputer.statistics_)
    add_matrix("constant_output", constant_output)
    add_vector("scaler_mean", scaler.mean_)
    add_vector("scaler_variance", scaler.var_)
    add_vector("scaler_scale", scaler.scale_)
    add_matrix("scaled_output", scaled_output)
    add_vector("variance_threshold", [threshold])
    add_vector("feature_variances", selector.variances_)
    add_vector("selected_indices", np.flatnonzero(selector.get_support()))
    add_matrix("selected_output", selected_output)

    data_path = fixture_dir / "preprocessing_v1.tsv"
    metadata_path = fixture_dir / "preprocessing_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "constant": -2.0,
            "features": 4,
            "samples": 5,
            "variance_threshold": threshold,
        },
        "environment": environment.metadata(),
        "fixture": "preprocessing_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.feature_selection.VarianceThreshold",
            "sklearn.impute.SimpleImputer",
            "sklearn.preprocessing.StandardScaler",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_linear_model_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

    x_train = np.array(
        [
            [-2.0, 0.5, 1.0],
            [-1.0, -1.5, 0.0],
            [0.0, 2.0, -0.5],
            [1.0, -0.5, 2.0],
            [2.0, 1.5, 1.0],
            [3.0, -2.0, -1.0],
            [4.0, 0.25, 0.5],
            [5.0, 2.5, -2.0],
        ],
        dtype=np.float64,
    )
    x_predict = np.array(
        [
            [-1.5, 0.0, 0.5],
            [0.5, 1.0, -1.0],
            [2.5, -1.0, 1.5],
            [6.0, 0.75, -0.25],
        ],
        dtype=np.float64,
    )
    regression_target = np.array(
        [-1.15, 0.4, -2.25, 2.6, 2.85, 8.3, 7.175, 4.0], dtype=np.float64
    )
    classification_target = np.array([-3, -3, -3, 7, 7, -3, 7, 7], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    ridge_alpha = 2.5
    logistic_c = 1.7

    linear = LinearRegression().fit(
        x_train, regression_target, sample_weight=sample_weight
    )
    ridge = Ridge(alpha=ridge_alpha, solver="svd").fit(
        x_train, regression_target, sample_weight=sample_weight
    )
    logistic = LogisticRegression(
        C=logistic_c,
        fit_intercept=True,
        solver="lbfgs",
        tol=1e-12,
        max_iter=1000,
    ).fit(x_train, classification_target, sample_weight=sample_weight)

    rows = ["# ModelKit sklearn linear-model reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("regression_target", regression_target)
    add_vector("classification_target", classification_target)
    add_vector("sample_weight", sample_weight)
    add_vector("ridge_alpha", [ridge_alpha])
    add_vector("logistic_c", [logistic_c])
    add_vector("linear_coefficients", linear.coef_)
    add_vector("linear_intercept", [linear.intercept_])
    add_vector("linear_prediction", linear.predict(x_predict))
    add_vector("ridge_coefficients", ridge.coef_)
    add_vector("ridge_intercept", [ridge.intercept_])
    add_vector("ridge_prediction", ridge.predict(x_predict))
    add_vector("logistic_classes", logistic.classes_)
    add_vector("logistic_coefficients", logistic.coef_[0])
    add_vector("logistic_intercept", logistic.intercept_)
    add_vector("logistic_decision", logistic.decision_function(x_predict))
    add_matrix("logistic_probabilities", logistic.predict_proba(x_predict))
    add_vector("logistic_prediction", logistic.predict(x_predict))

    data_path = fixture_dir / "linear_models_v1.tsv"
    metadata_path = fixture_dir / "linear_models_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "features": x_train.shape[1],
            "logistic_c": logistic_c,
            "logistic_max_iter": 1000,
            "logistic_solver": "lbfgs",
            "logistic_tol": 1e-12,
            "prediction_samples": x_predict.shape[0],
            "ridge_alpha": ridge_alpha,
            "samples": x_train.shape[0],
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "linear_models_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.linear_model.LinearRegression",
            "sklearn.linear_model.Ridge",
            "sklearn.linear_model.LogisticRegression",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    environment.validate()
    fixture_dir = ROOT / "test" / "fixtures" / "sklearn"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    generate_split_fixture(fixture_dir)
    generate_splitter_fixture(fixture_dir)
    generate_preprocessing_fixture(fixture_dir)
    generate_linear_model_fixture(fixture_dir)


if __name__ == "__main__":
    main()
