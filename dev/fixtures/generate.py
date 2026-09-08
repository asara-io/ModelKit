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


def generate_metrics_fixture(fixture_dir: Path) -> None:
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

    regression_truth = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    regression_prediction = np.array([1.5, 1.0, 5.0, 7.0, 18.0])
    regression_weight = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    classification_truth = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    classification_prediction = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    positive_probability = np.array([0.05, 0.8, 0.7, 0.6, 0.4, 0.2, 0.9, 0.3])
    classification_weight = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 2.5, 0.75, 1.25])

    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        classification_truth,
        positive_probability,
        sample_weight=classification_weight,
        drop_intermediate=False,
    )
    precision_values, recall_values, precision_recall_thresholds = (
        precision_recall_curve(
            classification_truth,
            positive_probability,
            sample_weight=classification_weight,
        )
    )
    values = {
        "regression_truth": regression_truth,
        "regression_prediction": regression_prediction,
        "regression_weight": regression_weight,
        "mean_absolute_error": [
            mean_absolute_error(
                regression_truth,
                regression_prediction,
                sample_weight=regression_weight,
            )
        ],
        "mean_squared_error": [
            mean_squared_error(
                regression_truth,
                regression_prediction,
                sample_weight=regression_weight,
            )
        ],
        "root_mean_squared_error": [
            root_mean_squared_error(
                regression_truth,
                regression_prediction,
                sample_weight=regression_weight,
            )
        ],
        "r2": [
            r2_score(
                regression_truth,
                regression_prediction,
                sample_weight=regression_weight,
            )
        ],
        "classification_truth": classification_truth,
        "classification_prediction": classification_prediction,
        "positive_probability": positive_probability,
        "classification_weight": classification_weight,
        "accuracy": [
            accuracy_score(
                classification_truth,
                classification_prediction,
                sample_weight=classification_weight,
            )
        ],
        "balanced_accuracy": [
            balanced_accuracy_score(
                classification_truth,
                classification_prediction,
                sample_weight=classification_weight,
            )
        ],
        "precision": [
            precision_score(
                classification_truth,
                classification_prediction,
                sample_weight=classification_weight,
            )
        ],
        "recall": [
            recall_score(
                classification_truth,
                classification_prediction,
                sample_weight=classification_weight,
            )
        ],
        "f1": [
            f1_score(
                classification_truth,
                classification_prediction,
                sample_weight=classification_weight,
            )
        ],
        "log_loss": [
            log_loss(
                classification_truth,
                positive_probability,
                sample_weight=classification_weight,
                labels=[0, 1],
            )
        ],
        "roc_auc": [
            roc_auc_score(
                classification_truth,
                positive_probability,
                sample_weight=classification_weight,
            )
        ],
        "roc_thresholds": roc_thresholds,
        "false_positive_rates": false_positive_rate,
        "true_positive_rates": true_positive_rate,
        "precision_recall_thresholds": precision_recall_thresholds,
        "precision_curve": precision_values,
        "recall_curve": recall_values,
    }
    rows = ["# ModelKit sklearn metric reference fixture v1"]
    rows.extend(f"{name}\t{float_values(value)}" for name, value in values.items())
    data_path = fixture_dir / "metrics_v1.tsv"
    metadata_path = fixture_dir / "metrics_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "positive_label": 1,
            "precision_recall_drop_intermediate": False,
            "roc_drop_intermediate": False,
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "metrics_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.metrics regression metrics",
            "sklearn.metrics binary classification metrics",
            "sklearn.metrics.roc_curve",
            "sklearn.metrics.precision_recall_curve",
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
    scaler_sample_weight = np.array([1.0, 2.0, 0.5, 3.0, 0.0], dtype=np.float64)
    weighted_scaler = StandardScaler().fit(
        mean_output, sample_weight=scaler_sample_weight
    )
    weighted_scaled_output = weighted_scaler.transform(mean_output)

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
    add_vector("scaler_sample_weight", scaler_sample_weight)
    add_vector("weighted_scaler_mean", weighted_scaler.mean_)
    add_vector("weighted_scaler_variance", weighted_scaler.var_)
    add_vector("weighted_scaler_scale", weighted_scaler.scale_)
    add_matrix("weighted_scaled_output", weighted_scaled_output)

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


def generate_univariate_selection_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.feature_selection import (
        SelectKBest,
        SelectPercentile,
        f_classif,
        f_regression,
    )

    regression_x = np.array(
        [
            [-2.0, 4.0, 0.3, 7.0],
            [-1.5, 2.25, -0.2, 7.0],
            [-1.0, 1.0, 1.2, 7.0],
            [-0.5, 0.25, 0.7, 7.0],
            [0.0, 0.0, -0.8, 7.0],
            [0.5, 0.25, 0.1, 7.0],
            [1.0, 1.0, -1.1, 7.0],
            [1.5, 2.25, 0.9, 7.0],
            [2.0, 4.0, -0.4, 7.0],
            [2.5, 6.25, 1.5, 7.0],
        ],
        dtype=np.float64,
    )
    regression_y = np.array(
        [-3.8, -2.4, -2.1, -0.4, 0.2, 1.4, 1.7, 3.5, 3.8, 5.4],
        dtype=np.float64,
    )
    regression_scores, _ = f_regression(regression_x, regression_y)
    regression_selector = SelectKBest(f_regression, k=2).fit(
        regression_x, regression_y
    )

    classification_x = np.array(
        [
            [-0.2, 0.0, 1.2, 4.0],
            [0.1, 1.0, -0.3, 5.0],
            [0.3, 2.0, 0.5, 3.0],
            [-0.1, 1.0, -1.0, 4.0],
            [2.8, 2.0, 0.1, 5.0],
            [3.1, 0.0, 1.5, 4.0],
            [3.3, 1.0, -0.7, 3.0],
            [2.9, 2.0, 0.8, 5.0],
            [6.2, 1.0, -1.4, 4.0],
            [5.8, 2.0, 0.2, 3.0],
            [6.1, 0.0, 1.0, 5.0],
            [5.9, 1.0, -0.1, 4.0],
        ],
        dtype=np.float64,
    )
    classification_y = np.repeat(np.array([-3, 4, 11], dtype=np.int64), 4)
    classification_scores, _ = f_classif(classification_x, classification_y)
    classification_selector = SelectPercentile(f_classif, percentile=50.0).fit(
        classification_x, classification_y
    )

    rows = ["# ModelKit sklearn univariate-selection reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("regression_x", regression_x)
    add_vector("regression_y", regression_y)
    add_vector("regression_scores", regression_scores)
    add_vector("regression_selected", regression_selector.get_support(indices=True))
    add_matrix("regression_output", regression_selector.transform(regression_x))
    add_matrix("classification_x", classification_x)
    add_vector("classification_y", classification_y)
    add_vector("classification_scores", classification_scores)
    add_vector(
        "classification_selected",
        classification_selector.get_support(indices=True),
    )
    add_matrix(
        "classification_output",
        classification_selector.transform(classification_x),
    )

    (fixture_dir / "univariate_selection_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "comparison": "F-scores, selected indices, and transformed dense matrices for count and percentile selection.",
        "configuration": {
            "classification_percentile": 50.0,
            "regression_k": 2,
        },
        "environment": environment.metadata(),
        "fixture": "univariate_selection_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.feature_selection.f_regression",
            "sklearn.feature_selection.f_classif",
            "sklearn.feature_selection.SelectKBest",
            "sklearn.feature_selection.SelectPercentile",
        ],
        "schema_version": 1,
    }
    (fixture_dir / "univariate_selection_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_model_based_selection_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LinearRegression, RidgeClassifier

    regression_x = np.array(
        [
            [-2.0, 4.0, 0.3, 1.0],
            [-1.5, 2.25, -0.2, 0.0],
            [-1.0, 1.0, 1.2, -1.0],
            [-0.5, 0.25, 0.7, 1.0],
            [0.0, 0.0, -0.8, 0.0],
            [0.5, 0.25, 0.1, -1.0],
            [1.0, 1.0, -1.1, 1.0],
            [1.5, 2.25, 0.9, 0.0],
            [2.0, 4.0, -0.4, -1.0],
            [2.5, 6.25, 1.5, 1.0],
        ],
        dtype=np.float64,
    )
    regression_y = (
        4.0 * regression_x[:, 0]
        - 0.8 * regression_x[:, 1]
        + 0.15 * regression_x[:, 2]
        + 0.02 * regression_x[:, 3]
        + 1.5
    )
    regression_selector = SelectFromModel(
        LinearRegression(), threshold="mean", max_features=2
    ).fit(regression_x, regression_y)
    regression_importances = np.abs(regression_selector.estimator_.coef_)

    classification_x = np.array(
        [
            [-0.2, 0.0, 1.2, 0.0],
            [0.1, 1.0, -0.3, 1.0],
            [0.3, 2.0, 0.5, -1.0],
            [-0.1, 1.0, -1.0, 0.5],
            [2.8, 2.0, 0.1, -0.5],
            [3.1, 0.0, 1.5, 1.0],
            [3.3, 1.0, -0.7, 0.0],
            [2.9, 2.0, 0.8, -1.0],
            [6.2, 1.0, -1.4, 0.5],
            [5.8, 2.0, 0.2, -0.5],
            [6.1, 0.0, 1.0, 1.0],
            [5.9, 1.0, -0.1, 0.0],
        ],
        dtype=np.float64,
    )
    classification_y = np.repeat(np.array([-3, 4, 11], dtype=np.int64), 4)
    classification_selector = SelectFromModel(
        RidgeClassifier(alpha=0.5),
        threshold="median",
        max_features=2,
        norm_order=1,
    ).fit(classification_x, classification_y)
    classification_importances = np.linalg.norm(
        classification_selector.estimator_.coef_, ord=1, axis=0
    )

    rows = ["# ModelKit sklearn model-based-selection reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("regression_x", regression_x)
    add_vector("regression_y", regression_y)
    add_vector("regression_importances", regression_importances)
    add_vector("regression_threshold", [regression_selector.threshold_])
    add_vector("regression_selected", regression_selector.get_support(indices=True))
    add_matrix("regression_output", regression_selector.transform(regression_x))
    add_matrix("classification_x", classification_x)
    add_vector("classification_y", classification_y)
    add_vector("classification_importances", classification_importances)
    add_vector("classification_threshold", [classification_selector.threshold_])
    add_vector(
        "classification_selected",
        classification_selector.get_support(indices=True),
    )
    add_matrix(
        "classification_output",
        classification_selector.transform(classification_x),
    )

    (fixture_dir / "model_based_selection_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "comparison": "Fitted coefficient importances, resolved thresholds, selected indices, and transformed dense matrices.",
        "configuration": {
            "classification_estimator": "RidgeClassifier(alpha=0.5)",
            "classification_norm_order": 1,
            "classification_threshold": "median",
            "max_features": 2,
            "regression_estimator": "LinearRegression()",
            "regression_threshold": "mean",
        },
        "environment": environment.metadata(),
        "fixture": "model_based_selection_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.feature_selection.SelectFromModel",
            "sklearn.linear_model.LinearRegression",
            "sklearn.linear_model.RidgeClassifier",
        ],
        "schema_version": 1,
    }
    (fixture_dir / "model_based_selection_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_recursive_feature_elimination_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression, RidgeClassifier

    regression_x = np.array(
        [
            [-2.0, 4.0, 0.3, 1.0, -1.0],
            [-1.5, 2.25, -0.2, 0.0, 1.0],
            [-1.0, 1.0, 1.2, -1.0, 0.0],
            [-0.5, 0.25, 0.7, 1.0, -1.0],
            [0.0, 0.0, -0.8, 0.0, 1.0],
            [0.5, 0.25, 0.1, -1.0, 0.0],
            [1.0, 1.0, -1.1, 1.0, -1.0],
            [1.5, 2.25, 0.9, 0.0, 1.0],
            [2.0, 4.0, -0.4, -1.0, 0.0],
            [2.5, 6.25, 1.5, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    regression_y = (
        4.0 * regression_x[:, 0]
        - 0.8 * regression_x[:, 1]
        + 0.15 * regression_x[:, 2]
        + 0.02 * regression_x[:, 3]
        + 0.4 * regression_x[:, 4]
        + 1.5
    )
    regression_selector = RFE(
        LinearRegression(), n_features_to_select=2, step=1
    ).fit(regression_x, regression_y)

    classification_x = np.array(
        [
            [-0.2, 0.0, 1.2, 0.0, 1.0],
            [0.1, 1.0, -0.3, 1.0, -1.0],
            [0.3, 2.0, 0.5, -1.0, 0.0],
            [-0.1, 1.0, -1.0, 0.5, 1.0],
            [2.8, 2.0, 0.1, -0.5, -1.0],
            [3.1, 0.0, 1.5, 1.0, 0.0],
            [3.3, 1.0, -0.7, 0.0, 1.0],
            [2.9, 2.0, 0.8, -1.0, -1.0],
            [6.2, 1.0, -1.4, 0.5, 0.0],
            [5.8, 2.0, 0.2, -0.5, 1.0],
            [6.1, 0.0, 1.0, 1.0, -1.0],
            [5.9, 1.0, -0.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    classification_y = np.repeat(np.array([-3, 4, 11], dtype=np.int64), 4)
    classification_selector = RFE(
        RidgeClassifier(alpha=0.5), n_features_to_select=2, step=0.4
    ).fit(classification_x, classification_y)

    rows = ["# ModelKit sklearn recursive-feature-elimination reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("regression_x", regression_x)
    add_vector("regression_y", regression_y)
    add_vector("regression_selected", regression_selector.get_support(indices=True))
    add_vector("regression_ranking", regression_selector.ranking_)
    add_vector("regression_importances", np.abs(regression_selector.estimator_.coef_))
    add_matrix("regression_output", regression_selector.transform(regression_x))
    add_matrix("classification_x", classification_x)
    add_vector("classification_y", classification_y)
    add_vector(
        "classification_selected",
        classification_selector.get_support(indices=True),
    )
    add_vector("classification_ranking", classification_selector.ranking_)
    add_vector(
        "classification_importances",
        np.linalg.norm(classification_selector.estimator_.coef_, ord=1, axis=0),
    )
    add_matrix(
        "classification_output",
        classification_selector.transform(classification_x),
    )

    (fixture_dir / "recursive_feature_elimination_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "comparison": "Selected indices, elimination rankings, final-estimator importances, and transformed dense matrices.",
        "configuration": {
            "classification_estimator": "RidgeClassifier(alpha=0.5)",
            "classification_feature_count": 2,
            "classification_step": 0.4,
            "regression_estimator": "LinearRegression()",
            "regression_feature_count": 2,
            "regression_step": 1,
        },
        "environment": environment.metadata(),
        "fixture": "recursive_feature_elimination_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.feature_selection.RFE",
            "sklearn.linear_model.LinearRegression",
            "sklearn.linear_model.RidgeClassifier",
        ],
        "schema_version": 1,
    }
    (fixture_dir / "recursive_feature_elimination_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_transform_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.impute import MissingIndicator
    from sklearn.preprocessing import (
        LabelEncoder,
        MaxAbsScaler,
        MinMaxScaler,
        Normalizer,
        OneHotEncoder,
        OrdinalEncoder,
        PolynomialFeatures,
        RobustScaler,
    )

    numeric = np.array(
        [
            [-4.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [8.0, 0.0, 100.0],
        ],
        dtype=np.float64,
    )
    normalized = np.array(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    categorical = np.array(
        [[2.0, 10.0], [1.0, 20.0], [2.0, 10.0]], dtype=np.float64
    )
    categorical_predict = np.array(
        [[3.0, 10.0], [1.0, 20.0]], dtype=np.float64
    )
    labels = np.array([42, -3, 42, 10], dtype=np.int64)
    polynomial = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float64)
    missing = np.array(
        [[np.nan, 1.0, np.nan], [2.0, 3.0, np.nan], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )

    min_max = MinMaxScaler(feature_range=(-1.0, 2.0)).fit(numeric)
    max_abs = MaxAbsScaler().fit(numeric)
    robust = RobustScaler(quantile_range=(25.0, 75.0)).fit(numeric)
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        categorical
    )
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(categorical)
    label = LabelEncoder().fit(labels)
    polynomial_features = PolynomialFeatures(degree=2, include_bias=True).fit(
        polynomial
    )
    interaction_features = PolynomialFeatures(
        degree=2, include_bias=True, interaction_only=True
    ).fit(polynomial)
    missing_only = MissingIndicator(features="missing-only", error_on_new=False).fit(
        missing
    )
    missing_all = MissingIndicator(features="all").fit(missing)

    rows = ["# ModelKit sklearn transform reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("numeric_input", numeric)
    add_vector("min_max_data_min", min_max.data_min_)
    add_vector("min_max_data_max", min_max.data_max_)
    add_vector("min_max_data_range", min_max.data_range_)
    add_vector("min_max_scale", min_max.scale_)
    add_vector("min_max_offset", min_max.min_)
    add_matrix("min_max_output", min_max.transform(numeric))
    add_vector("max_abs_max", max_abs.max_abs_)
    add_vector("max_abs_scale", max_abs.scale_)
    add_matrix("max_abs_output", max_abs.transform(numeric))
    add_vector("robust_center", robust.center_)
    add_vector("robust_scale", robust.scale_)
    add_matrix("robust_output", robust.transform(numeric))
    add_matrix("normalizer_input", normalized)
    for norm in ("l1", "l2", "max"):
        add_matrix(f"normalizer_{norm}_output", Normalizer(norm=norm).transform(normalized))
    add_matrix("categorical_input", categorical)
    add_matrix("categorical_predict", categorical_predict)
    for column, values in enumerate(one_hot.categories_):
        add_vector(f"one_hot_categories_{column}", values)
    add_matrix("one_hot_output", one_hot.transform(categorical_predict))
    for column, values in enumerate(ordinal.categories_):
        add_vector(f"ordinal_categories_{column}", values)
    add_matrix("ordinal_output", ordinal.transform(categorical_predict))
    add_vector("label_input", labels)
    add_vector("label_classes", label.classes_)
    encoded_labels = label.transform(labels)
    add_vector("label_encoded", encoded_labels)
    add_vector("label_decoded", label.inverse_transform(encoded_labels))
    add_matrix("polynomial_input", polynomial)
    add_matrix("polynomial_output", polynomial_features.transform(polynomial))
    add_matrix("interaction_output", interaction_features.transform(polynomial))
    add_matrix("missing_input", missing)
    add_vector("missing_only_features", missing_only.features_)
    add_matrix("missing_only_output", missing_only.transform(missing))
    add_matrix("missing_all_output", missing_all.transform(missing))

    data_path = fixture_dir / "transforms_v1.tsv"
    metadata_path = fixture_dir / "transforms_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "categorical_features": categorical.shape[1],
            "min_max_range": [-1.0, 2.0],
            "numeric_features": numeric.shape[1],
            "polynomial_degree": 2,
            "samples": numeric.shape[0],
        },
        "environment": environment.metadata(),
        "fixture": "transforms_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.impute.MissingIndicator",
            "sklearn.preprocessing.LabelEncoder",
            "sklearn.preprocessing.MaxAbsScaler",
            "sklearn.preprocessing.MinMaxScaler",
            "sklearn.preprocessing.Normalizer",
            "sklearn.preprocessing.OneHotEncoder",
            "sklearn.preprocessing.OrdinalEncoder",
            "sklearn.preprocessing.PolynomialFeatures",
            "sklearn.preprocessing.RobustScaler",
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


def generate_regularized_linear_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import ElasticNet, Lasso, enet_path, lasso_path

    x_train = np.array(
        [
            [-2.0, 0.5, 1.0, 0.0],
            [-1.0, -1.5, 0.0, 1.0],
            [0.0, 2.0, -0.5, 0.0],
            [1.0, -0.5, 2.0, 1.0],
            [2.0, 1.5, 1.0, 0.0],
            [3.0, -2.0, -1.0, 1.0],
            [4.0, 0.25, 0.5, 0.0],
            [5.0, 2.5, -2.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [-1.15, 0.4, -2.25, 2.6, 2.85, 8.3, 7.175, 4.0], dtype=np.float64
    )
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    x_predict = np.array(
        [
            [-1.5, 0.0, 0.5, 1.0],
            [0.5, 1.0, -1.0, 0.0],
            [2.5, -1.0, 1.5, 1.0],
            [6.0, 0.75, -0.25, 0.0],
        ],
        dtype=np.float64,
    )
    lasso_alpha = 0.15
    elastic_alpha = 0.12
    elastic_l1_ratio = 0.35
    path_alphas = np.array([0.8, 0.35, 0.12, 0.04], dtype=np.float64)

    lasso = Lasso(
        alpha=lasso_alpha,
        fit_intercept=True,
        selection="cyclic",
        tol=1e-12,
        max_iter=100000,
    ).fit(x_train, target, sample_weight=sample_weight)
    elastic = ElasticNet(
        alpha=elastic_alpha,
        l1_ratio=elastic_l1_ratio,
        fit_intercept=True,
        selection="cyclic",
        tol=1e-12,
        max_iter=100000,
    ).fit(x_train, target, sample_weight=sample_weight)

    centered_x = x_train - x_train.mean(axis=0)
    centered_target = target - target.mean()
    lasso_path_alphas, lasso_path_coefficients, _ = lasso_path(
        centered_x,
        centered_target,
        alphas=path_alphas,
        tol=1e-12,
        max_iter=100000,
    )
    elastic_path_alphas, elastic_path_coefficients, _ = enet_path(
        centered_x,
        centered_target,
        l1_ratio=elastic_l1_ratio,
        alphas=path_alphas,
        tol=1e-12,
        max_iter=100000,
    )

    rows = ["# ModelKit sklearn regularized-linear reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("target", target)
    add_vector("sample_weight", sample_weight)
    add_vector("lasso_alpha", [lasso_alpha])
    add_vector("lasso_coefficients", lasso.coef_)
    add_vector("lasso_intercept", [lasso.intercept_])
    add_vector("lasso_prediction", lasso.predict(x_predict))
    add_vector("elastic_alpha", [elastic_alpha])
    add_vector("elastic_l1_ratio", [elastic_l1_ratio])
    add_vector("elastic_coefficients", elastic.coef_)
    add_vector("elastic_intercept", [elastic.intercept_])
    add_vector("elastic_prediction", elastic.predict(x_predict))
    add_matrix("centered_x", centered_x)
    add_vector("centered_target", centered_target)
    add_vector("lasso_path_alphas", lasso_path_alphas)
    add_matrix("lasso_path_coefficients", lasso_path_coefficients.T)
    add_vector("elastic_path_alphas", elastic_path_alphas)
    add_matrix("elastic_path_coefficients", elastic_path_coefficients.T)

    data_path = fixture_dir / "regularized_linear_v1.tsv"
    metadata_path = fixture_dir / "regularized_linear_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "elastic_alpha": elastic_alpha,
            "elastic_l1_ratio": elastic_l1_ratio,
            "features": x_train.shape[1],
            "lasso_alpha": lasso_alpha,
            "max_iter": 100000,
            "path_alphas": path_alphas.tolist(),
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "selection": "cyclic",
            "tol": 1e-12,
            "weighted_estimators": True,
        },
        "environment": environment.metadata(),
        "fixture": "regularized_linear_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.linear_model.ElasticNet",
            "sklearn.linear_model.Lasso",
            "sklearn.linear_model.enet_path",
            "sklearn.linear_model.lasso_path",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_ridge_classifier_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

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
    binary_target = np.array([-4, -4, -4, 9, 9, -4, 9, 9], dtype=np.int64)
    multiclass_target = np.array([-4, -4, 2, 2, 9, -4, 9, 9], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    alpha = 2.5

    binary = RidgeClassifier(alpha=alpha, solver="svd").fit(
        x_train, binary_target, sample_weight=sample_weight
    )
    multiclass = RidgeClassifier(alpha=alpha, solver="svd").fit(
        x_train, multiclass_target, sample_weight=sample_weight
    )

    rows = ["# ModelKit sklearn ridge-classifier reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("sample_weight", sample_weight)
    add_vector("alpha", [alpha])
    add_vector("binary_target", binary_target)
    add_vector("binary_classes", binary.classes_)
    add_matrix("binary_coefficients", np.atleast_2d(binary.coef_))
    add_vector("binary_intercepts", np.atleast_1d(binary.intercept_))
    add_vector("binary_decisions", binary.decision_function(x_predict))
    add_vector("binary_predictions", binary.predict(x_predict))
    add_vector("multiclass_target", multiclass_target)
    add_vector("multiclass_classes", multiclass.classes_)
    add_matrix("multiclass_coefficients", multiclass.coef_)
    add_vector("multiclass_intercepts", multiclass.intercept_)
    add_matrix("multiclass_decisions", multiclass.decision_function(x_predict))
    add_vector("multiclass_predictions", multiclass.predict(x_predict))

    data_path = fixture_dir / "ridge_classifier_v1.tsv"
    metadata_path = fixture_dir / "ridge_classifier_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "alpha": alpha,
            "classes": [2, 3],
            "features": x_train.shape[1],
            "fit_intercept": True,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "solver": "svd",
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "ridge_classifier_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "reference": "sklearn.linear_model.RidgeClassifier",
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_multinomial_logistic_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

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
    target = np.array([-4, -4, 2, 2, 9, -4, 9, 9], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    c = 1.7
    tolerance = 1e-12
    max_iterations = 1000
    model = LogisticRegression(
        C=c,
        fit_intercept=True,
        solver="lbfgs",
        tol=tolerance,
        max_iter=max_iterations,
    ).fit(x_train, target, sample_weight=sample_weight)

    rows = ["# ModelKit sklearn multinomial-logistic reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("target", target)
    add_vector("sample_weight", sample_weight)
    add_vector("c", [c])
    add_vector("tolerance", [tolerance])
    add_vector("max_iterations", [max_iterations])
    add_vector("classes", model.classes_)
    add_matrix("coefficients", model.coef_)
    add_vector("intercepts", model.intercept_)
    add_matrix("decisions", model.decision_function(x_predict))
    add_matrix("probabilities", model.predict_proba(x_predict))
    add_vector("predictions", model.predict(x_predict))

    data_path = fixture_dir / "multinomial_logistic_v1.tsv"
    metadata_path = fixture_dir / "multinomial_logistic_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "c": c,
            "classes": len(model.classes_),
            "features": x_train.shape[1],
            "fit_intercept": True,
            "max_iterations": max_iterations,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "solver": "lbfgs",
            "tolerance": tolerance,
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "multinomial_logistic_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "reference": "sklearn.linear_model.LogisticRegression",
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_glm_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import PoissonRegressor, TweedieRegressor

    x_train = np.array(
        [
            [-2.0, 0.5],
            [-1.0, -1.5],
            [0.0, 2.0],
            [1.0, -0.5],
            [2.0, 1.5],
            [3.0, -2.0],
            [4.0, 0.25],
            [5.0, 2.5],
        ],
        dtype=np.float64,
    )
    x_predict = np.array(
        [[-1.5, 0.0], [0.5, 1.0], [2.5, -1.0], [6.0, 0.75]],
        dtype=np.float64,
    )
    target = np.array([0.25, 0.8, 1.3, 2.1, 4.2, 5.5, 9.0, 15.0])
    sample_weight = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25])
    alpha = 0.35
    tolerance = 1e-12
    max_iterations = 1000
    poisson = PoissonRegressor(
        alpha=alpha,
        tol=tolerance,
        max_iter=max_iterations,
    ).fit(x_train, target, sample_weight=sample_weight)
    power = 1.5
    tweedie = TweedieRegressor(
        power=power,
        alpha=alpha,
        link="log",
        tol=tolerance,
        max_iter=max_iterations,
    ).fit(x_train, target, sample_weight=sample_weight)

    rows = ["# ModelKit sklearn generalized-linear-model reference fixture v1"]

    def add_matrix(name: str, values) -> None:
        for index, row in enumerate(values):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("target", target)
    add_vector("sample_weight", sample_weight)
    add_vector("alpha", [alpha])
    add_vector("tolerance", [tolerance])
    add_vector("max_iterations", [max_iterations])
    add_vector("poisson_coefficients", poisson.coef_)
    add_vector("poisson_intercept", [poisson.intercept_])
    add_vector("poisson_predictions", poisson.predict(x_predict))
    add_vector("tweedie_power", [power])
    add_vector("tweedie_coefficients", tweedie.coef_)
    add_vector("tweedie_intercept", [tweedie.intercept_])
    add_vector("tweedie_predictions", tweedie.predict(x_predict))

    data_path = fixture_dir / "glm_v1.tsv"
    metadata_path = fixture_dir / "glm_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "alpha": alpha,
            "features": x_train.shape[1],
            "fit_intercept": True,
            "max_iterations": max_iterations,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "solver": "lbfgs",
            "tolerance": tolerance,
            "tweedie_link": "log",
            "tweedie_power": power,
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "glm_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.linear_model.PoissonRegressor",
            "sklearn.linear_model.TweedieRegressor",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_sgd_regressor_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import SGDRegressor

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
    target = np.array(
        [-1.15, 0.4, -2.25, 2.6, 2.85, 8.3, 7.175, 4.0], dtype=np.float64
    )
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
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
    eta0 = 0.01
    epochs = 6
    configuration = {
        "loss": "squared_error",
        "penalty": None,
        "alpha": 0.0,
        "fit_intercept": True,
        "max_iter": epochs,
        "tol": None,
        "shuffle": False,
        "learning_rate": "constant",
        "eta0": eta0,
        "average": False,
    }
    fitted = SGDRegressor(**configuration).fit(
        x_train, target, sample_weight=sample_weight
    )
    incremental = SGDRegressor(**configuration)
    for _ in range(epochs):
        incremental.partial_fit(x_train, target, sample_weight=sample_weight)

    rows = ["# ModelKit sklearn SGD-regressor reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("target", target)
    add_vector("sample_weight", sample_weight)
    add_vector("eta0", [eta0])
    add_vector("epochs", [epochs])
    add_vector("fit_coefficients", fitted.coef_)
    add_vector("fit_intercept", fitted.intercept_)
    add_vector("fit_prediction", fitted.predict(x_predict))
    add_vector("partial_coefficients", incremental.coef_)
    add_vector("partial_intercept", incremental.intercept_)
    add_vector("partial_prediction", incremental.predict(x_predict))

    data_path = fixture_dir / "sgd_regressor_v1.tsv"
    metadata_path = fixture_dir / "sgd_regressor_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "average": False,
            "epochs": epochs,
            "features": x_train.shape[1],
            "fit_intercept": True,
            "learning_rate": "constant",
            "loss": "squared_error",
            "penalty": None,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "shuffle": False,
            "weighted": True,
            "eta0": eta0,
        },
        "environment": environment.metadata(),
        "fixture": "sgd_regressor_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "reference": "sklearn.linear_model.SGDRegressor",
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_sgd_classifier_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import SGDClassifier

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
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    targets = {
        "binary": np.array([-4, -4, 2, 2, -4, 2, 2, -4], dtype=np.int64),
        "multiclass": np.array([-4, -4, 2, 2, 9, -4, 9, 9], dtype=np.int64),
    }
    eta0 = 0.05
    epochs = 6
    cases = [
        ("binary_hinge", "binary", "hinge"),
        ("binary_log", "binary", "log_loss"),
        ("multiclass_hinge", "multiclass", "hinge"),
        ("multiclass_log", "multiclass", "log_loss"),
    ]

    rows = ["# ModelKit sklearn SGD-classifier reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    def add_model(prefix: str, model, loss: str) -> None:
        decisions = model.decision_function(x_predict)
        if decisions.ndim == 1:
            decisions = decisions[:, np.newaxis]
        add_matrix(f"{prefix}_coefficients", model.coef_)
        add_vector(f"{prefix}_intercepts", model.intercept_)
        add_matrix(f"{prefix}_decisions", decisions)
        if loss == "log_loss":
            add_matrix(f"{prefix}_probabilities", model.predict_proba(x_predict))
        add_vector(f"{prefix}_predictions", model.predict(x_predict))

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("sample_weight", sample_weight)
    add_vector("eta0", [eta0])
    add_vector("epochs", [epochs])
    for name, target in targets.items():
        add_vector(f"{name}_target", target)
    for case, target_name, loss in cases:
        target = targets[target_name]
        configuration = {
            "loss": loss,
            "penalty": None,
            "alpha": 0.0,
            "fit_intercept": True,
            "max_iter": epochs,
            "tol": None,
            "shuffle": False,
            "learning_rate": "constant",
            "eta0": eta0,
            "average": False,
        }
        fitted = SGDClassifier(**configuration).fit(
            x_train, target, sample_weight=sample_weight
        )
        incremental = SGDClassifier(**configuration)
        classes = np.unique(target)
        for _ in range(epochs):
            incremental.partial_fit(
                x_train, target, classes=classes, sample_weight=sample_weight
            )
        add_vector(f"{case}_classes", fitted.classes_)
        add_model(f"{case}_fit", fitted, loss)
        add_model(f"{case}_partial", incremental, loss)

    data_path = fixture_dir / "sgd_classifier_v1.tsv"
    metadata_path = fixture_dir / "sgd_classifier_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "average": False,
            "cases": [case for case, _, _ in cases],
            "epochs": epochs,
            "eta0": eta0,
            "features": x_train.shape[1],
            "fit_intercept": True,
            "learning_rate": "constant",
            "losses": ["hinge", "log_loss"],
            "penalty": None,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "shuffle": False,
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "sgd_classifier_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "reference": "sklearn.linear_model.SGDClassifier",
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_class_weight_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.utils.class_weight import compute_class_weight

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
    binary = np.array([-4, -4, -4, 2, -4, -4, 2, -4], dtype=np.int64)
    multiclass = np.array([-4, -4, 2, 2, 9, -4, 9, 9], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25], dtype=np.float64
    )
    explicit = {-4: 0.5, 2: 3.0}
    c = 1.7
    tolerance = 1e-12
    max_iterations = 1000
    alpha = 0.7

    rows = ["# ModelKit sklearn class-weight reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    def logistic(class_weight):
        return LogisticRegression(
            C=c,
            fit_intercept=True,
            solver="lbfgs",
            tol=tolerance,
            max_iter=max_iterations,
            class_weight=class_weight,
        )

    add_matrix("x_train", x_train)
    add_matrix("x_predict", x_predict)
    add_vector("binary_target", binary)
    add_vector("multiclass_target", multiclass)
    add_vector("sample_weight", sample_weight)
    add_vector("explicit_classes", list(explicit.keys()))
    add_vector("explicit_weights", list(explicit.values()))
    add_vector("c", [c])
    add_vector("tolerance", [tolerance])
    add_vector("max_iterations", [max_iterations])
    add_vector("alpha", [alpha])
    add_vector(
        "balanced_class_weights",
        compute_class_weight(
            "balanced", classes=np.unique(binary), y=binary, sample_weight=sample_weight
        ),
    )
    add_vector(
        "unweighted_balanced_class_weights",
        compute_class_weight("balanced", classes=np.unique(binary), y=binary),
    )
    balanced_logistic = logistic("balanced").fit(
        x_train, binary, sample_weight=sample_weight
    )
    add_matrix("balanced_logistic_coefficients", balanced_logistic.coef_)
    add_vector("balanced_logistic_intercept", balanced_logistic.intercept_)
    add_matrix(
        "balanced_logistic_probabilities",
        balanced_logistic.predict_proba(x_predict),
    )
    explicit_logistic = logistic(explicit).fit(
        x_train, binary, sample_weight=sample_weight
    )
    add_matrix("explicit_logistic_coefficients", explicit_logistic.coef_)
    add_vector("explicit_logistic_intercept", explicit_logistic.intercept_)
    add_matrix(
        "explicit_logistic_probabilities",
        explicit_logistic.predict_proba(x_predict),
    )
    balanced_multinomial = logistic("balanced").fit(
        x_train, multiclass, sample_weight=sample_weight
    )
    add_matrix("balanced_multinomial_coefficients", balanced_multinomial.coef_)
    add_vector("balanced_multinomial_intercepts", balanced_multinomial.intercept_)
    add_matrix(
        "balanced_multinomial_probabilities",
        balanced_multinomial.predict_proba(x_predict),
    )
    # RidgeClassifier derives balanced weights from unweighted counts, so the
    # ridge case is generated without sample weights where both definitions
    # agree.
    balanced_ridge = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(
        x_train, multiclass
    )
    add_matrix("balanced_ridge_coefficients", balanced_ridge.coef_)
    add_vector("balanced_ridge_intercepts", balanced_ridge.intercept_)
    add_matrix("balanced_ridge_decisions", balanced_ridge.decision_function(x_predict))

    data_path = fixture_dir / "class_weight_v1.tsv"
    metadata_path = fixture_dir / "class_weight_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "alpha": alpha,
            "c": c,
            "explicit_class_weight": {str(k): v for k, v in explicit.items()},
            "features": x_train.shape[1],
            "max_iterations": max_iterations,
            "prediction_samples": x_predict.shape[0],
            "samples": x_train.shape[0],
            "solver": "lbfgs",
            "tolerance": tolerance,
            "weighted": True,
        },
        "environment": environment.metadata(),
        "fixture": "class_weight_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.linear_model.LogisticRegression",
            "sklearn.linear_model.RidgeClassifier",
            "sklearn.utils.class_weight.compute_class_weight",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_multiclass_metrics_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    truth = np.array([0, 2, 1, 2, 0, 1, 2, 2, 1, 0], dtype=np.int64)
    prediction = np.array([0, 2, 1, 1, 0, 2, 2, 2, 1, 1], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25, 1.0, 2.0], dtype=np.float64
    )
    # A label that appears only in predictions exercises zero-division handling.
    sparse_truth = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sparse_prediction = np.array([0, 0, 2, 1, 1], dtype=np.int64)
    classes = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
            [0.3, 0.4, 0.3],
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],
            [0.05, 0.15, 0.8],
            [0.1, 0.2, 0.7],
            [0.25, 0.5, 0.25],
            [0.45, 0.35, 0.2],
        ],
        dtype=np.float64,
    )

    rows = ["# ModelKit sklearn multiclass metrics reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_vector("truth", truth)
    add_vector("prediction", prediction)
    add_vector("sample_weight", sample_weight)
    add_vector("sparse_truth", sparse_truth)
    add_vector("sparse_prediction", sparse_prediction)
    add_vector("classes", classes)
    add_matrix("probabilities", probabilities)
    add_matrix(
        "confusion_matrix",
        confusion_matrix(truth, prediction, sample_weight=sample_weight),
    )
    add_matrix(
        "unweighted_confusion_matrix",
        confusion_matrix(truth, prediction, labels=[2, 0, 1]),
    )
    add_vector("accuracy", [accuracy_score(truth, prediction, sample_weight=sample_weight)])
    add_vector(
        "balanced_accuracy",
        [balanced_accuracy_score(truth, prediction, sample_weight=sample_weight)],
    )
    for average in ("micro", "macro", "weighted"):
        add_vector(
            f"precision_{average}",
            [precision_score(truth, prediction, average=average, sample_weight=sample_weight)],
        )
        add_vector(
            f"recall_{average}",
            [recall_score(truth, prediction, average=average, sample_weight=sample_weight)],
        )
        add_vector(
            f"f1_{average}",
            [f1_score(truth, prediction, average=average, sample_weight=sample_weight)],
        )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        truth, prediction, average=None, sample_weight=sample_weight
    )
    add_vector("class_precision", per_precision)
    add_vector("class_recall", per_recall)
    add_vector("class_f1", per_f1)
    add_vector("class_support", support)
    add_vector(
        "sparse_precision_macro",
        [precision_score(sparse_truth, sparse_prediction, average="macro", zero_division=0.0)],
    )
    add_vector(
        "sparse_recall_macro",
        [recall_score(sparse_truth, sparse_prediction, average="macro", zero_division=0.0)],
    )
    add_vector(
        "sparse_f1_weighted",
        [f1_score(sparse_truth, sparse_prediction, average="weighted", zero_division=0.0)],
    )
    add_vector(
        "sparse_balanced_accuracy",
        [balanced_accuracy_score(sparse_truth, sparse_prediction)],
    )
    add_vector(
        "log_loss",
        [log_loss(truth, probabilities, sample_weight=sample_weight, labels=classes)],
    )

    data_path = fixture_dir / "multiclass_metrics_v1.tsv"
    metadata_path = fixture_dir / "multiclass_metrics_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "averages": ["micro", "macro", "weighted"],
            "classes": 3,
            "samples": int(truth.shape[0]),
            "weighted": True,
            "zero_division": 0.0,
        },
        "environment": environment.metadata(),
        "fixture": "multiclass_metrics_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.metrics.accuracy_score",
            "sklearn.metrics.balanced_accuracy_score",
            "sklearn.metrics.confusion_matrix",
            "sklearn.metrics.f1_score",
            "sklearn.metrics.log_loss",
            "sklearn.metrics.precision_recall_fscore_support",
            "sklearn.metrics.precision_score",
            "sklearn.metrics.recall_score",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_ranking_metrics_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        dcg_score,
        ndcg_score,
        roc_auc_score,
        top_k_accuracy_score,
    )

    binary_truth = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
    binary_probabilities = np.array(
        [0.1, 0.9, 0.4, 0.4, 0.65, 0.35, 0.8, 0.55], dtype=np.float64
    )
    binary_weight = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25])
    truth = np.array([0, 2, 1, 2, 0, 1, 2, 2, 1, 0], dtype=np.int64)
    sample_weight = np.array(
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25, 1.0, 2.0], dtype=np.float64
    )
    classes = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
            [0.3, 0.4, 0.3],
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],
            [0.05, 0.15, 0.8],
            [0.1, 0.2, 0.7],
            [0.25, 0.5, 0.25],
            [0.45, 0.35, 0.2],
        ],
        dtype=np.float64,
    )
    # Tied scores and tied gains exercise the tie-averaged and tie-ignoring paths.
    relevance = np.array(
        [
            [3.0, 2.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ranking_scores = np.array(
        [
            [0.9, 0.9, 0.3, 0.1, 0.5],
            [0.2, 0.2, 0.2, 0.9, 0.1],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ],
        dtype=np.float64,
    )
    ranking_weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)

    rows = ["# ModelKit sklearn ranking metrics reference fixture v1"]

    def add_matrix(name: str, matrix) -> None:
        for index, row in enumerate(matrix):
            rows.append(f"{name}\t{index}\t{float_values(row)}")

    def add_vector(name: str, values) -> None:
        rows.append(f"{name}\t{float_values(values)}")

    add_vector("binary_truth", binary_truth)
    add_vector("binary_probabilities", binary_probabilities)
    add_vector("binary_weight", binary_weight)
    add_vector(
        "average_precision",
        [
            average_precision_score(
                binary_truth, binary_probabilities, sample_weight=binary_weight
            )
        ],
    )
    add_vector("truth", truth)
    add_vector("sample_weight", sample_weight)
    add_vector("classes", classes)
    add_matrix("probabilities", probabilities)
    for average in ("macro", "weighted", "micro"):
        add_vector(
            f"roc_auc_ovr_{average}",
            [
                roc_auc_score(
                    truth,
                    probabilities,
                    multi_class="ovr",
                    average=average,
                    sample_weight=sample_weight,
                    labels=classes,
                )
            ],
        )
    for average in ("macro", "weighted"):
        add_vector(
            f"roc_auc_ovo_{average}",
            [
                roc_auc_score(
                    truth, probabilities, multi_class="ovo", average=average, labels=classes
                )
            ],
        )
    for k in (1, 2):
        add_vector(
            f"top_{k}_accuracy",
            [
                top_k_accuracy_score(
                    truth, probabilities, k=k, sample_weight=sample_weight, labels=classes
                )
            ],
        )
    add_matrix("relevance", relevance)
    add_matrix("ranking_scores", ranking_scores)
    add_vector("ranking_weight", ranking_weight)
    for label, k in (("all", None), ("3", 3)):
        for ties, ignore_ties in (("averaged", False), ("ignored", True)):
            add_vector(
                f"dcg_{label}_{ties}",
                [
                    dcg_score(
                        relevance,
                        ranking_scores,
                        k=k,
                        sample_weight=ranking_weight,
                        ignore_ties=ignore_ties,
                    )
                ],
            )
            add_vector(
                f"ndcg_{label}_{ties}",
                [
                    ndcg_score(
                        relevance,
                        ranking_scores,
                        k=k,
                        sample_weight=ranking_weight,
                        ignore_ties=ignore_ties,
                    )
                ],
            )

    data_path = fixture_dir / "ranking_metrics_v1.tsv"
    metadata_path = fixture_dir / "ranking_metrics_v1.metadata.json"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "configuration": {
            "classes": 3,
            "ranking_columns": int(relevance.shape[1]),
            "ranking_cutoffs": [None, 3],
            "samples": int(truth.shape[0]),
            "top_k": [1, 2],
            "weighted": True,
            "one_vs_one_weighted": False,
        },
        "environment": environment.metadata(),
        "fixture": "ranking_metrics_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.metrics.average_precision_score",
            "sklearn.metrics.dcg_score",
            "sklearn.metrics.ndcg_score",
            "sklearn.metrics.roc_auc_score",
            "sklearn.metrics.top_k_accuracy_score",
        ],
        "schema_version": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_column_transformer_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    train = np.array(
        [
            [-4.0, 1.0, 10.0, np.nan, 90.0],
            [-2.0, 2.0, 20.0, 2.0, 91.0],
            [0.0, 1.0, 40.0, 4.0, 92.0],
            [8.0, 3.0, 100.0, np.nan, 93.0],
        ],
        dtype=np.float64,
    )
    test = np.array(
        [[100.0, 9.0, 1000.0, np.nan, -1.0], [-8.0, 1.0, 0.0, 9.0, -2.0]],
        dtype=np.float64,
    )
    cases = {
        "mixed": ColumnTransformer(
            [
                ("scale", StandardScaler(), [2, 0]),
                ("impute", SimpleImputer(strategy="mean"), [3]),
                ("category", "passthrough", [1]),
                ("discard", "drop", [4]),
            ],
            remainder="passthrough",
        ),
        "overlap": ColumnTransformer(
            [
                ("scale", StandardScaler(), [0]),
                ("raw", "passthrough", [0, 2]),
                ("discard", "drop", [3]),
            ],
            remainder="passthrough",
        ),
        "encoding": ColumnTransformer(
            [
                ("scale", StandardScaler(), [0]),
                (
                    "encode",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    [1],
                ),
                ("unused", StandardScaler(), []),
            ]
        ),
    }
    rows = ["# ModelKit sklearn column transformer reference fixture v1"]

    def add_matrix(name: str, values) -> None:
        for index, values in enumerate(values):
            rows.append(f"{name}\t{index}\t{float_values(values)}")

    add_matrix("train", train)
    add_matrix("test", test)
    for name, transformer in cases.items():
        add_matrix(f"{name}_train", transformer.fit_transform(train))
        add_matrix(f"{name}_test", transformer.transform(test))
        if name != "encoding":
            rows.append(f"{name}_names\t" + ",".join(transformer.get_feature_names_out()))

    (fixture_dir / "column_transformer_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "cases": list(cases),
            "dense_output": True,
            "fit_rows": 4,
            "inference_rows": 2,
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "naming": (
                "Exact names for scale, impute, passthrough, and remainder; "
                "encoders retain ModelKit's existing generated feature-name convention."
            ),
        },
        "environment": environment.metadata(),
        "fixture": "column_transformer_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.compose.ColumnTransformer",
            "sklearn.preprocessing.StandardScaler",
            "sklearn.preprocessing.OneHotEncoder",
            "sklearn.impute.SimpleImputer",
        ],
        "schema_version": 1,
    }
    (fixture_dir / "column_transformer_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_nested_composition_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler

    train = np.array([[np.nan, 1.0], [2.0, 2.0], [4.0, np.nan], [6.0, 8.0]])
    test = np.array([[10.0, np.nan], [np.nan, -3.0]])
    union = FeatureUnion(
        [
            ("scaled", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])),
            ("imputed", SimpleImputer()),
            ("unused", "drop"),
        ]
    )
    nested = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer()),
                                    (
                                        "views",
                                        FeatureUnion(
                                            [("scaled", StandardScaler()), ("raw", "passthrough")]
                                        ),
                                    ),
                                ]
                            ),
                            [0],
                        ),
                        ("other", SimpleImputer(), [1]),
                    ]
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    rows = ["# ModelKit sklearn nested composition reference fixture v1"]

    def add_matrix(name: str, values) -> None:
        for row, values in enumerate(values):
            rows.append(f"{name}\t{row}\t{float_values(values)}")

    add_matrix("train", train)
    add_matrix("test", test)
    for name, specification in [("union", union), ("nested", nested)]:
        add_matrix(f"{name}_train", specification.fit_transform(train))
        add_matrix(f"{name}_test", specification.transform(test))
        rows.append(f"{name}_names\t" + ",".join(specification.get_feature_names_out()))
    (fixture_dir / "nested_composition_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "dense_output": True,
            "fit_rows": 4,
            "inference_rows": 2,
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "cases": ["union", "nested"],
        },
        "environment": environment.metadata(),
        "fixture": "nested_composition_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "references": [
            "sklearn.pipeline.FeatureUnion",
            "sklearn.pipeline.Pipeline",
            "sklearn.compose.ColumnTransformer",
            "sklearn.impute.SimpleImputer",
            "sklearn.preprocessing.StandardScaler",
        ],
        "schema_version": 1,
    }
    (fixture_dir / "nested_composition_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_transformed_target_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression

    train = np.arange(6, dtype=np.float64)
    target = np.expm1([0.4, 0.9, 0.8, 1.8, 1.1, 2.0])
    weights = np.array([1., 2., 1., 3., 2., 1.])
    test = np.array([-1., 0.5, 6.])
    model = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1,
        check_inverse=True,
    ).fit(train[:, None], target, sample_weight=weights)
    rows = ["# ModelKit sklearn reference fixture v1"]
    for name, values in [
        ("train", train), ("target", target), ("weights", weights),
        ("test", test), ("prediction", model.predict(test[:, None])),
    ]:
        rows.append(f"{name}\t{float_values(values)}")
    (fixture_dir / "transformed_target_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {"absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
                          "func": "log1p", "inverse_func": "expm1", "weighted": True},
        "environment": environment.metadata(),
        "fixture": "transformed_target_v1", "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0", "schema_version": 1,
        "references": ["sklearn.compose.TransformedTargetRegressor", "sklearn.linear_model.LinearRegression"],
    }
    (fixture_dir / "transformed_target_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def generate_resampling_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.model_selection import (
        RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit,
        StratifiedShuffleSplit, train_test_split,
    )

    labels = np.repeat([0, 1, 2], [20, 12, 8])
    x = np.arange(len(labels))[:, None]
    rows = ["# ModelKit sklearn resampling semantics fixture v1"]

    def add(name, values):
        rows.append(f"{name}\t{comma_separated(values)}")

    add("labels", labels)
    train, test = train_test_split(np.arange(11), test_size=0.3, shuffle=False)
    add("holdout_train", train)
    add("holdout_test", test)
    for index, (train, test) in enumerate(
        ShuffleSplit(n_splits=3, test_size=0.3, random_state=1729).split(x)
    ):
        add(f"shuffle_sizes_{index}", [len(train), len(test)])
    for index, (train, test) in enumerate(
        StratifiedShuffleSplit(n_splits=3, train_size=20, test_size=10,
                               random_state=1729).split(x, labels)
    ):
        add(f"stratified_train_{index}", np.bincount(labels[train], minlength=3))
        add(f"stratified_test_{index}", np.bincount(labels[test], minlength=3))
    for index, (train, test) in enumerate(
        RepeatedKFold(n_splits=4, n_repeats=2, random_state=1729).split(x)
    ):
        add(f"repeated_sizes_{index}", [len(train), len(test)])
    for index, (_, test) in enumerate(
        RepeatedStratifiedKFold(n_splits=4, n_repeats=2,
                                random_state=1729).split(x, labels)
    ):
        add(f"repeated_stratified_test_{index}", np.bincount(labels[test], minlength=3))
    (fixture_dir / "resampling_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {"random_state": 1729, "folds": 4, "repeats": 2,
                          "shuffle_test_fraction": 0.3,
                          "stratified_train_count": 20, "stratified_test_count": 10},
        "comparison": "Exact unshuffled rows, partition sizes and untied class allocations; random row identities intentionally differ between RNG implementations.",
        "environment": environment.metadata(), "fixture": "resampling_v1",
        "generator": "dev/fixtures/generate.py", "license": "Apache-2.0",
        "schema_version": 1,
        "references": ["sklearn.model_selection.train_test_split",
                       "sklearn.model_selection.ShuffleSplit",
                       "sklearn.model_selection.StratifiedShuffleSplit",
                       "sklearn.model_selection.RepeatedKFold",
                       "sklearn.model_selection.RepeatedStratifiedKFold"],
    }
    (fixture_dir / "resampling_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def generate_partitioning_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.model_selection import (
        LeaveOneGroupOut, LeaveOneOut, PredefinedSplit, StratifiedGroupKFold,
    )

    assignments = np.array([-1, 8, 8, 2, 2, -1, 99, 99])
    leave_groups = np.array([10, -3, 10, 2, -3, 2])
    groups = np.repeat([-10, 2, 9, 20, 35, 50], [5, 4, 3, 4, 6, 2])
    labels = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1])
    rows = ["# ModelKit sklearn partitioning reference fixture v1"]

    def add(name, values):
        rows.append(f"{name}\t{comma_separated(values)}")

    add("assignments", assignments)
    add("leave_groups", leave_groups)
    add("groups", groups)
    add("labels", labels)
    cases = {
        "predefined": PredefinedSplit(assignments).split(),
        "leave_one_out": LeaveOneOut().split(np.zeros((4, 1))),
        "leave_one_group_out": LeaveOneGroupOut().split(
            np.zeros((len(leave_groups), 1)), groups=leave_groups),
        "stratified_group": StratifiedGroupKFold(n_splits=3).split(
            np.zeros((len(groups), 1)), labels, groups),
    }
    for name, splits in cases.items():
        for index, (train, test) in enumerate(splits):
            add(f"{name}_{index}_train", train)
            add(f"{name}_{index}_test", test)
    (fixture_dir / "partitioning_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {"stratified_group_folds": 3, "shuffle": False},
        "comparison": "Exact source row identities for unshuffled fixtures; grouped balancing is heuristic, with ModelKit additionally guaranteeing nonempty folds.",
        "environment": environment.metadata(), "fixture": "partitioning_v1",
        "generator": "dev/fixtures/generate.py", "license": "Apache-2.0",
        "schema_version": 1,
        "references": ["sklearn.model_selection.PredefinedSplit",
                       "sklearn.model_selection.LeaveOneOut",
                       "sklearn.model_selection.LeaveOneGroupOut",
                       "sklearn.model_selection.StratifiedGroupKFold"],
    }
    (fixture_dir / "partitioning_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def generate_randomized_search_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, RandomizedSearchCV

    x = np.arange(1., 13.)[:, None]
    y = np.array([3., 6., 6., 10., 11., 14., 14., 18., 19., 22., 22., 26.])
    model = RandomizedSearchCV(
        LinearRegression(), {"fit_intercept": [False, True]}, n_iter=2,
        cv=KFold(n_splits=3), random_state=19,
        scoring=["neg_mean_squared_error", "neg_mean_absolute_error"],
        refit="neg_mean_squared_error", return_train_score=True,
    ).fit(x, y)
    rows = ["# ModelKit sklearn randomized search reference fixture v1"]

    def add(name, values):
        rows.append(f"{name}\t{float_values(values)}")

    add("x", x[:, 0])
    add("y", y)
    add("test_x", [0., 13.])
    for index, parameters in enumerate(model.cv_results_["params"]):
        label = "intercept" if parameters["fit_intercept"] else "no_intercept"
        for metric in ["neg_mean_squared_error", "neg_mean_absolute_error"]:
            for partition in ["train", "test"]:
                add(f"{label}_{partition}_{metric}",
                    [model.cv_results_[f"mean_{partition}_{metric}"][index]])
    add("selected_intercept", [int(model.best_params_["fit_intercept"])])
    add("prediction", model.predict(np.array([[0.], [13.]])))
    (fixture_dir / "randomized_search_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {"random_state": 19, "folds": 3, "iterations": 2,
                          "fit_intercept": [False, True], "absolute_tolerance": 1e-10},
        "comparison": "Complete finite candidate set compared by parameter value, train/test means, named selection and full-data refit predictions; candidate order is RNG-specific.",
        "environment": environment.metadata(), "fixture": "randomized_search_v1",
        "generator": "dev/fixtures/generate.py", "license": "Apache-2.0",
        "schema_version": 1,
        "references": ["sklearn.model_selection.RandomizedSearchCV",
                       "sklearn.linear_model.LinearRegression"],
    }
    (fixture_dir / "randomized_search_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def generate_cross_val_prediction_fixture(fixture_dir: Path) -> None:
    import warnings

    import numpy as np
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    class FirstClassClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, x, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, x):
            return np.full(len(x), self.classes_[0])

        def predict_proba(self, x):
            probabilities = np.zeros((len(x), len(self.classes_)))
            probabilities[:, 0] = 1.0
            return probabilities

    regression_x = np.arange(12.0)[:, None]
    regression_y = np.array(
        [1.0, 3.2, 4.8, 7.1, 8.9, 11.2, 12.8, 15.1, 16.9, 19.2, 20.8, 23.1]
    )
    folds = KFold(n_splits=3, shuffle=False)
    regression_prediction = cross_val_predict(
        LinearRegression(), regression_x, regression_y, cv=folds
    )
    classification_x = np.arange(9.0)[:, None]
    classification_y = np.repeat([10, 20, 30], 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        class_probabilities = cross_val_predict(
            FirstClassClassifier(),
            classification_x,
            classification_y,
            cv=KFold(n_splits=3, shuffle=False),
            method="predict_proba",
        )
    rows = ["# ModelKit sklearn cross-validation prediction fixture v1"]

    def add(name, values):
        rows.append(f"{name}\t{float_values(values)}")

    add("regression_x", regression_x[:, 0])
    add("regression_y", regression_y)
    add("regression_prediction", regression_prediction)
    rows.append("classification_y\t" + comma_separated(classification_y))
    rows.append("classification_classes\t10,20,30")
    for row, probabilities in enumerate(class_probabilities):
        add(f"classification_probability_{row}", probabilities)
    (fixture_dir / "cross_val_prediction_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "folds": 3,
            "shuffle": False,
            "absolute_tolerance": 1e-9,
        },
        "comparison": "Regression predictions and globally aligned probability columns in original source-row order; the classifier deliberately omits one dataset class from every training fold.",
        "environment": environment.metadata(),
        "fixture": "cross_val_prediction_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "schema_version": 1,
        "references": [
            "sklearn.model_selection.cross_val_predict",
            "sklearn.linear_model.LinearRegression",
        ],
    }
    (fixture_dir / "cross_val_prediction_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_learning_curve_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, learning_curve

    x = np.arange(12.0)[:, None]
    y = np.array(
        [1.0, 3.2, 4.8, 7.1, 8.9, 11.2, 12.8, 15.1, 16.9, 19.2, 20.8, 23.1]
    )
    train_sizes, train_scores, test_scores = learning_curve(
        LinearRegression(),
        x,
        y,
        cv=KFold(n_splits=3, shuffle=False),
        scoring="neg_mean_absolute_error",
        train_sizes=np.array([0.25, 0.5, 1.0]),
        shuffle=False,
    )
    rows = ["# ModelKit sklearn learning-curve fixture v1"]
    rows.append("x\t" + float_values(x[:, 0]))
    rows.append("y\t" + float_values(y))
    rows.append("training_sizes\t" + comma_separated(train_sizes))
    for point, scores in enumerate(train_scores):
        rows.append(f"train_scores_{point}\t{float_values(scores)}")
    for point, scores in enumerate(test_scores):
        rows.append(f"test_scores_{point}\t{float_values(scores)}")
    (fixture_dir / "learning_curve_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "folds": 3,
            "scoring": "neg_mean_absolute_error",
            "shuffle": False,
            "train_sizes": [0.25, 0.5, 1.0],
            "absolute_tolerance": 1e-9,
        },
        "comparison": "Resolved training sizes and per-fold train/test scores over identical KFold partitions and nested training prefixes.",
        "environment": environment.metadata(),
        "fixture": "learning_curve_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "schema_version": 1,
        "references": [
            "sklearn.model_selection.learning_curve",
            "sklearn.linear_model.LinearRegression",
        ],
    }
    (fixture_dir / "learning_curve_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_validation_curve_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, validation_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    raw = np.arange(-4.0, 11.0)
    x = np.column_stack((raw, raw * raw))
    y = np.array(
        [
            15.2,
            9.1,
            5.4,
            2.8,
            1.2,
            0.7,
            1.5,
            3.4,
            6.8,
            11.1,
            16.9,
            23.7,
            31.8,
            41.0,
            51.6,
        ]
    )
    alphas = np.array([0.0, 0.5, 5.0])
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(fit_intercept=True, solver="svd")),
        ]
    )
    train_scores, test_scores = validation_curve(
        estimator,
        x,
        y,
        param_name="ridge__alpha",
        param_range=alphas,
        cv=KFold(n_splits=3, shuffle=False),
        scoring="neg_mean_squared_error",
    )
    rows = ["# ModelKit sklearn validation-curve fixture v1"]
    rows.append("feature_0\t" + float_values(x[:, 0]))
    rows.append("feature_1\t" + float_values(x[:, 1]))
    rows.append("target\t" + float_values(y))
    rows.append("alphas\t" + float_values(alphas))
    for point, scores in enumerate(train_scores):
        rows.append(f"train_scores_{point}\t{float_values(scores)}")
    for point, scores in enumerate(test_scores):
        rows.append(f"test_scores_{point}\t{float_values(scores)}")
    (fixture_dir / "validation_curve_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "absolute_tolerance": 1e-7,
            "alphas": alphas.tolist(),
            "folds": 3,
            "ridge_solver": "svd",
            "scoring": "neg_mean_squared_error",
            "shuffle": False,
            "standard_scaling": True,
        },
        "comparison": "Per-alpha, per-fold train/test scores over one shared KFold partition with fold-local standard scaling.",
        "environment": environment.metadata(),
        "fixture": "validation_curve_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "schema_version": 1,
        "references": [
            "sklearn.model_selection.validation_curve",
            "sklearn.linear_model.Ridge",
            "sklearn.preprocessing.StandardScaler",
        ],
    }
    (fixture_dir / "validation_curve_v1.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_permutation_test_fixture(fixture_dir: Path) -> None:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, permutation_test_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    target = np.repeat(np.arange(6, dtype=float), 2)
    groups = np.repeat(np.arange(6), 2)
    feature_0 = np.linspace(-2.0, 3.5, target.size)
    x = np.column_stack((feature_0, np.sin(feature_0)))
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=0.5, fit_intercept=True, solver="svd")),
        ]
    )
    folds = list(KFold(n_splits=3, shuffle=False).split(x, target))
    observed, permutation_scores, p_value = permutation_test_score(
        estimator,
        x,
        target,
        groups=groups,
        cv=folds,
        n_permutations=5,
        random_state=73,
        scoring="neg_mean_squared_error",
    )
    rows = [
        "# ModelKit sklearn permutation-test fixture v1",
        "feature_0\t" + float_values(x[:, 0]),
        "feature_1\t" + float_values(x[:, 1]),
        "target\t" + float_values(target),
        "groups\t" + comma_separated(groups),
        "observed_score\t" + float_value(observed),
        "permutation_scores\t" + float_values(permutation_scores),
        "p_value\t" + float_value(p_value),
    ]
    (fixture_dir / "permutation_test_v1.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "configuration": {
            "folds": 3,
            "groups": "two rows per group; target is constant within each group",
            "n_permutations": 5,
            "random_state": 73,
            "ridge_alpha": 0.5,
            "ridge_solver": "svd",
            "scoring": "neg_mean_squared_error",
            "shuffle": False,
            "standard_scaling": True,
        },
        "comparison": "Observed score, within-group permutation scores, and corrected upper-tail p-value.",
        "environment": environment.metadata(),
        "fixture": "permutation_test_v1",
        "generator": "dev/fixtures/generate.py",
        "license": "Apache-2.0",
        "schema_version": 1,
        "references": [
            "sklearn.model_selection.permutation_test_score",
            "sklearn.linear_model.Ridge",
            "sklearn.preprocessing.StandardScaler",
        ],
    }
    (fixture_dir / "permutation_test_v1.metadata.json").write_text(
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
    generate_resampling_fixture(fixture_dir)
    generate_partitioning_fixture(fixture_dir)
    generate_randomized_search_fixture(fixture_dir)
    generate_cross_val_prediction_fixture(fixture_dir)
    generate_learning_curve_fixture(fixture_dir)
    generate_validation_curve_fixture(fixture_dir)
    generate_permutation_test_fixture(fixture_dir)
    generate_metrics_fixture(fixture_dir)
    generate_preprocessing_fixture(fixture_dir)
    generate_univariate_selection_fixture(fixture_dir)
    generate_model_based_selection_fixture(fixture_dir)
    generate_recursive_feature_elimination_fixture(fixture_dir)
    generate_transform_fixture(fixture_dir)
    generate_column_transformer_fixture(fixture_dir)
    generate_nested_composition_fixture(fixture_dir)
    generate_transformed_target_fixture(fixture_dir)
    generate_linear_model_fixture(fixture_dir)
    generate_regularized_linear_fixture(fixture_dir)
    generate_ridge_classifier_fixture(fixture_dir)
    generate_multinomial_logistic_fixture(fixture_dir)
    generate_glm_fixture(fixture_dir)
    generate_sgd_regressor_fixture(fixture_dir)
    generate_sgd_classifier_fixture(fixture_dir)
    generate_class_weight_fixture(fixture_dir)
    generate_multiclass_metrics_fixture(fixture_dir)
    generate_ranking_metrics_fixture(fixture_dir)


if __name__ == "__main__":
    main()
