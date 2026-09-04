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


def main() -> None:
    environment.validate()
    fixture_dir = ROOT / "test" / "fixtures" / "sklearn"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    generate_split_fixture(fixture_dir)
    generate_splitter_fixture(fixture_dir)
    generate_metrics_fixture(fixture_dir)
    generate_preprocessing_fixture(fixture_dir)
    generate_transform_fixture(fixture_dir)
    generate_linear_model_fixture(fixture_dir)
    generate_regularized_linear_fixture(fixture_dir)
    generate_ridge_classifier_fixture(fixture_dir)
    generate_multinomial_logistic_fixture(fixture_dir)
    generate_glm_fixture(fixture_dir)
    generate_sgd_regressor_fixture(fixture_dir)


if __name__ == "__main__":
    main()
