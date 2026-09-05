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


def ridge_classifier(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    binary_score = x[:, 0] + 0.25 * x[:, 1] - 0.1 * x[:, 2]
    binary_target = np.where(binary_score > 0.0, 7, -3)
    multiclass_target = np.where(
        x[:, 0] + 0.25 * x[:, 1] > 1.0,
        9,
        np.where(x[:, 2] - 0.2 * x[:, 3] > 0.0, 2, -4),
    )
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    binary = RidgeClassifier(alpha=scenario["alpha"], solver="svd").fit(
        x, binary_target, sample_weight=sample_weight
    )
    multiclass = RidgeClassifier(alpha=scenario["alpha"], solver="svd").fit(
        x, multiclass_target, sample_weight=sample_weight
    )
    binary_decisions = binary.decision_function(x)
    binary_predictions = binary.predict(x)
    multiclass_decisions = multiclass.decision_function(x)
    multiclass_predictions = multiclass.predict(x)
    signature = np.asarray(
        [
            binary_decisions[0],
            binary_decisions[-1],
            binary_predictions[0],
            binary_predictions[-1],
            *multiclass_decisions[0],
            *multiclass_decisions[-1],
            multiclass_predictions[0],
            multiclass_predictions[-1],
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "binary_ridge_classification",
            "multiclass_ridge_classification",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def multinomial_logistic(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    target = np.where(
        x[:, 0] + 0.25 * x[:, 1] > 1.0,
        9,
        np.where(x[:, 2] - 0.2 * x[:, 3] > 0.0, 2, -4),
    )
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    fitted = LogisticRegression(
        C=scenario["c"],
        solver=scenario["solver"],
        tol=scenario["tolerance"],
        max_iter=scenario["max_iterations"],
    ).fit(x, target, sample_weight=sample_weight)
    decisions = fitted.decision_function(x)
    probabilities = fitted.predict_proba(x)
    predictions = fitted.predict(x)
    signature = np.asarray(
        [
            *decisions[0],
            *probabilities[0],
            *probabilities[-1],
            predictions[0],
            predictions[-1],
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "weighted_multinomial_logistic_fit",
            "decision_function",
            "predict_proba",
            "predict",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def glm(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import PoissonRegressor, TweedieRegressor

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    raw = 0.2 + 0.08 * x[:, 0] - 0.04 * x[:, 1] + 0.03 * x[:, 2]
    target = np.exp(raw) * (0.8 + (rows[:, 0] % 5).astype(np.float64) * 0.1)
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    common = {
        "alpha": scenario["alpha"],
        "solver": scenario["solver"],
        "tol": scenario["tolerance"],
        "max_iter": scenario["max_iterations"],
    }
    poisson = PoissonRegressor(**common).fit(
        x, target, sample_weight=sample_weight
    )
    tweedie = TweedieRegressor(
        power=scenario["power"], link="log", **common
    ).fit(x, target, sample_weight=sample_weight)
    poisson_predictions = poisson.predict(x)
    tweedie_predictions = tweedie.predict(x)
    signature = np.asarray(
        [
            poisson.coef_[0],
            poisson.intercept_,
            poisson_predictions[0],
            poisson_predictions[-1],
            tweedie.coef_[0],
            tweedie.intercept_,
            tweedie_predictions[0],
            tweedie_predictions[-1],
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "weighted_poisson_fit_predict",
            "weighted_tweedie_fit_predict",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def regularized_linear(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import ElasticNet, Lasso

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    target = (
        2.0
        + 1.5 * x[:, 0]
        - 0.8 * x[:, 1]
        + 0.3 * x[:, 2]
        + ((rows[:, 0] % 7).astype(np.float64) - 3.0) * 0.02
    )
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    common = {
        "fit_intercept": True,
        "tol": scenario["tolerance"],
        "max_iter": scenario["max_iterations"],
        "selection": "cyclic",
    }
    lasso = Lasso(alpha=scenario["alpha"], **common).fit(
        x, target, sample_weight=sample_weight
    )
    elastic = ElasticNet(
        alpha=scenario["alpha"], l1_ratio=scenario["l1_ratio"], **common
    ).fit(x, target, sample_weight=sample_weight)

    def path(estimator):
        coefficients = []
        for alpha in scenario["path_alphas"]:
            estimator.set_params(alpha=alpha)
            estimator.fit(x, target, sample_weight=sample_weight)
            coefficients.append(estimator.coef_.copy())
        return np.asarray(coefficients)

    lasso_path = path(Lasso(alpha=scenario["path_alphas"][0], warm_start=True, **common))
    elastic_path = path(
        ElasticNet(
            alpha=scenario["path_alphas"][0],
            l1_ratio=scenario["l1_ratio"],
            warm_start=True,
            **common,
        )
    )
    lasso_predictions = lasso.predict(x)
    elastic_predictions = elastic.predict(x)
    signature = np.asarray(
        [
            lasso.coef_[0],
            lasso.intercept_,
            lasso_predictions[0],
            lasso_predictions[-1],
            elastic.coef_[0],
            elastic.intercept_,
            elastic_predictions[0],
            elastic_predictions[-1],
            lasso_path[0, 0],
            lasso_path[-1, 0],
            elastic_path[0, 0],
            elastic_path[-1, 0],
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "weighted_lasso_fit_predict",
            "weighted_elastic_net_fit_predict",
            "weighted_lasso_path",
            "weighted_elastic_net_path",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def sgd_regression(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import SGDRegressor

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    target = (
        2.0
        + 1.5 * x[:, 0]
        - 0.8 * x[:, 1]
        + 0.3 * x[:, 2]
        + ((rows[:, 0] % 7).astype(np.float64) - 3.0) * 0.02
    )
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    configuration = {
        "loss": "squared_error",
        "penalty": None,
        "alpha": 0.0,
        "fit_intercept": True,
        "max_iter": scenario["epochs"],
        "tol": None,
        "shuffle": False,
        "learning_rate": "constant",
        "eta0": scenario["eta0"],
        "average": False,
    }
    fitted = SGDRegressor(**configuration).fit(
        x, target, sample_weight=sample_weight
    )
    incremental = SGDRegressor(**configuration)
    for _ in range(scenario["epochs"]):
        incremental.partial_fit(x, target, sample_weight=sample_weight)
    predictions = fitted.predict(x)
    incremental_predictions = incremental.predict(x)
    signature = np.asarray(
        [
            fitted.coef_[0],
            fitted.intercept_[0],
            predictions[0],
            predictions[-1],
            incremental.coef_[0],
            incremental.intercept_[0],
            incremental_predictions[0],
            incremental_predictions[-1],
            fitted.t_ - 1.0,
            incremental.t_ - 1.0,
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "weighted_sgd_regressor_fit_predict",
            "weighted_sgd_regressor_partial_fit_predict",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def sgd_classification(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import SGDClassifier

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    binary = np.where(x[:, 0] + 0.25 * x[:, 1] > 1.0, 9, -4)
    multiclass = np.where(
        x[:, 0] + 0.25 * x[:, 1] > 1.0,
        9,
        np.where(x[:, 2] - 0.2 * x[:, 3] > 0.0, 2, -4),
    )
    sample_weight = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25

    def configuration(loss: str) -> dict[str, object]:
        return {
            "loss": loss,
            "penalty": None,
            "alpha": 0.0,
            "fit_intercept": True,
            "max_iter": scenario["epochs"],
            "tol": None,
            "shuffle": False,
            "learning_rate": "constant",
            "eta0": scenario["eta0"],
            "average": False,
        }

    hinge = SGDClassifier(**configuration("hinge")).fit(
        x, binary, sample_weight=sample_weight
    )
    log_loss = SGDClassifier(**configuration("log_loss")).fit(
        x, multiclass, sample_weight=sample_weight
    )
    incremental = SGDClassifier(**configuration("log_loss"))
    classes = np.array([-4, 2, 9], dtype=np.int64)
    for _ in range(scenario["epochs"]):
        incremental.partial_fit(
            x, multiclass, classes=classes, sample_weight=sample_weight
        )
    hinge_decisions = hinge.decision_function(x)
    hinge_predictions = hinge.predict(x)
    probabilities = log_loss.predict_proba(x)
    predictions = log_loss.predict(x)
    incremental_probabilities = incremental.predict_proba(x)
    signature = np.asarray(
        [
            hinge.coef_[0, 0],
            hinge.intercept_[0],
            hinge_decisions[0],
            hinge_decisions[-1],
            hinge_predictions[0],
            hinge_predictions[-1],
            log_loss.coef_[0, 0],
            log_loss.intercept_[0],
            *probabilities[0],
            *probabilities[-1],
            predictions[0],
            predictions[-1],
            incremental_probabilities[-1, 2],
            log_loss.t_ - 1.0,
            incremental.t_ - 1.0,
        ],
        dtype="<f8",
    )
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "weighted_binary_hinge_sgd_fit_decision_predict",
            "weighted_multiclass_log_loss_sgd_fit_proba_predict",
            "weighted_multiclass_log_loss_sgd_partial_fit_proba",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def adapter_admission(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.utils import check_array, check_consistent_length
    from sklearn.utils.validation import _check_sample_weight

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = ((rows * 17 + columns * 31 + dataset["seed"]) % 1000).astype(np.float64)
    x /= 100.0
    missing = (columns > 0) & (
        (rows * 101 + columns * 53 + dataset["seed"]) % dataset["missing_modulus"] == 0
    )
    x[missing] = np.nan
    y = rows[:, 0] % 3
    weights = 1.0 + (rows[:, 0] % 5).astype(np.float64) * 0.25
    groups = rows[:, 0] // 10

    def admit(features):
        admitted = check_array(features, dtype=np.float64, ensure_all_finite="allow-nan")
        labels = check_array(y, ensure_2d=False, dtype=np.int64)
        admitted_weights = _check_sample_weight(weights, admitted)
        admitted_groups = check_array(groups, ensure_2d=False, dtype=np.int64)
        check_consistent_length(admitted, labels, admitted_weights, admitted_groups)
        return [
            float(admitted.shape[0]),
            float(admitted.shape[1]),
            float(np.nansum(admitted)),
            float(np.isnan(admitted).sum()),
            float(labels.sum()),
            float(admitted_weights.sum()),
            float(admitted_groups.sum()),
        ]

    tensor_signature = admit(x)
    table = {f"feature_{index}": x[:, index].copy() for index in range(x.shape[1])}
    table_signature = admit(np.column_stack(list(table.values())))
    signature = np.asarray([*tensor_signature, *table_signature], dtype="<f8")
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "features": x.shape[1],
        "operations": [
            "numpy_tensor_check_array_admission",
            "numpy_column_stack_check_array_admission",
        ],
        "samples": x.shape[0],
        "signature": signature.tolist(),
        "threadpools": threadpools(),
    }


def sparse_kernels(scenario: dict[str, object]) -> dict[str, object]:
    import time

    import numpy as np
    import scipy.sparse as sparse
    from sklearn.preprocessing import OneHotEncoder

    def timed(function, repeats):
        started = time.perf_counter_ns()
        result = function()
        for _ in range(repeats - 1):
            result = function()
        return result, {"allocated_words": None, "elapsed_ns": time.perf_counter_ns() - started}

    dataset = scenario["dataset"]
    seed = dataset["seed"]
    repeats = scenario["repeats"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    values = (((rows * 17 + columns * 31 + seed) % 1000).astype(np.float64) + 0.5) / 100.0 - 5.0
    hashes = (rows * 101 + columns * 53 + seed) % 10000
    operand = ((columns[0] * 7) % 13).astype(np.float64) / 13.0 - 0.5
    transposed_operand = ((rows[:, 0] * 3) % 11).astype(np.float64) / 11.0 - 0.5
    signature = []
    cases = []
    matrices = []
    for density in dataset["densities"]:
        threshold = int(round(density * 10000))
        dense = np.where(hashes < threshold, values, 0.0)
        csr = sparse.csr_matrix(dense)
        matrices.append(csr)
        csr_product, csr_timing = timed(lambda: csr @ operand, repeats)
        dense_product, dense_timing = timed(lambda: dense @ operand, repeats)
        csr_transposed, csr_transposed_timing = timed(
            lambda: csr.T @ transposed_operand, repeats
        )
        dense_transposed, dense_transposed_timing = timed(
            lambda: dense.T @ transposed_operand, repeats
        )
        signature.extend(
            [
                float(csr.nnz),
                float(csr_product.sum()),
                float(dense_product.sum()),
                float(csr_transposed.sum()),
                float(dense_transposed.sum()),
            ]
        )
        cases.append(
            {
                "csr_memory": {
                    "dense_equivalent_bytes": int(dense.nbytes),
                    "total_bytes": int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes),
                },
                "csr_product": csr_timing,
                "csr_transposed_product": csr_transposed_timing,
                "dense_product": dense_timing,
                "dense_transposed_product": dense_transposed_timing,
                "density": density,
                "nonzeros": int(csr.nnz),
            }
        )
    one_hot = scenario["one_hot"]
    one_hot_rows = np.arange(one_hot["samples"], dtype=np.int64)[:, np.newaxis]
    one_hot_features = np.arange(one_hot["features"], dtype=np.int64)[np.newaxis, :]
    one_hot_input = ((one_hot_rows * 13 + one_hot_features * 7) % one_hot["cardinality"]).astype(
        np.float64
    )
    dense_encoder = OneHotEncoder(handle_unknown="error", sparse_output=False).fit(one_hot_input)
    sparse_encoder = OneHotEncoder(handle_unknown="error", sparse_output=True).fit(one_hot_input)
    one_hot_dense, one_hot_dense_timing = timed(lambda: dense_encoder.transform(one_hot_input), 1)
    one_hot_csr, one_hot_csr_timing = timed(
        lambda: sparse_encoder.transform(one_hot_input).tocsr(), 1
    )
    source = matrices[len(matrices) // 2]
    even_rows = np.arange(0, dataset["samples"], 2)
    materialized, materialize_timing = timed(lambda: source[even_rows], 1)
    signature.extend(
        [
            float(one_hot_dense.shape[1]),
            float(one_hot_csr.nnz),
            float((one_hot_dense * np.arange(one_hot_dense.shape[1])).sum()),
            float((one_hot_csr.indices * one_hot_csr.data).sum()),
            float(materialized.shape[0]),
            float(materialized.nnz),
            float(materialized.data.sum()),
        ]
    )
    signature = np.asarray(signature, dtype="<f8")
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature.tobytes()).hexdigest(),
        "densities": cases,
        "features": dataset["features"],
        "materialization": {
            "allocated_bytes": None,
            "materialize": materialize_timing,
            "materialized_bytes": int(
                materialized.data.nbytes + materialized.indices.nbytes + materialized.indptr.nbytes
            ),
            "shared_bytes": None,
            "view_rows": int(materialized.shape[0]),
        },
        "one_hot": {
            "csr_memory": {
                "dense_equivalent_bytes": int(one_hot_dense.nbytes),
                "total_bytes": int(
                    one_hot_csr.data.nbytes + one_hot_csr.indices.nbytes + one_hot_csr.indptr.nbytes
                ),
            },
            "csr_transform": one_hot_csr_timing,
            "dense_bytes": int(one_hot_dense.nbytes),
            "dense_transform": one_hot_dense_timing,
            "output_columns": int(one_hot_dense.shape[1]),
        },
        "operations": [
            "csr_and_dense_feature_matrix_vector_products",
            "one_hot_dense_and_csr_transform",
            "csr_row_view_materialization",
        ],
        "repeats": repeats,
        "samples": dataset["samples"],
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


def cross_validation(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    raw = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    raw = (raw / 100.0) - 5.0
    x = raw.copy()
    missing = (columns > 0) & (
        (rows * 101 + columns * 53 + dataset["seed"])
        % dataset["missing_modulus"]
        == 0
    )
    x[missing] = np.nan
    score = raw[:, 0] + 0.25 * raw[:, 1] - 0.1 * raw[:, 2]
    y = np.where(score > 0.0, 7, -3)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=scenario["logistic_c"],
                    solver=scenario["logistic_solver"],
                    tol=scenario["logistic_tolerance"],
                    max_iter=scenario["logistic_max_iterations"],
                ),
            ),
        ]
    )
    result = cross_validate(
        pipeline,
        x,
        y,
        cv=StratifiedKFold(n_splits=scenario["folds"], shuffle=False),
        scoring=("accuracy", "balanced_accuracy", "neg_log_loss", "roc_auc"),
        n_jobs=scenario.get("execution_workers", 1),
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    signature = []
    for fold in range(scenario["folds"]):
        for scorer in (
            "accuracy",
            "balanced_accuracy",
            "neg_log_loss",
            "roc_auc",
        ):
            signature.extend(
                [
                    result[f"train_{scorer}"][fold],
                    result[f"test_{scorer}"][fold],
                ]
            )
        train_indices = result["indices"]["train"][fold]
        test_indices = result["indices"]["test"][fold]
        signature.extend(
            [
                len(train_indices),
                len(test_indices),
                np.sum(train_indices, dtype=np.int64),
                np.sum(test_indices, dtype=np.int64),
                float(result["estimator"][fold] is not None),
            ]
        )
    signature_array = np.asarray(signature, dtype="<f8")
    return {
        "allocated_words": None,
        "checksum": hashlib.sha256(signature_array.tobytes()).hexdigest(),
        "features": x.shape[1],
        "folds": scenario["folds"],
        "fold_workers": scenario.get("execution_workers", 1),
        "operations": [
            "mean_imputation",
            "standard_scaling",
            "binary_logistic_regression",
            "cross_validate",
            "multiple_scorers",
            "train_scores",
            "models",
            "indices",
        ],
        "samples": x.shape[0],
        "signature": signature_array.tolist(),
        "threadpools": threadpools(),
    }


def grid_search(scenario: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold

    dataset = scenario["dataset"]
    rows = np.arange(dataset["samples"], dtype=np.int64)[:, np.newaxis]
    columns = np.arange(dataset["features"], dtype=np.int64)[np.newaxis, :]
    x = (
        (rows * (17 + columns * 12) + columns * 31 + dataset["seed"]) % 1000
    ).astype(np.float64)
    x = (x / 100.0) - 5.0
    coefficients = ((columns[0] % 5) - 2).astype(np.float64) * 0.2
    noise = (((rows[:, 0] * 13 + 1729) % 11) - 5).astype(np.float64) * 0.01
    y = 1.25 + x @ coefficients + noise
    search = GridSearchCV(
        Ridge(solver="svd"),
        {
            "alpha": scenario["alphas"],
            "fit_intercept": scenario["fit_intercepts"],
        },
        cv=KFold(n_splits=scenario["folds"], shuffle=False),
        scoring=("neg_mean_squared_error", "r2"),
        refit="r2",
        return_train_score=True,
        error_score="raise",
        n_jobs=1,
    ).fit(x, y)
    signature = []
    for candidate, parameters in enumerate(search.cv_results_["params"]):
        signature.extend(
            [
                parameters["alpha"],
                float(parameters["fit_intercept"]),
                search.cv_results_["mean_train_neg_mean_squared_error"][candidate],
                search.cv_results_["mean_test_neg_mean_squared_error"][candidate],
                search.cv_results_["mean_train_r2"][candidate],
                search.cv_results_["mean_test_r2"][candidate],
                search.cv_results_["rank_test_r2"][candidate],
            ]
        )
    predictions = search.predict(x[[0, -1]])
    signature.extend([search.best_index_, predictions[0], predictions[1]])
    signature_array = np.asarray(signature, dtype="<f8")
    return {
        "allocated_words": None,
        "candidates": len(search.cv_results_["params"]),
        "checksum": hashlib.sha256(signature_array.tobytes()).hexdigest(),
        "features": x.shape[1],
        "folds": scenario["folds"],
        "operations": [
            "finite_grid_expansion",
            "cross_validate",
            "candidate_ranking",
            "best_model_refit",
        ],
        "samples": x.shape[0],
        "signature": signature_array.tolist(),
        "threadpools": threadpools(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("execution_workers", type=int, nargs="?")
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    if args.execution_workers is not None:
        scenario["execution_workers"] = args.execution_workers
    workload = scenario.get("workload", "dummy_cv")
    if workload == "dummy_cv":
        result = dummy_cv(scenario)
    elif workload == "preprocessing":
        result = preprocessing(scenario)
    elif workload == "linear_models":
        result = linear_models(scenario)
    elif workload == "ridge_classifier":
        result = ridge_classifier(scenario)
    elif workload == "multinomial_logistic":
        result = multinomial_logistic(scenario)
    elif workload == "glm":
        result = glm(scenario)
    elif workload == "regularized_linear":
        result = regularized_linear(scenario)
    elif workload == "sgd_regression":
        result = sgd_regression(scenario)
    elif workload == "sgd_classification":
        result = sgd_classification(scenario)
    elif workload == "adapter_admission":
        result = adapter_admission(scenario)
    elif workload == "sparse_kernels":
        result = sparse_kernels(scenario)
    elif workload == "splitters":
        result = splitters(scenario)
    elif workload == "metrics":
        result = metrics(scenario)
    elif workload in ("cross_validation", "parallel_cross_validation"):
        result = cross_validation(scenario)
    elif workload == "grid_search":
        result = grid_search(scenario)
    else:
        raise ValueError(f"unknown workload {workload!r}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
