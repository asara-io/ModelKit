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


def main() -> None:
    environment.validate()
    fixture_dir = ROOT / "test" / "fixtures" / "sklearn"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    generate_split_fixture(fixture_dir)
    generate_preprocessing_fixture(fixture_dir)


if __name__ == "__main__":
    main()
