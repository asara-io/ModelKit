from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "python"))

import environment


def comma_separated(values) -> str:
    return ",".join(str(int(value)) for value in values)


def main() -> None:
    environment.validate()

    from sklearn.model_selection import StratifiedKFold

    target = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=1729)
    splits = list(splitter.split([[0.0]] * len(target), target))

    fixture_dir = ROOT / "test" / "fixtures" / "sklearn"
    fixture_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
