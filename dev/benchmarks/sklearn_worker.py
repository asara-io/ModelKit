from pathlib import Path
import argparse
import hashlib
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))

    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from threadpoolctl import threadpool_info

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
    checksum = hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest()
    threadpools = [
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
    print(
        json.dumps(
            {
                "checksum": checksum,
                "folds": len(result["test_accuracy"]),
                "threadpools": threadpools,
            }
        )
    )


if __name__ == "__main__":
    main()
