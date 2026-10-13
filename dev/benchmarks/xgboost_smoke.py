from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1"}
    worker = ROOT / "_build/default/bench/ocaml/xgboost_worker.exe"
    completed = subprocess.run(
        [str(worker)], env=environment, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(f"XGBoost smoke worker failed:\n{completed.stderr}")
    report = json.loads(completed.stdout)
    report["generator"] = "bench/ocaml/xgboost_worker.ml"
    report["environment"] = {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python_harness": platform.python_version(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    report["methodology"] = {
        "timing": "operation wall time; excludes startup, data generation, GC preparation, and response checks",
        "allocation": "OCaml GC allocation counters; float32 buffer payload is reported separately; native allocations and peak RSS are unmeasured",
        "fit": "includes admission, native training, and detachment from training caches",
        "predict": "regression values or classification probabilities; includes feature conversion and output copying",
        "validation": "finite responses, exact repeated-fit signatures, exact conversion payload, sorted original labels, probability normalization and label alignment",
        "comparison": "noncomparative smoke evidence; no timing or allocation regression threshold",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Verified XGBoost {report['native']} smoke: {arguments.output}")


if __name__ == "__main__":
    main()
