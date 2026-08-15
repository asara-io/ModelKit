from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "python"))

import environment
import psutil


def resident_set_size(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for member in processes:
        try:
            total += member.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def measure(command: list[str], process_environment: dict[str, str]) -> dict[str, object]:
    started = time.perf_counter_ns()
    child = subprocess.Popen(
        command,
        env=process_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process = psutil.Process(child.pid)
    peak_rss_bytes = 0
    while child.poll() is None:
        peak_rss_bytes = max(peak_rss_bytes, resident_set_size(process))
        time.sleep(0.001)
    stdout, stderr = child.communicate()
    elapsed_ns = time.perf_counter_ns() - started
    if child.returncode != 0:
        raise RuntimeError(f"benchmark worker failed:\n{stderr}")
    return {
        "elapsed_ns": elapsed_ns,
        "peak_rss_bytes": peak_rss_bytes,
        "worker": json.loads(stdout),
    }


def platform_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    return f"{system}-{machine}"


def worker_commands(scenario: dict[str, object], scenario_path: Path) -> dict[str, list[str]]:
    sklearn_worker = ROOT / "dev" / "benchmarks" / "sklearn_worker.py"
    commands = {"sklearn": [sys.executable, str(sklearn_worker), str(scenario_path)]}
    workload = scenario.get("workload")
    if workload == "preprocessing":
        modelkit_worker = (
            ROOT / "_build" / "default" / "bench" / "ocaml" / "preprocessing_worker.exe"
        )
        if not modelkit_worker.exists():
            raise RuntimeError(
                "ModelKit benchmark worker is missing; run "
                "`opam exec -- dune build bench/ocaml/preprocessing_worker.exe`"
            )
        dataset = scenario["dataset"]
        commands["modelkit"] = [
            str(modelkit_worker),
            str(dataset["samples"]),
            str(dataset["features"]),
            str(dataset["seed"]),
            str(dataset["missing_modulus"]),
            str(scenario["variance_threshold"]),
            str(scenario["imputation_constant"]),
        ]
    elif workload == "linear_models":
        modelkit_worker = (
            ROOT / "_build" / "default" / "bench" / "ocaml" / "linear_models_worker.exe"
        )
        if not modelkit_worker.exists():
            raise RuntimeError(
                "ModelKit benchmark worker is missing; run "
                "`opam exec -- dune build bench/ocaml/linear_models_worker.exe`"
            )
        dataset = scenario["dataset"]
        commands["modelkit"] = [
            str(modelkit_worker),
            str(dataset["samples"]),
            str(dataset["features"]),
            str(dataset["seed"]),
            str(scenario["ridge_alpha"]),
            str(scenario["logistic_c"]),
            str(scenario["logistic_tolerance"]),
            str(scenario["logistic_max_iterations"]),
        ]
    elif workload == "splitters":
        modelkit_worker = (
            ROOT / "_build" / "default" / "bench" / "ocaml" / "splitters_worker.exe"
        )
        if not modelkit_worker.exists():
            raise RuntimeError(
                "ModelKit benchmark worker is missing; run "
                "`opam exec -- dune build bench/ocaml/splitters_worker.exe`"
            )
        dataset = scenario["dataset"]
        commands["modelkit"] = [
            str(modelkit_worker),
            str(dataset["samples"]),
            str(scenario["folds"]),
            str(dataset["classes"]),
            str(dataset["group_size"]),
            str(scenario["time_test_size"]),
            str(scenario["time_gap"]),
        ]
    elif workload == "metrics":
        modelkit_worker = (
            ROOT / "_build" / "default" / "bench" / "ocaml" / "metrics_worker.exe"
        )
        if not modelkit_worker.exists():
            raise RuntimeError(
                "ModelKit benchmark worker is missing; run "
                "`opam exec -- dune build bench/ocaml/metrics_worker.exe`"
            )
        commands["modelkit"] = [
            str(modelkit_worker),
            str(scenario["dataset"]["samples"]),
        ]
    elif workload == "cross_validation":
        modelkit_worker = (
            ROOT
            / "_build"
            / "default"
            / "bench"
            / "ocaml"
            / "cross_validation_worker.exe"
        )
        if not modelkit_worker.exists():
            raise RuntimeError(
                "ModelKit benchmark worker is missing; run "
                "`opam exec -- dune build bench/ocaml/cross_validation_worker.exe`"
            )
        dataset = scenario["dataset"]
        commands["modelkit"] = [
            str(modelkit_worker),
            str(dataset["samples"]),
            str(dataset["features"]),
            str(dataset["seed"]),
            str(scenario["folds"]),
            str(dataset["missing_modulus"]),
            str(scenario["logistic_c"]),
            str(scenario["logistic_tolerance"]),
            str(scenario["logistic_max_iterations"]),
        ]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=ROOT
        / "dev"
        / "benchmarks"
        / "scenarios"
        / "sklearn_dummy_cv_smoke.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    environment.validate()

    scenario_path = args.scenario.resolve()
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    output = args.output or (
        ROOT
        / "dev"
        / "benchmarks"
        / "results"
        / f"{scenario['scenario']}.{platform_name()}.json"
    )
    commands = worker_commands(scenario, scenario_path)
    process_environment = os.environ.copy()
    thread_limit = str(scenario["thread_limit"])
    for variable in (
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        process_environment[variable] = thread_limit

    for _ in range(scenario["warmup_runs"]):
        for command in commands.values():
            measure(command, process_environment)
    runs = {implementation: [] for implementation in commands}
    for _ in range(scenario["measured_runs"]):
        for implementation, command in commands.items():
            runs[implementation].append(measure(command, process_environment))
    for implementation, implementation_runs in runs.items():
        checksums = {run["worker"]["checksum"] for run in implementation_runs}
        if len(checksums) != 1:
            raise RuntimeError(
                f"{implementation} benchmark worker produced inconsistent results"
            )
        if any(run["peak_rss_bytes"] <= 0 for run in implementation_runs):
            raise RuntimeError(
                f"benchmark harness did not observe {implementation} resident memory"
            )
    signature_tolerance = {
        "preprocessing": 1e-12,
        "linear_models": 1e-7,
        "splitters": 0.0,
        "metrics": 1e-7,
        "cross_validation": 1e-7,
    }.get(scenario.get("workload"))
    if signature_tolerance is not None:
        signatures = {
            implementation: implementation_runs[0]["worker"]["signature"]
            for implementation, implementation_runs in runs.items()
        }
        reference = signatures["sklearn"]
        for implementation, signature in signatures.items():
            if len(signature) != len(reference) or any(
                not math.isclose(
                    expected,
                    observed,
                    rel_tol=signature_tolerance,
                    abs_tol=signature_tolerance,
                )
                for expected, observed in zip(reference, signature, strict=True)
            ):
                raise RuntimeError(
                    f"{implementation} output signature does not match sklearn"
                )
    observed_thread_counts = {
        pool["num_threads"]
        for implementation_runs in runs.values()
        for run in implementation_runs
        for pool in run["worker"]["threadpools"]
    }
    if observed_thread_counts != {scenario["thread_limit"]}:
        raise RuntimeError(
            f"expected thread count {scenario['thread_limit']}, "
            f"observed {sorted(observed_thread_counts)}"
        )

    report = {
        "claim_eligible": scenario["claim_eligible"],
        "environment": {
            **environment.metadata(),
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "operating_system": platform.platform(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "measurement": (
                "interleaved fresh-process wall time and sampled resident set size"
            ),
            "rss_sample_interval_seconds": 0.001,
            "signature_tolerance": (
                {"absolute": signature_tolerance, "relative": signature_tolerance}
                if signature_tolerance is not None
                else None
            ),
            "thread_limit": scenario["thread_limit"],
            "warmup_runs": scenario["warmup_runs"],
        },
        "runs": runs,
        "scenario": scenario,
        "schema_version": 2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
