from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
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
    worker = ROOT / "dev" / "benchmarks" / "sklearn_worker.py"
    command = [sys.executable, str(worker), str(scenario_path)]
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
        measure(command, process_environment)
    runs = [
        measure(command, process_environment)
        for _ in range(scenario["measured_runs"])
    ]
    checksums = {run["worker"]["checksum"] for run in runs}
    if len(checksums) != 1:
        raise RuntimeError("benchmark worker produced inconsistent results")
    if any(run["peak_rss_bytes"] <= 0 for run in runs):
        raise RuntimeError("benchmark harness did not observe resident memory")
    observed_thread_counts = {
        pool["num_threads"]
        for run in runs
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
            "measurement": "fresh process wall time and sampled resident set size",
            "rss_sample_interval_seconds": 0.001,
            "thread_limit": scenario["thread_limit"],
            "warmup_runs": scenario["warmup_runs"],
        },
        "runs": runs,
        "scenario": scenario,
        "schema_version": 1,
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
