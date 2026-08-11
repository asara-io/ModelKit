from importlib.metadata import PackageNotFoundError, version
import platform
import sys


EXPECTED_PYTHON = (3, 14, 3)
EXPECTED_PACKAGES = {
    "joblib": "1.5.3",
    "narwhals": "2.24.0",
    "numpy": "2.5.2",
    "psutil": "7.1.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "threadpoolctl": "3.6.0",
}


def validate() -> None:
    observed_python = sys.version_info[:3]
    if observed_python != EXPECTED_PYTHON:
        expected = ".".join(map(str, EXPECTED_PYTHON))
        observed = ".".join(map(str, observed_python))
        raise RuntimeError(f"expected Python {expected}, observed {observed}")

    mismatches = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            observed = version(package)
        except PackageNotFoundError:
            observed = "not installed"
        if observed != expected:
            mismatches.append(f"{package}: expected {expected}, observed {observed}")
    if mismatches:
        raise RuntimeError("development dependency mismatch:\n" + "\n".join(mismatches))


def metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "packages": {
            package: version(package) for package in sorted(EXPECTED_PACKAGES)
        },
    }
