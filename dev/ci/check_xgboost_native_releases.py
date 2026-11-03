#!/usr/bin/env python3
"""Exercise checksum-pinned native libraries without installing Python XGBoost."""

import hashlib
import json
import pathlib
import platform
import subprocess
import tempfile
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = pathlib.Path(__file__).with_name("xgboost-native-wheels.json")


def unpack(wheel: pathlib.Path, destination: pathlib.Path, system: str) -> pathlib.Path:
    member = "xgboost/lib/libxgboost." + ("dylib" if system == "Darwin" else "so")
    with zipfile.ZipFile(wheel) as archive:
        archive.extract(member, destination)
        for name in archive.namelist():
            if name.startswith("xgboost_cpu.libs/") and not name.endswith("/"):
                archive.extract(name, destination)
    library = destination / "xgboost/lib"
    aliases = (
        ["libxgboost.3.dylib"]
        if system == "Darwin"
        else ["libxgboost.so.0", "libxgboost.so.3"]
    )
    for alias in aliases:
        (library / alias).symlink_to(pathlib.Path(member).name)
    return library


def main() -> None:
    system = platform.system()
    key = system + "/" + platform.machine()
    releases = json.loads(MANIFEST.read_text())[key]
    with tempfile.TemporaryDirectory(prefix="modelkit-native-releases-") as scratch:
        for version, source in releases.items():
            destination = pathlib.Path(scratch) / version
            destination.mkdir()
            wheel = destination / "native.whl"
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "-o", str(wheel), source["url"]],
                check=True,
            )
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != source["sha256"]:
                raise RuntimeError("Native wheel checksum mismatch: " + version)
            library = unpack(wheel, destination, system)
            variable = "DYLD_LIBRARY_PATH" if system == "Darwin" else "LD_LIBRARY_PATH"
            search_path = str(library)
            if system == "Linux":
                search_path += ":" + str(destination / "xgboost_cpu.libs")
            result = subprocess.run(
                ["opam", "exec", "--", "env", variable + "=" + search_path,
                 "dune", "runtest", "--force", "--only-packages",
                 "modelkit,modelkit-parallel,modelkit-xgboost", "-j", "2", "adapters/xgboost"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            # A missing loader alias can silently select the system library instead.
            if "XGBoost regression (native " + version + ")" not in result.stdout:
                raise RuntimeError("Tests did not load the requested native library: " + version)


if __name__ == "__main__":
    main()
