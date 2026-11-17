"""Explicitly build the patched GUDHI CRT competitor; never build on import."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, test=False):
    """Verify pinned sources, patch fresh copies, and build the CRT driver.

    ``test=True`` also compiles and runs the focused arithmetic regression.
    Build products and local compiler provenance remain under ``build/``.
    """
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(ROOT / "vendor" / name) != expected:
            raise RuntimeError(f"source checksum mismatch: {name}")
    destination = ROOT / "build"
    destination.mkdir(exist_ok=True)
    include = destination / "include"
    if include.exists():
        shutil.rmtree(include)
    shutil.copytree(ROOT / "vendor", include)
    patches = sorted((ROOT / "patches").glob("*.patch"))
    for patch in patches:
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
            cwd=include, check=True,
        )
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra"]
    sources = [(ROOT / "src/backend.cpp", "backend")]
    if test:
        sources.append((ROOT / "tests/arithmetic.cpp", "arithmetic_test"))
    for source, executable in sources:
        subprocess.run(
            [*compiler, *flags, "-I", str(include), str(source), "-o",
             str(destination / executable), "-lflint", "-lgmpxx", "-lgmp"],
            check=True,
        )
    record = {
        "method": "patched GUDHI CRT",
        "compiler": subprocess.check_output([*compiler, "--version"], text=True).splitlines()[0],
        "flags": flags,
        "source_commit": manifest["commit"],
        "patches": {p.name: digest(p) for p in patches},
        "sources": {str(p.relative_to(ROOT)): digest(p) for p, _ in sources},
        "executables": {name: digest(destination / name) for _, name in sources},
    }
    (destination / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    if test:
        subprocess.run([str(destination / "arithmetic_test")], check=True)
        subprocess.run([sys.executable, str(ROOT / "tests/test_backend.py")], check=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="also run the arithmetic regression")
    print(json.dumps(build(test=parser.parse_args().test), indent=2))
