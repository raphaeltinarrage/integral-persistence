"""Compile the internal invariant test without exposing mutable matrices in Python."""

from pathlib import Path
import os
import shlex
import subprocess


def test_native_kernel_against_independent_dense_oracles(tmp_path):
    package = Path(__file__).resolve().parents[1]
    executable = tmp_path / "native_kernel"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(
        [*compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
         f"-I{package / 'cpp' / 'include'}", str(package / "tests" / "native_kernel.cpp"),
         "-lgmpxx", "-lgmp", "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=90,
    )
    completed = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=60,
    )
    assert "independent dense oracles passed" in completed.stdout
