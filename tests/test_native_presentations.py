"""Check native presentation invariants using independent dense arithmetic."""

import os
from pathlib import Path
import shlex
import subprocess


def test_presentation_prefix_lattices_and_chain_bases(tmp_path):
    package = Path(__file__).resolve().parents[1]
    executable = tmp_path / "presentation_invariants"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(
        [*compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{package / 'cpp' / 'include'}",
         str(package / "tests" / "presentation_invariants.cpp"),
         "-lgmpxx", "-lgmp", "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=90,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=60,
    )
    assert "presentation prefix lattice and chain-basis invariants passed" in result.stdout
