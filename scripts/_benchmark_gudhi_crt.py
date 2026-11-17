"""Subprocess interface to the patched GUDHI CRT competitor."""

import json
import math
from numbers import Real
import operator
import os
from pathlib import Path
import re
import resource
import subprocess
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1] / "vendor" / "gudhi_crt"
EXECUTABLE = ROOT / "build" / "backend"
PHASES = ("input_preparation", "initialization", "computation", "output")
EXCLUDED_OUTPUT = ("native_cleanup", "native_sorting", "native_json_encoding")


def available():
    """Return whether the optional CRT executable has been explicitly built."""
    return EXECUTABLE.is_file()


def capabilities():
    """Describe the pinned competitor and its native computation scope."""
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    return {
        "method": "patched GUDHI CRT", "version": manifest["version"],
        "source_commit": manifest["commit"], "available": available(),
        "arbitrary_precision_primes_elements_masks": True,
        "prime_validation": "FLINT fmpz_is_prime",
        "computed_degrees": "all; persistence_dim_max=False omits maximal homology",
        "source_license": "MIT", "threads": 1,
    }


def _integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _memory_limit_bytes(memory_mb):
    if memory_mb is None:
        return None
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, Real):
        raise TypeError("memory_mb must be None or a finite positive real number")
    try:
        if not math.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("memory_mb must be finite and positive")
        size = int(memory_mb * 1024 * 1024)
    except OverflowError as error:
        raise ValueError("memory_mb is too large for an address-space cap") from error
    if size < 1:
        raise ValueError("memory_mb must specify at least one byte")
    return size


def _limits(memory_bytes):
    def apply():
        if memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_native(simplices, primes, degree=1, timeout=60, memory_mb=1024,
               *, persistence_dim_max=True):
    """Compute ordered simplex-index pairs over exactly the requested primes.

    Supply a face-before-coface sequence of simplices with increasing vertices,
    and distinct, increasing integer primes. Native validation proves primes
    and checks the filtration. Invalid inputs raise ``ValueError``. Operational
    failures return a status such as ``timeout`` or ``memory_limit`` so the
    comparison runner can record failed trials without inventing results.

    Returned ``barcodes`` maps decimal prime strings to pairs ``[birth, death]``;
    death ``None`` is essential. Native ``phases['output']`` measures requested-
    degree extraction, including allocation and exact mask expansion. Cleanup,
    sorting, encoding, and FLINT validation have separate timers. The driver
    still computes all degrees unless ``persistence_dim_max=False``; on triangle
    inputs that flag omits H2 while retaining triangles that kill H1. Leave it
    true for graph inputs if H1 is requested.

    ``memory_mb`` caps the child's address space in MiB, not RSS; ``None`` adds
    no cap. The child remains in the parent's process group so the comparison
    worker watchdog can stop the whole trial. Empty prime lists bypass CRT.
    """
    degree = _integer(degree, "degree")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if not isinstance(persistence_dim_max, bool):
        raise TypeError("persistence_dim_max must be a bool")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, Real)
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be None or finite and positive")
    memory_bytes = _memory_limit_bytes(memory_mb)
    started = perf_counter()
    fields = tuple(_integer(prime, "prime") for prime in primes)
    metadata = {
        "method": "patched GUDHI CRT",
        "memory_limit_mb": memory_mb,
        "memory_limit_scope": "native_address_space",
        "persistence_dim_max": persistence_dim_max,
    }
    if not fields:
        return {
            "status": "success", "barcodes": {}, "phases": dict.fromkeys(PHASES, 0.0),
            "process_seconds": 0.0, "process_overhead_seconds": 0.0,
            "empty_fields_bypassed": True, "stderr": "", "returncode": 0,
            "prime_validation_seconds": 0.0, "python_serialization_seconds": 0.0,
            "python_output_seconds": 0.0,
            "output_scope": "native_degree_pair_extraction",
            "excluded_output_seconds": dict.fromkeys(EXCLUDED_OUTPUT, 0.0), **metadata,
        }
    if not available():
        return {"status": "unavailable_dependency",
                "reason": "run python vendor/gudhi_crt/build.py", **metadata}
    ordered = tuple(tuple(_integer(vertex, "vertex") for vertex in simplex) for simplex in simplices)
    protocol = f"{len(ordered)} {len(fields)} {degree}\n" + " ".join(map(str, fields)) + "\n"
    protocol += "".join(f"{len(simplex)} {' '.join(map(str, simplex))}\n" for simplex in ordered)
    serialization = perf_counter() - started
    command = [str(EXECUTABLE)]
    if not persistence_dim_max:
        command.append("--no-top-dimensional-persistence")
    start = perf_counter()
    try:
        process = subprocess.run(
            command, input=protocol, text=True, capture_output=True, timeout=timeout,
            preexec_fn=_limits(memory_bytes),
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        return {"status": "timeout", "process_seconds": perf_counter() - start,
                "stderr": stderr, "returncode": None, **metadata}
    duration = perf_counter() - start
    if process.returncode == 2:
        raise ValueError(process.stderr.strip())
    read_started = perf_counter()
    if process.returncode:
        allocation_failed = process.returncode == 3 or re.search(
            r"^GNU MP: Cannot allocate memory \(size=[0-9]+\)\r?$",
            process.stderr, re.MULTILINE,
        ) is not None
        result = {"status": "memory_limit" if allocation_failed else "crash"}
    else:
        result = json.loads(process.stdout)
        expected_degrees = "all_including_maximal" if persistence_dim_max else "below_maximal"
        if (result.get("computed_degrees") != expected_degrees
                or result.get("output_scope") != "native_degree_pair_extraction"):
            raise RuntimeError("incompatible CRT executable; rebuild with python vendor/gudhi_crt/build.py")
    readout = perf_counter() - read_started
    result.update(
        process_seconds=duration, stderr=process.stderr, returncode=process.returncode,
        python_serialization_seconds=serialization, python_output_seconds=readout, **metadata,
    )
    result["process_overhead_seconds"] = (
        duration - sum(result.get("phases", {}).values())
        - sum(result.get("excluded_output_seconds", {}).values())
        - result.get("prime_validation_seconds", 0.0)
    )
    return result
