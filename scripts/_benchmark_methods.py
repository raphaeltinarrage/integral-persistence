"""Benchmark methods, fresh-process trials, and result summaries.

benchmark.py imports these methods and launches this file for each trial.
The worker reads a JSON request from stdin and writes its result to stdout.
It imports no experiment or plotting code.
"""

import contextlib
import gc
import json
import math
import os
import signal
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from statistics import mean
from time import perf_counter

# Make the shared comparison helpers available when launched directly.
scripts_directory = Path(__file__).resolve().parent
if str(scripts_directory) not in sys.path:
    sys.path.insert(0, str(scripts_directory))

import integral_persistence as ip
from integral_persistence import _native
from _benchmark_filtrations import TriangleOrder, build_filtration

# %% Fresh trials and result summaries

METHODS = ("brute_force", "modular", "omnifield", "ours")
PHASES = ("prime_discovery", "method_initialization", "main_computation", "output")


def _validate_modular_memory_mb(value):
    """Validate the requested CRT address-space cap, with ``None`` unlimited."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("modular_memory_mb must be a number or None")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("modular_memory_mb must be positive and finite")
    if value < 1 / (1024 * 1024):
        raise ValueError("modular_memory_mb must specify at least one byte")


def execute_worker_request(request, runners):
    """Run exactly one method after preparing cold caches and a fixed GC policy.

    Input validation, imports, and garbage collection precede
    measurement. Cyclic collection is disabled during the method; ordinary
    reference-counted destruction remains enabled. The supplied runner retains
    its existing definition of each measured phase.
    """
    method = request["method"]
    if method not in METHODS or method not in runners:
        raise ValueError(f"unknown benchmark method: {method!r}")
    modular_memory_mb = request.get("modular_memory_mb", 1024)
    _validate_modular_memory_mb(modular_memory_mb)
    order = TriangleOrder.from_dict(request["order"])
    model = request["model"]
    dependency = {
        "brute_force": "gudhi",
        "modular": "_benchmark_gudhi_crt",
        "omnifield": "dionysus",
        "ours": "integral_persistence",
    }[method]
    with contextlib.redirect_stdout(sys.stderr):
        import_module(dependency)
        gc.collect()
        metadata = {
            "pid": os.getpid(),
            "input_sha256": order.digest,
            "python_executable": sys.executable,
            "modular_memory_mb": modular_memory_mb,
            "cache_policy": "cold",
            "gc_policy": "collect_then_disable",
        }
        enabled = gc.isenabled()
        gc.disable()
        started = perf_counter()
        try:
            result = runners[method](order, model=model)
            metadata["runner_wall_seconds"] = perf_counter() - started
        finally:
            if enabled:
                gc.enable()
    result["worker_metadata"] = metadata
    return result


def run_fresh_trial(method, order, *, script=None, model="LM", timeout=300,
                    on_timeout="raise",
                    modular_memory_mb=1024, on_error="raise"):
    """Launch one isolated trial with the current interpreter and exact input.

    By default, launch this file (``_benchmark_methods.py``). An explicit
    ``script`` may override this; it must accept ``--worker`` and run one trial
    without starting any experiments. Worker failures raise errors by
    default; ``on_error="record"`` preserves memory-limit failures and other
    unsuccessful computations with unknown phase timings and primes.
    With ``on_timeout="record"``, a timeout instead returns a censored record
    with unknown phase timings and primes. A timeout kills the worker's process
    group on POSIX, including any nested CRT executable.
    ``fresh_process_wall_seconds`` includes startup/import/transport costs and
    is independent of ``phases_seconds``; it is not an extra stacked phase.
    ``modular_memory_mb`` requests the native CRT address-space cap in MiB,
    specifying at least one byte; ``None`` disables that cap. The worker records
    this requested policy even when it fails before completing a computation.
    """
    if method not in METHODS:
        raise ValueError(f"unknown benchmark method: {method!r}")
    if on_timeout not in ("raise", "record"):
        raise ValueError("on_timeout must be 'raise' or 'record'")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    _validate_modular_memory_mb(modular_memory_mb)
    if not isinstance(order, TriangleOrder):
        raise TypeError("order must be an immutable TriangleOrder")
    if script is None:
        script = Path(__file__)
    script = Path(script).resolve(strict=True)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    request = {
        "method": method,
        "order": order.to_dict(),
        "model": model,
        "modular_memory_mb": modular_memory_mb,
    }
    payload = json.dumps(request)
    started = perf_counter()
    with subprocess.Popen(
        [sys.executable, str(script), "--worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=(os.name == "posix"),
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1"},
    ) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
            reason = f"{method} worker exceeded {timeout:g} seconds"
            if on_timeout == "raise":
                raise TimeoutError(reason) from error
            timed_out = True
        elapsed = perf_counter() - started
        if timed_out or process.returncode:
            if not timed_out:
                reason = (
                    f"{method} worker exited with code {process.returncode}:\n"
                    f"{stderr or stdout}"
                )
                if on_error == "raise":
                    raise RuntimeError(reason)
            return {
                "status": "timeout" if timed_out else "error",
                "method": method,
                "phases_seconds": {phase: None for phase in PHASES},
                "primes": None,
                "timeout_seconds": float(timeout),
                "timeout_scope": "worker_wall",
                "reason": reason,
                "fresh_process_wall_seconds": elapsed,
                "worker_stderr": stderr,
                "worker_stdout": stdout,
                "worker_returncode": process.returncode,
                "worker_metadata": {
                    "pid": process.pid,
                    "input_sha256": order.digest,
                    "python_executable": sys.executable,
                    "modular_memory_mb": modular_memory_mb,
                    "cache_policy": "cold",
                    "gc_policy": "collect_then_disable",
                    # Parent-supplied input and requested policies are known;
                    # the unfinished worker has not verified them in a result.
                    "input_verified": False,
                },
            }
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{method} worker returned invalid JSON:\n{stdout}\n{stderr}") from error
    if (not isinstance(result, dict)
            or result.get("status") not in ("success", "memory_limit", "timeout", "error")
            or result.get("method") != method):
        raise RuntimeError(f"{method} worker returned an unsuccessful or mismatched result: {result}")
    if result.get("worker_metadata", {}).get("input_sha256") != order.digest:
        raise RuntimeError(f"{method} worker did not preserve the exact triangle order")
    if result["worker_metadata"].get("pid") != process.pid:
        raise RuntimeError(f"{method} worker reported a different process identifier")
    if result["worker_metadata"].get("modular_memory_mb", modular_memory_mb) != modular_memory_mb:
        raise RuntimeError(f"{method} worker reported a different modular memory limit")
    phases = result.get("phases_seconds")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise RuntimeError(f"{method} worker returned invalid phase timings")
    if result["status"] == "success":
        for phase in PHASES:
            value = phases[phase]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"{method} worker returned an invalid {phase} timing")
        result["primes"] = tuple(result["primes"])
    else:
        if result.get("primes") is not None or any(value is not None for value in phases.values()):
            raise RuntimeError(f"{method} unsuccessful worker must leave phases and primes unknown")
        if on_error == "raise":
            raise RuntimeError(
                f"{method} worker returned {result['status']}: {result.get('reason', result)}"
            )
    result["fresh_process_wall_seconds"] = elapsed
    result["timeout_seconds"] = float(timeout)
    result["timeout_scope"] = "worker_wall"
    result["worker_metadata"]["input_verified"] = True
    result["worker_metadata"]["modular_memory_mb"] = modular_memory_mb
    result["worker_stderr"] = stderr
    result["worker_returncode"] = process.returncode
    return result


def summarize_trials(method, samples, *, seed, order, reference_primes):
    """Aggregate one method and filtration without inventing failure timings.

    Every successful trial contributes equally to each phase mean. Unsuccessful
    trials contribute to their separate outcome counts and the process-wall mean,
    but not to the phase means; with no successes, all phase means and the
    method's prime list are unknown (``None``). ``reference_primes`` records the
    screening result or another completed canonical-discovery worker's list;
    None means no reference is available. It is never attributed to a failed
    method. A partial result is consequently conditional on successful trials.
    """
    if method not in METHODS:
        raise ValueError(f"unknown benchmark method: {method!r}")
    if not isinstance(order, TriangleOrder):
        raise TypeError("order must be an immutable TriangleOrder")
    samples = tuple(samples)
    if not samples:
        raise ValueError("cannot summarize an empty set of trials")
    successful = []
    limits = set()
    memory_limits = set()
    counts = dict.fromkeys(("success", "timeout", "memory_limit", "error"), 0)
    for sample in samples:
        if sample.get("method") != method:
            raise ValueError("trial method does not match the requested method")
        if sample.get("status") not in counts:
            raise ValueError("trial status must be 'success', 'timeout', 'memory_limit', or 'error'")
        counts[sample["status"]] += 1
        digest = sample.get(
            "input_sha256", sample.get("worker_metadata", {}).get("input_sha256")
        )
        worker_digest = sample.get("worker_metadata", {}).get("input_sha256", digest)
        if digest != order.digest or worker_digest != order.digest:
            raise ValueError("trial input digest does not match the triangle order")
        wall_seconds = sample.get("fresh_process_wall_seconds")
        if (not isinstance(wall_seconds, (int, float))
                or not math.isfinite(wall_seconds) or wall_seconds < 0):
            raise ValueError("trial process-wall timing must be finite and nonnegative")
        limit = sample.get("timeout_seconds")
        if (not isinstance(limit, (int, float))
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("trial timeout limit must be positive and finite")
        limits.add(float(limit))
        metadata = sample.get("worker_metadata", {})
        if method == "modular" and "modular_memory_mb" in metadata:
            memory_limit = metadata["modular_memory_mb"]
            _validate_modular_memory_mb(memory_limit)
            memory_limits.add(memory_limit)
        phases = sample.get("phases_seconds", {})
        if sample["status"] == "success":
            if sample.get("primes") is None:
                raise ValueError("a successful trial must report its primes")
            for phase in PHASES:
                value = phases.get(phase)
                if (not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError(f"successful trial has an invalid {phase} timing")
            successful.append(sample)
        elif sample.get("primes") is not None or any(
            phases.get(phase) is not None for phase in PHASES
        ):
            raise ValueError("unsuccessful trials must leave phases and primes unknown")
    primes = tuple(successful[0]["primes"]) if successful else None
    if any(tuple(sample["primes"]) != primes for sample in successful):
        raise ValueError("successful trials disagree on the prime list")
    completed = len(successful)
    if completed:
        status = "success" if completed == len(samples) else "partial"
    elif counts["timeout"] == len(samples):
        status = "timeout"
    elif counts["memory_limit"] == len(samples):
        status = "memory_limit"
    else:
        status = "failed"
    summary = {
        "status": status,
        "method": method,
        "phases_seconds": {
            phase: mean(sample["phases_seconds"][phase] for sample in successful)
            if successful else None
            for phase in PHASES
        },
        "primes": primes,
        "reference_primes": None if reference_primes is None else tuple(reference_primes),
        "seed": seed,
        "input_sha256": order.digest,
        "trials": len(samples),
        "successful_trials": completed,
        "timed_out_trials": counts["timeout"],
        "memory_limited_trials": counts["memory_limit"],
        "failed_trials": counts["error"],
        "fresh_process_wall_seconds": mean(
            sample["fresh_process_wall_seconds"] for sample in samples
        ),
        "timeout_scope": "worker_wall",
    }
    if len(limits) == 1:
        summary["timeout_seconds"] = limits.pop()
    else:
        summary["timeout_limits_seconds"] = sorted(limits)
    if len(memory_limits) == 1:
        summary["memory_limit_mb"] = memory_limits.pop()
        summary["memory_limit_scope"] = "native_address_space"
    elif memory_limits:
        summary["memory_limits_mb"] = sorted(
            memory_limits, key=lambda value: (value is not None, value or 0),
        )
        summary["memory_limit_scope"] = "native_address_space"
    return summary

# %% run_brute_force

def run_brute_force(order, model="LM"):
    result = dict(status="success", method="brute_force",
                  phases_seconds=dict.fromkeys(PHASES, 0.0), primes=[])
    gudhi = import_module("gudhi")
    simplices, times = build_filtration(order, model=model)

    started = perf_counter()
    data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
    result["input_preparation_seconds"] = perf_counter() - started
    discovery = ip.rational_reduction(data.matrix)
    primes = tuple(discovery.primes)
    result["primes"] = primes
    phases = discovery.timings
    result["phases_seconds"]["prime_discovery"] = phases["rational_reduction"] + phases["factorization"]
    del discovery, data
    if not primes:
        return result

    started = perf_counter()
    st = gudhi.SimplexTree()
    for position, simplex in enumerate(simplices):
        st.insert(simplex, filtration=position)
    result["phases_seconds"]["method_initialization"] = perf_counter() - started
    del simplices, times

    # Process triangle boundaries for H1, but omit H2. Keep H1 on graph prefixes.
    include_maximal = st.dimension() <= 1
    started = perf_counter()
    for prime in primes:
        st.compute_persistence(homology_coeff_field=prime, min_persistence=0,
                               persistence_dim_max=include_maximal)
    result["phases_seconds"]["main_computation"] = perf_counter() - started
    return result

# %% run_modular

def run_modular(order, model="LM", timeout=60, *, memory_mb=1024):
    """Time CRT over the discovered primes; memory_mb=None disables its cap."""
    from _benchmark_gudhi_crt import run_native
    result = dict(status="success", method="modular",
                  phases_seconds=dict.fromkeys(PHASES, 0.0), primes=[],
                  memory_limit_mb=memory_mb, memory_limit_scope="native_address_space")
    simplices, times = build_filtration(order, model=model)

    started = perf_counter()
    data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
    result["input_preparation_seconds"] = perf_counter() - started
    discovery = ip.rational_reduction(data.matrix)
    primes = tuple(discovery.primes)
    result["primes"] = primes
    phases = discovery.timings
    result["phases_seconds"]["prime_discovery"] = phases["rational_reduction"] + phases["factorization"]
    del discovery, data
    if not primes:
        return result

    computed = run_native(
        simplices, primes, degree=1,
        persistence_dim_max=not any(len(simplex) == 3 for simplex in simplices),
        timeout=timeout, memory_mb=memory_mb,
    )
    if computed["status"] != "success":
        result.update(
            status=computed["status"] if computed["status"] in ("memory_limit", "timeout") else "error",
            reason=computed.get("reason") or computed.get("stderr") or computed["status"],
            discovery_primes=primes,
            completed_phases_seconds={"prime_discovery": result["phases_seconds"]["prime_discovery"]},
            primes=None, phases_seconds=dict.fromkeys(PHASES), native_outcome=computed,
        )
        return result

    timings = computed["phases"]
    phases = result["phases_seconds"]
    phases["method_initialization"] = timings["input_preparation"] + timings["initialization"]
    phases["main_computation"] = timings["computation"]
    phases["output"] = timings["output"]
    result["output_scope"] = "native_h1_pair_extraction"
    result["excluded_output_seconds"] = {
        **computed["excluded_output_seconds"],
        "python_json_decoding": computed["python_output_seconds"],
    }
    result["native_prime_validation_seconds"] = computed["prime_validation_seconds"]
    result["native_process_overhead_seconds"] = computed["process_overhead_seconds"]
    return result

# %% run_omnifield

def run_omnifield(order, model="LM"):
    METHOD = "omnifield"
    result = {
        "status": "success",
        "method": METHOD,
        "phases_seconds": {name: 0.0 for name in PHASES},
        "primes": [],
    }

    # Import
    dio = import_module("dionysus")

    # prime_discovery
    started_prime = perf_counter()
    result["phases_seconds"]["prime_discovery"] = perf_counter() - started_prime

    # method_initialization
    started_initialization = perf_counter()
    if model == "LM":
        simplices = [(i,) for i in range(order.n)] + [(i, j) for i in range(order.n) for j in range(i + 1, order.n)] + list(order.triangles)
    else:
        raise ValueError(f"Unsupported model: {model}")
    filtration = dio.Filtration([dio.Simplex(simplex, i) for i, simplex in enumerate(simplices)])
    result["phases_seconds"]["method_initialization"] = perf_counter() - started_initialization

    # main_computation
    started_main = perf_counter()
    persistence = dio.omnifield_homology_persistence(filtration)
    result["phases_seconds"]["main_computation"] = perf_counter() - started_main

    # output
    started_output = perf_counter()
    primes = persistence.primes()
    result["primes"] = primes
    for prime in [1]+primes:
        _ = dio.init_diagrams(persistence, filtration, prime)
    result["phases_seconds"]["output"] = perf_counter() - started_output
    del filtration, persistence

    return result

# %% run_ours

def run_ours(order, model="LM"):
    """Time native reduction and the native H1 pair-extraction kernel."""
    result = dict(status="success", method="ours",
                  phases_seconds=dict.fromkeys(PHASES, 0.0), primes=[])
    simplices, times = build_filtration(order, model=model)
    started = perf_counter()
    data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
    result["input_preparation_seconds"] = perf_counter() - started
    state = ip.reduction(data.matrix)

    result["primes"] = tuple(state.primes)
    phases = state.timings
    result["phases_seconds"]["prime_discovery"] = phases["rational_reduction"] + phases["factorization"]
    result["phases_seconds"]["method_initialization"] = phases["exhaustive_rational_reduction"]
    result["phases_seconds"]["main_computation"] = phases["exceptional_correction"]
    started_output = perf_counter()
    exceptional = state.exceptional_pivots
    sequences = [state.generic_pivots] + [exceptional[p] for p in result["primes"]]
    pairs, extraction_seconds = _native.ordinary_pairs_timed(sequences, data.dimensions, 1)
    result["phases_seconds"]["output"] = extraction_seconds
    result["output_scope"] = "native_h1_pair_extraction"
    result["excluded_output_seconds"] = {
        "pivot_packaging_and_cleanup": phases["profile_packaging_and_cleanup"],
        "preparation_validation_python_conversion": max(0.0, perf_counter() - started_output - extraction_seconds),
    }
    del pairs, sequences, exceptional, data, state
    return result


# Run one trial when launched by benchmark.py through run_trial_fresh().
if __name__ == "__main__":
    request = json.load(sys.stdin)
    result = execute_worker_request(request, {
        "brute_force": run_brute_force,
        "modular": lambda order, model: run_modular(
            order, model=model, timeout=None,
            memory_mb=request.get("modular_memory_mb", 1024),
        ),
        "omnifield": run_omnifield,
        "ours": run_ours,
    })
    print(json.dumps(result))
