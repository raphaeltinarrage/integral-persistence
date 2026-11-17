from collections import Counter
import gc
import math
from time import perf_counter

import gudhi
import numpy as np
import integral_persistence as ip

def build_rips_filtration(points, edge_fraction=1/10):
    """Keep the prescribed fraction of shortest edges and expand their triangles.

    Exactly floor(edge_fraction * binom(n, 2)) edges are retained. Ties are
    broken lexicographically by vertex pair, including a tie at the stopping
    edge. Triangle grades are the maximum of their three edge lengths.
    Preserve these radii and the GUDHI simplex order for both barcode readings.
    If the budget splits a distance tie, the final graph is only a prefix of
    that full Rips layer.
    """
    if not 0 <= edge_fraction <= 1:
        raise ValueError("Require 0 <= edge_fraction <= 1")
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or min(points.shape) < 1
            or not np.isfinite(points).all()):
        raise ValueError("Require a nonempty, finite two-dimensional point array")
    n = len(points)
    possible_edges = math.comb(n, 2)
    target_edges = math.floor(edge_fraction * possible_edges)
    first, second = np.triu_indices(n, k=1)  # lexicographic vertex-pair order
    distances = np.empty(possible_edges)
    offset = 0
    # Only one O(n * dimension) coordinate-difference array is needed at once.
    for vertex in range(n - 1):
        stop = offset + n - vertex - 1
        distances[offset:stop] = np.linalg.norm(points[vertex + 1:] - points[vertex], axis=1)
        offset = stop
    edge_order = np.argsort(distances, kind="stable")
    tree = gudhi.SimplexTree()
    for vertex in range(n):
        tree.insert([vertex], filtration=0.0)
    for index in edge_order[:target_edges]:
        tree.insert([int(first[index]), int(second[index])], filtration=float(distances[index]))
    del first, second, distances, edge_order, points
    edges = tree.num_simplices() - tree.num_vertices()
    tree.expansion(2)
    counts = {
        "edges": edges,
        "reduced_edges": edges,
        "target_edges": target_edges,
        "possible_edges": possible_edges,
        "edge_fraction": float(edge_fraction),
        "edge_selection": "shortest_edges_fixed_count",
        "triangles": tree.num_simplices() - tree.num_vertices() - edges,
        "simplices": tree.num_simplices(),
    }
    return tree, counts

def betti_jumps(barcode):
    """Record the pointwise dimension function, preserving multiplicity/ties."""
    events = Counter()
    for birth, death in barcode:
        events[birth] += 1
        if death != math.inf:
            events[death] -= 1
    return {time: count for time, count in events.items() if count}

def detect_torsion(data):
    """Reduce once; decode original and simplexwise times from the same state."""
    state = ip.reduction(data.matrix)
    actual = ip.decode_barcodes(state, data)
    refined = ip.decode_barcodes(state, data, times=list(range(data.source_count)))
    generic_jumps = betti_jumps(actual.barcode())
    return dict(
        status="success", field_dependent=not actual.field_independent,
        rips_exceptional_primes=list(actual.exceptional_primes),
        rips_betti_changing_primes=[p for p in actual.matrix_primes
                                   if betti_jumps(actual.barcode(p)) != generic_jumps],
        simplexwise_exceptional_primes=list(refined.exceptional_primes),
        stage_seconds=dict(state.timings),
    )

def _uncomputed_module(status, reason=None):
    record = dict(status=status, field_dependent=None, rips_exceptional_primes=None,
                  rips_betti_changing_primes=None, simplexwise_exceptional_primes=None)
    if reason:
        record["reason"] = reason
    return record

def run_trial(n_points, dimension, trial, edge_fraction=1/10, seed=0):
    """Check H1 field/tensor dependence, then free H1 only for affected clouds.

    Random streams use exactly SeedSequence([seed, 1, dimension, trial]). Free
    failures preserve completed field/tensor results. Uncomputed lists remain
    None; factorization/proof failures are errors, never provisional results.
    """
    result = _uncomputed_module("pending")
    result["tensor"] = _uncomputed_module("not_run")
    result["free"] = _uncomputed_module("skipped", "tensor_not_completed")
    tree = data = None
    stage = "filtration"
    try:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, dimension, trial]))
        started = perf_counter()
        tree, counts = build_rips_filtration(rng.random((n_points, dimension)), edge_fraction)
        result.update(counts)
        result["construction_seconds"] = perf_counter() - started
        stage = "tensor_detection"
        started = perf_counter()
        # H0 has no integral torsion, so the field H1 barcode is tensor H1.
        data = ip.build_matrices(tree, degrees=[1], modules=["field"])[1]["field"]
        tensor = detect_torsion(data)
        tensor["detection_seconds"] = perf_counter() - started
        result["tensor"] = tensor
        result.update(tensor)
        data = None
        if not tensor["rips_exceptional_primes"]:
            result["free"] = _uncomputed_module("skipped", "no_tensor_rips_exceptional_primes")
            return result
        stage = "free_presentation"
        started = perf_counter()
        data = ip.build_matrices(tree, degrees=[1], modules=["free"])[1]["free"]
        presentation_seconds = perf_counter() - started
        tree = None
        stage = "free_detection"
        started = perf_counter()
        result["free"] = detect_torsion(data)
        result["free"].update(presentation_seconds=presentation_seconds,
                              detection_seconds=perf_counter() - started)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        failed = result["tensor"] if stage in ("filtration", "tensor_detection") else result["free"]
        failed.update(status="error", error_stage=stage, error=message)
        result.update(status="error", error_stage=stage, error=message)
    finally:
        tree = data = None
        gc.collect()
    return result
