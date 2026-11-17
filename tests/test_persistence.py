"""Independent barcode and persistence-map checks for all three modules."""

from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import inf

import pytest

import integral_persistence as ip


RP2 = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]


def filtration(facets, grades):
    """Independent closure and valid deterministic order for small fixtures."""
    faces = {}
    for facet, time in zip(facets, grades, strict=True):
        for size in range(1, len(facet) + 1):
            for face in combinations(sorted(facet), size):
                faces[face] = min(faces.get(face, time), time)
    simplices = sorted(faces, key=lambda face: (faces[face], len(face), face))
    return simplices, [faces[face] for face in simplices]


def scalar(value, prime):
    return Fraction(value) if prime is None else value % prime


def divide(a, b, prime):
    return a / b if prime is None else a * pow(b, -1, prime) % prime


def rref(rows, ncols, prime):
    rows = [[scalar(value, prime) for value in row] for row in rows]
    pivots = []
    for column in range(ncols):
        found = next((i for i in range(len(pivots), len(rows)) if rows[i][column]), None)
        if found is None:
            continue
        pivot = len(pivots)
        rows[pivot], rows[found] = rows[found], rows[pivot]
        coefficient = rows[pivot][column]
        rows[pivot] = [divide(value, coefficient, prime) for value in rows[pivot]]
        for i in range(len(rows)):
            if i == pivot:
                continue
            coefficient = rows[i][column]
            rows[i] = [scalar(a - coefficient * b, prime)
                       for a, b in zip(rows[i], rows[pivot], strict=True)]
        pivots.append(column)
    return rows, pivots


def column_rank(columns, rows, prime):
    matrix = [[column[i] for column in columns] for i in range(rows)]
    return len(rref(matrix, len(columns), prime)[1])


def boundary_columns(chains, cochains):
    indices = {simplex: i for i, simplex in enumerate(chains)}
    result = []
    for simplex in cochains:
        column = [0] * len(chains)
        if len(simplex) > 1:
            for omitted in range(len(simplex)):
                face = simplex[:omitted] + simplex[omitted + 1:]
                column[indices[face]] = (-1) ** omitted
        result.append(column)
    return result


def independent_map_rank(simplices, times, degree, source, target, prime):
    """rank H_d(K_source) -> H_d(K_target), via cycles plus target boundaries."""
    source_chains = [simplex for simplex, time in zip(simplices, times, strict=True)
                     if len(simplex) == degree + 1 and time <= source]
    source_lower = [simplex for simplex, time in zip(simplices, times, strict=True)
                    if len(simplex) == degree and time <= source]
    target_chains = [simplex for simplex, time in zip(simplices, times, strict=True)
                     if len(simplex) == degree + 1 and time <= target]
    target_cochains = [simplex for simplex, time in zip(simplices, times, strict=True)
                       if len(simplex) == degree + 2 and time <= target]
    columns = boundary_columns(source_lower, source_chains)
    rows = [[column[i] for column in columns] for i in range(len(source_lower))]
    reduced, pivots = rref(rows, len(source_chains), prime)
    target_indices = {simplex: i for i, simplex in enumerate(target_chains)}
    cycles = []
    for free in set(range(len(source_chains))) - set(pivots):
        cycle = [scalar(0, prime)] * len(target_chains)
        cycle[target_indices[source_chains[free]]] = scalar(1, prime)
        for i, pivot in enumerate(pivots):
            cycle[target_indices[source_chains[pivot]]] = scalar(-reduced[i][free], prime)
        cycles.append(cycle)
    boundaries = boundary_columns(target_chains, target_cochains)
    return (column_rank(cycles + boundaries, len(target_chains), prime)
            - column_rank(boundaries, len(target_chains), prime))


def independent_barcodes(simplices, times, degree, prime):
    """Ordinary dense boundary reduction over Q or F_p, without augmentation."""
    indices = {simplex: i for i, simplex in enumerate(simplices)}
    columns, owners, births, deaths = [], {}, [], {}
    for j, simplex in enumerate(simplices):
        column = [scalar(0, prime)] * len(simplices)
        if len(simplex) > 1:
            for omitted in range(len(simplex)):
                face = simplex[:omitted] + simplex[omitted + 1:]
                column[indices[face]] = scalar((-1) ** omitted, prime)
        while any(column):
            pivot = max(i for i, value in enumerate(column) if value)
            if pivot not in owners:
                owners[pivot] = j
                deaths[pivot] = j
                break
            previous = columns[owners[pivot]]
            coefficient = divide(column[pivot], previous[pivot], prime)
            column = [scalar(a - coefficient * b, prime)
                      for a, b in zip(column, previous, strict=True)]
        else:
            births.append(j)
        columns.append(column)
    return Counter((times[birth], times[deaths[birth]] if birth in deaths else inf)
                   for birth in births if len(simplices[birth]) == degree + 1
                   and (birth not in deaths or times[birth] != times[deaths[birth]]))


def barcode_map_rank(bars, source, target):
    return sum(birth <= source and target < death for birth, death in bars)


@pytest.mark.parametrize("prime", [None, 2, 3, 5])
def test_arbitrary_degrees_and_full_persistence_maps(prime):
    # Two tetrahedra with overlapping faces and a disconnected circle. Every
    # integral prefix group is free, so field, tensor and free modules agree.
    facets = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
              (0, 1, 2, 3), (1, 2, 3, 4), (6, 7), (7, 8), (6, 8)]
    simplices, times = filtration(facets, [0, 0, 0, 0, 1, 2, 0, 0, 1])
    result = ip.persistence(simplices, times, degrees=[0, 1, 2, 3, 8])
    for degree, outputs in result.items():
        expected = independent_barcodes(simplices, times, degree, prime)
        for module in ("field", "tensor", "free"):
            bars = getattr(outputs, module).barcode(prime=prime)
            assert Counter(bars) == expected
            for source in sorted(set(times)):
                for target in sorted(set(times)):
                    if source <= target:
                        assert barcode_map_rank(bars, source, target) == independent_map_rank(
                            simplices, times, degree, source, target, prime)


def test_tensor_and_tor_torsion_in_projective_plane_and_cone():
    for cone in (False, True):
        facets = RP2 + ([face + (6,) for face in RP2] if cone else [])
        simplices, times = filtration(facets, [0] * len(RP2) + ([1] * len(RP2) if cone else []))
        result = ip.persistence(simplices, times, degrees=[0, 1, 2, 3])
        end = 1 if cone else inf
        for degree in (1, 2):
            assert result[degree].field.barcode() == []
            assert result[degree].field.barcode(prime=2) == [(0, end)]
            assert result[degree].field.exceptional_primes == [2]
            assert result[degree].free.barcode(prime=2) == []
        assert result[1].tensor.barcode(prime=2) == [(0, end)]
        assert result[2].tensor.barcode(prime=2) == []
        assert result[0].field.barcode(prime=2) == [(0, inf)]
        for degree in result:
            for prime in (None, 2, 3):
                bars = result[degree].field.barcode(prime=prime)
                assert Counter(bars) == independent_barcodes(simplices, times, degree, prime)
                for source in sorted(set(times)):
                    for target in sorted(set(times)):
                        if source <= target:
                            assert barcode_map_rank(bars, source, target) == independent_map_rank(
                                simplices, times, degree, source, target, prime)


def test_map_torsion_without_pointwise_torsion():
    triangles = [(0, 1, 2), (1, 2, 3), (2, 3, 4),
                 (3, 4, 5), (1, 4, 5), (0, 1, 5)]
    counts = Counter(edge for triangle in triangles for edge in combinations(triangle, 2))
    boundary = [edge for edge, count in counts.items() if count == 1]
    simplices, times = filtration(boundary + triangles, [0] * len(boundary) + [1] * len(triangles))
    output = ip.persistence(simplices, times, degrees=[1])[1]
    for module in ("field", "tensor", "free"):
        result = getattr(output, module)
        assert result.barcode() == [(0, inf)]
        assert result.barcode(prime=3) == [(0, inf)]
        assert result.barcode(prime=2) == [(0, 1), (1, inf)]
        assert result.exceptional_primes == [2]
        assert barcode_map_rank(result.barcode(prime=2), 0, 1) == 0
        assert barcode_map_rank(result.barcode(), 0, 1) == 1
    assert independent_map_rank(simplices, times, 1, 0, 1, 2) == 0
    assert independent_map_rank(simplices, times, 1, 0, 1, None) == 1


def test_reduce_once_decode_tied_and_simplexwise_times():
    facets = RP2 + [face + (6,) for face in RP2]
    # Keep the projective-plane prefix before the cone. The later decoding
    # deliberately collapses both stages to one time without changing order.
    simplices, _ = filtration(facets, [0] * len(RP2) + [1] * len(RP2))
    matrices = ip.build_matrices(simplices, degrees=[1, 2], modules=["field", "tensor", "free"])
    assert matrices[1]["field"].matrix is matrices[2]["field"].matrix
    states = {}
    for degree, modules in matrices.items():
        for module, data in modules.items():
            assert data.times is None
            identity = data.matrix.identity
            if identity not in states:
                states[identity] = ip.reduction(data.matrix)
            state = states[identity]
            before = (state.phase, state.timings.copy(), state.primes, state.generic_pivots)
            tied = ip.decode_barcodes(state, data, times=[0] * len(simplices))
            refined = ip.decode_barcodes(state, data, times=list(range(len(simplices))))
            assert tied.barcode() == []
            assert tied.barcode(prime=2) == []
            assert tied.field_independent
            assert tied.exceptional_primes == []
            assert (state.phase, state.timings, state.primes, state.generic_pivots) == before
            assert data.matrix.consumed
            if module == "field":
                assert 2 in tied.matrix_primes
                assert 2 in refined.exceptional_primes
                for prime in (None, 2, 3):
                    assert Counter(refined.barcode(prime=prime)) == independent_barcodes(
                        simplices, list(range(len(simplices))), degree, prime)


def test_requested_degrees_shared_state_and_matrix_consumption():
    simplices, times = filtration([(0, 1, 2)], [1])
    matrices = ip.build_matrices(simplices, times, degrees=[0, 1, 6], modules=["field"])
    matrix = matrices[0]["field"].matrix
    assert matrix is matrices[1]["field"].matrix is matrices[6]["field"].matrix
    copy = matrix.copy()
    state_copy = ip.reduction(copy)
    assert copy.consumed and not matrix.consumed
    assert ip.decode_barcodes(state_copy, matrices[0]["field"]).barcode() == [(1, inf)]
    state = ip.reduction(matrix)
    assert matrices[6]["field"].matrix.consumed
    with pytest.raises(RuntimeError, match="consumed"):
        ip.reduction(matrix)
    with pytest.raises(RuntimeError, match="consumed"):
        matrix.copy()
    unrelated = ip.build_matrices(simplices, times, degrees=[0], modules=["field"])
    with pytest.raises(ValueError, match="different matrix"):
        ip.decode_barcodes(state, unrelated[0]["field"])
    outputs = ip.persistence(simplices, times, degrees=[6, 0, 1], modules=["field"])
    assert set(outputs) == {0, 1, 6}
    assert outputs[0].states["field"] is outputs[1].states["field"] is outputs[6].states["field"]
    assert not hasattr(outputs[0], "tensor")
    assert outputs[6].field.barcode() == []
    assert ip.persistence(simplices, times, degrees=[], modules=["field"]) == {}


@pytest.mark.parametrize("times", [
    [2**53, 2**53 + 1, 2**53 + 2],
    [2**53 + 1, float(2**53 + 2), 2**53 + 3],
    [-(2**180), -(2**179), 2**200],
])
def test_exact_integer_float_endpoints_and_snapshot(times):
    simplices = [[0], [1], [0, 1]]
    original = list(times)
    data = ip.build_matrices(simplices, times, degrees=[0], modules=["field"])[0]["field"]
    times[:] = [0, 0, 0]
    simplices[:] = []
    state = ip.reduction(data.matrix)
    bars = ip.decode_barcodes(state, data).barcode()
    assert bars == [(original[0], inf), (original[1], original[2])]
    assert type(bars[0][0]) is type(original[0])
    assert type(bars[1][0]) is type(original[1])
    assert type(bars[1][1]) is type(original[2])
    returned = data.times
    returned[0] = 0
    assert data.times == original


def test_ties_and_barcode_multiplicity():
    simplices = [(0,), (1,), (2,), (0, 1), (1, 2)]
    result = ip.persistence(simplices, [0, 0.0, 0, 1.0, 1], degrees=[0])[0]
    for module in ("field", "tensor", "free"):
        bars = getattr(result, module).barcode()
        assert Counter(bars) == Counter({(0, 1): 2, (0, inf): 1})
        assert isinstance(bars, list)
        assert all(isinstance(pair, tuple) for pair in bars)
    tied = ip.persistence([(0,), (1,), (0, 1)], [1, 1.0, 1], degrees=[0])[0]
    assert tied.field.barcode() == [(1, inf)]


def test_arbitrary_integer_vertex_labels_do_not_change_modules():
    simplices, times = filtration(RP2, [0] * len(RP2))
    labels = [2**180, -100, 2, 0, -(2**200), 500]
    relabelled = [tuple(labels[v] for v in reversed(simplex)) for simplex in simplices]
    expected = ip.persistence(simplices, times, degrees=[0, 1, 2])
    actual = ip.persistence(relabelled, times, degrees=[0, 1, 2])
    for degree in expected:
        for module in ("field", "tensor", "free"):
            for prime in (None, 2, 3):
                assert getattr(actual[degree], module).barcode(prime=prime) == getattr(
                    expected[degree], module).barcode(prime=prime)


def test_empty_filtration_and_invalid_regrading():
    outputs = ip.persistence([], [], degrees=[0, 1, 9])
    for result in outputs.values():
        for module in ("field", "tensor", "free"):
            assert getattr(result, module).barcode() == []
            assert getattr(result, module).field_independent
    data = ip.build_matrices([(0,), (1,), (0, 1)], degrees=[0], modules=["field"])[0]["field"]
    state = ip.reduction(data.matrix)
    for times in ([0, 1], [0, 2, 1]):
        with pytest.raises(ValueError):
            ip.decode_barcodes(state, data, times=times)
    assert state.phase == "complete"
    assert ip.decode_barcodes(state, data).barcode() == [(0, inf), (1, 2)]


@pytest.mark.parametrize("times", [
    [0, Fraction(1, 3), 1], [0, float("nan"), 1], [0, 1, inf],
    [0, True, 1], [2**53 + 1, float(2**53), 2**53 + 2],
])
def test_reject_unsupported_or_decreasing_times(times):
    with pytest.raises((TypeError, ValueError)):
        ip.build_matrices([(0,), (1,), (0, 1)], times)


@pytest.mark.parametrize("simplices", [
    [(0, 1)], [(0,), (0,)], [(0,), (1,), (0, 0, 1)], [()],
    [(0,), (1,), (0, 1, 2)], [(0,), (1,), (0.0, 1)],
])
def test_validate_face_closed_filtration(simplices):
    with pytest.raises((TypeError, ValueError)):
        ip.build_matrices(simplices)


def test_gudhi_simplex_tree_matches_explicit_input_and_preserves_ties():
    gudhi = pytest.importorskip("gudhi")
    tree = gudhi.SimplexTree()
    tree.insert([0, 1], filtration=0.1)
    tree.insert([0, 2], filtration=0.2)
    last = 0.1 + 0.2
    tree.insert([1, 2], filtration=last)
    tree.expansion(2)
    assert tree.filtration([1, 2]) == tree.filtration([0, 1, 2]) == last
    simplices, times = zip(*tree.get_filtration(), strict=True)
    explicit = ip.persistence(simplices, times, degrees=[0, 1, 2])
    streamed = ip.persistence(tree, degrees=[0, 1, 2])
    for degree in explicit:
        for module in ("field", "tensor", "free"):
            a = getattr(explicit[degree], module)
            b = getattr(streamed[degree], module)
            for prime in (None, 2, 3):
                assert a.barcode(prime=prime) == b.barcode(prime=prime)
    assert streamed[1].field.barcode() == []
