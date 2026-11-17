"""Public matrix contracts checked against independent dense field arithmetic."""

from copy import deepcopy
from fractions import Fraction
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import random
import sys

import pytest
import integral_persistence as ip
from integral_persistence import _native


def _rational_oracle(matrix, ncols=None):
    rows = len(matrix)
    count = len(matrix[0]) if rows else (ncols or 0)
    columns = [
        [Fraction(i == j) for i in range(count)]
        + [Fraction(matrix[i][j]) for i in range(rows)]
        for j in range(count)
    ]
    owner = {}
    pivots, minors = [], []
    minor = Fraction(1)
    for j, column in enumerate(columns):
        row = max(i for i, value in enumerate(column) if value)
        while row in owner:
            source = columns[owner[row]]
            factor = column[row] / source[row]
            column[:] = [a - factor * b for a, b in zip(column, source)]
            row = max(i for i, value in enumerate(column) if value)
        owner[row] = j
        pivots.append(row)
        minor *= column[row]
        assert minor.denominator == 1 and minor != 0
        minors.append(minor.numerator)
    return pivots, minors


def _modular_oracle(matrix, prime, ncols=None):
    rows = len(matrix)
    count = len(matrix[0]) if rows else (ncols or 0)
    columns = [
        [int(i == j) for i in range(count)]
        + [matrix[i][j] % prime for i in range(rows)]
        for j in range(count)
    ]
    owner, pivots = {}, []
    for j, column in enumerate(columns):
        row = max(i for i, value in enumerate(column) if value)
        while row in owner:
            source = columns[owner[row]]
            factor = column[row] * pow(source[row], -1, prime) % prime
            column[:] = [(a - factor * b) % prime for a, b in zip(column, source)]
            row = max(i for i, value in enumerate(column) if value)
        owner[row] = j
        pivots.append(row)
    return pivots


def _small_prime_divisors(number):
    number = abs(number)
    factors = set()
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors.add(divisor)
            number //= divisor
        divisor += 1
    if number > 1:
        factors.add(number)
    return factors


def _sparse_columns(matrix):
    return [
        {row: values[col] for row, values in enumerate(matrix) if values[col]}
        for col in range(len(matrix[0]) if matrix else 0)
    ]


def test_imports_are_the_clean_package_and_compiled_extension():
    package = Path(__file__).resolve().parents[1]
    source = Path(ip.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    assert source.is_relative_to(package / "src") or source.is_relative_to(environment)
    assert any(str(_native.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    assert Path(_native.__file__).resolve().is_relative_to(environment)
    assert not hasattr(ip, "Filtration")


def test_random_rectangular_profiles_and_certified_prime_discovery():
    generator = random.Random(93125)
    for _ in range(35):
        rows, columns = generator.randrange(1, 7), generator.randrange(1, 7)
        matrix = [[generator.randrange(-12, 13) for _ in range(columns)] for _ in range(rows)]
        before = deepcopy(matrix)
        generic, minors = _rational_oracle(matrix)
        expected_primes = sorted(set().union(*(_small_prime_divisors(x) for x in minors)))
        result = ip.reduction(matrix)
        assert matrix == before
        assert result.phase == "complete"
        assert result.shape == (rows, columns)
        assert result.generic_pivots == generic
        assert result.pivot_profile() == generic
        assert result.primes == expected_primes
        assert set(result.exceptional_pivots) == set(expected_primes)
        for prime in sorted(set(expected_primes) | {2, 3, 5, 7, 11, 13}):
            expected = _modular_oracle(matrix, prime)
            assert result.pivot_profile(prime=prime) == expected
            if prime in expected_primes:
                assert result.exceptional_pivots[prime] == expected
            else:
                assert expected == generic


def test_dense_sparse_and_staged_reduction_are_identical():
    matrix = [[2, 4, 6], [3, 9, 12]]
    columns = _sparse_columns(matrix)
    before = deepcopy(columns)
    direct = ip.reduction(matrix)
    state = ip.rational_reduction(columns=columns, nrows=2)
    identity = id(state)
    assert columns == before
    assert state.phase == "rational"
    assert state.primes == [2, 3]
    with pytest.raises(RuntimeError, match="stage"):
        state.pivot_profile(2)
    with pytest.raises(RuntimeError, match="stage"):
        _ = state.exceptional_pivots
    assert ip.exhaustive_rational_reduction(state) is None
    assert state.phase == "exhaustive"
    assert ip.exceptional_correction(state) is None
    assert state.phase == "complete" and id(state) == identity
    assert state.primes == direct.primes
    assert state.generic_pivots == direct.generic_pivots
    assert state.exceptional_pivots == direct.exceptional_pivots
    assert set(state.timings) >= {
        "rational_reduction", "factorization", "exhaustive_rational_reduction", "exceptional_correction"
    }
    assert state.timings["profile_packaging_and_cleanup"] >= 0
    assert all(value >= 0 for value in state.timings.values())


@pytest.mark.parametrize("matrix", [[], [[]], [[], []], [[0, 0], [0, 0]], [[1, 0], [0, -1]]])
def test_empty_zero_and_unimodular_inputs(matrix):
    state = ip.rational_reduction(matrix)
    generic, _ = _rational_oracle(matrix)
    assert state.primes == []
    assert ip.exhaustive_rational_reduction(state) is None
    assert ip.exceptional_correction(state) is None
    assert state.pivot_profile(101) == generic
    assert state.exceptional_pivots == {}


def test_zero_rows_and_nonzero_column_count_are_preserved():
    state = ip.reduction(columns=[{}, {}, {}], nrows=0)
    assert state.shape == (0, 3)
    assert state.generic_pivots == [0, 1, 2]
    assert state.primes == []


def test_coefficients_beyond_python_decimal_conversion_limit():
    # Conversion must not rely on Python's bounded decimal str(int) path.
    huge = -(1 << 16000)
    state = ip.reduction([[huge]])
    assert state.primes == [2]
    assert state.generic_pivots == [1]
    assert state.pivot_profile(2) == [0]
    assert state.pivot_profile(3) == [1]


def test_primes_beyond_machine_integer_range_are_proved():
    prime = (1 << 127) - 1
    state = ip.reduction([[prime]])
    assert state.primes == [prime]
    assert state.pivot_profile(prime) == [0]
    assert state.pivot_profile(2) == [1]
    assert ip.reduction([[1]]).pivot_profile(prime) == [1]


@pytest.mark.parametrize("prime", [-7, 0, 1, 4, 561, 1105, (1 << 127) - 3])
def test_composites_and_nonpositive_characteristics_are_rejected(prime):
    with pytest.raises(ValueError, match="prime"):
        ip.reduction([[2]]).pivot_profile(prime)


@pytest.mark.parametrize("prime", [True, 2.0, "2", Fraction(2)])
def test_characteristic_requires_an_exact_integer(prime):
    with pytest.raises(TypeError, match="integer"):
        ip.reduction([[2]]).pivot_profile(prime)


@pytest.mark.parametrize("matrix", [[[True]], [[1.0]], [[Fraction(1)]], [["1"]], [[None]]])
def test_matrix_coefficients_require_exact_integers(matrix):
    with pytest.raises(TypeError, match="integer"):
        ip.reduction(matrix)


@pytest.mark.parametrize("kwargs", [
    {"matrix": [[1], [2, 3]]},
    {"matrix": [[], [1]]},
    {"matrix": [[1]], "nrows": 1},
    {"matrix": [[1]], "columns": [{0: 1}], "nrows": 1},
    {"columns": [{0: 1}]},
    {"columns": [[(0, 1), (0, 2)]], "nrows": 1},
    {"columns": [{1: 1}], "nrows": 1},
    {"columns": [{-1: 1}], "nrows": 1},
    {"columns": [{0: 1}], "nrows": -1},
    {"columns": [[(0, 1, 2)]], "nrows": 1},
])
def test_invalid_matrix_shapes_and_sparse_entries_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ip.reduction(**kwargs)


def test_unsorted_sparse_pairs_and_explicit_zeros_are_valid():
    result = ip.reduction(columns=[[(2, 3), (0, 2), (1, 0)], [(1, 4), (0, 7)]], nrows=3)
    direct = ip.reduction([[2, 7], [0, 4], [3, 0]])
    assert result.generic_pivots == direct.generic_pivots
    assert result.exceptional_pivots == direct.exceptional_pivots
    assert result.primes == direct.primes


def _native_boundary():
    return ip.build_matrices([[0], [1], [0, 1]], degrees=[0], modules=["field"])[0]["field"]


def test_native_aliases_are_consumed_but_explicit_copy_preserves_input():
    data = _native_boundary()
    matrix = data.matrix
    alias = data.matrix
    copy = matrix.copy()
    first = ip.reduction(copy)
    assert copy.consumed
    assert not matrix.consumed and not alias.consumed
    second = ip.reduction(matrix)
    assert alias.consumed and data.matrix.consumed
    assert first.generic_pivots == second.generic_pivots
    for consumed in [copy, matrix, alias, data.matrix]:
        with pytest.raises(RuntimeError, match="consumed"):
            ip.reduction(consumed)
        with pytest.raises(RuntimeError, match="consumed"):
            consumed.copy()
    # Shape and decoding metadata remain useful after coefficient consumption.
    assert matrix.shape == (3, 3)
    assert data.source_count == 3
    assert ip.decode_barcodes(first, data).barcode() == ip.decode_barcodes(second, data).barcode()


def test_invalid_stage_calls_do_not_destroy_a_valid_state():
    state = ip.rational_reduction([[6]])
    with pytest.raises(RuntimeError, match="stage"):
        ip.exceptional_correction(state)
    assert state.phase == "rational"
    ip.exhaustive_rational_reduction(state)
    with pytest.raises(RuntimeError, match="stage"):
        ip.exhaustive_rational_reduction(state)
    assert state.phase == "exhaustive"
    ip.exceptional_correction(state)
    with pytest.raises(RuntimeError, match="stage"):
        ip.exceptional_correction(state)
    assert state.phase == "complete"
    assert state.pivot_profile(2) == [0]


@pytest.mark.parametrize("stage", ["exhaustive", "exceptional"])
def test_cancellation_invalidates_mutable_state(stage):
    state = ip.rational_reduction([[6]])
    operation = ip.exhaustive_rational_reduction
    if stage == "exceptional":
        ip.exhaustive_rational_reduction(state)
        operation = ip.exceptional_correction
    with pytest.raises(InterruptedError, match="cancelled"):
        operation(state, progress=lambda event: False)
    assert state.phase == "failed"
    with pytest.raises(RuntimeError, match="stage"):
        state.pivot_profile(2)
    with pytest.raises(RuntimeError, match="stage"):
        _ = state.exceptional_pivots
    with pytest.raises(RuntimeError, match="stage"):
        ip.exhaustive_rational_reduction(state)


def test_callback_exception_propagates_and_fails_state():
    class StopRequested(Exception):
        pass

    state = ip.rational_reduction([[2]])

    def stop(event):
        raise StopRequested("stop the calculation")

    with pytest.raises(StopRequested, match="stop the calculation"):
        ip.exhaustive_rational_reduction(state, progress=stop)
    assert state.phase == "failed"


def test_cancelled_discovery_returns_no_partial_result_and_consumes_native_input():
    data = _native_boundary()

    def stop_at_factorization(event):
        return event["phase"] != "factorization"

    with pytest.raises(InterruptedError, match="cancelled"):
        ip.rational_reduction(data.matrix, progress=stop_at_factorization)
    assert data.matrix.consumed
    with pytest.raises(RuntimeError, match="consumed"):
        ip.reduction(data.matrix)


def test_invalid_progress_callback_is_rejected_before_native_consumption():
    data = _native_boundary()
    with pytest.raises(TypeError, match="progress"):
        ip.reduction(data.matrix, progress="verbose")
    assert not data.matrix.consumed


def test_progress_is_optional_and_stage_events_are_available(capsys):
    ip.reduction([[2]])
    assert capsys.readouterr().out == ""
    events = []
    result = ip.reduction([[2]], progress=events.append)
    assert result.phase == "complete"
    assert {event["phase"] for event in events} >= {
        "rational_reduction", "factorization", "exhaustive_rational_reduction", "exceptional_correction"
    }
    assert all(0 <= event["completed"] <= event["total"] for event in events)
    assert all(event["elapsed_seconds"] >= 0 for event in events)


def test_result_lists_and_properties_cannot_mutate_native_state():
    result = ip.reduction([[6]])
    profile = result.generic_pivots
    profile[0] = -1
    primes = result.primes
    primes.clear()
    exceptional = result.exceptional_pivots
    exceptional[2][0] = -1
    assert result.generic_pivots == [1]
    assert result.primes == [2, 3]
    assert result.pivot_profile(2) == [0]
    with pytest.raises(AttributeError):
        result.phase = "rational"
    with pytest.raises(AttributeError):
        result.generic_pivots = []


def test_zero_row_dense_shape_and_generic_sparse_mapping():
    from collections import UserDict

    class ShapedRows:
        shape = (0, 3)
        def __iter__(self):
            return iter(())

    empty = ip.reduction(ShapedRows())
    assert empty.shape == (0, 3)
    assert empty.generic_pivots == [0, 1, 2]
    sparse = ip.reduction(columns=[UserDict({0: 2}), UserDict({1: 3})], nrows=2)
    assert sparse.primes == [2, 3]
    assert sparse.generic_pivots == [2, 3]


def test_numpy_dense_shapes_when_available():
    np = pytest.importorskip("numpy")
    assert ip.reduction(np.empty((0, 3), dtype=int)).shape == (0, 3)
    assert ip.reduction(np.empty((3, 0), dtype=int)).shape == (3, 0)
    assert ip.reduction(np.array([[2, 0], [0, 3]], dtype=object)).primes == [2, 3]
    with pytest.raises(ValueError, match="two dimensions"):
        ip.reduction(np.array([], dtype=int))
