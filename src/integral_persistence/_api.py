"""Thin orchestration of native matrices, reduction states and decoded results."""
from types import SimpleNamespace
from . import _native


def _progress(callback):
    if callback is True:
        return lambda event: print(f"{event['phase']}: {event['completed']}/{event['total']}")
    if callback is False:
        return None
    return callback


def rational_reduction(matrix=None, *, columns=None, nrows=None, progress=None):
    """Reduce an integer matrix over Q and discover all exceptional primes.

    Supply dense rows, or sparse ``columns`` (mappings or (row, value) pairs)
    with ``nrows``. Coefficients are arbitrary-size integers. Native matrix
    handles are consumed; ordinary Python input is unchanged. The returned
    explicit state is ready for ``exhaustive_rational_reduction``.
    """
    return _native.rational_reduction(matrix, columns, nrows, _progress(progress))


def exhaustive_rational_reduction(state, *, progress=None):
    """Mutate a rational state to exhaustive form; return None.

    This is stage 2. It uses the generic pivot-induced row priority and needs
    no filtration times. With no exceptional primes it advances without work.
    """
    _native.exhaustive_rational_reduction(state, _progress(progress))


def exceptional_correction(state, *, progress=None):
    """Mutate an exhaustive state to completed all-fields profiles; return None.

    Grouped rational corrections share complete pivot prefixes. Completion
    releases algebraic workspace. Errors invalidate the mutable state.
    """
    _native.exceptional_correction(state, _progress(progress))


def reduction(matrix=None, *, columns=None, nrows=None, progress=None):
    """Compose rational discovery, exhaustive reduction and prime correction.

    Return a completed state exposing ``primes``, ``generic_pivots``,
    ``exceptional_pivots``, ``pivot_profile(prime=None)`` and stage ``timings``.
    Pivots refer to [I_m; A]: dummy rows 0..m-1, matrix rows m..m+n-1.
    This function does not require grades or decode barcodes.
    """
    state = rational_reduction(matrix, columns=columns, nrows=nrows, progress=progress)
    exhaustive_rational_reduction(state, progress=progress)
    exceptional_correction(state, progress=progress)
    return state


def build_matrices(simplices, times=None, *, degrees=(1,),
                   modules=("field", "tensor", "free"), progress=None):
    """Build requested native matrices and separate source-index metadata.

    Input is a face-closed sequence of nonempty simplices (each face before
    its cofaces), optionally with nondecreasing integer/finite-float times,
    or a GUDHI SimplexTree. Vertex labels are exact integers. No public
    filtration object is required. Omitted times allow later decoding at
    user-supplied times, or at simplex indices by default.

    Return ``matrices[degree][module]``. Each entry has ``matrix``, source
    metadata and construction ``timings``. Field entries share one matrix;
    reduce that matrix only once. Free construction finishes before any
    tensor matrix is consumed. H1 uses the exact spanning-forest constructor.
    """
    return _native.build_matrices(simplices, times, degrees, modules, _progress(progress))


def decode_barcodes(state, data, *, times=None):
    """Decode a completed reduction using its MatrixData and optional regrading.

    Times must preserve the fixed simplex order and may tie. They are used
    only here, so this call can be repeated without reduction. Equal-endpoint
    intervals disappear; multiplicities and infinite deaths are preserved.
    The result provides ``barcode(prime=None)``, ``exceptional_primes``
    (barcode-changing primes), ``matrix_primes`` and ``field_independent``.
    """
    return _native.decode_barcodes(state, data, times)


def persistence(simplices, times=None, *, degrees=(1,),
                modules=("field", "tensor", "free"), progress=None):
    """Build, reduce and decode the requested modules in the requested degrees.

    ``result[d].field.barcode()`` returns a rational barcode, and
    ``result[d].tensor.barcode(prime=2)`` a tensor barcode in characteristic 2.
    Only requested modules are present. Each degree object also provides
    ``states`` and ``matrices`` for repeated decoding without fresh reduction.
    """
    matrices = build_matrices(simplices, times, degrees=degrees, modules=modules,
                              progress=progress)
    states = {}
    result = {}
    for degree, entries in matrices.items():
        outputs = {}
        degree_states = {}
        for module, data in entries.items():
            key = data.matrix.identity
            if key not in states:
                states[key] = reduction(data.matrix, progress=progress)
            degree_states[module] = states[key]
            outputs[module] = decode_barcodes(states[key], data)
        result[degree] = SimpleNamespace(**outputs, states=degree_states, matrices=entries)
    return result
