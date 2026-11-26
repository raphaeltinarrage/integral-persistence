""" Main functions, and the three examples of torsion (map, tensor, Tor) from the paper. """

# %% Import

from itertools import combinations
import gudhi as gd
import numpy as np

import integral_persistence as ip

# %% ------ Integer matrix input ------------------------------------------------------

# Integer matrix (dense rows)
matrix = [[2, 1], [0, 3]]

# Full algorithm
print("\nRun full algorithm on matrix...")
state = ip.reduction(matrix)

# Print attributes
for name in dir(state):
    if not name.startswith("_"):
        value = getattr(state, name)
        if not callable(value):
            print(f"{name} = {value!r}")

"""
Run full algorithm on matrix...
exceptional_pivots = {2: [0, 3], 3: [2, 1]}
generic_pivots = [2, 3]
phase = 'complete'
primes = [2, 3]
shape = (2, 2)
timings = {'exceptional_correction': 2.7107e-05, 'exhaustive_rational_reduction': 7.006e-06, 'factorization': 3.0892e-05, 'profile_packaging_and_cleanup': 5.086e-06, 'rational_reduction': 3.2073e-05}
"""

# Full algorithm step by step
print("\nRun full algorithm on matrix step by step...")
state = ip.rational_reduction(matrix)   # Step 1: rational reduction and prime discovery 
print("Phase:", state.phase)
ip.exhaustive_rational_reduction(state) # Step 2: exhaustive reduction
print("Phase:", state.phase)
ip.exceptional_correction(state)        # Step 3: correction at exceptional primes
print("Phase:", state.phase)

"""
Run full algorithm on matrix step by step...
Phase: rational
Phase: exhaustive
Phase: complete
"""

# Print generic and exceptional pivots
print("Generic pivots:", state.generic_pivots)
print("Exceptional pivots:", state.exceptional_pivots)

"""
Generic pivots: [2, 3]
Exceptional pivots: {2: [0, 3], 3: [2, 1]}
"""

# %% ------ Simplicial filtration input (list) ------------------------------------------------------

# Simplicial filtration (list)
simplices = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
times = [0, 0, 0, 1, 1, 1, 2]

# Full algorithm
print("\nRun full algorithm on filtration...")
results = ip.persistence(simplices, times=times, degrees=[0,1], modules=("field", "tensor", "free"))
for degree in [0, 1]:
    for module in ("field", "tensor", "free"):
        result = getattr(results[degree], module)
        print(f"H{degree} {module:<6}:", result.barcode())

"""
Run full algorithm on filtration...
H0 field : [(0, 1), (0, 1), (0, inf)]
H0 tensor: [(0, 1), (0, 1), (0, inf)]
H0 free  : [(0, 1), (0, 1), (0, inf)]
H1 field : [(1, 2)]
H1 tensor: [(1, 2)]
H1 free  : [(1, 2)]
"""

# Full algorithm step by step
print("\nRun full algorithm on filtration step by step...")
data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"] 
state = ip.reduction(data.matrix)

# Decode barcodes with original times and simplexwise times
actual = ip.decode_barcodes(state, data)
refined = ip.decode_barcodes(state, data, times=list(range(len(simplices))))
print("H1 original times:", actual.barcode())
print("H1 simplexwise times:", refined.barcode())

"""
Run full algorithm on filtration step by step...
H1 original times: [(1, 2)]
H1 simplexwise times: [(5, 6)]
"""

# %% ------ Simplicial filtration input (simplex tree) ------------------------------------------------------

# Triangulation of RP^2 in 6 vertices
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
st = gd.SimplexTree()
for vertex in range(6):
    st.insert([vertex], filtration=0)
for edge in combinations(range(6), 2):
    st.insert(edge, filtration=1)
for triangle in triangles:
    st.insert(triangle, filtration=2)

# Full algorithm
print("\nRun full algorithm on filtration...")
results = ip.persistence(st, degrees=[1], modules=("field", "tensor", "free"))
for module in ("field", "tensor", "free"):
    result = getattr(results[1], module)
    print(f"\nH1 {module}:")
    print("Field independent:", result.field_independent)
    if not result.field_independent:
        print("Exceptional primes:", result.exceptional_primes)
        print(f"Over Q: ", result.barcode())
        print("Over F2:", result.barcode(prime=2))

"""
Run full algorithm on filtration...

H1 field:
Field independent: False
Exceptional primes: [2]
Over Q:  [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
Over F2: [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, inf)]

H1 tensor:
Field independent: False
Exceptional primes: [2]
Over Q:  [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
Over F2: [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, inf)]

H1 free:
Field independent: True
"""

# %% ------ Example of map torsion: torus knot ------------------------------------------------------

# Torus knot (1, 3) in R^4
theta = np.linspace(0, 2 * np.pi, num=50, endpoint=False)
points = np.column_stack((np.cos(theta), np.sin(theta), np.cos(3 * theta), np.sin(3 * theta)))
alpha = gd.AlphaComplex(points=points, precision="exact")
st = alpha.create_simplex_tree()

# Decide field-dependence in H1
degree = 1
results = ip.persistence(st, degrees=[degree], modules=("field", "tensor", "free"))
for module in ("field", "tensor", "free"):
    result = getattr(results[degree], module)
    print(f"H{degree} {module:<6}", end='')
    print(" --- Field independent:", result.field_independent, end='')
    if not result.field_independent:
        print(" --- Exceptional primes:", result.exceptional_primes)
    else:
        print()

"""
H1 field  --- Field independent: False --- Exceptional primes: [3]
H1 tensor --- Field independent: False --- Exceptional primes: [3]
H1 free   --- Field independent: False --- Exceptional primes: [3]
"""

# %% ------ Example of tensor torsion: projective plane ------------------------------------------------------

# Projective plane: essential circle at time 0, completed to RP2 at time 1
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
st = gd.SimplexTree()
for edge in [(0, 1), (1, 4), (0, 4)]:
    st.insert(edge, filtration=0)
for triangle in triangles:
    st.insert(triangle, filtration=1)

# Decide field-dependence in H1
degree = 1
results = ip.persistence(st, degrees=[degree], modules=("field", "tensor", "free"))
for module in ("field", "tensor", "free"):
    result = getattr(results[degree], module)
    print(f"H{degree} {module:<6}", end='')
    print(" --- Field independent:", result.field_independent, end='')
    if not result.field_independent:
        print(" --- Exceptional primes:", result.exceptional_primes)
    else:
        print()

"""
H1 field  --- Field independent: False --- Exceptional primes: [2]
H1 tensor --- Field independent: False --- Exceptional primes: [2]
H1 free   --- Field independent: True
"""

# Print barcodes over Q and F2
degree, module = 1, "tensor"
result = getattr(results[degree], module)
print(f"\nBarcodes for H{degree} {module}:")
print("Over Q: ", result.barcode())
for prime in result.exceptional_primes:
    print(f"Over F{prime}:", result.barcode(prime=prime))

"""
Barcodes for H1 tensor:
Over Q:  [(0.0, 1.0)]
Over F2: [(0.0, inf)]
"""

# %% ------ Example of Tor torsion: Klein bottle ------------------------------------------------------

# Klein bottle at time 0 with apex at time 1
n = 4
def vertex(i, j):
    if j == n:
        return (-i) % n   # Reversed top-to-bottom identification
    return n * j + i % n  # Periodic left-to-right identification
st = gd.SimplexTree()
for j in range(n):
    for i in range(n):
        a = vertex(i, j)
        b = vertex(i + 1, j)
        c = vertex(i + 1, j + 1)
        d = vertex(i, j + 1)
        st.insert([a, b, c], filtration=0)
        st.insert([a, c, d], filtration=0)
apex = n * n
for i in range(n):
    st.insert([apex, vertex(i, 0), vertex(i + 1, 0)], filtration=1)

# Decide field-dependence in H1 and H2
degrees = [1,2]
results = ip.persistence(st, degrees=degrees, modules=("field", "tensor", "free"))
for degree in degrees:
    for module in ("field", "tensor", "free"):
        result = getattr(results[degree], module)
        print(f"H{degree} {module:<6}", end='')
        print(" --- Field independent:", result.field_independent, end='')
        if not result.field_independent:
            print(" --- Exceptional primes:", result.exceptional_primes)
        else:
            print()
    print()

"""
H1 field  --- Field independent: False --- Exceptional primes: [2]
H1 tensor --- Field independent: False --- Exceptional primes: [2]
H1 free   --- Field independent: True

H2 field  --- Field independent: False --- Exceptional primes: [2]
H2 tensor --- Field independent: True
H2 free   --- Field independent: True
"""

# Print barcodes over Q and F2
degree, module = 2, "field"
result = getattr(results[degree], module)
print(f"Barcodes for H{degree} {module}:")
print("Over Q: ", result.barcode())
for prime in result.exceptional_primes:
    print(f"Over F{prime}:", result.barcode(prime=prime))

"""
Barcodes for H2 field:
Over Q:  [(1.0, inf)]
Over F2: [(0.0, inf)]
"""
