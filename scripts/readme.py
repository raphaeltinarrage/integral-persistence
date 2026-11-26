""" Reproduce examples in README.md. """

# %%

from itertools import combinations
import gudhi as gd
import numpy as np
import integral_persistence as ip

print("\nMatrix input...")
matrix = [[7, 1], [0, 11]]
state = ip.reduction(matrix)
print(state.primes)                    # [7, 11] (exceptional primes)
print(state.generic_pivots)            # [2, 3]  (generic pivots)
print(state.pivot_profile(prime=2))    # [2, 3]  (pivots over F_2)
print(state.pivot_profile(prime=7))    # [0, 3]  (pivots over F_7)
print(state.pivot_profile(prime=11))   # [2, 1]  (pivots over F_11)

print("\nFiltration input: list of simplices and times...")
# Filled triangle
simplices = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
times = [0, 0, 0, 1, 1, 1, 2]
result = ip.persistence(simplices, times, degrees=[0,1,2], modules=("field",))
print(result[1].field.exceptional_primes)   # []
print(result[1].field.field_independent)    # True

print("\nFiltration input: GUDHI simplex tree...")
# Projective plane
vertices = [[i] for i in range(6)]
edges = list(combinations(range(6), 2))
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
simplices = vertices + edges + triangles
st = gd.SimplexTree()
for i, simplex in enumerate(simplices): # simplex-wise filtration
    st.insert(simplex, filtration=i)
degree = 1
result = ip.persistence(st, degrees=[degree], modules=("field",))
print(result[degree].field.exceptional_primes)   # [2]
print(result[degree].field.field_independent)    # False

print("\nMap torsion...")
# Torus knot (1, 3) in R^4
theta = np.linspace(0, 2 * np.pi, num=50, endpoint=False)
points = np.column_stack((np.cos(theta), np.sin(theta), np.cos(3 * theta), np.sin(3 * theta)))
# Alpha filtration
alpha = gd.AlphaComplex(points=points, precision="exact")
st = alpha.create_simplex_tree()
# Decide field-dependence in H1
degree = 1
result = ip.persistence(st, degrees=[degree], modules=("field", "tensor", "free"))
print(result[degree].field.field_independent)    # False
print(result[degree].tensor.field_independent)   # False
print(result[degree].free.field_independent)     # False

print("\nTensor torsion...")
# Projective plane
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
# Filtration: essential circle at time 0, completed to RP2 at time 1
st = gd.SimplexTree()
for edge in [(0, 1), (1, 4), (0, 4)]:
    st.insert(edge, filtration=0)
for triangle in triangles:
    st.insert(triangle, filtration=1)
# Decide field-dependence in H1
degree = 1
result = ip.persistence(st, degrees=[degree], modules=("field", "tensor", "free"))
print(result[degree].field.field_independent)    # False
print(result[degree].tensor.field_independent)   # False
print(result[degree].free.field_independent)     # True

print("\nTor torsion...")
# Klein bottle at time 0
n = 4
def vertex(i, j):
    if j == n:
        return (-i) % n   # Top-to-bottom identification
    return n * j + i % n  # Left-to-right identification
st = gd.SimplexTree()
for j in range(n):
    for i in range(n):
        a = vertex(i, j)
        b = vertex(i + 1, j)
        c = vertex(i + 1, j + 1)
        d = vertex(i, j + 1)
        st.insert([a, b, c], filtration=0)
        st.insert([a, c, d], filtration=0)
# Attach a disk along the torsion loop at time 1
apex = n * n
for i in range(n):
    st.insert([apex, vertex(i, 0), vertex(i + 1, 0)], filtration=1)
# Decide field-dependence in H2
degree = 2
result = ip.persistence(st, degrees=[degree], modules=("field", "tensor", "free"))
print(result[degree].field.field_independent)    # False
print(result[degree].tensor.field_independent)   # True
print(result[degree].free.field_independent)     # True
# %%
