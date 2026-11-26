# Computing integral persistence efficiently

Authors: [Tamal K. Dey](https://www.cs.purdue.edu/homes/tamaldey/) and [Raphaël Tinarrage](https://raphaeltinarrage.github.io/)

Compute the persistent homology of an integer matrix or a simplicial filtration over all fields at once. 

Given a simplicial filtration, the package can build the usual persistent homology module as well as the associated tensor and free modules.

The core code is in `C++`.
It uses `GMP` for exact arithmetic and `FLINT` for prime factorization.
The interface is in `Python`.

## Usage

See [scripts/readme.py](scripts/readme.py) to reproduce the examples below and [scripts/demo.py](scripts/demo.py) for more detailed examples.

### Matrix input

```python
import integral_persistence as ip

matrix = [[7, 1], [0, 11]]

state = ip.reduction(matrix)
print(state.primes)                    # [7, 11] (exceptional primes)
print(state.generic_pivots)            # [2, 3]  (generic pivots)
print(state.pivot_profile(prime=2))    # [2, 3]  (pivots over F_2)
print(state.pivot_profile(prime=7))    # [0, 3]  (pivots over F_7)
print(state.pivot_profile(prime=11))   # [2, 1]  (pivots over F_11)
```

The input matrix $D$ has shape $n\times m$.
Its augmented matrix is $`\widetilde{D} = \begin{pmatrix}I_m\\D\end{pmatrix}`$; it has shape $(n+m)\times m$.
In the cell above, pivots refer to $\widetilde{D}$.

### Filtration input: list of simplices and times

```python
# Filled triangle
simplices = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
times = [0, 0, 0, 1, 1, 1, 2]

degree = 1
result = ip.persistence(simplices, times, degrees=[degree], modules=("field",))
print(result[degree].field.field_independent)    # True
print(result[degree].field.exceptional_primes)   # []
```

### Filtration input: GUDHI simplex tree

```python
from itertools import combinations
import gudhi as gd

# Projective plane
vertices = [[i] for i in range(6)]
edges = list(combinations(range(6), 2))
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
simplices = vertices + edges + triangles

# GUDHI filtration
st = gd.SimplexTree()
for i, simplex in enumerate(simplices): # simplex-wise filtration
    st.insert(simplex, filtration=i)

degree = 1
result = ip.persistence(st, degrees=[degree], modules=("field",))
print(result[degree].field.field_independent)    # False
print(result[degree].field.exceptional_primes)   # [2]
```

## Three sources of torsion

### **Map torsion** in torus knots

```python
import numpy as np

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
```

### **Tensor torsion** in the projective plane

```python
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
```

### **Tor torsion** in the Klein bottle

```python
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
```

## Additional experiments

- Benchmark against other methods: [scripts/benchmark.py](scripts/benchmark.py).
- Torsion in random Rips filtrations: [scripts/random_rips.py](scripts/random_rips.py).
- Torsion in torus knots in $\mathbb{R}^4$: [scripts/torus_knots.py](scripts/torus_knots.py).

## Linux installation

On Ubuntu, install the core build dependencies with:

```sh
sudo apt install build-essential cmake python3-dev python3-venv libgmp-dev libflint-dev
```

Then create an environment and install the package with the dependencies needed by the demo:

```sh
git clone https://github.com/raphaeltinarrage/integral-persistence.git
cd integral-persistence
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ".[experiments]"
python scripts/demo.py
```

For the scripts, the following packages are required: `numpy>=2`, `gudhi>=3.11`, `dionysus>=2.1`, `matplotlib>=3.9`, and `tqdm>=4.66`.
