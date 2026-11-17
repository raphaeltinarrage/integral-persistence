# Computing integral persistence efficiently

Compute the persistent homology of an integer matrix or a simplicial filtration over all fields at once. 

Given a simplicial filtration, the package can build the usual persistent homology module as well as the associated tensor and free modules.

The core code is in `C++`.
It uses `GMP` for exact arithmetic and `FLINT` for prime factorization.
The interface is in `Python`.

## Matrix input

```python
import integral_persistence as ip

result = ip.reduction([[2, 1], [0, 3]])
print(result.primes)                   # [2, 3]
print(result.generic_pivots)           # [2, 3]
print(result.pivot_profile(prime=2))   # [0, 3]
print(result.pivot_profile(prime=3))   # [2, 1]
```

Pivots refer to the augmented matrix `[I_m; A]`.

## Filtration input

```python
# Filled triangle
simplices = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
times = [0, 0, 0, 1, 1, 1, 2]

result = ip.persistence(simplices, times, degrees=[1], modules=("field",))
print(result[1].field.exceptional_primes)   # []
print(result[1].field.field_independent)    # True
```

```python
# Projective plane
vertices = [[i] for i in range(6)]
edges = list(combinations(range(6), 2))
triangles = [
    (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
    (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
]
simplices = vertices + edges + triangles
times = list(range(len(simplices)))

result = ip.persistence(simplices, times, degrees=[1], modules=("field",))
print(result[1].field.exceptional_primes)   # [2]
print(result[1].field.field_independent)    # False
```

## Linux installation

On Ubuntu, install the core build dependencies with:

```sh
sudo apt install build-essential cmake python3-dev python3-venv libgmp-dev libflint-dev
```

Then create an environment and install the package with the dependencies needed by the demo:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[experiments]"
python scripts/demo.py
```

For the scripts, the following packages are required: `numpy>=2`, `gudhi>=3.11`, `dionysus>=2.1`, `matplotlib>=3.9`, and `tqdm>=4.66`.
