"""Persistence of torus knots (a,b) in R^4 with alpha filtrations."""

# %% Imports

from math import gcd
import gudhi
import numpy as np
import integral_persistence as ip

def generate_torus_knot_points(N, a, b):
    """Return N points on the torus knot (a,b) in R^4."""
    theta = np.linspace(0, 2 * np.pi, num=N, endpoint=False)
    return np.column_stack((np.cos(a * theta), np.sin(a * theta),
                            np.cos(b * theta), np.sin(b * theta)))

# %% Example on the torus knot (1,3)

import gudhi
import numpy as np

# Generate data
N, a, b = 50, 1, 3
points = generate_torus_knot_points(N, a, b)
alpha = gudhi.AlphaComplex(points=points, precision="exact")
st = alpha.create_simplex_tree()

# Decide field-dependence in H1
degree = 1
modules = ("field", "free")
results = ip.persistence(st, degrees=[degree], modules=modules)
for module in modules:
    result = getattr(results[degree], module)
    print(f"H{degree} {module:<6}", end='')
    print(" --- Field independent:", result.field_independent, end='')
    if not result.field_independent:
        print(" --- Exceptional primes:", result.exceptional_primes)
    else:
        print()

"""
H1 field  --- Field independent: False --- Exceptional primes: [3]
H1 free   --- Field independent: False --- Exceptional primes: [3]
"""

# Show difference between the Q and F3 H1-barcodes for 
barcode_free_rational = results[degree].free.barcode()
barcode_free_mod3 = results[degree].free.barcode(prime=3)
barcode_free_rational_difference = [b for b in barcode_free_rational if b not in barcode_free_mod3]
barcode_free_mod3_difference = [b for b in barcode_free_mod3 if b not in barcode_free_rational]
print("Bars over Q but not over F3:", barcode_free_rational_difference)
print("Bars over F3 but not over Q:", barcode_free_mod3_difference)

"""
Bars over Q but not over F3: [(0.039054406398635666, 1.2734340459139588), (0.7285980652182205, 1.0102682032339034)]
Bars over F3 but not over Q: [(0.039054406398635666, 1.0102682032339034), (0.7285980652182205, 1.2734340459139588)]
"""

# %% More examples

PARAMETERS = [(a, b) for a in range(1, 6) for b in range(a + 1, 6) if gcd(a, b) == 1]
for a, b in PARAMETERS:
    N = b * 25
    print(f"Torus knot ({a}, {b}) with N={N:<3} points", end='')

    # Generate data
    points = generate_torus_knot_points(N, a, b)
    alpha = gudhi.AlphaComplex(points=points, precision="exact")
    st = alpha.create_simplex_tree()

    # Decide field-dependence in H1
    degree = 1
    module = "free"
    results = ip.persistence(st, degrees=[degree], modules=[module])
    result = getattr(results[degree], module)
    print(f" --- H{degree} {module}", end='')
    print(" --- Field independent:", result.field_independent, end='')
    if not result.field_independent:
        print(" --- Exceptional primes:", result.exceptional_primes)
    else:
        print()

"""
Torus knot (1, 2) with N=50  points --- H1 free --- Field independent: False --- Exceptional primes: [2]
Torus knot (1, 3) with N=75  points --- H1 free --- Field independent: False --- Exceptional primes: [3]
Torus knot (1, 4) with N=100 points --- H1 free --- Field independent: False --- Exceptional primes: [2]
Torus knot (1, 5) with N=125 points --- H1 free --- Field independent: False --- Exceptional primes: [5]
Torus knot (2, 3) with N=75  points --- H1 free --- Field independent: False --- Exceptional primes: [3]
Torus knot (2, 5) with N=125 points --- H1 free --- Field independent: False --- Exceptional primes: [5]
Torus knot (3, 4) with N=100 points --- H1 free --- Field independent: False --- Exceptional primes: [2]
Torus knot (3, 5) with N=125 points --- H1 free --- Field independent: False --- Exceptional primes: [5]
Torus knot (4, 5) with N=125 points --- H1 free --- Field independent: False --- Exceptional primes: [5]
"""
