"""Small CRT checks with an independent dense modular persistence oracle.

Run directly, or use ``python vendor/gudhi_crt/build.py --test``. No installed
GUDHI, Dionysus, or production reduction implementation is used by the oracle.
"""

from itertools import combinations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from _gudhi_crt import run_native


def closure(facets):
    return sorted({face for facet in facets for size in range(1, len(facet) + 1)
                   for face in combinations(facet, size)}, key=lambda x: (len(x), x))


def dense_barcode(simplices, prime, degree):
    """Ordinary left-to-right dense column reduction over a prime field."""
    index = {simplex: i for i, simplex in enumerate(simplices)}
    columns, owners, births, deaths = [], {}, [], {}
    for j, simplex in enumerate(simplices):
        column = [0] * len(simplices)
        if len(simplex) > 1:
            for i in range(len(simplex)):
                column[index[simplex[:i] + simplex[i + 1:]]] = (-1) ** i % prime
        while any(column):
            pivot = max(i for i, value in enumerate(column) if value)
            if pivot not in owners:
                owners[pivot] = j
                deaths[pivot] = j
                break
            previous = columns[owners[pivot]]
            scale = column[pivot] * pow(previous[pivot], -1, prime) % prime
            column = [(a - scale * b) % prime for a, b in zip(column, previous)]
        if not any(column):
            births.append(j)
        columns.append(column)
    return [[birth, deaths.get(birth)] for birth in births
            if len(simplices[birth]) == degree + 1]


class BackendTests(unittest.TestCase):
    def test_complete_pairs_against_independent_oracle(self):
        projective = closure([
            (0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
            (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5),
        ])
        cases = [closure([(0, 1), (0, 2), (1, 2)]), closure([(0, 1, 2)]),
                 closure(combinations(range(4), 3)), projective]
        primes = (2, 3, 18446744073709551629)
        for simplices in cases:
            for degree in range(3):
                with self.subTest(size=len(simplices), degree=degree):
                    result = run_native(simplices, primes, degree=degree)
                    self.assertEqual(result["status"], "success")
                    for prime in primes:
                        self.assertEqual(result["barcodes"][str(prime)],
                                         dense_barcode(simplices, prime, degree))
                    self.assertGreaterEqual(result["prime_validation_seconds"], 0)

    def test_omit_top_homology_retains_h1_deaths(self):
        simplices = closure([(0, 1, 2)])
        result = run_native(simplices, (2, 3), persistence_dim_max=False)
        self.assertEqual(result["computed_degrees"], "below_maximal")
        self.assertEqual(result["barcodes"]["2"], [[5, 6]])
        with self.assertRaises(ValueError):
            run_native(simplices[:-1], (2,), persistence_dim_max=False)

    def test_strict_native_validation(self):
        for primes in [(4,), (2, 2), (3, 2), (-7,)]:
            with self.subTest(primes=primes), self.assertRaises(ValueError):
                run_native([(0,)], primes)
        for simplices in [[(0, 1)], [(0,), (0,)], [(0,), (1,), (1, 0)]]:
            with self.subTest(simplices=simplices), self.assertRaises(ValueError):
                run_native(simplices, (2,))

    def test_empty_fields_bypass(self):
        result = run_native([], [])
        self.assertTrue(result["empty_fields_bypassed"])
        self.assertEqual(result["barcodes"], {})


if __name__ == "__main__":
    unittest.main()
