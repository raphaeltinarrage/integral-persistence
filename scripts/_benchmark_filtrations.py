"""Generate Linial–Meshulam filtrations."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import comb
import operator
import random


def exact_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return operator.index(value)


def _vertex_count(n: int) -> int:
    n = exact_integer(n, "ambient vertex count")
    if n < 0:
        raise ValueError("ambient vertex count must be nonnegative")
    return n


def _triangle(n, vertices):
    triangle = tuple(exact_integer(vertex, "vertex") for vertex in vertices)
    if len(triangle) != 3 or not 0 <= triangle[0] < triangle[1] < triangle[2] < n:
        raise ValueError("triangles must have three increasing vertices in range(n)")
    return triangle


def triangle_unrank(n: int, rank: int) -> tuple[int, int, int]:
    """Invert lexicographic triangle indices using two integer binary searches.

    There are ``C(n,3)-C(n-a,3)`` triangles whose first vertex is below a.
    The analogous cumulative count fixes the second vertex; the last vertex
    is the residual offset. All counts remain arbitrary-precision integers.
    """
    n = _vertex_count(n)
    rank = exact_integer(rank, "triangle rank")
    total = comb(n, 3)
    if not 0 <= rank < total:
        raise ValueError("triangle rank outside the ambient triangle set")
    low, high = 0, n - 2
    while low + 1 < high:
        middle = (low + high) // 2
        if total - comb(n - middle, 3) <= rank:
            low = middle
        else:
            high = middle
    a = low
    rank -= total - comb(n - a, 3)
    block = comb(n - a - 1, 2)
    low, high = a + 1, n - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if block - comb(n - middle, 2) <= rank:
            low = middle
        else:
            high = middle
    b = low
    return a, b, b + 1 + rank - (block - comb(n - b, 2))


@dataclass(frozen=True)
class TriangleOrder:
    """The exact triangle order shared by all methods and saved with results."""

    n: int
    triangles: tuple[tuple[int, int, int], ...]
    seed: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "n", _vertex_count(self.n))
        triangles = tuple(_triangle(self.n, vertices) for vertices in self.triangles)
        if len(set(triangles)) != len(triangles):
            raise ValueError("triangle arrivals must be distinct")
        object.__setattr__(self, "triangles", triangles)
        if self.seed is not None:
            object.__setattr__(self, "seed", exact_integer(self.seed, "seed"))

    def __len__(self) -> int:
        return len(self.triangles)

    @property
    def digest(self) -> str:
        """SHA-256 of canonical JSON containing n and the ordered triangles."""
        payload = json.dumps({"n": self.n, "triangles": self.triangles}, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("ascii")).hexdigest()

    def to_dict(self) -> dict:
        """Return the input and its checksum as a JSON-ready dictionary."""
        return {
            "n": self.n,
            "triangles": [list(triangle) for triangle in self.triangles],
            "seed": self.seed,
            "sha256": self.digest,
        }

    @classmethod
    def from_dict(cls, record: dict) -> "TriangleOrder":
        """Load an input, accepting older metadata-rich records too."""
        seed = record.get("seed")
        if isinstance(seed, str):  # Older files stored the seed as a string.
            seed = int(seed)
        result = cls(record["n"], record["triangles"], seed=seed)
        if record.get("sha256") != result.digest:
            raise ValueError("triangle record checksum does not match its input")
        return result


def generate_triangle_order(n, count, seed=0):
    """Preserve the original lazy Fisher--Yates permutation exactly."""
    n = _vertex_count(n)
    count, seed = exact_integer(count, "count"), exact_integer(seed, "seed")
    total = comb(n, 3)
    if not 0 <= count <= total:
        raise ValueError("count must lie between zero and C(n,3)")
    rng, swaps, triangles = random.Random(seed), {}, []
    for position in range(count):
        chosen = rng.randrange(position, total)
        rank = swaps.pop(chosen, chosen)
        if chosen != position:
            replacement = swaps.pop(position, position)
            if replacement != chosen:
                swaps[chosen] = replacement
        triangles.append(triangle_unrank(n, rank))
    return TriangleOrder(n, triangles, seed=seed)


def build_filtration(order, model="LM"):
    """Return simplices and actual event times: graph at zero, triangle j at j."""
    if model != "LM":
        raise ValueError("The retained comparison uses model='LM'")
    simplices = list(combinations(range(order.n), 1)) + list(combinations(range(order.n), 2))
    times = [0] * len(simplices) + list(range(1, len(order) + 1))
    return simplices + list(order.triangles), times
