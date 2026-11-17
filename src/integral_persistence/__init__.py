"""Exact persistence over all fields, computed natively in C++."""
from ._api import (build_matrices, decode_barcodes, exceptional_correction,
                   exhaustive_rational_reduction, persistence,
                   rational_reduction, reduction)

__all__ = ["build_matrices", "decode_barcodes", "exceptional_correction",
           "exhaustive_rational_reduction", "persistence", "rational_reduction",
           "reduction"]
__version__ = "0.1.0"
