"""CRUST 1.0 global crustal model reader (Laske et al., 2013).

Ships as part of the shakermaker package so the data travels with `pip install`.
The citation is printed on import; please cite the source if you use it.
"""

from .crust1 import Crust1, CRUST1_CITATION

print(CRUST1_CITATION)

__all__ = ["Crust1", "CRUST1_CITATION"]
