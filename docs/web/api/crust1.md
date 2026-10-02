# CRUST 1.0

A bundled reader for the [CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html)
global crustal model (Laske et al., 2013) — a 1° × 1° grid of 9 layers (water,
ice, 3 sediments, 3 crystalline, mantle). It ships inside the package
(`shakermaker/crust1/`), so the data grids travel with `pip install` and the
reader needs no path configuration.

```python
from shakermaker.crust1 import Crust1   # prints CRUST1_CITATION on import

crust1 = Crust1()                                  # zero-config
profile = crust1.profile_at(-33.42, -70.61)        # 9-layer column at a lat/lon
crust1.print_shakermaker((-33.42, -70.61))         # ready-to-paste CrustModel snippet
```

::: shakermaker.crust1.crust1.Crust1

See also the example `examples/01_crustmodel/crust1_sites.py`.
