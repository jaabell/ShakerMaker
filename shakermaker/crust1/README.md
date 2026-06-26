# `shakermaker/crust1/` — CRUST 1.0 global crustal model (bundled)

This subpackage holds the **CRUST 1.0** reader and data grids, packaged with
ShakerMaker so the data travels with `pip install`.

[CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html) (Laske et al., 2013) is a
global crustal model on a 1° × 1° grid (9 layers: water, ice, 3 sediments,
3 crystalline, mantle).

| Path | What |
|------|------|
| `crust1.py`     | Self-contained `Crust1` reader (numpy + matplotlib only) |
| `crust1.0/`     | The CRUST 1.0 binary grids (`crust1.vp/.vs/.rho/.bnds`) + type add-on |
| `__init__.py`   | Reexports `Crust1`, prints `CRUST1_CITATION` on import |

The reader auto-detects the data because it sits **next to** `crust1.py`
(it resolves `Path(__file__).parent / "crust1.0"`), so no path hacks are needed:

```python
from shakermaker.crust1 import Crust1   # CRUST1_CITATION prints on import

crust = Crust1()                          # zero-config: finds crust1.0/ next to crust1.py
# crust = Crust1(data_dir=...)            # or point it elsewhere

profile = crust.profile_at(-33.42, -70.61)        # 9-layer column at a lat/lon
crust.print_shakermaker((-33.42, -70.61))         # ready-to-paste CrustModel snippet
```

Used by `examples/01_crustmodel/crust1_sites.py`.

**Source / license:** CRUST 1.0 is distributed freely by UCSD (G. Laske et al.).
It is redistributed here unmodified for convenience. Please cite the source
(see `CRUST1_CITATION` in `crust1.py`).
