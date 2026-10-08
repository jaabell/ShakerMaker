# Crust Model

## What this is

`CrustModel` is the 1-D, horizontally-layered viscoelastic medium that the
whole FK (Frequency-Wavenumber) engine runs on. Every simulation needs
exactly one `CrustModel`: a stack of flat layers (plus a final half-space)
defined by thickness and elastic/anelastic properties. There is no lateral
variation — it's a 1-D medium, as required by the FK method.

There are three ways to obtain one:

1. **By hand**, layer by layer, with `CrustModel.add_layer(...)`.
2. **From a packaged preset** (`shakermaker/cm_library/`) — already-validated
   benchmark models (SCEC LOH.1/LOH.3) or models from prior studies (Abell's
   thesis, Southern California).
3. **From CRUST 1.0** (`shakermaker.crust1.Crust1`) — query the real global
   model at any lat/lon and generate a ready-to-paste `CrustModel` snippet.

## Source of truth

- `shakermaker/crustmodel.py` (381 lines) — the `CrustModel` class.
- `shakermaker/cm_library/LOH.py` — `SCEC_LOH_1()`, `SCEC_LOH_3()`.
- `shakermaker/cm_library/AbellThesis.py` — `AbellThesis(split=1)`.
- `shakermaker/cm_library/SOCal_LF.py` — `SOCal_LF()`.
- `shakermaker/crust1/crust1.py` (636 lines) — the `Crust1` class.

## Full API reference

### `CrustModel` (`shakermaker/crustmodel.py`)

```python
from shakermaker.crustmodel import CrustModel

CrustModel(nlayers)                      # nlayers = int, total number of layers to define

add_layer(self, d, vp, vs, rho, qp, qs)
# d   : layer thickness, km. d=0 ONLY on the last layer -> infinite half-space.
# vp  : Vp, km/s
# vs  : Vs, km/s
# rho : density, g/cm^3
# qp  : quality factor Qp (dimensionless; use a high value, ~10000, for "no attenuation")
# qs  : quality factor Qs (same idea)
# Must be called exactly nlayers times, stacking from the surface downward.

modify_layer(self, layer_idx, d=None, vp=None, vs=None, rho=None, qp=None, qs=None)
# layer_idx: 0-based index. Only the kwargs that are != None get modified.
# GOTCHA: the kwarg names are qp=/qs=, NOT gp=/gs= (an easy typo to make).

properties_at_depths(self, z, kind="previous")
# z: depth or array of depths (km). Returns (a, b, rho, qa, qb) = (Vp, Vs, rho, Qa, Qb)
# interpolated with scipy.interpolate.interp1d, kind='previous' by default (step function,
# not linear interpolation between layers -- returns the active layer's property at that z).

split_at_depth(self, z, tol=0.01)
# Splits the layer that contains depth z into two layers identical in properties,
# incrementing nlayers by 1. Does nothing if a layer interface already exists within
# [z-tol, z+tol]. Useful for "carving out" a zone with different properties inside an
# existing layer: first split_at_depth(z_top), split_at_depth(z_bottom), then
# modify_layer(idx, ...) on the newly created intermediate layer.

get_layer(self, z, tol=0.01)
# Returns the (0-based) index of the layer whose top interface sits at depth z
# (within tol), or None if there is no interface there. This is NOT "which layer
# contains z"; it's "is there a layer interface exactly at z".

plot(self, figsize=(6, 4))          # -> matplotlib Figure. Stratigraphic column (stacked bars).
plot_profile(self, figsize=(11, 5), halfspace_extra=10.0)
# -> matplotlib Figure. 3 subplots (Vp, Vs, rho) vs depth, step-style.
# halfspace_extra: how many km of the half-space to draw (which is in theory infinite).

# Properties (numpy arrays, size nlayers): nlayers, d, a (=Vp), b (=Vs), rho, qa, qb
# __str__ prints a table: Layer | Depth | Thick | Vp | Vs | rho | Qa | Qb
```

**Both figures (`plot`, `plot_profile`) are returned, not shown automatically** —
you need to call `fig.savefig(...)` or, if the backend is interactive,
`plt.show()` afterward.

### Presets in `shakermaker/cm_library/`

```python
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.cm_library.AbellThesis import AbellThesis
from shakermaker.cm_library.SOCal_LF import SOCal_LF

SCEC_LOH_1()          # -> CrustModel, 2 layers
SCEC_LOH_3()          # -> CrustModel, 2 layers
AbellThesis(split=1)  # -> CrustModel, 11*split + 1 layers
SOCal_LF()             # -> CrustModel, 14 layers
```

| Preset | Layers | Description verified in source |
|---|---|---|
| `SCEC_LOH_1()` | 2 (slow layer + half-space) | The official **LOH.1** SCEC test-suite benchmark (Day et al. 2001): a slow layer over a half-space, **no attenuation** (Qa=Qb=10000 in both layers — anelastic attenuation approximated as zero via a very high Q, not a real Q). Layer 1: Vp=4.0, Vs=2.0, rho=2.6, d=1.0 km. Layer 2 (half-space): Vp=6.0, Vs=3.464, rho=2.7. This is **the** model used in `examples/12_validation/` (validation against the analytical solution) and in most of `examples/05..11`. |
| `SCEC_LOH_3()` | 2 (slow layer + half-space) | The **LOH.3** benchmark: same geometry as LOH.1 but **with real attenuation** — Qa=54.65/Qb=137.95 in layer 1, Qa=69.3/Qb=120 in the half-space. Same Vp/Vs/rho as LOH.1. Use it when the example/test needs non-trivial attenuation. |
| `AbellThesis(split=1)` | `11*split + 1` | Model from J. A. Abell's PhD thesis (UC Davis, 2016) on nuclear-plant SSI. `split` subdivides each of the 11 upper layers into `split` identical sub-layers (the last layer, the half-space, is never subdivided — the code forces `split=1` for it). Useful to refine the layer mesh without changing the physics. Qa=Qb=1000 fixed across all layers. |
| `SOCal_LF()` | 14 | Southern California low-frequency crustal model (source: SCEC Broadband Platform). Vs ranges from 0.86 km/s (surface layer) to 4.5 km/s (half-space), with Qa/Qb varying per layer (18 to ~2700). The last layer (`thickness=0`) is the half-space. |

All 4 are zero-argument functions (except `AbellThesis(split=1)`) that
return an already-populated `CrustModel` — no need to call `add_layer`
manually.

### `Crust1` — global CRUST 1.0 (`shakermaker/crust1/crust1.py`)

The global crustal model of Laske et al. (2013), 1°×1° grid, 9 layers
(water, ice, 3 sediment, 3 crystalline crust, mantle), packaged and shipped
with `pip install` — no separate data download needed.

```python
from shakermaker.crust1 import Crust1

crust1 = Crust1(data_dir=None)     # data_dir=None -> use the folder packaged next to the module

# --- point query ---
cell_index(self, lat, lon)          # -> (row, col) in the 180x360 grid
cell_midpoint(self, lat, lon)       # -> (lat, lon) of that 1-degree cell's center
profile_at(self, lat, lon)          # -> dict with the 9 layers + moho_depth_km, avg_vp/vs/rho, etc.
type_at(self, lat, lon)             # -> dict {'code','name','group'}, or None if the type addon is missing

# --- text ---
print_tables(self, sites)                       # descriptive table per site (sites: (lat,lon) or [(lat,lon,label), ...])
print_shakermaker(self, sites, *, Qa=1000.0, Qb=1000.0)   # prints a ready-to-paste CrustModel snippet

# --- 6 plot methods, all return a Figure, all default to show=True/save_path=None ---
plot_profile(self, sites, *, figsize=(12,6.5), halfspace_extra=20.0, show=True, save_path=None)
plot_global_topo(self, sites=None, *, figsize=(11,5.5), show=True, save_path=None)
plot_regional_topo(self, sites, *, zoom_deg='auto', figsize=(9,7), show=True, save_path=None)
plot_regional_geological(self, sites, *, zoom_deg='auto', figsize=(11,7), show=True, save_path=None)
plot_global_velocity(self, sites=None, *, layer=5, figsize=(18,5), show=True, save_path=None)
plot_stacked_columns(self, sites, *, figsize=(9,6), show=True, save_path=None)
```

`sites` accepts: a single tuple `(lat, lon)` or `(lat, lon, "label")`, or a
list of several — `Crust1._norm` normalizes either form. With 1 site,
`plot_profile` draws layer-colored backgrounds; with >1, it draws one
overlaid curve per site.

`Crust1.BENCHMARK_SITES` ships 3 example sites already loaded: Santiago,
San Francisco, and Meyrin/CERN.

Importing `shakermaker.crust1` prints `CRUST1_CITATION` (a mandatory
citation to Laske et al. 2013).

`plot_regional_geological` requires the geological-type addon
(`CNtype1-1.txt` + `CNtype1_key.txt`); if it's not present, it raises
`RuntimeError("type addon missing (CNtype1-1.txt)")`.

## Minimal working example

### By hand

```python
from shakermaker.crustmodel import CrustModel

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)     # slow layer, 1 km
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)   # half-space (d=0)
print(crust)
```

### From a preset

```python
from shakermaker.cm_library.LOH import SCEC_LOH_1
crust = SCEC_LOH_1()
```

### From CRUST 1.0 (real pattern used in `examples/14_SFSI/example_documented.ipynb`)

```python
from shakermaker.crust1 import Crust1
from shakermaker.crustmodel import CrustModel

c1 = Crust1()
p = c1.profile_at(lat=40.85, lon=-124.13)   # near Samoa Beach, Humboldt Co., CA

crust = CrustModel(5)
# water and ice are SKIPPED on purpose (see Gotchas) -- only sediments + crust + mantle
crust.add_layer(p['thickness'][2] + p['thickness'][3] + p['thickness'][4],
                 p['vp'][2], p['vs'][2], p['rho'][2], 1000., 1000.)   # combined sediments (simplified)
crust.add_layer(p['thickness'][5], p['vp'][5], p['vs'][5], p['rho'][5], 1000., 1000.)  # upper crust
crust.add_layer(p['thickness'][6], p['vp'][6], p['vs'][6], p['rho'][6], 1000., 1000.)  # middle crust
crust.add_layer(p['thickness'][7], p['vp'][7], p['vs'][7], p['rho'][7], 1000., 1000.)  # lower crust
crust.add_layer(0.0, p['vp'][8], p['vs'][8], p['rho'][8], 1000., 1000.)                # mantle (half-space)
```

Or, simpler and less error-prone: let `print_shakermaker` generate the exact
snippet (with the water/ice layers already commented out and the
zero-thickness ones already excluded) and paste it:

```python
c1.print_shakermaker((40.85, -124.13, "Samoa Beach"), Qa=1000., Qb=1000.)
# prints a ready-to-copy `crust = CrustModel(N); crust.add_layer(...)` block
```

## Known gotchas

- **`modify_layer` uses `qp=`/`qs=`, not `gp=`/`gs=`.** Confirmed in the
  real signature (`crustmodel.py:88`). It's the easiest typo to make.
- **`d=0.0` is only valid on the last layer** (the half-space). `add_layer`
  does not explicitly validate this for intermediate layers — it's the
  caller's responsibility to respect the "surface → depth, half-space last"
  ordering.
- **Water and ice from CRUST 1.0 must be skipped.** `Crust1`'s own
  `_print_shakermaker` code (`crust1.py:369`) carries the literal comment
  `"water - skipped (ShakerMaker FK fails with Vs=0)"` — the FK engine
  cannot tolerate `Vs=0` (a water/fluid layer). If you build the
  `CrustModel` by hand from `profile_at(...)` instead of using
  `print_shakermaker`, you must manually exclude layers `k=0` (water) and
  `k=1` (ice) from the dict `profile_at` returns. The same pattern appears
  independently in `examples/14_SFSI/example_documented.ipynb` (water/ice
  layers commented out by hand with the note "ShakerMaker FK fails with
  Vs=0").
- **`get_layer(z)` is not "which layer contains z"**, it's "is there a
  layer interface at exactly this depth (± tol)". To get properties at an
  arbitrary depth inside a layer, use `properties_at_depths(z)`.
- **`split_at_depth` is a silent no-op** if an interface already exists
  near `z` (within `tol`) — it does not raise or warn.
- **`SCEC_LOH_1`/`SCEC_LOH_3` "no attenuation" doesn't mean truly infinite
  Q**, it means Q=10000 (the benchmark's standard numerical approximation,
  documented in `SCEC_LOH_1`'s own docstring).
- **CRUST 1.0 is a 1°×1° grid** — `cell_index`/`profile_at` return the
  property of the whole cell containing the point, not a smooth
  interpolation; two sites within the same degree of lat/lon give exactly
  the same profile.

## Combines with

- **`02_sources_and_stf.md`**: the `CrustModel` is passed together with the
  `FaultSource` to the `ShakerMaker(crust, fault, receivers)` constructor —
  it's mandatory in every simulation, no exceptions.
- **`03_ffsp_source.md`**: `FFSPSource` also takes a `CrustModel` as a
  mandatory argument (`crust_model=...`) for its own Fortran kernel.
- **`05_check_parameters.md`**: several of `check_parameters`'s
  recommendations (e.g. `Vs_min`, `Vp_surf`) are derived directly from
  `crust.a`/`crust.b` — a poorly-built `CrustModel` (layers in the wrong
  order, Vs=0) produces nonsensical downstream recommendations.
- **`09_sw4_export.md`**: `export_sw4`/`export_sw4_topo` use the same
  `CrustModel` to write the SW4 materials file.

## See also

- `examples/01_crustmodel/crustmodel_build.py`, `examples/01_crustmodel/crust1_sites.py`,
  `examples/01_crustmodel/notebooks/crustmodel.ipynb` — runnable examples.
- `examples/14_SFSI/example_documented.ipynb` — a full real-site case using CRUST 1.0.
- `examples/EXAMPLES_REFERENCE.md` (folder `examples/`, section "01. Crust Models") —
  file-by-file review of every example in this folder.
