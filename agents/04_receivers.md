# Receivers (Station, StationList, DRMBox, SurfaceGrid, PointCloudDRMReceiver)

## What this is

Every receiver in ShakerMaker (a surface station, a DRM box, a QA grid, a
point from a FEM mesh) is, underneath, a `StationList` — a collection of
`Station` objects passed as the third argument to
`ShakerMaker(crust, fault, receivers)`. The 4 classes covered in this file
are the only ways to produce an object compatible with that position:

- `Station` + `StationList` — the base case: standalone stations, built by
  hand.
- `DRMBox` — a Bielak-style 3-D box, the canonical way to generate a
  regular DRM boundary (two layers of nodes, internal/external) without
  writing the geometry by hand.
- `SurfaceGrid` — a regular grid (a single plane, a filled volume, or just
  a box's shell), reusable both as a simple "DRM mesh" and to export a
  reference free-field motion.
- `PointCloudDRMReceiver` — same role as `DRMBox` but with node positions
  **imported** from a real FEM mesh (STKO), not generated.

## Source of truth

- `shakermaker/station.py` — `Station`, `StationObserver`, `interpolator()`.
- `shakermaker/stationlist.py` — `StationList`.
- `shakermaker/sl_extensions/DRMBox.py` — `DRMBox`, `Plane` (internal).
- `shakermaker/sl_extensions/SurfaceGrid.py` — `SurfaceGrid`.
- `shakermaker/sl_extensions/PointCloudDRMReceiver.py` — `PointCloudDRMReceiver`.

`DRMBox`, `SurfaceGrid`, and `PointCloudDRMReceiver` are **not re-exported**
from `shakermaker/__init__.py` — import them from the submodule:
`from shakermaker.sl_extensions import DRMBox, SurfaceGrid, PointCloudDRMReceiver`.

## Full API reference

### `Station(x=None, internal=False, metadata=None)`

Stores the response **in memory** (numpy arrays), unfiltered (filtering, if
requested, is applied afterward via metadata — the raw stored data is never
overwritten).

| Method/property | Signature | What it does |
|---|---|---|
| `x` | property | position `[x,y,z]` in km (x=North, y=East, z=Down positive) |
| `metadata` | property | metadata dict |
| `is_internal` | property | internal/external flag (relevant in DRM) |
| `add_to_response(z,e,n,t,tmin=0,tmax=100.)` | — | accumulates the response (sums if data already present — used to sum each subfault's contribution); tolerates `t[0]<0` (an acausal STF tail) |
| `get_response()` | `-> (z,e,n,t)` | returns the accumulated response |
| `attach(observer)` / `detach(observer)` | — | Observer pattern; `StationList` auto-attaches to every station it adds |
| `clear_response()` | — | frees `_z/_e/_n/_t` from memory and resets `_initialized`. Called internally by `run_fast` (Stage 2) in `writer_mode='progressive'` to keep RAM O(1) regardless of station count (`shakermaker/station.py:156-168`) |
| `add_greens_function(z,e,n,t,tdata,t0,subfault_id)` | — | stores the GF **only if** `metadata={"save_gf": True}` (opt-in, `station.py:170-183`). Used by the engine at `shakermaker.py:648` inside `run()` |
| `get_greens_functions()` | `-> dict` | `{subfault_id: (z,e,n,t,tdata,t0)}` |
| `save(npzfilename)` / `load(npzfilename)` | — | native `.npz` persistence, no `h5py` dependency |

**Metadata recognized by the engine** (keys `ShakerMaker` interprets, not
merely decorative):

| Key | Effect |
|---|---|
| `name` | human-readable label, appears in logs/writers |
| `filter_results=True` + `filter_parameters={"fmax": ...}` | post-process low-pass filter; the raw unfiltered data remains available |
| `save_gf=True` | enables `add_greens_function` (stores the GF per subfault — expensive in RAM for large faults, use only for debugging/validation) |

### `StationList(stations, metadata)`

| Method/property | What it does |
|---|---|
| `add_station(station)` | adds it and calls `station.attach(self)` |
| `get_station_by_id(id)` | raises `IndexError` if out of range |
| `nstations` / `is_finalized` | properties |
| `finalize()` | sets `is_finalized=True` (does not free anything on its own) |
| `__iter__` | iterates the `Station`s in insertion order |

This is the base class of `DRMBox`, `SurfaceGrid`, and
`PointCloudDRMReceiver` — each of the three is, in practice, a
`StationList` with geometry-generation logic added in its constructor.

### `DRMBox(pos, nelems, h, metadata=None)`

A Bielak-style 3-D box with **two layers of stations** (internal/external)
plus one QA station. **Has no `azimuth` argument** — the real signature
only accepts `pos, nelems, h, metadata`.

- `pos` = `[x,y,z]` box center, km.
- `nelems` = `[nx,ny,nz]` number of elements (cells) per axis.
- `h` = `[hx,hy,hz]` spacing per axis, km.

Exact geometry (`_create_DRM_stations`, `DRMBox.py:129-212`): 5 planes per
layer — `-Y, +Y, -X, +X, -Z` (the box is **open at the top**, `+Z`, because
`z` positive points down and the box top coincides with the free surface).
The internal shell measures `[Nx·hx, Ny·hy, Nz·hz]`; the external shell is
one cell larger in each lateral direction and one deeper:
`[(Nx+2)·hx, (Ny+2)·hy, (Nz+1)·hz]`.

At the end of the constructor a **QA station** is added automatically
(`metadata={"name": "QA"}`, `internal=True`) at the box center `pos` — a
reference station that reproduces the free-field solution at that point, to
validate the DRM mesh against a direct calculation (see `08_drm_workflow.md`).

**Practical sizing rule** (seen in
`examples/legacy_examples/example2_drm.py`, not enforced by the code):

```python
dx = vs / fmax / 15   # vs = site's minimum Vs (km/s), fmax = usable max frequency (Hz)
```

Gives ~15 points per minimum wavelength (`lambda_min = vs/fmax`). Same idea
as `n_per_wavelength` in `check_parameters` (`05_check_parameters.md`), but
applied to the DRM element size rather than to the downstream FEM mesh —
they're the same number by design, both being "points per wavelength," just
for different meshes.

`nplanes` / `planes` expose the internal `Plane` objects (advanced use /
introspection, not needed for a normal workflow).

### `SurfaceGrid(x0, nelems, h, mode='plane', plane_x=None, plane_y=None, plane_z=None, metadata=None)`

A regular grid of stations, compatible with the same interface as `DRMBox`
(shares the `drmbox_*` metadata keys, see below). `h` can be a scalar
(same spacing on all 3 axes) or `[hx,hy,hz]`.

Three modes (`SurfaceGrid.py:104-192`):

| `mode` | What it generates | When to use it |
|---|---|---|
| `'plane'` | **a single plane** — requires exactly one of `plane_x`/`plane_y`/`plane_z` (error if 0 or more than 1 is given). `plane_z=Z0` → XY plane at that elevation (uses `nx,ny`); `plane_y=Y0` → XZ plane (uses `nx,nz`); `plane_x=X0` → YZ plane (uses `ny,nz`). | a surface (free-field) motion field, or a reference cross-section |
| `'filled'` | full 3-D grid, `(nx+1)×(ny+1)×(nz+1)` points | a verification/QA volume, or a dense field for 3-D post-processing |
| `'hollow'` | only the box's faces (like a `DRMBox` but without the internal/external double layer — a single shell) | a simple closed boundary without `DRMBox`'s internal/external detail |

Always appends one QA station at the end (`metadata={"name":"QA"}`,
`internal=False`, at the center `x0`) — same convention as `DRMBox`.

**Deliberate metadata note** (`SurfaceGrid.py:193-212`, explicit comment in
the code): `SurfaceGrid` writes its bounding-box keys with the `drmbox_*`
prefix (not `surfacegrid_*`) **on purpose**, so that
`ShakerMaker.export_drm_geometry` (which looks for `drmbox_*` keys) works
identically for a `SurfaceGrid` as for a `DRMBox`. `nelems` and `mode` do
keep the `surfacegrid_*` prefix, since they have no `DRMBox` equivalent.

### `PointCloudDRMReceiver(point_cloud_file, crd_scale, x0_fem, drmbox_x0, metadata=None)`

Same role as `DRMBox` but positions come from a text file (TSV) exported
from a FEM model (typically STKO), not from a generated grid.

**Exact expected file format**: tab-separated, columns
`Node_ID  X  Y  Z  Type`, with `X,Y,Z` in the FEM's own units (typically
mm) and `Type` in `{'internal', 'external'}` (case-insensitive,
`.str.lower()` is applied).

**Coordinate transform — fixed, not configurable**
(`PointCloudDRMReceiver.py:62-79`):

```
T = [[0, 1, 0],
     [1, 0, 0],
     [0, 0, -1]]

xyz_km = T^-1 . ((xyz_fem - x0_fem) * crd_scale) + drmbox_x0
```

This is the fixed STKO convention (swaps X/Y, flips Z) — **not a generic
transform**, it is *hard-coded* in the class. If the point cloud comes from
a different tool with a different axis convention, this class won't work
as-is without adapting `T`.

- `crd_scale`: unit scale factor from FEM units to km (mm→km: `1/1e6`,
  since `1 mm = 0.001 m = 1e-6 km`).
- `x0_fem`: reference origin in FEM units (e.g. the top-center of the DRM
  box as defined in STKO).
- `drmbox_x0`: DRM domain center in ShakerMaker coordinates (km) — the same
  value you'd pass as `pos` to a `DRMBox`.

Automatically appends one QA station at `drmbox_x0` (`internal=True`,
`metadata={"name":"QA"}`) — final `nstations` = file rows + 1.

## Minimal working example

### Single station

```python
from shakermaker.station import Station
from shakermaker.stationlist import StationList

sta = Station([6, 8, 0], metadata={"name": "S1"})
stations = StationList([sta], {})
```

### DRMBox sized by the practical rule

```python
from shakermaker.sl_extensions import DRMBox

vs, fmax = 1.0, 5.0          # km/s, Hz
dx = vs / fmax / 15          # ~13 m
drm = DRMBox(pos=[6, 8, 1.0], nelems=[10, 10, 4], h=[dx, dx, dx],
             metadata={"name": "drmbox"})
```

### SurfaceGrid as a free-field surface plane

```python
from shakermaker.sl_extensions import SurfaceGrid

grid = SurfaceGrid(x0=[6, 8, 0], nelems=[20, 20, 0], h=0.05,
                    mode='plane', plane_z=0.0, metadata={"name": "free_field"})
```

### PointCloudDRMReceiver from an STKO export

```python
from shakermaker.sl_extensions import PointCloudDRMReceiver

drm = PointCloudDRMReceiver(
    point_cloud_file='drm_nodes.txt',
    crd_scale=1 / 1e6,                    # mm -> km
    x0_fem=[22000.0, 15500.0, 0.0],        # FEM origin (mm)
    drmbox_x0=[6.0, 8.0, 0.0],             # center in ShakerMaker (km)
    metadata={"name": "fem_drm"})
```

## Known gotchas

- **`DRMBox` does not accept `azimuth`.** If you're porting code from an
  older reference that passes that kwarg, drop it.
- The `DRMBox` shell is **open at `+Z`** (the free surface) — it does not
  generate a top face; only 5 planes per layer, not 6.
- `DRMBox`/`PointCloudDRMReceiver`'s QA station has `internal=True`;
  `SurfaceGrid`'s has `internal=False` — a minor inconsistency across
  classes, but the name `"QA"` is uniform across all three and is what
  `DRMHDF5StationListWriter` uses to split `/DRM_Data` from `/DRM_QA_Data`
  (`shakermaker/slw_extensions/drmhdf5stationlistwriter.py:203`,
  `is_QA = station.metadata.get("name") == "QA"`).
- `SurfaceGrid(mode='plane')` requires **exactly one** of
  `plane_x/plane_y/plane_z` — passing 0 or 2+ raises `ValueError`. Passing
  any of the three in `'filled'`/`'hollow'` mode also raises `ValueError`
  (they're mutually exclusive with `mode != 'plane'`).
- `PointCloudDRMReceiver`'s transform is fixed to the STKO convention (mm,
  Z-up → km, Z-down with X/Y swapped) — it won't work for point clouds
  using a different axis convention without modifying the class.
- `add_to_response` **sums** successive contributions (does not overwrite)
  — this is how the response from multiple subfaults accumulates on the
  same station. If you call it twice intending to "restart," call
  `clear_response()` first.
- `save_gf=True` in a station's metadata with **many subfaults** (e.g. an
  FFSP source with hundreds of subfaults) can consume a lot of RAM — this
  is a debugging tool, not for production.

## Combines with

- **`06_engine_run_modes.md`** — `DRMBox`/`SurfaceGrid`/
  `PointCloudDRMReceiver` work with both the legacy engine
  (`model.run(...)`) and the OP pipeline (`model.run_nearest(...)`); for
  geometries with many stations (a real DRM boundary), the OP pipeline is
  practically mandatory for performance.
- **`07_writers_persistence.md`** — all 3 DRM-oriented receivers inherit
  from `StationList` and in principle accept any writer, but only produce a
  **physically valid DRM boundary** for OpenSees when combined with
  `DRMHDF5StationListWriter`. Using `HDF5StationListWriter` with a `DRMBox`
  is valid but only makes sense as a reference plane/box, not as a real DRM
  boundary condition.
- **`05_check_parameters.md`** — `n_per_wavelength` in `check_parameters`
  guides the downstream FEM mesh; the `dx=vs/fmax/15` rule for
  `DRMBox`/`SurfaceGrid` is the same idea applied to the receiver mesh.
- **`08_drm_workflow.md`** — the full DRM flow: sizing the box, exporting
  geometry only (`export_drm_geometry`, no FK run) vs. running the full
  engine, interpreting `/DRM_Data` vs `/DRM_QA_Data`.
- **`RECIPES.md`** — concrete combinations: single station + nearest
  method, DRM + nearest method, DRM + legacy, SurfaceGrid + geometry-only
  export.

## See also

- `examples/04_receivers/` — `single_station.py`, `drmbox.py`,
  `surface_grid.py`, `pointcloud_drm.py`,
  `notebooks/receivers_geometry.ipynb`.
- `examples/08_drm/` — real `DRMBox` usage in a full MPI DRM workflow.
- `examples/legacy_examples/example2_drm.py` — origin of the
  `dx=vs/fmax/15` rule.
- `examples/EXAMPLES_REFERENCE.md` (sections 04, 08, 14) — observed
  behavior of these scripts, already verified in an earlier review this
  session.
