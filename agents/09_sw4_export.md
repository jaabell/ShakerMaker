# SW4 Export

## What this is

`shakermaker/sw4_exporter/` bridges a ShakerMaker model to a run of **SW4** (the
finite-difference wave-propagation code) and back. SW4 and ShakerMaker share
the same frame (`x = North, y = East, z = down`); how SW4 records compare with
ShakerMaker outputs is in `12_coordinates_and_conventions.md`. ShakerMaker never runs SW4 itself —
SW4 is an external tool the user must install/compile separately. This subpackage only:

1. Writes SW4 input files + a compact HDF5 transport bundle (`ShakerMaker.export_sw4(...)`
   or `.export_sw4_topo(...)`, no FK core run, no SW4 run).
2. Lets a downstream tool unpack that bundle into a real SW4 case directory
   (`unpack_sw4_package_h5(...)`).
3. After the user has run SW4 externally, reassembles SW4's own `.txt` receiver output
   plus the same bundle into a `.h5drm` file for an OpenSees DRM boundary condition
   (`build_h5drm_from_sw4_case(...)`).

Use this when you want to cross-validate the FK engine against a finite-difference
solution, or when you need topography / full 3-D heterogeneity that the 1-D layered FK
method can't represent, and OpenSees needs the resulting motions as `.h5drm`.

## Source of truth

- `shakermaker/shakermaker.py`: `ShakerMaker.export_sw4(...)` (~line 2297),
  `ShakerMaker.export_sw4_topo(...)` (~line 2368).
- `shakermaker/sw4_exporter/README.md` — the authoritative description of the package
  format, coordinate convention, and module map. Read it before touching this subpackage.
- `shakermaker/sw4_exporter/h5drm_from_sw4.py` — `build_h5drm_from_sw4_case(...)`
  (defined **twice** in this file — lines 191 and 581; verify which one is actually
  exported/used if you need to trace behavior beyond this doc's summary, e.g. via
  `python -c "from shakermaker.sw4_exporter import build_h5drm_from_sw4_case as f; import inspect; print(inspect.getsourcefile(f))"`).
- `shakermaker/sw4_exporter/{exporter,config,coordinates,grid,materials,sources,
  receivers,topography,input_writer,package_h5,geometry_plot}.py` — one module per
  concern, listed in `references/api.md` of the shakemaker-skill.
- Worked examples: `examples/09_sw4_export/` (see `examples/EXAMPLES_REFERENCE.md` for a
  file-by-file breakdown), `examples/13_shakermaker_sw4/` (cross-validation against a
  real SW4 run).

**Stale reference in the README, verified**: `shakermaker/sw4_exporter/README.md`
mentions `examples/sw4_2_h5drm.py` and `examples/sw4_export_smoke_test.py` — **neither
file exists in this repo** (checked directly). These are leftover paths from before the
2026-06-07 reorganization of `examples/` into numbered topic folders. The real,
current equivalents are `examples/09_sw4_export/build_h5drm_from_sw4_case.py` and
`examples/09_sw4_export/package_h5_roundtrip.py`.

## Full API reference

### `ShakerMaker.export_sw4(...)` — no topography

```python
model.export_sw4(
    path=None,                    # output dir, default os.getcwd()
    h=50,                         # SW4 grid spacing, metres
    size_domain=None,             # [x, y, z] metres; None per-axis = auto from geometry
    tmax=50,                      # SW4 simulation end time, s
    m0=1,                         # moment scale factor written to SW4 source lines
    fileio_path="shakermaker2sw4_fileio",   # SW4 result dir name (relative)
    supergrid_gp=30, supergrid_pad_gp=10,   # absorbing supergrid sizing (grid points)
    interface_blocks=True, interface_block_delta=1.0,   # SW4 `block` per crust interface
    refine_fmax=None,             # Hz; if set, derive mesh refinement from stratigraphy
    refine_n_per_wavelength=10.0, # points-per-wavelength target when refine_fmax is set
    refine_round_zmax="outward",  # rounding rule for refinement zmax lines
    station_prefix="sf",          # SW4 receiver name prefix
    shakermaker_stations=True,    # write ShakerMaker's own StationList as SW4 receivers
    domain_sw4=False, domain_sw4_size=None,   # add a regular DRM-style domain grid
    plot_geometry=False, plot_geometry_sw4=False,   # PyVista viewer (needs pyvista)
    plot_stratigraphy=False, stratigraphy_flat=False,
    stratigraphy_max_points_per_layer=300000,
    stratigraphy_max_display_depth_m=None,
    h5_export_name="sw4_package.h5",
)
```

Domain is built from the union of sources + stations. `refine_fmax` is the recommended
way to size the mesh: instead of forcing a uniform `h` fine enough for the *softest*
layer everywhere (expensive), it derives per-layer SW4 `refinement zmax=...` lines from
the crust stratigraphy so shallow soft layers get a finer local mesh — see
`shakermaker.sw4_exporter.refinement.compute_layer_refinement`.

### `ShakerMaker.export_sw4_topo(...)` — with Cartesian topography

Same parameters as `export_sw4`, plus:

```python
topo_file=None,                       # path to a local Cartesian topo file (required)
topo_zmax=None,                       # max elevation used for domain sizing
write_topography_z0_stations=False,   # also add depth=0 receivers at every topo node
shakermaker_stations=False,           # NOTE: default differs from export_sw4 (True there)
shakermaker_stations_to_surface=False,# force ShakerMaker stations to the topo surface
domain_sw4=False, domain_sw4_size=None,
```

Domain is built from topography **plus** model geometry. `plot_geometry` shows the
original (ShakerMaker) coordinates; `plot_geometry_sw4` shows the local SW4 box.

### `unpack_sw4_package_h5(package, output_dir)`

Recreates the `sw4/` file tree on disk from the embedded `sw4_package.h5` (which itself
holds every text payload gzipped, keyed by relpath — `SW4Exporter.write()` never touches
`sw4/` directly, only the HDF5 bundle + a standalone unpacker script).

### `build_h5drm_from_sw4_case(...)` — after SW4 has actually run

```python
from shakermaker.sw4_exporter import build_h5drm_from_sw4_case

build_h5drm_from_sw4_case(
    case_path,                    # same `path` given to export_sw4/export_sw4_topo
    package_h5=None,              # auto-detected in <case_path>/shakermakerexports/ if None
    output_name="motions.h5drm",  # written into <case_path>/shakermakerexports/
    use_filter=False,             # ObsPy bandpass; ObsPy only imported if True
    freqmin=0.25, freqmax=10.0, corners=4, zerophase=True,
    move_2_shakermaker_coor=False,   # False: SW4 local km; True: ShakerMaker/UTM km
)
```

`case_path` must contain `sw4/` (SW4's own result `.txt` files, one per receiver, under
`sw4/<fileio_path>/`) and `shakermakerexports/` (the compact package this reads for
geometry). If a receiver's `.txt` file is missing, that row is filled with **zeros** and
a `WARNING` is printed — it does not raise. Component mapping is stored as metadata
`component_map`: `E=SW4_Y, N=SW4_X, Z=SW4_Z(down+)`.

## Coordinate convention (exact)

ShakerMaker is the master frame: every source, station, and topography node is stored in
ShakerMaker coordinates (georeferenced, **kilometres**). SW4 only understands a local
Cartesian box whose origin sits at one corner, in **metres**. The exporter:

1. Resolves the SW4 box (extent + origin) from the union of sources, stations, and
   topography.
2. Builds a `CoordinateTransform(domain_origin_m)` that is a **pure translation** — no
   rotation:
   ```
   P_sw4_m          = P_shakermaker_m - domain_origin_m
   P_shakermaker_m  = P_sw4_m + domain_origin_m
   ```
3. Writes the SW4 `.in` file in SW4 local metres.
4. Stores the offset both directions, in both metres and kilometres, inside the package
   (`/coordinates/sw4_origin_in_shakermaker_m`, `/coordinates/shakermaker_to_sw4_offset_m`)
   so any downstream tool can convert back without guessing.

**Topography only**: SW4's topo-file convention is `x=East, y=North`; ShakerMaker uses
`x=North, y=East`. `rotate_topography_to_shakermaker` swaps those two columns on read;
`rebuild_cartesian_topography` then re-sorts the grid to row-major order.

## HDF5 package layout (`sw4_package.h5`)

One bundle, always the same structure — **there is no separate "compact" format**;
`manifest`, `config`, `coordinates`, `crust`, `stations`, `sw4_input`, `sources`,
`topography`, `receivers`, `drm_template`, `files` (every text file the SW4 case needs,
gzipped). Attributes: `package_version="1.0"`, `generator="ShakerMaker.sw4_exporter"`,
`purpose="transport_unpack_to_sw4_files"` (used by auto-detection).

Receiver rows are tagged by `kind`: `shakermaker` (true z), `shakermaker_surface`
(forced z=0), `topography_surface` (one depth=0 per topo node), `topography_to_z0`
(vertical fill), `sw4_domain` (regular DRM-style grid). DRM-aware receiver lists
(`DRMBox`, `SurfaceGrid`, `PointCloudDRMReceiver`) carry a trailing QA station tagged
`is_qa=True`; a plain `StationList` has no QA station, and the bounding-box centre is
used as a fallback (`/drm_template/qa_xyz_km`).

## Minimal working example

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Gaussian

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 1000., 1000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 1000., 1000.)

src = PointSource([0, 0, 2], [0, 90, 0], stf=Gaussian(t0=0.36, freq=5.0, M0=1e17))
fault = FaultSource([src], metadata={"name": "src"})
stas = StationList([Station([5, 5, 0], metadata={"name": "S1"})], {})

model = ShakerMaker(crust, fault, stas)

# Step 1 — write SW4 inputs + package (no FK run, no SW4 run here)
model.export_sw4(path="./sw4_case", h=100.0, size_domain=[20000, 20000, 10000], tmax=20.0)

# Step 2 — (outside Python) run SW4 against the unpacked case:
#   python sw4_case/shakermakerexports/unpack_sw4_package.py
#   <sw4-binary> sw4_case/sw4/shakermaker2sw4.in

# Step 3 — after SW4 finishes, rebuild a .h5drm from its .txt output:
from shakermaker.sw4_exporter import build_h5drm_from_sw4_case
build_h5drm_from_sw4_case("./sw4_case")   # -> sw4_case/shakermakerexports/motions.h5drm
```

## Known gotchas

- **SW4 is never run by this repo.** `export_sw4`/`export_sw4_topo` only write inputs;
  the user must install SW4 separately and invoke it themselves against the unpacked
  `.in` file.
- **No "compact" vs "full" package format exists.** Any documentation (including one
  example folder's own README) that calls the bundle "compact" is using informal
  language for "one summary file" — the actual HDF5 structure is always the same full
  bundle described above.
- **`shakermaker_stations` default differs between the two entry points**: `True` in
  `export_sw4`, `False` in `export_sw4_topo` — check explicitly if your topo export needs
  ShakerMaker's own stations written as SW4 receivers.
- **`build_h5drm_from_sw4_case` silently zero-fills missing receivers.** A `.h5drm` built
  before SW4 finished writing all stations will look complete but contain zeros for the
  missing ones — only a printed `WARNING` flags it.
- **`shakermaker/sw4_exporter/README.md` has stale example paths** (see "Source of
  truth" above) — trust `examples/09_sw4_export/` on disk over that README's file table.

## Combines with

- `01_crust_model.md` — the crust becomes SW4 `block` material lines (`materials.py`).
- `04_receivers.md` — `DRMBox`/`SurfaceGrid`/`PointCloudDRMReceiver` receiver lists get a
  QA station tagged in the package and are the natural target for a later `.h5drm`
  rebuild; a plain `StationList` also works but has no real QA point.
- `07_writers_persistence.md` — the resulting `motions.h5drm` has the same
  `DRM_Data`/`DRM_QA_Data`/`DRM_Metadata` shape family as `DRMHDF5StationListWriter`'s
  output, so downstream consumers (OpenSees `H5DRMLoadPattern`, `ShakerMakerResults`)
  treat them the same way.
- `RECIPES.md` — "SW4 export → run externally → rebuild .h5drm" recipe.

## See also

- `examples/09_sw4_export/` and `examples/13_shakermaker_sw4/` (full script/notebook
  breakdown in `examples/EXAMPLES_REFERENCE.md`).
- `shakermaker/sw4_exporter/README.md` for the package layout table and module map.
