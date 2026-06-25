# SW4 export

ShakerMaker can hand a complete model, crust, sources, receivers, and
optionally topography, to **SW4**, the LLNL/CIG 3-D finite-difference solver.
This is the other half of the coupling story: where [H5DRM](drm.md) injects FK
motions into a *local* FE model, the SW4 export sets up a *regional* 3-D
finite-difference run with the same sources and medium.

## What the exporter does

ShakerMaker is georeferenced in **kilometres**; SW4 runs in a **local cartesian
box in metres**. The exporter bridges the two:

1. **Resolves the SW4 domain** (extent + origin) from the union of sources,
   stations, and topography.
2. Builds a **pure-translation** transform between the frames, no rotation.
3. Writes the SW4 input file (`.in`) in SW4 local metres, plus a slip-rate
   file per source.
4. Packs everything into a single **HDF5 transport bundle** alongside a
   standalone unpacking script.

```python
P_sw4_m         = P_shakermaker_m - domain_origin_m
P_shakermaker_m = P_sw4_m + domain_origin_m
```

The offset is stored both ways in the package so any downstream tool can return
to ShakerMaker coordinates without guessing.

!!! note "Coordinate frames"
    ShakerMaker uses `x = North, y = East, z = down`. SW4 topography files use
    `x = East, y = North`, so the exporter swaps those columns on read. Keep
    this in mind if you supply your own topography.

## Entry points

Both build a `SW4ExportConfig`, hand it to `SW4Exporter`, and call `.write()`.

```python
# Flat domain
model.export_sw4(
    path="/run/dir",
    h=50,                       # SW4 grid spacing (m)
    size_domain=[20000, 20000, 10000],   # box size (m); auto from geometry if None
    tmax=60,                    # simulation duration (s)
    plot_geometry=True,         # preview the layout before running
)

# With topography (SW4 cartesian file, x=East y=North)
model.export_sw4_topo(
    path="/run/dir",
    topo_file="topo.txt",
    topo_zmax=1500,             # max topographic elevation (m)
    write_topography_z0_stations=True,
)
```

### Key configuration

| Argument | Units | Meaning |
|---|---|---|
| `h` | m | SW4 grid spacing (the finest resolvable wavelength scales with it) |
| `size_domain` | m | `[x, y, z]` box size; if `None`, derived from the geometry |
| `tmax` | s | simulation duration |
| `m0` | – | moment scaling applied to the SW4 sources |
| `supergrid_gp` | grid pts | width of the absorbing super-grid layer |
| `station_prefix` | – | prefix for SW4 receiver records |
| `topo_file`, `topo_zmax` | –, m | topography input and its cap (`export_sw4_topo`) |
| `plot_geometry`, `plot_geometry_sw4` | bool | PyVista previews in each frame |
| `h5_export_name` | – | name of the transport bundle (`sw4_package.h5`) |

## What `.write()` produces

```
shakermakerexports/
    sw4_package.h5            single HDF5 transport bundle
    unpack_sw4_package.py     standalone unpacker, no ShakerMaker dependency
```

The actual SW4 run tree appears only after you **unpack** the bundle:

```python
from shakermaker.sw4_exporter import unpack_sw4_package_h5
unpack_sw4_package_h5("shakermakerexports/sw4_package.h5", "sw4/")
```

```
sw4/
    shakermaker2sw4.in            the SW4 input file
    sources/source_NNNNNN.txt     per-source slip-rate files
    sources/sources_summary.csv
    topo/<name>_local.txt         only if topography was exported
```

The two-step design (bundle → unpack) keeps the transport portable: you move
one `.h5` to the cluster, unpack it there, and run SW4, the unpacker has no
ShakerMaker dependency.

## Receiver families

The exporter tags each receiver record by `kind`, so SW4 records the right
quantity at the right place:

| Family | Where |
|---|---|
| `shakermaker` | the ShakerMaker stations, in the SW4 box |
| `shakermaker_surface` | the same, projected to the free surface |
| `topography_surface` | on the topographic surface |
| `topography_to_z0` | topographic nodes mapped to `z = 0` |
| `sw4_domain` | a regular grid spanning the SW4 box |

DRM-aware receiver lists also carry a QA station (`is_qa=True`) for validation.

## The round trip: SW4 → H5DRM

The reverse direction closes the loop with [DRM](drm.md): after an SW4 run, the
motions on a box can be turned back into an `.h5drm` for a local OpenSees
model. The helper is `examples/09_sw4_export/build_h5drm_from_sw4_case.py`
(`build_h5drm_from_sw4_case(...)`), which handles the SW4-local-m ↔
ShakerMaker/UTM-km conversion.

## Public API (`shakermaker.sw4_exporter`)

| Symbol | Role |
|---|---|
| `SW4Exporter` | orchestrator (`.write()`) |
| `SW4ExportConfig` | dataclass of every knob |
| `unpack_sw4_package_h5(package, output_dir)` | explode the bundle into the SW4 tree |

See `shakermaker/sw4_exporter/README.md` for the full HDF5 layout.

## Reference

[DRM](drm.md) · [Outputs & writers](outputs.md) · [ShakerMaker engine API](../api/shakermaker.md)
