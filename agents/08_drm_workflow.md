# DRM workflow — driving an OpenSees Domain Reduction Method boundary

## What this is

The Domain Reduction Method (DRM) excites a finite-element model with
free-field ground motion applied only at the boundary of a sub-domain
(a "DRM box"), rather than at every node. ShakerMaker's role is to compute
the free-field motion at every DRM-boundary node with the FK engine and
write it into a `.h5drm` file that OpenSees' `H5DRMLoadPattern` reads
directly. This file walks the concepts that don't belong to any single
class (`04_receivers.md` covers `DRMBox`/`SurfaceGrid`/`PointCloudDRMReceiver`
in isolation; `07_writers_persistence.md` covers `DRMHDF5StationListWriter`'s
schema) — this is the "how the pieces fit together for a real DRM campaign"
layer. For the positions/motion order inside the `.h5drm` and the OpenSees
matrix `T` (STKO Local X = (0, 1, 0), Local Y = (1, 0, 0) for a Z-up model),
read `12_coordinates_and_conventions.md` first.

## Source of truth

`shakermaker/sl_extensions/` (receiver geometry), `shakermaker/
slw_extensions/drmhdf5stationlistwriter.py` (writer), `shakermaker/
shakermaker.py::export_drm_geometry` (line 2206).

## The three receiver options for a DRM boundary

| Receiver | What it represents | When to use |
|---|---|---|
| `DRMBox` | A closed 3-D box (Bielak-style), auto-generated boundary nodes | You want ShakerMaker to generate the whole DRM boundary geometry itself, on a regular grid |
| `SurfaceGrid` (mode='filled'/'hollow') | A regular box, either full volume or shell-only | Same idea as `DRMBox` but with explicit control over fill mode; `'hollow'` is closer to what a real DRM boundary needs (only the shell) |
| `PointCloudDRMReceiver` | Boundary nodes **imported from an existing FEM mesh** | You already meshed the structure/soil domain elsewhere (e.g. in STKO/OpenSees) and need motion at *exactly* those node coordinates |

Every one of the three appends a **QA station** automatically (identified
internally by `station.metadata.get("name") == "QA"`) — a single reference
station, typically at the box center, meant for visual/numerical sanity
checks against a direct (non-DRM) computation at the same point, not as
part of the DRM boundary itself.

## Sizing a `DRMBox` (or any DRM-oriented grid)

The spacing rule seen throughout the repo (e.g.
`examples/legacy_examples/example2_drm.py`): pick element size from the
highest frequency you need to resolve and the slowest shear-wave velocity
in the model —

```python
dx = vs_min / fmax / 15   # km; "vs_min"/"fmax" in the same units
```

i.e. **≥15 points per shortest S-wavelength**. This is the DRM-specific
sibling of `check_parameters`'s more general FEM-mesh advice
(`lambda_min/n_per_wavelength`, default `n_per_wavelength=10`) — see
`05_check_parameters.md`. If you already ran `check_parameters`, prefer its
`dx_fem` recommendation (it accounts for the actual `Vp_surf`/CFL step too);
the `/15` rule above is a quick standalone estimate when you haven't.

## Two ways to drive a DRM box: geometry-only vs. real physics

1. **Geometry-only, no FK run — `model.export_drm_geometry(filename=
   "drm_geometry.h5drm")`**. Requires a `DRMBox`/`SurfaceGrid`/
   `PointCloudDRMReceiver` receiver (raises `TypeError` otherwise). Writes
   real station coordinates and `internal`/`external` flags, but only
   **synthetic** 2-sample linear-ramp (0→10) data for
   velocity/displacement/acceleration — near-instant, useful to check node
   count and geometry (e.g. visually in STKO) *before* spending real FK
   compute. Never use this data for anything physical.
2. **Real physics — run the FK engine (legacy `run()` or the OP pipeline)
   with `writer=DRMHDF5StationListWriter(...)`.** This is what actually
   produces usable motion. For anything beyond a toy box (hundreds+ nodes),
   use the OP pipeline (`run_nearest`/`gen_pairs`+`compute_gf`+`run_fast`,
   see `06_engine_run_modes.md`) — DRM boxes routinely have far more
   receivers than a legacy pair-by-pair run can handle economically, since
   most boundary nodes sit at very similar (horizontal distance, source
   depth, receiver depth) triples and dedupe heavily under the OP pipeline.

## Minimal working example — small DRM box, OP pipeline, real physics

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import Gaussian
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)  # d=0 -> half-space

sigma = 0.06
stf = Gaussian(t0=6 * sigma, freq=1 / sigma, M0=1e18 / 5e14 / 2)
src = PointSource([0, 0, 2.0], [0., 90., 0.], stf=stf)
fault = FaultSource([src], metadata={"name": "src"})

fmax, vs_min = 5.0, 2.0     # Hz, km/s
dx = vs_min / fmax / 15     # box element size (km)
drm = DRMBox([6, 8, 0], [4, 4, 2], [dx, dx, dx],
             metadata={"name": "drmbox"})   # NOTE: no `azimuth` kwarg

model = ShakerMaker(crust, fault, drm)
model.check_parameters(dt=0.01, nfft=4096, dk=0.1, tb=200, tmax=20)

writer = DRMHDF5StationListWriter("motions.h5drm")
model.run_nearest(stage='all', h5_database_name='gf_db.h5',
                   delta_h=0.04, delta_v_rec=0.005, delta_v_src=0.2,
                   dt=0.01, nfft=4096, dk=0.1, tb=200, tmax=20,
                   writer=writer, writer_mode='progressive')
```

Run with `mpiexec -n N python script.py` for a real-size box (see
`11_mpi_and_hpc.md`).

## Importing an existing FEM mesh boundary (`PointCloudDRMReceiver`)

```python
from shakermaker.sl_extensions import PointCloudDRMReceiver

stations = PointCloudDRMReceiver(
    point_cloud_file='drm_nodes.txt',   # TSV: Node_ID X Y Z Type (Type: internal/external)
    crd_scale=1 / 1e3,                  # e.g. FEM mesh in metres -> ShakerMaker km
    x0_fem=[0., 0., 0.],                # FEM-local origin
    drmbox_x0=[6.0, 8.0, 0.0],          # target center in ShakerMaker km coords
    metadata={"name": "fem_drm"})
```
`nstations` = number of rows in the file + 1 (the auto-appended QA
station). See `04_receivers.md` for the full parameter reference.

## Gotchas

- **`DRMBox` has no `azimuth` parameter** — its real signature is
  `DRMBox(pos, nelems, h, metadata=None)`. If you're porting an older
  script/notebook that passes `azimuth=`, drop it.
- **`export_drm_geometry`'s data is synthetic** — a 2-sample linear ramp,
  not physics. Only use it to sanity-check geometry/node count, never as
  simulation output.
- **Fixed bug (was: NaN in `/DRM_QA_Data`)**: NaN appeared partway through the record
  in the **horizontal** (E, N) components of the QA station when it sat exactly above
  the source (zero epicentral distance, a 0/0 in the Bessel terms of `subfk.f`). Fixed by
  commit `2a83ca6` (upstream PR #48); `/DRM_Data` was never affected.
- **`SurfaceGrid` + `DRMHDF5StationListWriter` is not always a "real" DRM
  boundary.** A single-plane `SurfaceGrid` (`mode='plane'`) paired with the
  DRM writer (seen in `examples/14_SFSI/Surface/surface_SSFI.py`) does not
  represent a genuine closed DRM boundary (which needs a box, not a plane)
  — it's sometimes used purely to get the `.h5drm` container format for
  compatibility with a downstream reader tool (`ShakerMakerResults`), not
  because the physics calls for it. Don't assume every `.h5drm` file was
  produced from a real DRM box; check what receiver type produced it.
- **Tolerance constants for the OP pipeline's Stage 0 (`delta_h`,
  `delta_v_rec`, `delta_v_src`) must be physically meaningful.** A working
  reference (`examples/08_drm/drm_loh1.py`) uses `delta_h=40m,
  delta_v_rec=5m, delta_v_src=200m` (via `_m=0.001` km-per-metre). A
  tolerance many orders of magnitude smaller than your coordinate precision
  (seen as a bug in two other example scripts, from an accidental extra
  `/1e12` factor) effectively disables Green's-Function reuse and defeats
  the point of the OP pipeline.
- **MPI robustness for large DRM campaigns**: see `11_mpi_and_hpc.md` for
  the historical Stage 2 hang bug, the still-partially-open "hang #2," and
  the cross-node HDF5 visibility-lag retry — all directly relevant once a
  DRM box grows past a handful of MPI ranks.

## Combines with

- `04_receivers.md` — full parameter reference for `DRMBox`, `SurfaceGrid`,
  `PointCloudDRMReceiver`.
- `07_writers_persistence.md` — `DRMHDF5StationListWriter`'s exact schema
  (`/DRM_Data`, `/DRM_QA_Data`, `/DRM_Metadata`).
- `06_engine_run_modes.md` — legacy `run()` for tiny debug boxes, OP
  pipeline for anything real-sized.
- `05_check_parameters.md` — always check before running; its FEM-mesh
  advice (`n_per_wavelength`, `courant`) is directly relevant to DRM box
  sizing.
- `11_mpi_and_hpc.md` — MPI launch pattern and known robustness issues at
  DRM-campaign scale.
- `RECIPES.md` — a full worked "DRM box + OP pipeline" and "PointCloudDRMReceiver
  + OP pipeline" recipe.

## See also

- `examples/08_drm/` — `drm_loh1.py` (full OP+MPI+DRM, also the bug repro),
  `drm_vs_direct.py` (sanity check), `export_drm_geometry.py`,
  `notebooks/drm.ipynb`.
- `examples/04_receivers/` — isolated construction of each receiver type.
- `examples/14_SFSI/DRM/drm.py` — real HPC/SLURM production script driving
  a `PointCloudDRMReceiver` from an exported FEM mesh.
