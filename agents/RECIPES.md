# Recipes — combining ShakerMaker's building blocks

## What this is

Every ShakerMaker script is an assembly of independent choices:

1. **Crust** (`01_crust_model.md`): hand-built `CrustModel`, a `cm_library`
   preset, or a `Crust1` (CRUST 1.0) lookup.
2. **Source** (`02_sources_and_stf.md` / `03_ffsp_source.md`): one or more
   `PointSource`s with a standard STF, or a stochastic `FFSPSource` bridged
   to `PointSource`s by hand.
3. **Receiver** (`04_receivers.md`): `Station`/`StationList`, `DRMBox`,
   `SurfaceGrid`, or `PointCloudDRMReceiver`.
4. **Run tier** (`06_engine_run_modes.md`): legacy `run()`, or the OP
   pipeline (`gen_pairs`/`compute_gf`/`run_fast`/`run_nearest`).
5. **Writer** (`07_writers_persistence.md`): none (in-memory), plain
   `HDF5StationListWriter`, or `DRMHDF5StationListWriter`.
6. **Export target** (`09_sw4_export.md`): none (stay in ShakerMaker), or
   hand off to SW4 and rebuild a `.h5drm` afterward.

Almost every combination of these six is valid — that's the point of this
file: instead of re-deriving a new script from first principles every time,
pick one row/option from each dimension below, then use the closest worked
recipe as a template. **Always call `check_parameters(...)` first** in every
recipe (`05_check_parameters.md`) — omitted below only for brevity.

## Compatibility matrix

| Receiver | Legacy `run()` | OP pipeline | Best writer | Notes |
|---|---|---|---|---|
| `Station`/`StationList` | ✅ common | ✅ common (many stations) | `HDF5StationListWriter` or none | Simplest case |
| `DRMBox` | ✅ debug/small only | ✅ **recommended** for real size | `DRMHDF5StationListWriter` | See `08_drm_workflow.md` |
| `SurfaceGrid` (`plane`/`filled`/`hollow`) | ✅ small | ✅ **recommended** for real size | `HDF5StationListWriter` (generic array) or `DRMHDF5StationListWriter` (if feeding a DRM-shaped consumer) | `'hollow'` is closest to a real DRM shell; `'plane'`+DRM-writer is a format-convenience pattern, not a real DRM boundary — see `08_drm_workflow.md` gotchas |
| `PointCloudDRMReceiver` | ⚠️ only for tiny meshes | ✅ **required** in practice (meshes are large) | `DRMHDF5StationListWriter` | Needs MPI at real scale — see `11_mpi_and_hpc.md` |

| Source | Compatible with | Notes |
|---|---|---|
| `PointSource` + standard STF | any receiver, any run tier | The default case |
| `FaultSource` of many `PointSource`s (hand-built grid) | any receiver, any run tier | See `02_sources_and_stf.md` |
| `FFSPSource` → manual `FaultSource` bridge | any receiver, any run tier | Bridge is always hand-written (`03_ffsp_source.md`); OP pipeline's Stage 0-1/Stage 2 split is what makes multi-realization ensembles cheap |

Any crust option (`01_crust_model.md`) combines freely with any of the
above — crust choice is orthogonal to receiver/writer/run-tier choice.

## Recipe 1 — Single station, quickest possible check

**When**: sanity-checking a source/crust, no persistence needed.

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Gaussian
from shakermaker.tools.plotting import ZENTPlot

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)

sigma = 0.06
src = PointSource([0, 0, 4], [90, 90, 0],
                   stf=Gaussian(t0=6*sigma, freq=1/sigma, M0=1e18/5e14/2))
fault = FaultSource([src], metadata={"name": "mainshock"})
sta = Station([0, 4, 0], metadata={"name": "S1"})
stas = StationList([sta], {})

model = ShakerMaker(crust, fault, stas)
model.check_parameters(dt=0.01, nfft=4096, dk=0.1, tb=200, tmax=30)
model.run(dt=0.01, nfft=4096, dk=0.1, tb=200, tmax=30)
ZENTPlot(sta, xlim=[0, 30], show=True)
```
No writer, no HDF5 — result lives only in `sta.get_response()`.

## Recipe 2 — Many stations, OP pipeline, persisted to HDF5

**When**: a modest StationList (tens to low-hundreds of stations), want a
reusable Green's-Function database and a portable output file.

```python
from shakermaker.slw_extensions import HDF5StationListWriter
# ... crust/fault/stations built as in Recipe 1, but `stations` = many Station objects

model = ShakerMaker(crust, fault, stations)
model.check_parameters(dt=0.02, nfft=4096, dk=0.15, tb=500, tmax=40)

writer = HDF5StationListWriter("output.h5")
model.run_nearest(stage='all', h5_database_name='gf_db.h5',
                   dt=0.02, nfft=4096, dk=0.15, tb=500, tmax=40,
                   writer=writer, writer_mode='progressive')
```
Produces `gf_db_map.h5` + `gf_db_gf.h5` (Green's-Function database, reusable
for a different writer/output run against the *same* geometry) and
`output.h5` (`/Data/*` + `/Metadata/*` schema — see `07_writers_persistence.md`).

## Recipe 3 — DRM box for OpenSees, real size, MPI

**When**: a proper DRM boundary, hundreds+ of nodes, meant to drive a real
FE model. See `08_drm_workflow.md` for the full walkthrough and sizing
rule (`dx = vs_min/fmax/15`) — this is the canonical combination
(`DRMBox` + OP pipeline + `DRMHDF5StationListWriter`) documented there.

```bash
mpiexec -n 16 python drm_campaign.py
```

## Recipe 4 — Import a FEM mesh boundary (`PointCloudDRMReceiver`) + OP pipeline

**When**: the structure/soil domain was already meshed elsewhere (e.g.
STKO) and you have its boundary-node coordinates as a text file.

```python
from shakermaker.sl_extensions import PointCloudDRMReceiver
from shakermaker.slw_extensions import DRMHDF5StationListWriter

stations = PointCloudDRMReceiver(
    point_cloud_file='drm_nodes.txt', crd_scale=1/1e3,
    x0_fem=[0., 0., 0.], drmbox_x0=[6.0, 8.0, 0.0],
    metadata={"name": "fem_drm"})

model = ShakerMaker(crust, fault, stations)
model.check_parameters(dt=0.005, nfft=8192, dk=0.4, tb=400, tmax=60)
writer = DRMHDF5StationListWriter('motions.h5drm')
model.run_nearest(stage='all', h5_database_name='gf.h5',
                   delta_h=0.04, delta_v_rec=0.005, delta_v_src=0.2,
                   dt=0.005, nfft=8192, dk=0.4, tb=400, tmax=60,
                   writer=writer, writer_mode='progressive',
                   verbose=False, showProgress=True)
```
Launch under SLURM/MPI for a real mesh (thousands of nodes) — see
`11_mpi_and_hpc.md` for a full SLURM script template.

## Recipe 5 — Geometry-only DRM preview (no FK compute)

**When**: you just want to sanity-check node count / box geometry (e.g. in
STKO) before spending real compute.

```python
model = ShakerMaker(crust, fault, drm_or_surfacegrid_or_pointcloud_receiver)
model.export_drm_geometry("preview.h5drm")   # near-instant; data is a synthetic ramp
```

## Recipe 6 — FFSP: single realization → legacy run

**When**: exploring one stochastic finite-fault realization end to end,
model small enough for the legacy engine.

```python
from shakermaker.ffspsource import FFSPSource
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import SRF2

source = FFSPSource(id_sf_type=8, freq_min=0.01, freq_max=24.0,
    fault_length=30.0, fault_width=16.0, x_hypc=15.0, y_hypc=8.0,
    depth_hypc=8.0, xref_hypc=0.0, yref_hypc=0.0, magnitude=6.0,
    fc_main_1=0.09, fc_main_2=3.0, rv_avg=3.0, ratio_rise=0.3,
    strike=358.0, dip=40.0, rake=113.0, pdip_max=15.0, prake_max=30.0,
    nsubx=16, nsuby=8, nb_taper_trbl=[5,5,5,5], seeds=[52,448,4446],
    id_ran1=1, id_ran2=1, angle_north_to_x=0.0, is_moment=3,
    crust_model=crust, output_name="FFSP_OUTPUT", verbose=True)
source.run()
source.set_active_realization(0)
sf = source.get_subfaults()

MINSLIP = 0.5  # practical runtime limiter, not a physical threshold — see 03_ffsp_source.md
dt = 0.01
sources = []
for i in range(len(sf["x"])):
    if sf["slip"][i] < MINSLIP:
        continue
    Tr = float(sf["rise_time"][i])
    stf = SRF2(Tr=Tr, Tp=0.15*Tr, Te=0.7*Tr, dt=dt,
               slip=float(sf["slip"][i]), a=1.0, b=100.0)
    sources.append(PointSource(
        [sf["x"][i]/1e3, sf["y"][i]/1e3, sf["z"][i]/1e3],   # METERS -> km, mandatory
        [float(sf["strike"][i]), float(sf["dip"][i]), float(sf["rake"][i])],
        stf=stf, tt=float(sf["rupture_time"][i])))

fault = FaultSource(sources, metadata={"name": "ffsp_rlz_0"})
model = ShakerMaker(crust, fault, stations)
model.run(dt=dt, nfft=8192, dk=0.2, tb=0, tmin=0., tmax=150.)
```

## Recipe 7 — FFSP ensemble, efficient (Stage 0-1 once, Stage 2 per realization)

**When**: multiple FFSP realizations against the **same** receivers — the
Green's Functions only depend on geometry, which is shared across
realizations, so compute them once.

```python
model0 = ShakerMaker(crust, fault_rlz0, stations)   # any one realization's fault, for geometry
model0.run_nearest(stage='0_1', h5_database_name='gf.h5',
                    dt=dt, nfft=8192, dk=0.2, tb=0)

for k in range(n_realizations):
    source.set_active_realization(k)
    fault_k = build_fault_from_subfaults(source.get_subfaults())  # same pattern as Recipe 6
    writer_k = DRMHDF5StationListWriter(f'motions_rlz{k}.h5drm')
    ShakerMaker(crust, fault_k, stations).run_nearest(
        stage=2, h5_database_name='gf.h5', dt=dt, nfft=8192, dk=0.2, tb=0,
        writer=writer_k, writer_mode='progressive')
```
See `03_ffsp_source.md` for the full reasoning.

## Recipe 8 — Export to SW4, run externally, rebuild `.h5drm`

**When**: you want a finite-difference cross-check or a topography-aware
run that ShakerMaker's 1-D FK engine can't do.

```python
# Step 1 (ShakerMaker, no FK run):
model = ShakerMaker(crust, fault, stations)
model.export_sw4(path="/run/dir", h=100, size_domain=[40000,40000,25000],
                  tmax=40, h5_export_name="sw4_package.h5")

# Step 2 (external, NOT part of this repo): unpack + run SW4
#   python /run/dir/shakermakerexports/unpack_sw4_package.py
#   <sw4-binary> /run/dir/sw4/shakermaker2sw4.in

# Step 3 (after SW4 finishes writing its receiver .txt files):
from shakermaker.sw4_exporter import build_h5drm_from_sw4_case
build_h5drm_from_sw4_case(case_path="/run/dir", output_name="motions.h5drm")
```
See `09_sw4_export.md` for the coordinate convention and every parameter.

## Recipe 9 — Crust from a real site (CRUST 1.0 lookup)

**When**: you have a lat/lon and want a physically-grounded starting crust
instead of a synthetic/benchmark one.

```python
from shakermaker.crust1 import Crust1
c1 = Crust1()
profile = c1.profile_at(lat, lon)
c1.print_shakermaker([(lat, lon)])   # prints a ready-to-paste CrustModel snippet
```
Remember: layers with `Vs=0` (water/ice) must be skipped by hand when
building the `CrustModel` — "ShakerMaker FK fails with Vs=0" (see
`01_crust_model.md`, confirmed pattern from `examples/14_SFSI/
example_documented.ipynb`). Feed the resulting `CrustModel` into any recipe
above.

## Recipe 10 — Save Green's Functions per subfault (`save_gf`)

**When**: you need the raw per-subfault Green's-Function tensors (e.g. to
manually verify a convolution, or inspect the DD/DS/SS elementary
components), not just the final convolved response.

```python
sta = Station([6, 8, 0], metadata={"name": "S1", "save_gf": True})  # opt-in
model = ShakerMaker(crust, fault, StationList([sta], {}))
model.run(dt=0.01, nfft=4096, dk=0.1, tb=200, tmax=30)   # legacy run() only

gfs = sta.get_greens_functions()   # dict keyed by subfault id
```
`save_gf` is a `Station`-level, legacy-`run()`-only opt-in (see
`04_receivers.md`) — it stores every subfault's raw `tdata` tensor in
memory, so only use it on small models.

## How to generate a combination not listed here

1. Pick your row from each dimension in the compatibility matrix.
2. Open the corresponding atom doc for anything you're unsure of the exact
   signature/units for (`01`–`04`, `07`, `09`).
3. Start from the closest recipe above and swap the receiver/writer/run-tier
   pieces — the object-construction code for crust/source is identical
   across every recipe; only receiver, writer, and the `run`/`run_nearest`
   call change.
4. Always call `check_parameters(...)` first with your final `dt/nfft/dk/tb/
   tmax` (`05_check_parameters.md`).
5. If it involves MPI/HPC, read `11_mpi_and_hpc.md` before launching.

## See also

- `00_orientation.md` — the decision tree this file assumes you already
  walked through.
- `examples/EXAMPLES_REFERENCE.md` — every existing script/notebook in the
  repo, in case one of them is already the exact combination you need
  (check there before writing anything new).
