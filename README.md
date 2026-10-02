![ShakerMaker](docs/source/images/logo.png)

---
Author   = Jose A. Abell, Jorge Crempien D., and Matias Recabarren

Modified = Patricio Palacios B. | Nicolás Mora Bowen | José Abell | Ladruno Team - 2026

---

ShakerMaker is a Python framework for computing earthquake ground motions using the frequency-wavenumber (FK) method. It provides a complete pipeline from crustal model definition and earthquake source specification to ground motion computation and export in HDF5, NumPy, and H5DRM (Domain Reduction Method) formats. Computations are parallelized with MPI and scale from personal workstations to supercomputing clusters.

The FK method is implemented in Fortran (originally from http://www.eas.slu.edu/People/LZhu/home.html with several modifications) and interfaced with python through f2py wrappers. Classes are built on top of this wrapper to simplify common modeling tasks such as crustal model specification, generation of source faults (from simple point sources to full kinematic rupturespecifications), generating single recording stations, grids and other arrays of recording stations and stations arranged to meet the requirements of the DRM. Filtering and simple plotting tools are provided to ease model setup. 

ShakerMaker includes the Finite Fault Stochastic Process tool, (FFSP developed in Fortran), which allows for the idealization of a fault with a determined area and an associated event magnitude, with specific properties of strike, dip, and rake. Easy-to-use graphical functions are added for visualization of calculated metrics, as well as the determined statistics of the stochastic space performed to select the best model.

Finally, computation of motion traces is done by pairing all sources and all receivers, which is parallelized using MPI. This means that ShakerMaker can run on simple personal computers all the way up to large supercomputing clusters. 

---

## Key Features

- **FK ground motion synthesis** — full-wavefield Green's functions in 1D layered viscoelastic media
- **Domain Reduction Method (DRM)** — compute boundary motions for sub-domain simulations; export directly to H5DRM format (See H5DRMLoadPattern https://github.com/OpenSees/OpenSees)
- **Stochastic finite fault ruptures (FFSP)** — generate spatially-correlated slip distributions with the Finite Fault Stochastic Process tool (Fortran 90); uses magnitude-area scaling relations and configurable random seeds
- **Source time functions** — Brune pulse, Gaussian pulse, Dirac delta, discrete arbitrary functions, and SRF2 format
- **Pre-packaged crustal models** — LOH.1 (SCEC), Southern California (low-frequency), and AbellThesis; extendable with custom 1D layered models
- **Multiple receiver geometries** — individual stations, surface grids, DRM boxes, and point clouds
- **MPI parallelism** — `mpi4py`-based parallelization over source–receiver pairs
- **Filtering and plotting** — low-pass/high-pass filtering, ZENT three-component seismogram plots, station and source geometry plots

---

## How It Works

The FK method computes the complete seismic wavefield for a point source embedded in a 1D layered halfspace. ShakerMaker organises the workflow into three components:

| Component | Role |
|---|---|
| **CrustModel** | Defines the 1D velocity structure: layer thickness, Vp, Vs, density, and Q factors |
| **Source** | Earthquake source: a `PointSource` (single point with strike/dip/rake) or an `FaultSource` (collection of point sources forming an extended fault), optionally driven by a `SourceTimeFunction` |
| **Receiver** | Recording location: `Station` (single point), `StationList` (collection), `DRMBox` (3D box for DRM), `SurfaceGrid` (regular grid), or `PointCloudDRMReceiver` (arbitrary points) |

These three are combined into a `ShakerMaker` instance and dispatched via `.run()` (sequential) or `.gen_pairs()` → `.compute_gf()` → `.run_fast()` (three-stage, MPI-parallel). Results are stored in each `Station` object and optionally written to disk.

---

## Installation

For now, only though the git repo:

Use the `setup.py` script, using setuptools, to compile and install::


```bash
git clone git@github.com:jaabell/ShakerMaker.git
cd ShakerMaker
python setup.py install
```

If you dont' have sudo, you can install locally for your user with::

```bash
python setup.py install --user
```

### Dependencies

| Package | Required | Notes |
|---|---|---|
| `numpy` | Yes | |
| `scipy` | Yes | Signal processing, interpolation |
| `h5py` | Yes | HDF5 I/O |
| `f2py` | Yes | Fortran–Python interface (bundled with `numpy`) |
| `matplotlib` | No | Plotting (`ZENTPlot`, `StationPlot`, `SourcePlot`) |
| `mpi4py` | No | MPI parallelism (recommended for large simulations) |


You can get all these packages with `pip`::
```bash
sudo pip install mpi4py h5py f2py numpy scipy matplotlib
```
or, for your user::

```bash
sudo pip install --user mpi4py f2py h5py numpy scipy matplotlib
```


---

## Core Components

### CrustModel

Defines a 1D layered viscoelastic halfspace. Each layer has thickness (km), Vp (km/s), Vs (km/s), density (g/cm³), and quality factors Qp, Qs. A zero-thickness layer represents the halfspace.

```python
from shakermaker.crustmodel import CrustModel

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)
```

Pre-defined models in `shakermaker.cm_library`:
- `SCEC_LOH_1()` — LOH.1 benchmark model
- `AbellThesis()` — model from Abell's thesis

### PointSource

A point earthquake source defined by its location `[x, y, z]` (km) and fault angles `[strike, dip, rake]` (degrees). Optionally accepts a `SourceTimeFunction` and an initial time offset `tt`.

```python
from shakermaker.pointsource import PointSource

source = PointSource([0, 0, 4], [90, 90, 0])
```

### FaultSource

A collection of `PointSource` objects forming an extended fault. Each sub-source can have its own location, angles, STF, and timing.

```python
from shakermaker.faultsource import FaultSource

sources = [PointSource([0, 0, 4], [90, 90, 0])]
fault = FaultSource(sources, metadata={"name": "mainshock"})
```

### Source Time Functions

Available in `shakermaker.stf_extensions`:

| STF | Description |
|---|---|
| `Brune(f0, t0)` | Brune pulse with corner frequency `f0` (Hz) and peak time `t0` (s) |
| `Gaussian(t0, freq, M0)` | Gaussian pulse; optionally its derivative for moment-rate representation |
| `Dirac(dt)` | Unit impulse at t=0, compatible with FK Green's function |
| `Discrete(t, data)` | Arbitrary user-defined time-history |
| `SRF2(srf2_file)` | Loads SRF2-format source description |

Each STF is convolved with the computed Green's function via `fftconvolve` to produce the final ground motion.

| SRF2 | Dirac |
|---|---|
| ![SRF2](docs/source/images/SRF2.png) | ![Brune](docs/source/images/stf_dirac.png) |
| Brune | Gaussian |
| ![Brune](docs/source/images/stf_brune.png) | ![Gaussian](docs/source/images/stf_gaussian.png) |

### Station and StationList

A `Station` records the three-component ground motion (Z, E, N) at a single location. Results are stored internally and accessible via `get_response()`. Stations can be saved/loaded to/from `.npz` files.

```python
from shakermaker.station import Station
from shakermaker.stationlist import StationList

s = Station([0, 4, 0], metadata={"name": "station-01"})
stations = StationList([s])
```

### Receiver Geometries (`sl_extensions`)

| Class | Description |
|---|---|
| `DRMBox(origin, shape, spacing)` | 3D box of stations for DRM boundary conditions |
| `SurfaceGrid(origin, shape, spacing)` | Regular 2D grid on the free surface |
| `PointCloudDRMReceiver(points)` | Arbitrary set of DRM receiver points |

![DRM box geometry](docs/source/images/drmbox.png)

### FFSPSource

The Finite Fault Stochastic Process tool generates spatially-correlated stochastic slip distributions on a fault plane. Parameters include magnitude, fault dimensions, hypocenter location, corner frequencies, random seeds, and rise-time ratio. The source can be run independently and results exported to HDF5 or legacy FFSP text format.

```python
from shakermaker.ffspsource import FFSPSource

source = FFSPSource(
    id_sf_type=8, freq_min=0.01, freq_max=24.0,
    fault_length=30.0, fault_width=16.0,
    magnitude=6.5, strike=358.0, dip=40.0, rake=113.0,
    nsubx=256, nsuby=128,
    crust_model=crustal,
    ...
)
subfaults = source.run()
source.write_hdf5("results.h5")
```

### ShakerMaker Engine

The `ShakerMaker` class orchestrates the simulation:

```python
from shakermaker.shakermaker import ShakerMaker

model = ShakerMaker(crust, fault, stations)
model.run(dt=0.005, nfft=2048, dk=0.1, tb=500)
```

Key parameters:
- `dt` — output time step (s)
- `nfft` — number of frequency samples
- `dk` — wavenumber discretisation
- `tb` — initial zero-padding samples
- `smth` — smoothing flag
- `writer` — `StationListWriter` instance for direct HDF5 output

The three-stage pipeline (`gen_pairs()` → `compute_gf()` → `run_fast()`) separates pair generation from computation, enabling MPI-parallel execution.

### Output Writers (`slw_extensions`)

| Writer | Format | Use case |
|---|---|---|
| `DRMHDF5StationListWriter` | H5DRM | DRM boundary motions for external solvers |

### Plotting Tools (`tools.plotting`)

- `ZENTPlot(station, xlim, show)` — three-component seismogram overlay (Z, E, N)
- `StationPlot(stations)` — station geometry visualisation
- `SourcePlot(source)` — source geometry visualisation

---

## Examples

The [`examples/`](examples) folder is organised by **topic**, one folder per
concept. Each script is a small, self-contained input → result; most end in
`print("PASS")` so they double as smoke tests. A full walkthrough lives in the
[documentation site](https://ppalacios92.github.io/ShakerMaker/examples/).

| Folder | Topic | Highlights |
|--------|-------|-----------|
| `01_crustmodel/` | Layered velocity models | `crustmodel_build.py`, `crust1_sites.py` (CRUST1.0) |
| `02_sources/` | Moment-tensor sources | `pointsource.py`, `faultsource_srf2.py` |
| `03_stf/` | Source time functions | `stf_gallery.py` (Dirac, Discrete, Brune, Gaussian, SRF2) |
| `04_receivers/` | Receiver layouts | `single_station.py`, `drmbox.py`, `surface_grid.py`, `pointcloud_drm.py` |
| `05_engine_direct/` | Classic FK pipeline | `run_simple.py`, `check_parameters.py`, `core_subgreen.py` |
| `06_nearest_method/` | Green's-function reuse | `nearest_all.py`, `stage_by_stage.py`, `legacy_migration.py` |
| `07_writers/` | Persisting results | `hdf5_writer.py`, `drm_writer.py`, `save_load_station.py`, `explore_h5_output.py` |
| `08_drm/` | Domain Reduction Method | `drm_vs_direct.py`, `export_drm_geometry.py` |
| `09_sw4_export/` | SW4 finite-difference export | `export_sw4.py`, `export_sw4_topo.py`, `package_h5_roundtrip.py`, `build_h5drm_from_sw4.py` |
| `10_ffsp/` | Stochastic finite faults | `ffsp_run.py`, `ffsp_io.py` |
| `11_plotting/` | Plotting helpers | `plotting_tools.py` |
| `12_validation/` | SCEC LOH.1 benchmark | `LOH1.py`, `LOH1_check.py` |
| `legacy_examples/` | José Abell's original upstream examples (kept unmodified for regression) |

Every folder ships a `notebooks/` subdirectory whose `.ipynb` files re-run the
same recipes interactively and save `.png` previews next to them.

Run everything at once with the smoke runner:

```bash
cd examples
python run_all_smoke.py          # fast subset (reports PASS / SKIP / FAIL)
python run_all_smoke.py --full   # also run the slow 12_validation cases
```

### Quick start

A two-layer crustal model with a strike-slip point source at 4 km depth
and a single receiver, plotted with `ZENTPlot`
(see `examples/legacy_examples/example0_readme_example.py`):

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)

source = PointSource([0, 0, 4], [90, 90, 0])
fault = FaultSource([source], metadata={"name": "single-point-source"})

s = Station([0, 4, 0], metadata={"name": "a station"})
stations = StationList([s], metadata=s.metadata)

model = ShakerMaker(crust, fault, stations)
model.run()

ZENTPlot(s, xlim=[0, 60], show=True)
```

---

## ShakerMaker Results

[**ShakerMaker Results**](https://github.com/ppalacios92/ShakerMakerResults) is a companion library for interactive visualisation of the HDF5 (`.h5`) files generated by ShakerMaker. It provides browser-based views of wave propagation, station responses, and spectral content, enabling rapid exploration of simulation outputs without writing plotting code.

---

