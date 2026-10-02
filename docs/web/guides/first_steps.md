# Getting started

ShakerMaker turns four ingredients, a crust, a source, a source time
function, and receivers, into three-component ground motion. This page
installs it and runs the smallest meaningful simulation.

## Installation

ShakerMaker compiles a Fortran FK kernel (and the FFSP generator), so it must
be **built for your platform and Python version** before it imports. The full
procedure, Linux (`gfortran`) and Windows (Intel `ifx` + MSVC), with MPI, is
in [Installation & compilation](installation.md).

Quick version once the toolchain is in place:

```bash
pip install "setuptools<60.0" "numpy<2.0" wheel scipy h5py mpi4py matplotlib  # numpy 1.x: build-from-source only
pip install . --no-build-isolation
python -c "import shakermaker; print('ok')"   # Fortran core loaded
```

| Package | Required | For |
|---|---|---|
| `numpy` | yes | arrays (runs on 2.x; only the build-from-source needs 1.x) |
| `scipy` | yes | signal processing, interpolation |
| `h5py` | yes | HDF5 output (incl. the DRM layout) |
| `matplotlib` | no | plotting |
| `mpi4py` | no | MPI parallelism (large runs) |

## The FK pipeline at a glance

```
CrustModel ─┐
PointSource ─┼─► ShakerMaker ──run()──► Station (z, e, n, t)
Station    ─┘
```

- **`CrustModel`**, the 1-D layered medium.
- **`PointSource` / `FaultSource`**, the earthquake, with a source time function.
- **`Station` / `StationList` / `DRMBox`**, where motion is recorded.
- **`ShakerMaker(crust, source, receivers).run(...)`**, computes the traces.

Run it sequentially with `run()`, or scale up with the optimised MPI
pipeline `run_nearest()`, which computes each unique pair-geometry once and
reuses it across thousands of receivers. See
[Running a simulation](running.md).

## Your first simulation

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0,   2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)

source = PointSource([0, 0, 4], [90, 90, 0])
fault  = FaultSource([source], metadata={"name": "single-point-source"})

sta = Station([0, 4, 0], metadata={"name": "STA01"})
stations = StationList([sta], metadata=sta.metadata)

model = ShakerMaker(crust, fault, stations)
model.run()

ZENTPlot(sta, xlim=[0, 60], show=True)
```

This is [Exercise 1](../exercises/01_first_run.md), where the output is read
in full.

![Quick example output](../assets/images/example_0_quick_example.png){ width=520 }

## Units & conventions

| Quantity | Unit |
|---|---|
| Distance / coordinates | km |
| Depth `z` | km, **positive down** |
| Velocity (Vp, Vs) | km/s |
| Density | g/cm³ |
| Angles (strike, dip, rake) | degrees (Aki–Richards) |
| Time | s |
| Output motion | velocity — units follow the STF source |

Coordinates are right-handed with `x`, `y` horizontal and `z` downward; the
free surface is `z = 0`.

## Next steps

- [The FK method](../background/fk_method.md), what is being computed.
- [Crust model](crust_model.md), [Sources](sources.md), [Receivers](receivers.md), the inputs in depth.
- [Exercises](../exercises/index.md), a graded, hands-on path.
