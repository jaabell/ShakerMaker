# Exercise 5: DRM box → H5DRM

**Goal.** Produce the boundary motions for a Domain Reduction Method analysis
and write them in the H5DRM format that the OpenSees `H5DRMLoadPattern`
consumes. This is ShakerMaker's flagship engineering workflow.

## The idea

The DRM splits a wave-propagation problem into a regional FK simulation and a
local detailed (FE/FD) model. ShakerMaker computes the three-component motion
on a **box of stations** surrounding the local domain; OpenSees then injects
those motions on the box boundary. See [the background](../background/finite_fault.md)
and [the DRM guide](../guides/drm.md).

## The model

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import Brune
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter

# --- Medium ---
crust = CrustModel(1)
crust.add_layer(0.0, 6.0, 3.5, 2.7, 10000., 10000.)   # uniform half-space

# --- Source: strike-slip at 1 km depth, Brune STF at 2 Hz ---
stf    = Brune(f0=2.0, t0=0.0)
source = PointSource([0, 0, 1.0], [0, 90, 0], tt=0, stf=stf)
fault  = FaultSource([source], metadata={"name": "fault"})

# --- DRM box centred at (10, 10, 0): 10 x 10 x 4 stations ---
fmax = 10.0
h    = 3.5 / fmax / 15                      # spacing ~ Vs / fmax / 15
drm  = DRMBox([10., 10., 0.], [10, 10, 4], [h, h, h],
              metadata={"name": "site-box"})

# --- Stream results straight to an .h5drm file ---
writer = DRMHDF5StationListWriter("motions.h5drm")
model  = ShakerMaker(crust, fault, drm)
model.run(dt=1/(2*fmax), nfft=2048, tb=500, dk=0.1, writer=writer)
```

## DRM box geometry, made explicit

`DRMBox(pos, nelems, h)` takes the box **centre**, the **station counts**,
and the **spacings**, not a corner and a size. The side lengths are:

| | Formula | This example |
|---|---|---|
| Interior | `[Nx·hx, Ny·hy, Nz·hz]` | $10h \times 10h \times 4h$ |
| Exterior boundary | `[(Nx+2)·hx, (Ny+2)·hy, (Nz+1)·hz]` | adds one element ring |

Pick `h ≈ Vs / fmax / 15` so the box resolves the shortest wavelength in the
band, and set `dt ≈ 1/(2·fmax)` to honour Nyquist.

![DRM box geometry](../assets/images/drmbox.png){ width=460 }

## What you get

A file `motions.h5drm` with the layout OpenSees expects:

```
/Coordinates    (n_nodes, 3)        node positions (km)
/Time           (nt,)               time vector (s)
/Velocity       (n_nodes, nt, 3)    three-component velocity
/Displacement   (n_nodes, nt, 3)
/Acceleration   (n_nodes, nt, 3)
/DRM_Information group               box dimensions, boundary index
```

Inspect it from the shell:

```bash
h5ls -r motions.h5drm        # list the groups
h5dump -H motions.h5drm      # header only (shapes + attrs)
```

## Things to try

1. **Export the geometry alone** without running:
   `model.export_drm_geometry("drm_geometry.h5drm")`.
2. **Validate** with [`examples/08_drm/drm_vs_direct.py`](../examples/index.md):
   the DRM-injected motion must match a direct FK computation at the same
   point within the numerical tolerance.

## Checkpoint

You can size a DRM box correctly and produce a valid `.h5drm`. Next:
[FFSP stochastic rupture](06_ffsp.md).
