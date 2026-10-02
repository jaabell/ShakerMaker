# 08 - Export DRM geometry (no FK run) for a DRMBox and a SurfaceGrid.
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox, SurfaceGrid

try:
    import h5py
except Exception:
    print("SKIP: h5py not available"); raise SystemExit(0)

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

dx = 0.5

# DRMBox geometry: writes coords + synthetic ramp; no FK run -> fast.
drm = DRMBox([6, 8, 0], [3, 3, 2], [dx, dx, dx], metadata={"name": "drm_box"})
model_box = ShakerMaker(crust, fault, drm)
f_box = "drm_geometry_box.h5drm"
model_box.export_drm_geometry(f_box)

# SurfaceGrid geometry (XY plane at z=0).
sg = SurfaceGrid([6, 8, 0], [4, 4, 1], dx, mode="plane", plane_z=0.0,
                 metadata={"name": "drm_surface"})
model_sg = ShakerMaker(crust, fault, sg)
f_sg = "drm_geometry_surface.h5drm"
model_sg.export_drm_geometry(f_sg)

assert os.path.exists(f_box)
assert os.path.exists(f_sg)
print("PASS")
