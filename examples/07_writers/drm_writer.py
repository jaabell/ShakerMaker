# 07 - Run a small DRMBox and write a .h5drm with DRMHDF5StationListWriter.
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter

try:
    import h5py
except Exception:
    print("SKIP: h5py not available"); raise SystemExit(0)

dt, nfft, tb, dk, tmax = 0.025, 2048, 1000, 0.1, 30

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

# Small DRM box -> well under 50 stations (fast).
dx = 0.5
drm = DRMBox([6, 8, 0], [1, 1, 1], [dx, dx, dx], metadata={"name": "drm_small"})
assert drm.nstations <= 50

model = ShakerMaker(crust, fault, drm)

f_drm = "drm_writer.h5drm"
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
writer = DRMHDF5StationListWriter(f_drm)
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax,
          writer=writer, writer_mode="progressive")

assert os.path.exists(f_drm)
print("PASS")
