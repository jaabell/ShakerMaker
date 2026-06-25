# 08 - Compute motion at one point directly and inside a tiny DRM box; compare.
# 2026-06-06

import numpy as np
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox

dt, nfft, tb, dk, tmax = 0.025, 2048, 1000, 0.1, 30

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2


def make_fault():
    stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
    src = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
    return FaultSource([src], metadata={"name": "src"})


center = [6, 8, 0]

# --- Direct single station at the DRM box center ---
s = Station(center, metadata={"name": "direct"})
model_direct = ShakerMaker(crust, make_fault(), StationList([s], {}))
model_direct.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
model_direct.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax)
zd, ed, nd, td = s.get_response()

# --- Tiny DRM box centered at the same point ---
dx = 0.5
drm = DRMBox(center, [1, 1, 1], [dx, dx, dx], metadata={"name": "drm"})
assert drm.nstations <= 50
model_drm = ShakerMaker(crust, make_fault(), drm)
model_drm.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
model_drm.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax)

# The QA station (last) sits at the box center -> same point as the direct run.
qa = drm.get_station_by_id(drm.nstations - 1)
zq, eq, nq, tq = qa.get_response()

assert zd.size > 0 and zq.size > 0
assert np.max(np.abs(zd)) > 0 and np.max(np.abs(zq)) > 0
print("direct  max|z| =", np.max(np.abs(zd)))
print("drm QA  max|z| =", np.max(np.abs(zq)))
print("PASS")
