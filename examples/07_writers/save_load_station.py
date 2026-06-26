# 07 - Run one station, save to .npz, reload into a fresh Station and compare.
# 2026-06-06

import numpy as np
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

dt, nfft, tb, dk, tmax = 0.025, 2048, 1000, 0.1, 30

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

s = Station([6, 8, 0], metadata={"name": "sta01"})
stations = StationList([s], {})

model = ShakerMaker(crust, fault, stations)
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax)

s.save("sta.npz")

s2 = Station()
s2.load("sta.npz")

z0, e0, n0, t0_ = s.get_response()
z1, e1, n1, t1_ = s2.get_response()

assert np.allclose(z0, z1)
assert np.allclose(e0, e1)
assert np.allclose(n0, n1)
assert np.allclose(t0_, t1_)
print("PASS")
