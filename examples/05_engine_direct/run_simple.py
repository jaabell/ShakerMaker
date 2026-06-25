# 05 - Direct engine run: SCEC_LOH_1 + 1 Gaussian source + 3 stations.
# 2026-06-06

from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)

source = PointSource([0, 0, 2], [0., 90., 0.], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

s1 = Station([6.0, 8.0, 0.0], metadata={"name": "s1"})
s2 = Station([8.0, 8.0, 0.0], metadata={"name": "s2"})
s3 = Station([6.0, 6.0, 0.0], metadata={"name": "s3"})
stations = StationList([s1, s2, s3], {})

model = ShakerMaker(crust, fault, stations)

dt, nfft, dk, tb, tmax = 0.025, 2048, 0.1, 1000, 30.
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax)

z, e, n, t = s1.get_response()
assert len(t) > 0
print("PASS")
