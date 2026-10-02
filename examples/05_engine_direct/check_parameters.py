# 05 - check_parameters: pure-arithmetic pre-run report (no FK run).
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
stations = StationList([s1], {})

model = ShakerMaker(crust, fault, stations)

res = model.check_parameters(dt=0.025, nfft=2048, dk=0.1, tb=1000, tmax=30)

assert isinstance(res, dict)
print("PASS")
