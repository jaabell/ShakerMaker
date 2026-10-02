# 12 - SCEC LOH.1 benchmark run (single receiver at (6,8,0)).
# 2026-06-06

from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

# LOH.1 Gaussian source time function.
sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2

crust = SCEC_LOH_1()

z = 2.0
s, d, r = 0., 90., 0.
src = PointSource([0, 0, z], [s, d, r],
                  stf=Gaussian(t0=t0, freq=1/sigma, M0=M0, derivative=False))
fault = FaultSource([src], metadata={"name": "LOH1_source"})

# The LOH.1 receiver.
sta = Station([6.0, 8.0, 0.0], metadata={"name": "loh1", "save_gf": True})
stations = StationList([sta], {})

model = shakermaker.ShakerMaker(crust, fault, stations)

dt = 0.025
nfft = 4096
dk = 0.1
tb = 1000          # large tb; a small tb clips the near field
tmax = nfft * dt   # output window

model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax, smth=1, verbose=True)

sta.save("loh1_station.npz")

zc, ec, nc, t = sta.get_response()
assert len(t) > 0 and len(zc) == len(t)
print("PASS")
