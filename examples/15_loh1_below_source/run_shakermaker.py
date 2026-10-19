# 15 - LOH.1 with receivers above and below the source (ShakerMaker side).
#
# Same source, crust and FK parameters as 12_validation/LOH1.py (validated against the
# Prose solution at the surface), with the receivers of receivers.txt: two epicentral
# offsets, from the surface down to 4 km below the 2 km deep source. Receivers deeper
# than the source go through the model flip of subgreen (receiver below the source).
#
#   python run_shakermaker.py [OUT_DIR]   # writes sm_<name>.npz for every receiver (default _output/)
#
# The first TSAVE seconds are kept (SW4 simulates 9 s); data/sm/ holds the files of the reference run.

import os
import sys

import numpy as np

from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_output")
os.makedirs(OUT, exist_ok=True)
TSAVE = 12.0

# LOH.1 Gaussian source time function (as 12_validation/LOH1.py).
sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2

crust = SCEC_LOH_1()
src = PointSource([0, 0, 2.0], [0., 90., 0.],
                  stf=Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False))
fault = FaultSource([src], metadata={"name": "LOH1_source"})

rows = [l.split() for l in open(os.path.join(HERE, "receivers.txt")) if l.strip() and not l.startswith("#")]
stations = [Station([float(n), float(e), float(z)], metadata={"name": name}) for name, n, e, z in rows]
model = shakermaker.ShakerMaker(crust, fault, StationList(stations, {}))

dt = 0.025
nfft = 4096
dk = 0.1
tb = 1000
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=nfft * dt, smth=1)

for sta in stations:
    z, e, n, t = sta.get_response()
    assert len(t) > 0 and np.all(np.isfinite(z)) and np.all(np.isfinite(e)) and np.all(np.isfinite(n))
    keep = t <= TSAVE
    np.savez(os.path.join(OUT, f"sm_{sta.metadata['name']}.npz"), z=z[keep], e=e[keep], n=n[keep], t=t[keep],
             x=np.array(sta.x))
print("PASS")
