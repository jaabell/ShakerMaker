# 07 - Run a small model and store results with HDF5StationListWriter (legacy + progressive).
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.slw_extensions import HDF5StationListWriter

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

s1 = Station([6, 8, 0], metadata={"name": "sta01"})
s2 = Station([8, 8, 0], metadata={"name": "sta02"})
stations = StationList([s1, s2], {})

model = ShakerMaker(crust, fault, stations)

f_legacy = "hdf5_writer_legacy.h5"
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
writer = HDF5StationListWriter(f_legacy)
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax, writer=writer, writer_mode="legacy")

f_prog = "hdf5_writer_progressive.h5"
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
writer = HDF5StationListWriter(f_prog)
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax, writer=writer, writer_mode="progressive")

assert os.path.exists(f_legacy)
assert os.path.exists(f_prog)
print("PASS")
