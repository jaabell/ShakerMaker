# 06 - run_nearest (OP pipeline, stage='all') on a small surface grid.
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

_m = 1e-3  # 1 m in km

crust = SCEC_LOH_1()

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)

source = PointSource([0, 0, 2], [0., 90., 0.], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

# XY plane at z=0: (nx+1)*(ny+1)=36 stations + 1 QA (large dx -> few stations)
dx = 2.0
stations = SurfaceGrid([6.0, 8.0, 0.0], [5, 5, 0], [dx, dx, dx],
                       mode='plane', plane_z=0.0,
                       metadata={"name": "grid"})

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_nearest_all")
os.makedirs(folder, exist_ok=True)
gf_databasename = os.path.join(folder, "gf_database.h5")
h5drm_output = os.path.join(folder, "surface.h5drm")

model = ShakerMaker(crust, fault, stations)

dt, nfft, dk, tb, tmax, tmin = 0.025, 2048, 0.1, 1000, 30., 0.
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

writer = DRMHDF5StationListWriter(h5drm_output)

model.run_nearest(
    stage='all',
    h5_database_name=gf_databasename,
    delta_h=2.5 * _m,
    delta_v_rec=2.5 * _m,
    delta_v_src=2.5 * _m,
    dt=dt,
    nfft=nfft,
    tb=tb,
    dk=dk,
    smth=1,
    tmin=tmin,
    tmax=tmax,
    writer=writer,
    writer_mode='progressive',
    verbose=False,
    showProgress=True,
)

assert os.path.exists(h5drm_output)
print("PASS")
