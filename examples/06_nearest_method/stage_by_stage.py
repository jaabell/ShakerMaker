# 06 - OP pipeline run stage-by-stage: gen_pairs -> compute_gf -> run_fast.
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

dx = 2.0
stations = SurfaceGrid([6.0, 8.0, 0.0], [5, 5, 0], [dx, dx, dx],
                       mode='plane', plane_z=0.0,
                       metadata={"name": "grid"})

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_stage_by_stage")
os.makedirs(folder, exist_ok=True)
gf_databasename = os.path.join(folder, "gf_database.h5")
h5drm_output = os.path.join(folder, "surface.h5drm")
base = gf_databasename.replace('.h5', '')

model = ShakerMaker(crust, fault, stations)

dt, nfft, dk, tb, tmax, tmin = 0.025, 2048, 0.1, 1000, 30., 0.
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

# Stage 0: geometry only -> writes <base>_map.h5
model.gen_pairs(gf_databasename,
                delta_h=2.5 * _m, delta_v_rec=2.5 * _m, delta_v_src=2.5 * _m)

# Stage 1: compute Green's functions -> writes <base>_gf.h5
model.compute_gf(gf_databasename, dt=dt, nfft=nfft, tb=tb, dk=dk, smth=1)

# Stage 2: assemble station responses with the precomputed GFs
writer = DRMHDF5StationListWriter(h5drm_output)
model.run_fast(gf_databasename, dt=dt, nfft=nfft, tb=tb, dk=dk, smth=1,
               tmin=tmin, tmax=tmax, writer=writer, writer_mode='progressive')

assert os.path.exists(base + '_map.h5')
assert os.path.exists(base + '_gf.h5')
assert os.path.exists(h5drm_output)
print("PASS")
