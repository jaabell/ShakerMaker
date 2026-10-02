# DRM boxes at SCEC LOH.1 station locations.

import numpy as np
from mpi4py import MPI

from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter


# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


# Station coordinates (local km)
#   utmx = East coordinate (km)   -> SM y
#   utmy = North coordinate (km)  -> SM x
#   Source is at (0, 0, 2), so Centro at (0, 0) sits above the epicenter.
#   Distances along azimuth ~53 deg (unit vector 0.6 N, 0.8 E):
#     1 km  -> (0.6, 0.8),  3 km -> (1.8, 2.4),  10 km -> (6.0, 8.0)

utmx = np.array([0.0,   6.0,   ])  # East (km)
utmy = np.array([0.0,   8.0,   ])  # North (km)

station_names = ['Centro', 's1']

# Select which stations to run:
selected_stations = ['Centro']
# selected_stations = ['s1']
# selected_stations = station_names


# LOH.1 source

sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2



crust = CrustModel(2)

# Slow layer
vp, vs, rho, thick, Qa, Qb = 4.000, 2.000, 2.600, 1.0, 10000.0, 10000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)

# Half-space
vp, vs, rho, thick, Qa, Qb = 6.000, 3.464, 2.700, 0, 10000.0, 10000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)



z_src = 2.0
s, d, r = 0., 90., 0.
src = PointSource([0, 0, z_src], [s, d, r],
                  stf=Gaussian(t0=t0, freq=1/sigma, M0=M0, derivative=False))
fault = FaultSource([src], metadata={"name": "LOH1_source"})


# DRM box geometry (tunable)

Lx_drm = 0.105  #km
Ly_drm = 0.105  #km
Lz_drm = 0.030  #km
dx_drm = 2.5e-3       # km 
nx_drm = int(Lx_drm / dx_drm)
ny_drm = int(Ly_drm / dx_drm)
nz_drm = int(Lz_drm / dx_drm)


# Simulation parameters

dt = 0.005
nfft = 4096
dk = 0.1
tb = 20              # large tb; a small tb clips the near field
tmin = 0.
tmax = 12       # output window

# Nearest-method (OP) parameters
_m = 0.001
delta_h = 40 * _m
delta_v_rec = 5.0 * _m
delta_v_src = 200 * _m
npairs_max = 200000


# Output folder
import os
output_folder = './drm_loh1_output'
if rank == 0:
    os.makedirs(output_folder, exist_ok=True)


# Run a DRMBox at each selected station

for name in selected_stations:
    idx = station_names.index(name)
    # SM: x = North = utmy, y = East = utmx (already in km, no /1e3)
    x_km = utmy[idx]
    y_km = utmx[idx]

    drm = DRMBox(
        [x_km, y_km, 0.0],
        [nx_drm, ny_drm, nz_drm],
        [dx_drm, dx_drm, dx_drm],
        metadata={"name": f"DRM_{name}"},
    )

    h5drm_name = f'{output_folder}/drm_{name}_sta{idx}.h5drm'
    writer = DRMHDF5StationListWriter(h5drm_name)
    gf_db = f'{output_folder}/gf_db_{name}_sta{idx}.h5'

    if rank == 0:
        dist = np.sqrt(x_km**2 + y_km**2)
        print(f"\n{'='*60}")
        print(f"DRM box at station {name}")
        print(f"  SM coords: (x={x_km:.1f} km N, y={y_km:.1f} km E)")
        print(f"  Distance from source: {dist:.1f} km")
        print(f"  Box: {nx_drm}x{ny_drm}x{nz_drm} elems, dx={dx_drm*1000:.0f} m")
        print(f"  Total DRM nodes: {drm.nstations}")
        print(f"  Output: {h5drm_name}")

    model = shakermaker.ShakerMaker(crust, fault, drm)


    model.run_nearest(
        stage='all',
        h5_database_name=gf_db,
        # Stage 0 params
        delta_h=delta_h,
        delta_v_rec=delta_v_rec,
        delta_v_src=delta_v_src,
        npairs_max=npairs_max,
        # Core params
        dt=dt,
        nfft=nfft,
        dk=dk,
        tb=tb,
        # Stage 1 & 2 params
        smth=1,
        # Stage 2 only
        tmin=tmin,
        tmax=tmax,
        writer=writer,
        writer_mode='progressive',
        # General
        verbose=False,
        debugMPI=False,
        showProgress=True,
    )

    if rank == 0:
        print(f"Done: {name}")

if rank == 0:
    print(f"\nAll DRM stations complete. Output in {output_folder}/")
