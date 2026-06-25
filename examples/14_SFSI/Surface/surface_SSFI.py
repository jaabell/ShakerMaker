
import numpy as np
import matplotlib.pyplot as plt


# Core
from shakermaker import shakermaker
# Crust model
from shakermaker.crustmodel import CrustModel
# Source
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
# STF
from shakermaker.stf_extensions.gaussian import Gaussian

# Station
from shakermaker.station import Station
from shakermaker.stationlist import StationList

# source
x_source, y_source = 491222.3167 , 4512564.5625

# Source
sigma=0.06
t0=6*sigma
stf=Gaussian(t0=t0, freq=1/sigma, M0=1 ,derivative=False)

M0=(1e18/5e14/20)
z = 30.
s,d,r = 340., 20., 0.


source = PointSource(   [y_source/1e3,x_source/1e3 , z], 
                        [s,d,r],
                        stf = Gaussian(t0=t0, freq=1/sigma, M0=M0 , derivative=False),
                    )
fault = FaultSource([source], metadata={"name":"LOH1_source"})



crust = CrustModel(5)
# water - skipped (ShakerMaker FK fails with Vs=0)
# vp, vs, rho, thick, Qa, Qb = 1.50, 0.00, 1.02, 0.000, 1000.0, 1000.0
# crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# ice - skipped
# vp, vs, rho, thick, Qa, Qb = 3.81, 1.94, 0.92, 0.000, 1000.0, 1000.0
# crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# upper sediments
vp, vs, rho, thick, Qa, Qb = 2.50, 1.07, 2.11, 0.200, 1000.0, 1000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# upper crystalline crust
vp, vs, rho, thick, Qa, Qb = 6.30, 3.63, 2.79, 12.620, 1000.0, 1000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# middle crystalline crust
vp, vs, rho, thick, Qa, Qb = 6.60, 3.80, 2.86, 11.120, 1000.0, 1000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# lower crystalline crust
vp, vs, rho, thick, Qa, Qb = 7.00, 3.99, 2.95, 6.310, 1000.0, 1000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)
# mantle half-space
vp, vs, rho, thick, Qa, Qb = 7.93, 4.41, 3.27, 0.000, 1000.0, 1000.0
crust.add_layer(thick, vp, vs, rho, Qa, Qb)



# ---------------------------------------------------------------------------
# DRM receiver from the extracted FEM point cloud
#   crd_scale = 1/1000  (FEM metres -> km)
#   x0_fem    = [0,0,0] (FEM origin = top-center of the DRM box at the surface)
#   drmbox_x0 = site centre in ShakerMaker km coords
# ---------------------------------------------------------------------------


from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import SurfaceGrid

x_source, y_source = 491222.3167 , 4512564.5625
drmbox_x0  = [y_source / 1e3, x_source / 1e3, 0.0]


Lx = 100/1000  # km
Ly = 100/1000  # km
Lz= 10/1000   # km
dx = 10/1000  # km
nx, ny, nz = int(Lx/dx), int(Ly/dx), int(Lz/dx)

stations = SurfaceGrid(drmbox_x0,
                    [nx, ny, nz],
                    [dx, dx, dx],
                    mode='plane', 
                    plane_z=0.0,
                    metadata={"name": "plane_z0"})

print(f"DRM receivers: {stations.nstations} stations")
print(f"drmbox_x0 (km): {drmbox_x0}")

# Create model
model = shakermaker.ShakerMaker(crust, fault, stations)

writer = DRMHDF5StationListWriter('ssfi_h5drm_surface.h5drm')

#Units 
_m = 0.001/1e12
delta_h=2.5*_m
delta_v_rec=2.5*_m
delta_v_src=2.5*_m
npairs_max = 100000     # max pairs per batch


dt=0.005
nfft=8192 * 4      # N timesteps
dk=0.4         # wavenumber discretization
tb=400          # Initial zero-padding
tmin=0
tmax=100


model.run_nearest(
    stage='all',
    # stage=2,
    h5_database_name='ssfi_gf_surface.h5',
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
    # sigma=2,
    # taper=0.9,
    # wc1=1,
    # wc2=2,
    # pmin=0,
    # pmax=1,
    # nx=1,
    # kc=15.0,
    # Stage 2 only
    # tmin=tmin,
    # tmax=tmax,
    writer=writer,
    writer_mode='progressive',
    # General
    verbose=False,
    debugMPI=False,
    showProgress=True,
)