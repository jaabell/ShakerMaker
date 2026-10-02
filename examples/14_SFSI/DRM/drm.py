
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

# Station coordinates (UTM)

utmx = np.array([
    405811.5368, 405964.3274, 406557.2285, 406426.7430, 406280.7166,
    405977.8081, 405127.1656, 405668.1475, 404622.1449, 409033.3607,
    407839.3333, 408723.1503, 403610.9892, 402302.4274, 402902.2053,
    401993.5796, 402018.3799, 401095.9612, 400727.0942, 398358.9065,
    396758.8127, 397603.6277, 403615.2897, 402629.1866, 401966.8675,
    400353.3506, 399642.4017, 401525.3814, 405454.8196, 400619.7621,
    400938.7643, 399723.3834, 394699.6510, 397093.9913, 397250.1158,
    396922.4553, 397460.4121, 396713.3378, 385160.5117, 396580.2275,
    395804.8655, 409884.0883, 407904.1925, 407927.0648, 407807.8240,
    408134.8808, 408758.8132, 409155.2322, 413697.5278,
])

utmy = np.array([
    4543767.5139, 4543383.8271, 4542594.1469, 4540734.9324, 4538861.2662,
    4536548.2666, 4532902.9715, 4530797.5603, 4531522.5129, 4530264.8118,
    4523392.8364, 4519997.5509, 4517635.8541, 4523962.3467, 4525257.4723,
    4523550.5246, 4522961.0724, 4522217.3703, 4520957.5788, 4517172.5535,
    4513518.6022, 4514187.7071, 4517780.8973, 4518072.6088, 4518054.3808,
    4519457.3798, 4519533.3956, 4517869.6110, 4516622.4532, 4517527.6550,
    4517735.5514, 4516205.7360, 4509963.0737, 4510585.8559, 4510827.7328,
    4509102.5527, 4505470.2236, 4503633.9505, 4456319.0643, 4446890.6842,
    4447048.4264, 4437532.3723, 4433610.7015, 4432661.0598, 4431644.9821,
    4431457.9654, 4430819.4681, 4431103.9011, 4428122.6967,
])

utm_order = [
    "Luffenholtz Beach and County Park", "Houda Point Access", "Moonstone County Park", "Little River State Beach", "Clam Beach County Park",
    "McKinleyville Vista Point", "Mad River Bluffs Park", "Hammond Trail", "Mad River Beach County Park", "Azalea State Reserve",
    "Arcata Marsh and Wildlife Sanctuary", "Eureka KOA", "Ebb Tide R.V. Park", "Ma-Le'l Dunes South", "Ma-Le'l Dunes North",
    "Humboldt Coastal Nature Center", "Manila Community Park", "Manila Dunes Recreation Area", "North Spit", "Accessways to Ocean Beach",
    "Samoa Dunes Recreation Area", "Samoa Boat Ramp County Park", "Eureka Slough Boat Ramp", "Samoa Bridge Launching Facility", "Woodley Island Marina",
    "Humboldt Bay Maritime Museum", "Samoa Beach", "Eureka Boardwalk", "Eureka Slough / Dead Mouse Marsh", "Eureka Public Marina and Wharfinger Building",
    "Eureka Mooring Basin", "Del Norte Street Pier", "Mike Thompson Wildlife Area / South Spit", "King Salmon", "King Salmon Shoreline Access",
    "Fields Landing County Park", "Humboldt Bay National Wildlife Refuge", "Hookton Slough", "Punta Gorda Lighthouse", "King Range National Conservation Area (Humboldt)",
    "Lost Coast Trail", "Tolkan Campground", "Black Sands Beach", "Little Black Sands Beach", "Abalone Point",
    "Seal Rock", "Mal Coombs Park", "Shelter Cove", "King Range National Conservation Area (Mendocino)",
]

# source
x_source, y_source = 491222.3167 , 4512564.5625


import folium
from pyproj import Transformer
# UTM 10N -> WGS84
transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)

# Stations 
# selected_stations = utm_order
selected_stations = ['Samoa Beach']
station_objs = []
for name in selected_stations:
    idx = utm_order.index(name)
    y_km = (utmx[idx] ) / 1e3
    x_km = (utmy[idx] ) / 1e3

    print(f"{idx} - Adding station {name} at (x={x_km:.2f} km, y={y_km:.2f} km)")
    # stations
    lons, lats = transformer.transform(y_km*1e3, x_km*1e3)
    print(f"    UTM (x={x_km:.2f} km, y={y_km:.2f} km) -> WGS84 (lon={lons:.5f}, lat={lats:.5f})")


    station_objs.append(Station([x_km, y_km, 0.0], metadata={"name": name, "save_gf": True}))

stations = StationList(station_objs, {})


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

from shakermaker.sl_extensions import PointCloudDRMReceiver
from shakermaker.slw_extensions import DRMHDF5StationListWriter


x_source, y_source = 491222.3167 , 4512564.5625
drmbox_x0  = [y_source / 1e3, x_source / 1e3, 0.0]

stations = PointCloudDRMReceiver(
    point_cloud_file='./drm_nodes.txt',
    crd_scale=1 / 1e3,
    x0_fem=[0., 0., 0.],
    drmbox_x0=drmbox_x0,
    metadata={"name": "SSIOS_DRM"})

print(f"DRM receivers: {stations.nstations} stations")
print(f"drmbox_x0 (km): {drmbox_x0}")

# Create model
model = shakermaker.ShakerMaker(crust, fault, stations)

writer = DRMHDF5StationListWriter('ssfi_h5drm.h5drm')

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
    h5_database_name='ssfi_gf.h5',
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