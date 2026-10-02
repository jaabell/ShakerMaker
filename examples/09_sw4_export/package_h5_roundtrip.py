# 09 - Export the SW4 package, then unpack it into an SW4 file tree.
# 2026-06-06

import os

try:
    import h5py
except Exception:
    print("SKIP: h5py not available")
    raise SystemExit(0)

from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sw4_exporter import unpack_sw4_package_h5

# --- Real UTM station data (meters). index 0 = 'Centro' = the source. ---
utmx = [359958.1764612976, 359909.210734884, 352972.9064965788, 356785.00778720574, 357388.4436483849, 343765.088895043, 349518.5389304694, 346324.7094952749, 333266.0855400809, 337477.02458516695, 336224.1507329495, 358265.57164]
utmy = [6294124.525366314, 6302625.576311215, 6302517.54834607, 6293263.310727202, 6283866.121857278, 6306996.823725268, 6293815.479730421, 6282778.22420124, 6304244.590630547, 6292761.800243624, 6278203.501919601, 6300158.0242]
names = ['Centro', 'H1_s0', 'N1_s1', 'N2_s2', 'N3_s3', 'I1_s4', 'I2_s5', 'I3_s6', 'F1_s7', 'F2_s8', 'F3_s9', 'CEN']

# --- UTM -> local km frame centered on CENTRO (index 0). ---
# ShakerMaker axes: x=North, y=East, z=Down. UTM: x=East, y=North.
x_km = [(utmy[i] - utmy[0]) / 1000.0 for i in range(len(utmy))]
y_km = [(utmx[i] - utmx[0]) / 1000.0 for i in range(len(utmx))]

# --- 4-layer crust. thickness, vp, vs, rho, Qa, Qb (km, km/s, g/cm3). ---
crust = CrustModel(4)
crust.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
crust.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
crust.add_layer(14.500, 5.50, 3.14, 2.50, 1000.0, 1000.0)
crust.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)

# --- Single point source at CENTRO (local 0,0), depth z = 10 km. ---
stf = Gaussian(t0=0.36, freq=16.6667, M0=1.0)
stf.dt = 0.01
src = PointSource([x_km[0], y_km[0], 10.0], [0.0, 90.0, 0.0], stf=stf)
fault = FaultSource([src], metadata={"name": "centro_source"})

# --- Stations at the OTHER UTM coordinates (skip index 0 = source). ---
stations = []
for i in range(1, len(names)):
    stations.append(Station([x_km[i], y_km[i], 0.0], metadata={"name": names[i]}))
stationlist = StationList(stations, metadata={"name": "stg_stations"})

model = ShakerMaker(crust, fault, stationlist)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sw4_out_roundtrip")

model.export_sw4(
    path=out_dir,
    h=100.0,
    size_domain=[40000.0, 40000.0, 25000.0],
    tmax=40.0,
    m0=1.0,
    station_prefix="sf",
    shakermaker_stations=True,
    h5_export_name="sw4_package.h5",
)

package = os.path.join(out_dir, "shakermakerexports", "sw4_package.h5")
assert os.path.exists(package), f"missing package: {package}"

unpacked = os.path.join(out_dir, "unpacked")
unpack_sw4_package_h5(package, unpacked)

sw4_input = os.path.join(unpacked, "sw4", "shakermaker2sw4.in")
assert os.path.exists(sw4_input), f"missing unpacked input: {sw4_input}"
print("PASS")
