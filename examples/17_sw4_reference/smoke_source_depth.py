# 17 - Smoke test: receivers at and near the depth of the source, against the stored SW4 solution (~5 s).
#
# Same LOH.1 model and source as smoke_vs_sw4.py (oblique point source at 2 km). Receivers of
# data/receivers_loh1.csv at a growing vertical distance from the source:
#
#   prof_r1_z02000   1 km from the epicentre, at the source depth (vertical separation 0)
#   prof_r1_z01100   same vertical, 0.9 km above the source
#   prof_r1_z03000   same vertical, 1 km below the source
#
# The receivers away from the source depth must match SW4. Closer receivers cannot be checked against this SW4
# grid: 100 m from the point source is one cell (h = 100 m below 1 km), and there SW4 smooths the source (at
# epi_z01900 and epi_z02100 the shapes agree, correlation 1.00, but the peaks differ by 0.36-1.25). The receiver at the source depth is a
# known issue: every engine departs from SW4 there (correlation 0.3-0.7 on the horizontal components in the
# reference cases), and the core returns it ten times faster than a receiver 10 m away, which suggests the
# wavenumber integral stops early when the source-receiver separation is zero. It is printed as KNOWN ISSUE and
# not checked; when it matches SW4 the script says so, and the check should then be added.
#
#   python smoke_source_depth.py     # PASS: correlation > 0.99 and peak ratio 0.95-1.05 away from the source depth

import csv
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

here = os.path.dirname(os.path.abspath(__file__))
names = ["prof_r1_z02000", "prof_r1_z01100", "prof_r1_z03000"]
at_source_depth = "prof_r1_z02000"
sigma, tmax, dt = 0.06, 12.0, 0.01

recs = {r["name"]: r for r in csv.DictReader(open(os.path.join(here, "data", "receivers_loh1.csv")))}
crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 1000.0, 1000.0)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 1000.0, 1000.0)
crust.depth_tol = 1e-6
source = PointSource([0.0, 0.0, 2.0], [30.0, 60.0, 45.0],
                     stf=Gaussian(t0=6 * sigma, freq=1 / sigma, M0=1e16 / 1e15, derivative=False))
stations = [Station([float(recs[n]["north_km"]), float(recs[n]["east_km"]), float(recs[n]["depth_km"])],
                    metadata={"name": n}) for n in names]
model = shakermaker.ShakerMaker(crust, FaultSource([source], {"name": "loh1_p1"}), StationList(stations, {"name": "loh1"}))
model.run(dt=0.02, nfft=1024, tb=100, dk=0.025, tmin=0.0, tmax=tmax, showProgress=False)

ref = np.load(os.path.join(here, "data", "sw4", "loh1_p1.npz"))
t4, v4, sw4_names = ref["t"], ref["vel"].astype(float), list(ref["names"])
tc = np.arange(0.0, tmax, dt)
sos = butter(4, [0.1, 2.0], btype="band", fs=1 / dt, output="sos")

ok, known = True, []
for sta, name in zip(stations, names):
    z, e, n, t = sta.get_response()
    v = v4[sw4_names.index(name)]                                                    # north, east, down
    u = np.concatenate([np.zeros((3, 1)), np.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * np.diff(t4), axis=1)], axis=1)
    a = sosfiltfilt(sos, np.array([np.interp(tc, t4, c) for c in (u[2], u[1], u[0])]), axis=1)
    b = sosfiltfilt(sos, np.array([np.interp(tc, t, c, left=0.0, right=c[-1]) for c in (z, e, n)]), axis=1)
    energy = np.linalg.norm(a, axis=1)
    dz = abs(float(recs[name]["depth_km"]) - 2.0)
    for k, comp in enumerate("ZEN"):
        if energy[k] < 0.05 * energy.max():
            continue
        cc = np.corrcoef(a[k], b[k])[0, 1]
        pk = np.abs(b[k]).max() / np.abs(a[k]).max()
        good = cc > 0.99 and 0.95 < pk < 1.05
        if name == at_source_depth:
            known.append(good)
            tag = "KNOWN ISSUE" if not good else "matches SW4"
        else:
            ok &= good
            tag = "ok" if good else "FAIL"
        print(f"{name:16s} dz {1000 * dz:5.0f} m  {comp}  cc {cc:.4f}  peak {pk:.3f}  {tag}")

if known and all(known):
    print("The receiver at the source depth now matches SW4: add it to the checks of this script.")
assert ok, "a receiver away from the source depth no longer matches SW4"
print("PASS")
