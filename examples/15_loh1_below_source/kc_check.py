# 15 - Wavenumber cut-off check for the receivers 0.5 km above and below the source.
#
# In the FK integral the kernels decay with wavenumber like exp(-k * dz), with dz the
# source-receiver depth separation, so receivers close to the source depth need a high
# kmax. subfk.f cuts the integral at kmax = kc / hs; subgreen passes hs = the source-receiver
# separation, as fk.f does (it used to pass the total thickness of the layered model, and then
# the vertical component 0.5 km below the source near the epicentre had a peak ratio of 1.36
# with kc = 15 and needed kc >= 30).
# This script reruns the three receivers at dz = 0.5 km with kc = 15, 30, 60 and 150 and
# compares them with SW4 (same band, window and amplitude factor as compare.py): the results
# must not depend on kc.
#
#   python kc_check.py       # writes _output/kc_check.csv

import contextlib
import csv
import io
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

with contextlib.redirect_stdout(io.StringIO()):
    from shakermaker import shakermaker
    from shakermaker.cm_library.LOH import SCEC_LOH_1
    from shakermaker.pointsource import PointSource
    from shakermaker.faultsource import FaultSource
    from shakermaker.station import Station
    from shakermaker.stationlist import StationList
    from shakermaker.stf_extensions.gaussian import Gaussian

HERE = os.path.dirname(os.path.abspath(__file__))
SW4_DIR = os.path.join(HERE, "data", "sw4")
OUT = os.path.join(HERE, "_output")
os.makedirs(OUT, exist_ok=True)
FMIN, FMAX, TMAX, DT = 0.1, 1.2, 9.0, 0.01
SCALE = 0.97789          # amplitude factor fitted at o6_z0000 by compare.py
RECEIVERS = {"o06_z1500": (0.6, 0.8, 1.5), "o06_z2500": (0.6, 0.8, 2.5), "o6_z2500": (6.0, 8.0, 2.5)}
KCS = (15.0, 30.0, 60.0, 150.0)

tc = np.arange(0.0, TMAX, DT)
sos = butter(4, [FMIN, FMAX], btype="band", fs=1.0 / DT, output="sos")


def sw4_displacement(name):
    a = np.loadtxt(os.path.join(SW4_DIR, f"{name}.txt"), comments="#")
    t = a[:, 0]
    out = {}
    for k, j in (("N", 1), ("E", 2), ("Z", 3)):
        u = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:, j] + a[:-1, j]) * np.diff(t))])
        out[k] = sosfiltfilt(sos, np.interp(tc, t, u))
    return out


rows = []
for name, x in RECEIVERS.items():
    ref = sw4_displacement(name)
    for kc in KCS:
        with contextlib.redirect_stdout(io.StringIO()):
            src = PointSource([0, 0, 2.0], [0., 90., 0.],
                              stf=Gaussian(t0=0.36, freq=1 / 0.06, M0=1e18 / 5e14 / 2, derivative=False))
            sta = Station(list(x), metadata={"name": name})
            model = shakermaker.ShakerMaker(SCEC_LOH_1(), FaultSource([src], metadata={"name": "s"}),
                                            StationList([sta], {}))
            model.run(dt=0.025, nfft=4096, tb=1000, dk=0.1, kc=kc, tmax=4096 * 0.025, smth=1)
        z, e, n, t = sta.get_response()
        sm = {"Z": z, "E": e, "N": n}
        row = dict(receiver=name, side="below" if x[2] > 2.0 else "above", kc=kc)
        for k in "ZEN":
            a = ref[k]
            b = SCALE * sosfiltfilt(sos, np.interp(tc, t, sm[k]))
            row[f"cc_{k}"] = float(np.corrcoef(a, b)[0, 1])
            row[f"peak_{k}"] = float(np.abs(b).max() / np.abs(a).max())
            row[f"nrms_{k}"] = float(np.linalg.norm(b - a) / np.linalg.norm(a))
        rows.append(row)
        print(f"{name:10s} kc {kc:5.0f} " + " | ".join(
            f"{k}: cc {row[f'cc_{k}']:.3f} peak {row[f'peak_{k}']:.2f} nrms {row[f'nrms_{k}']:.2f}" for k in "ZEN"),
              flush=True)

with open(os.path.join(OUT, "kc_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("PASS")
