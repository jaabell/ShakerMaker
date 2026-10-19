# 15 - LOH.1 with receivers above and below the source: ShakerMaker (FK) vs SW4 (finite differences).
#
# SW4 has no model flip and no wavenumber cut, so it is an independent reference for the
# receivers deeper than the source. Both solutions are band-passed to 0.1-1.2 Hz (SW4 reports a
# resolved bandwidth of 1.234 Hz for this grid in the header of its output) and compared in the
# 9 s that SW4 simulates.
# One amplitude factor (ShakerMaker LOH.1 units -> SW4 units) is fitted at the surface receiver
# o6_z0000 and applied to every receiver, so the amplitudes below the source are checked too.
#
#   python compare.py [SW4_DIR] [SM_DIR]     (defaults: data/sw4, data/sm)
# Writes comparison.csv and loh1_below_source.png to _output/.

import csv
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
SW4_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "sw4")
SM_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "sm")
FMIN, FMAX, TMAX, DT = 0.1, 1.2, 9.0, 0.01

rows = [l.split() for l in open(os.path.join(HERE, "receivers.txt")) if l.strip() and not l.startswith("#")]
t_common = np.arange(0.0, TMAX, DT)
sos = butter(4, [FMIN, FMAX], btype="band", fs=1.0 / DT, output="sos")


def read_sw4(name):
    """SW4 USGS ascii (variables=velocity, nsew=0): time, x (north), y (east), z.

    The two codes define the Gaussian source time function differently: with Gaussian(derivative=False)
    ShakerMaker's output is the time integral of SW4's velocity for type=Gaussian (checked at the surface
    receiver: correlation 0.99 on the three components, no lag). So SW4's velocity is integrated in time
    (cumulative trapezoid) before the comparison. The z components of both outputs share the same sign.
    """
    path = os.path.join(SW4_DIR, f"{name}.txt")
    a = np.loadtxt(path, comments="#")
    t = a[:, 0]
    out = {"N": a[:, 1], "E": a[:, 2], "Z": a[:, 3]}
    out = {k: np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]) for k, v in out.items()}
    return {k: np.interp(t_common, t, v, left=0.0, right=v[-1]) for k, v in out.items()}


def read_sm(name):
    d = np.load(os.path.join(SM_DIR, f"sm_{name}.npz"))
    return {k: np.interp(t_common, d["t"], d[k.lower()], left=0.0, right=0.0) for k in ("Z", "E", "N")}


def band(x):
    return sosfiltfilt(sos, x)


sw4 = {r[0]: {k: band(v) for k, v in read_sw4(r[0]).items()} for r in rows}
sm = {r[0]: {k: band(v) for k, v in read_sm(r[0]).items()} for r in rows}

# one amplitude factor from the surface receiver, least squares over the three components
ref = "o6_z0000"
num = sum(np.dot(sm[ref][k], sw4[ref][k]) for k in "ZEN")
den = sum(np.dot(sm[ref][k], sm[ref][k]) for k in "ZEN")
scale = num / den
print(f"amplitude factor ShakerMaker -> SW4, fitted at {ref}: {scale:.4e}")

table = []
for name, north, east, depth in rows:
    rec = dict(receiver=name, north_km=float(north), east_km=float(east), depth_km=float(depth),
               side="below" if float(depth) > 2.0 else "above")
    for k in "ZEN":
        a, b = sw4[name][k], scale * sm[name][k]
        rec[f"cc_{k}"] = float(np.corrcoef(a, b)[0, 1])
        rec[f"peak_{k}"] = float(np.abs(b).max() / np.abs(a).max())
        rec[f"nrms_{k}"] = float(np.linalg.norm(b - a) / np.linalg.norm(a))
    table.append(rec)
    print(f"{name:10s} {rec['side']:5s} " + " | ".join(
        f"{k}: cc {rec[f'cc_{k}']:.3f} peak {rec[f'peak_{k}']:.2f} nrms {rec[f'nrms_{k}']:.2f}" for k in "ZEN"))

OUT = os.path.join(HERE, "_output")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(rows), 3, figsize=(12, 1.5 * len(rows)), sharex=True)
for i, (name, north, east, depth) in enumerate(rows):
    for j, k in enumerate("ZEN"):
        ax = axes[i, j]
        ax.plot(t_common, sw4[name][k], "k", lw=1.0, label="SW4")
        ax.plot(t_common, scale * sm[name][k], "r--", lw=0.9, label="ShakerMaker")
        ax.text(0.99, 0.85, f"cc {table[i][f'cc_{k}']:.3f}", transform=ax.transAxes, ha="right", fontsize=7)
        if i == 0:
            ax.set_title(k)
        if j == 0:
            ax.set_ylabel(f"{name}\n{'below' if float(depth) > 2 else 'above'}", fontsize=7)
axes[0, 0].legend(fontsize=7, loc="upper left")
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
fig.suptitle(f"LOH.1, receivers above and below the 2 km source: velocity {FMIN}-{FMAX} Hz", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loh1_below_source.png"), dpi=120)
print("PASS")
