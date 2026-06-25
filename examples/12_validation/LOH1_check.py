# 12 - Compare ShakerMaker LOH.1 result against the Prose reference solution.
# 2026-06-06

import os
import numpy as np
from shakermaker.station import Station

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "data", "LOH.1_prose3")
NPZ = os.path.join(HERE, "loh1_station.npz")

if not os.path.exists(NPZ):
    print("SKIP: run LOH1.py first")
    raise SystemExit(0)

# Reference columns: time, vertical (x -1e5), radial (x 1e5), transverse (x 1e5).
# The semi-analytical Prose solution is convolved with the LOH.1 source ramp
# (1/T^2) t exp(-t/T) and a Gaussian (SCEC LOH.1 recipe) to get the seismogram.
sig, T = 0.06, 0.1
A = np.loadtxt(REF)
nt = A.shape[0]
t = A[:, 0]
dt = t[1] - t[0]
vert = A[:, 1] * (-1e5)
rad = A[:, 2] * (1e5)
trans = A[:, 3] * (1e5)

ramp = (1.0 / T**2) * t * np.exp(-t / T)
rad = dt * np.convolve(rad, ramp, "full")[:nt]
trans = dt * np.convolve(trans, ramp, "full")[:nt]
vert = dt * np.convolve(vert, ramp, "full")[:nt]

tau = t - 6 * sig
factor = 1 - (2 * T / sig**2) * tau - ((T / sig)**2) * (1 - (tau / sig)**2)
gauss = (1.0 / (np.sqrt(2 * np.pi) * sig)) * factor * np.exp(-0.5 * (tau / sig)**2)
rad = dt * np.convolve(rad, gauss, "full")[:nt]
trans = dt * np.convolve(trans, gauss, "full")[:nt]
vert = dt * np.convolve(vert, gauss, "full")[:nt]

# ShakerMaker result, rotated from (Z, E, N) to (radial, transverse, vertical).
# Source (0,0) -> receiver (6 North, 8 East): unit vector (rN, rE) = (0.6, 0.8).
sta = Station()
sta.load(NPZ)
z, e, n, tsm = sta.get_response()
rN, rE = 0.6, 0.8
sm_radial = rN * n + rE * e
sm_transv = -rE * n + rN * e
sm_vert = z


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


corrs = []
for name, smc, refc in [("radial", sm_radial, rad),
                        ("transverse", sm_transv, trans),
                        ("vertical", sm_vert, vert)]:
    smi = np.interp(t, tsm, smc)        # SM onto the reference time grid
    cc = corr(refc, smi)
    corrs.append(cc)
    print(f"{name:11s}: corr={cc:+.3f}")

assert min(corrs) > 0.9
print("PASS")
