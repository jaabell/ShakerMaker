# 12 - Compare ShakerMaker LOH.3 results against the Prose reference solution.
# 2026-09-26
#
# Run LOH3.py first; it writes loh3_station.npz and loh3_station_qref.npz.
#
# The reference is data/LOH.3_prose_corrected, the semi-analytical solution
# distributed with the SW4 matlab tools. The recipe below is a transcription of
# `loh3exact.m`: it is identical to the LOH.1 one except sigma = 0.05 s.
#
# Why two models are compared: see the header of LOH3.py. Short version, the FK
# core anchors the Futterman Q dispersion at 1 Hz and the benchmark anchors it
# at 2.5 Hz; the difference is a ~16 ms travel-time bias that caps the
# correlation at ~0.95 no matter how fine dt is.

import os
import sys

# Run against this working tree, not whatever snapshot happens to be installed.
# ShakerMaker is installed non-editable in some environments (e.g. the
# `clark_kent` venv carries a frozen copy under site-packages), so a plain
# `import shakermaker` from this directory picks up that copy instead. That is
# exactly how a stale `SCEC_LOH_3` produces a wrong-but-plausible answer.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from shakermaker.station import Station

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "data", "LOH.3_prose_corrected")

# Reference columns: time, vertical (x -1e5), radial (x 1e5), transverse (x 1e5).
# The semi-analytical Prose solution is convolved with the LOH.3 source ramp
# (1/T^2) t exp(-t/T) and then with the operator that turns that ramp into a
# Gaussian of spread sigma. That operator is (1 + T d/dt)^2 applied to the
# Gaussian, i.e. G + 2T G' + T^2 G'', which is the bracket below.
sig, T = 0.05, 0.1
A = np.loadtxt(REF)
nt = A.shape[0]
t = A[:, 0]
dt = t[1] - t[0]
vert = A[:, 1] * (-1e5)
rad = A[:, 2] * (1e5)
trans = A[:, 3] * (1e5)


def conv(x, k):
    return dt * np.convolve(x, k, "full")[:nt]


ramp = (1.0 / T**2) * t * np.exp(-t / T)
rad, trans, vert = conv(rad, ramp), conv(trans, ramp), conv(vert, ramp)

tau = t - 6 * sig
factor = 1 - (2 * T / sig**2) * tau - ((T / sig)**2) * (1 - (tau / sig)**2)
gauss = (1.0 / (np.sqrt(2 * np.pi) * sig)) * factor * np.exp(-0.5 * (tau / sig)**2)
rad, trans, vert = conv(rad, gauss), conv(trans, gauss), conv(vert, gauss)

# Compare over the window the reference is meaningful in.
win = (t >= 0) & (t <= 9)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def best_lag(a, b, K=40):
    """Correlation after the best integer shift, and that shift in seconds."""
    a = a - a.mean()
    b = b - b.mean()
    out = []
    for k in range(-K, K + 1):
        aa, bb = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        out.append((float(np.corrcoef(aa, bb)[0, 1]), k))
    c, k = max(out)
    return c, k * dt


def report(npz, label):
    if not os.path.exists(npz):
        print(f"SKIP {label}: run LOH3.py first ({os.path.basename(npz)} missing)")
        return None

    # ShakerMaker result, rotated from (Z, E, N) to (radial, transverse, vertical).
    # Source (0,0) -> receiver (6 North, 8 East): unit vector (rN, rE) = (0.6, 0.8).
    sta = Station()
    sta.load(npz)
    z, e, n, tsm = sta.get_response()
    rN, rE = 0.6, 0.8

    print(f"\n{label}")
    print(f"  {'component':11s} {'corr':>8s} {'peak SM/ref':>12s} "
          f"{'lsq scale':>10s} {'L2 misfit':>10s} {'corr@bestlag':>14s}")
    corrs = []
    for name, smc, refc in [("radial", rN * n + rE * e, rad),
                            ("transverse", -rE * n + rN * e, trans),
                            ("vertical", z, vert)]:
        smi = np.interp(t, tsm, smc)            # SM onto the reference time grid
        a, b = refc[win], smi[win]
        cc = corr(a, b)
        ac, bc = a - a.mean(), b - b.mean()
        scale = float(ac @ bc / (ac @ ac))      # least-squares amplitude ratio
        misfit = float(np.linalg.norm(b - a) / np.linalg.norm(a))
        pk = np.abs(b).max() / np.abs(a).max()
        cb, lag = best_lag(a, b)
        corrs.append(cc)
        print(f"  {name:11s} {cc:+8.4f} {pk:12.4f} {scale:10.4f} {misfit:10.4f} "
              f"{cb:+8.4f} @{lag:+.3f}s")
    return corrs


spec = report(os.path.join(HERE, "loh3_station.npz"),
              "LOH.3 as specified  (FK core anchors Q dispersion at 1 Hz)")
qref = report(os.path.join(HERE, "loh3_station_qref.npz"),
              "LOH.3 with Q dispersion re-anchored at 2.5 Hz  (benchmark convention)")

if spec is None or qref is None:
    raise SystemExit(0)

print("""
Reading the table
-----------------
The as-specified run correlates ~0.95 and its best correlation sits at a
negative lag: ShakerMaker arrives early, because above 1 Hz the core's medium
is faster than the benchmark's. Re-anchoring the dispersion at the benchmark's
own 2.5 Hz removes the lag and takes the correlation to ~0.999.

Both runs come out ~3.5% high in amplitude (`lsq scale` ~ 1.035). Part of that
is not specific to LOH.3: the elastic LOH.1 run, same source, same receiver,
same metric at dt = 0.004 s, already sits ~1% high (lsq scale 1.010-1.012).
So roughly 2.5 points of the excess are attributable to the attenuation model -
the core's Futterman constant-Q against whatever the semi-analytical solution
used - and the rest is the common bias of the comparison itself.""")

# The physics test: re-anchoring must fix the waveform, not just nudge it.
assert min(spec) > 0.94, spec
assert min(qref) > 0.99, qref
print("\nPASS")
