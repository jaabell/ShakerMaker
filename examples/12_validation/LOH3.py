# 12 - SCEC LOH.3 benchmark run (single receiver at (6,8,0)).
# 2026-09-26
#
# LOH.3 is LOH.1 plus anelastic attenuation: same 1 km slow layer over a
# half-space, same buried double-couple, same receiver 10 km away, but now
#
#     Q_S = Vs[m/s] / 50,     Q_P = (3/4) (Vp/Vs)^2 Q_S
#
# (the second rule is the statement that the bulk is lossless and all
# dissipation is in shear), and a narrower source time function, sigma = 0.05 s
# instead of 0.06 s.
#
# Two models are run and saved:
#
#   loh3_station.npz        the benchmark exactly as specified
#   loh3_station_qref.npz   same, with the Q dispersion re-anchored at 2.5 Hz
#
# Why the second one. The FK core carries Q through the Futterman complex
# velocity of Aki & Richards p.182 (`shakermaker/core/subfk.f:92`),
#
#     c(f) = c0 [ 1 + ln(f)/(pi*Q) + i/(2*Q) ],
#
# whose logarithm is anchored at 1 Hz: c(1 Hz) = c0, and the medium is faster
# than c0 above 1 Hz. The benchmark anchors its phase velocities at 2.5 Hz
# instead - that is what `attenuation phasefreq=2.5` means in the SW4 reference
# deck `examples/scec/LOH.3-h50.in`. Feeding c0' = c0/(1 + ln(f_ref)/(pi*Q))
# moves the anchor to f_ref without touching Q, so the amplitude decay
# exp(-pi*f*t/Q) is unchanged. It is a 0.4-0.7% change in wave speed, worth
# ~16 ms of travel time over these 10 km, and it is the whole difference
# between correlation 0.95 and correlation 0.999 against the reference
# solution (see LOH3_check.py).
#
# Run this first, then LOH3_check.py.

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

import shakermaker as _shakermaker_pkg
from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_3
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

# LOH.3 Gaussian source time function: sigma = 0.05 s (LOH.1 uses 0.06 s).
sigma = 0.05
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2

# LOH.3 layer properties, same as shakermaker.cm_library.LOH.SCEC_LOH_3.
VP, VS, RHO, THK = (4.000, 6.000), (2.000, 3.464), (2.600, 2.700), (1.0, 0.0)
QP, QS = (120.0, 155.9), (40.0, 69.3)

F_REF = 2.5     # Hz, the benchmark's phase-velocity reference frequency

# FK parameters.
#
# dt = 0.004 s is chosen from the frequency band, not from the output rate.
# `fk.f:73` tapers the spectrum with a raised cosine that starts at
# (1-taper)*f_Nyq = 0.1*f_Nyq and reaches zero at f_Nyq, so the band that comes
# out flat is only f <= 0.05/dt. The sigma = 0.05 s Gaussian has its corner at
# 1/(2*pi*sigma) = 3.2 Hz and useful energy to ~8 Hz, so dt = 0.004 s
# (flat to 12.5 Hz) is needed; dt = 0.025 s would only be flat to 2 Hz and
# would quietly low-pass the answer. 0.004 s also divides the 0.008 s sampling
# of the reference solution, so comparing needs no real resampling.
DT = 0.004
NFFT = 8192      # record nfft*dt = 32.8 s, comfortably past the 16.4 s reference
TB = 1000        # pre-arrival padding in samples; a small tb clips the near field
DK = 0.1
TMAX = 16.0      # the reference solution ends at 16.376 s


def scec_loh_3_qref(f_ref=F_REF):
    """LOH.3 crust with the Futterman Q dispersion anchored at ``f_ref`` [Hz].

    The FK core anchors it at 1 Hz, the benchmark at 2.5 Hz. Dividing each
    speed by ``1 + ln(f_ref)/(pi*Q)`` makes ``c(f_ref) == c0``, so the model
    the core integrates carries the benchmark's velocities at the benchmark's
    own reference frequency. Q is untouched.
    """
    model = CrustModel(2)
    for i in range(2):
        kp = 1.0 + np.log(f_ref) / (np.pi * QP[i])
        ks = 1.0 + np.log(f_ref) / (np.pi * QS[i])
        model.add_layer(THK[i], VP[i] / kp, VS[i] / ks, RHO[i], QP[i], QS[i])
    return model


def run_loh3(crust, npz, verbose=True):
    """Run one LOH.3 model and save the station to ``npz``."""
    z = 2.0
    s, d, r = 0., 90., 0.
    src = PointSource([0, 0, z], [s, d, r],
                      stf=Gaussian(t0=t0, freq=1/sigma, M0=M0, derivative=False))
    fault = FaultSource([src], metadata={"name": "LOH3_source"})

    # The LOH.3 receiver: sta10 of the SCEC suite, 10 km from the epicentre.
    sta = Station([6.0, 8.0, 0.0], metadata={"name": "loh3", "save_gf": True})
    stations = StationList([sta], {})

    model = shakermaker.ShakerMaker(crust, fault, stations)
    model.check_parameters(dt=DT, nfft=NFFT, dk=DK, tb=TB, tmax=TMAX)
    model.run(dt=DT, nfft=NFFT, tb=TB, dk=DK, tmax=TMAX, smth=1, verbose=verbose)

    sta.save(npz)
    zc, ec, nc, t = sta.get_response()
    assert len(t) > 0 and len(zc) == len(t)
    print(f"  -> {npz}   nt={len(t)}  peak|E|={np.abs(ec).max():.4e}")
    return sta


if __name__ == "__main__":
    print("shakermaker package:", _shakermaker_pkg.__file__)
    print(SCEC_LOH_3())

    print("=" * 70)
    print("LOH.3 as specified (the FK core anchors Q dispersion at 1 Hz)")
    print("=" * 70)
    run_loh3(SCEC_LOH_3(), "loh3_station.npz")

    print("=" * 70)
    print(f"LOH.3 with the Q dispersion re-anchored at f_ref = {F_REF} Hz")
    print("=" * 70)
    run_loh3(scec_loh_3_qref(), "loh3_station_qref.npz")

    print("PASS")
