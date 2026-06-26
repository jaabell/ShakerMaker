import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.stf_extensions.brune import Brune
from shakermaker.stf_extensions.gaussian import Gaussian

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")
DT = 0.001

stfs = [
    ("Brune f0=1 Hz", Brune(f0=1.0, t0=0.5)),
    ("Brune f0=3 Hz", Brune(f0=3.0, t0=0.5)),
    ("Gaussian", Gaussian(t0=0.5, freq=4.0)),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for name, stf in stfs:
    stf.dt = DT
    t = np.asarray(stf.t)
    s = np.asarray(stf.data, dtype=float)
    peak = max(abs(s).max(), 1e-12)
    ax1.plot(t, s / peak, linewidth=1.6, label=name)
    f = np.fft.rfftfreq(len(s), DT)
    S = np.abs(np.fft.rfft(s))
    ax2.loglog(f, S / S.max(), linewidth=1.6, label=name)

ax1.set_xlim(0, 2)
ax1.set_xlabel("t (s)")
ax1.set_ylabel("normalised STF")
ax1.set_title("Time domain")
ax1.legend()
ax2.set_xlim(0.1, 50)
ax2.set_ylim(1e-3, 1.5)
ax2.set_xlabel("f (Hz)")
ax2.set_ylabel("normalised |S(f)|")
ax2.set_title("Spectrum")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "stf_overview.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("stf gallery OK")
