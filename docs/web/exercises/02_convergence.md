# Exercise 2: Numerical convergence

**Goal.** Build intuition for the two parameters that most often go wrong:
the wavenumber spacing `dk` and the FFT length `nfft`. You will *see* each
failure mode and learn the fix.

## Part A: convergence in `dk`

`dk` is the wavenumber sampling step. Too coarse and the discrete-wavenumber
sum aliases in space, producing a spurious oscillatory tail.

```python
import matplotlib.pyplot as plt
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList

def run_one(dk):
    crust = CrustModel(2)
    crust.add_layer(1.0, 4.0, 2.0,   2.6, 10000., 10000.)
    crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)
    src = PointSource([0, 0, 4], [90, 90, 0])
    sta = Station([0, 4, 0], metadata={"name": f"dk={dk}"})
    m = ShakerMaker(crust, FaultSource([src], {"name": "s"}), StationList([sta]))
    m.run(dt=0.005, nfft=2048, dk=dk, tb=500)
    return sta

fig = plt.figure()
for dk in [0.05, 0.1, 0.2, 0.4]:
    sta = run_one(dk)
    z, e, n, t = sta.get_response()
    plt.plot(t, z, label=f"dk = {dk}")
plt.xlim(0, 30); plt.legend(); plt.xlabel("t (s)"); plt.ylabel("v_Z")
plt.show()
```

**What you should see.** At `dk = 0.05` and `0.1` the traces overlap, they
have converged. At `dk = 0.2` and especially `0.4`, a long oscillatory tail
appears after the S arrival ($t \sim r/V_S$). That tail is **spatial
aliasing**, not physics.

**Rule of thumb.** Halve `dk` until the trace stops changing. Smaller `dk`
costs more (more wavenumber points) but is safe.

## Part B: convergence in `nfft`

`nfft` sets the record length $T = N_\text{FFT}\cdot dt$. Too short and the
wavefield wraps around, energy that should arrive after $T$ folds back to
the start.

```python
for nfft in [512, 1024, 2048, 4096]:
    crust = CrustModel(2)
    crust.add_layer(1.0, 4.0, 2.0,   2.6, 10000., 10000.)
    crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)
    src = PointSource([0, 0, 4], [90, 90, 0])
    sta = Station([0, 4, 0])
    m = ShakerMaker(crust, FaultSource([src], {"name": "s"}), StationList([sta]))
    m.run(dt=0.005, nfft=nfft, dk=0.1, tb=500)
    z, e, n, t = sta.get_response()
    plt.plot(t, z, label=f"nfft = {nfft}  (T = {nfft*0.005:.1f} s)")
plt.legend(); plt.xlabel("t (s)"); plt.show()
```

**What you should see.** With `nfft = 512` ($T = 2.56$ s) the record ends
before the surface waves, the late energy wraps to the front as spurious
early signal. By `nfft = 2048` ($T = 10.2$ s) the full wave train fits.
Doubling beyond that costs time with no benefit.

## The diagnostic map

| Symptom | Parameter | Fix |
|---|---|---|
| Oscillatory tail after S | `dk` too large | halve `dk` |
| Energy wraps / abrupt end at $T$ | `nfft` too small | increase `nfft` (or `sigma`) |
| Noisy decaying tail | `sigma` too large | reduce `sigma` |

See the full [diagnostic checklist](../guides/running.md#diagnostic-checklist).

## Checkpoint

You can recognise aliasing vs. wrap-around at a glance and know which knob
fixes each. Next: [a sedimentary basin](03_basin.md).
