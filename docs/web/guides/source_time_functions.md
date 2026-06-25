# Source time functions

The STF is *how* the moment is released in time. The FK kernel always
computes the impulse (Dirac) response; the STF is convolved with it
afterwards, so one Green's function serves many STFs.

## Inputs: the five STFs

Real constructor signatures (from `shakermaker.stf_extensions`):

| STF | Signature | Key inputs |
|---|---|---|
| `Dirac` | `Dirac()` | none, the bare Green's function |
| `Brune` | `Brune(slip=1.0, f0=0.0, t0=0.0, dsigma=0.0, M0=1.0, Vs=0.0)` | corner freq `f0` **or** stress drop `dsigma`+`M0`+`Vs` |
| `Gaussian` | `Gaussian(t0=0.36, freq=16.6667, M0=1.0, derivative=False)` | pulse centre `t0`, width via `freq` |
| `Discrete` | `Discrete(data, t)` | your own samples `data` on time vector `t` |
| `SRF2` | `SRF2(srf2_file)` | path to an SRF2 file |

```python
from shakermaker.stf_extensions import Brune, Gaussian, Dirac

stf = Brune(f0=2.0, t0=0.0)            # corner frequency 2 Hz
src = PointSource([0, 0, 4], [90, 90, 0], stf=stf)
```

### Input details

| STF | Argument | Units | Meaning |
|---|---|---|---|
| `Brune` | `f0` | Hz | corner frequency (controls spectral roll-off) |
| `Brune` | `dsigma` | Pa | stress drop, alternative to `f0` (needs `M0`, `Vs`) |
| `Brune` | `slip`, `M0` | –, N·m | amplitude scaling |
| `Gaussian` | `t0` | s | pulse centre time |
| `Gaussian` | `freq` | Hz | inverse pulse width |
| `Gaussian` | `derivative` | bool | use the derivative (moment-rate) form |
| `Discrete` | `data`, `t` | –, s | arbitrary samples + time vector |

> `Brune` can be defined two ways: pass `f0` directly, **or** pass a stress
> drop `dsigma` together with `M0` and `Vs` and the corner frequency is
> derived. Don't mix the two.

## The full gallery

| SRF2 | Dirac |
|---|---|
| ![SRF2](../assets/images/SRF2.png) | ![Dirac](../assets/images/stf_dirac.png) |
| **Brune** | **Gaussian** |
| ![Brune](../assets/images/stf_brune.png) | ![Gaussian](../assets/images/stf_gaussian.png) |

*Reproduce with [`stf_gallery.py`](../examples/index.md#03-source-time-functions).*

## Reference

[Source time functions API →](../api/stf.md)
