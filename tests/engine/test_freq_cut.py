"""freq_cut of the FK core.

The core integrates every frequency on its own (the frequency loop of subfk.f does
not mix frequencies), then inverse-transforms and multiplies the trace by
exp(sigma*(t0 + t)) to undo the damping of the complex frequency. So:

1. before that correction (the damped trace), stopping the loop at 1.4 * fcut leaves
   the spectrum below fcut exactly as it was, and zero above 1.4 * fcut;
2. after it, the exp(sigma t) product spreads the cut a little into lower
   frequencies; the leak decays with the distance to the cut in units of
   df = 1 / (nfft dt), so with enough margin the band-limited trace (displacement,
   velocity, acceleration) is unchanged.
"""

import numpy as np
import pytest

from shakermaker import core
from shakermaker.crustmodel import CrustModel

DT, TB, SIGMA = 0.01, 40, 2.0


def _crust():
    c = CrustModel(3)
    c.add_layer(0.2, 1.32, 0.75, 2.4, 1000.0, 1000.0)
    c.add_layer(0.8, 2.75, 1.57, 2.5, 1000.0, 1000.0)
    c.add_layer(0.0, 5.50, 3.14, 2.5, 1000.0, 1000.0)
    return c


def _gf(fcut, nfft, zs=1.8, zr=0.0, xs=(2.0, 2.5, 3.1)):
    c = _crust(); c.split_at_depth(zs); c.split_at_depth(zr)
    td, z, e, n, t0 = core.subgreen(
        c.nlayers, c.get_layer(zs) + 1, c.get_layer(zr) + 1, 2, 0, c.d, c.a, c.b, c.rho, c.qa, c.qb,
        DT, nfft, TB, len(xs), SIGMA, 1, 1, 2, 0.0, 1.0, 0.1, 15.0, 0.9, np.array(xs), 0.0, np.pi / 2, 0.0,
        0.0, 0.0, xs[0], 0.0, 0.0, float(fcut))
    # the trace has nfft samples (the rest of the 2 nfft array is zero)
    return np.array(td[:, :, :nfft], dtype=np.float64), np.array(t0, dtype=np.float64)


def _undamped(td, t0, nfft):
    """The trace before the exp(sigma (t0 + t)) correction of subfk."""
    sig = SIGMA / (nfft * DT)
    t = np.arange(td.shape[-1]) * DT
    return td * np.exp(-sig * (t0[:, None, None] + t[None, None, :]))


@pytest.mark.parametrize("fcut", [5.0, 12.0, 20.0])
def test_damped_spectrum_unchanged_below_cut(fcut):
    nfft = 512
    full, t0a = _gf(0.0, nfft)
    cut, t0b = _gf(fcut, nfft)
    assert np.array_equal(t0a, t0b)
    F = np.fft.rfft(_undamped(full, t0a, nfft), axis=-1)
    C = np.fft.rfft(_undamped(cut, t0b, nfft), axis=-1)
    f = np.fft.rfftfreq(full.shape[-1], DT)
    low = f <= fcut
    assert np.abs(C[..., low] - F[..., low]).max() <= 1e-5 * np.abs(F[..., low]).max()
    high = f >= 1.41 * fcut
    assert np.abs(C[..., high]).max() <= 1e-5 * np.abs(F).max()


def test_band_limited_traces_equal_with_margin():
    """fcut 1.25 x band and a window long enough for many df of margin: the final
    trace low-passed to the band is unchanged in displacement, velocity, acceleration."""
    nfft, band = 4096, 16.0
    full, t0 = _gf(0.0, nfft); cut, _ = _gf(1.25 * band, nfft)
    f = np.fft.rfftfreq(full.shape[-1], DT)
    mask = (f <= band).astype(float)
    for d in range(3):
        w = (2j * np.pi * f) ** d
        a = np.fft.irfft(np.fft.rfft(full, axis=-1) * w * mask, full.shape[-1], axis=-1)
        b = np.fft.irfft(np.fft.rfft(cut, axis=-1) * w * mask, full.shape[-1], axis=-1)
        assert np.abs(a - b).max() <= 1e-3 * np.abs(a).max(), d


def test_zero_cut_is_previous_behaviour():
    nfft = 512
    a, _ = _gf(0.0, nfft)
    c = _crust(); c.split_at_depth(1.8); c.split_at_depth(0.0)
    td = core.subgreen(c.nlayers, c.get_layer(1.8) + 1, c.get_layer(0.0) + 1, 2, 0, c.d, c.a, c.b, c.rho, c.qa,
                       c.qb, DT, nfft, TB, 3, SIGMA, 1, 1, 2, 0.0, 1.0, 0.1, 15.0, 0.9, np.array((2.0, 2.5, 3.1)),
                       0.0, np.pi / 2, 0.0, 0.0, 0.0, 2.0, 0.0)[0]       # optional arguments omitted
    assert np.array_equal(a, np.array(td[:, :, :nfft], dtype=np.float64))
