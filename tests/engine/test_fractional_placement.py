# Placement of a trace at a real time, fraction of a sample included.
#
# add_shifted / Station.add_to_response place each trace with a 4-point Lagrange
# interpolator. Checked against the exact band-limited delay (FFT phase ramp) of a
# smooth pulse, for the whole-sample case (must be exact) and for fractions in (0, 1).

import numpy as np
import pytest

from shakermaker.station import Station, add_shifted, lagrange4_taps

DT = 0.0025


def _pulse(t, tc, f0):
    """Ricker wavelet centred at tc with peak frequency f0."""
    a = (np.pi * f0 * (t - tc)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _exact_on_grid(nout, t0, f0, n):
    """The pulse of a trace starting at t0 (n samples) sampled on the output grid."""
    tout = np.arange(nout) * DT
    y = _pulse(tout, t0 + 0.2, f0)
    y[(tout < t0) | (tout > t0 + (n - 1) * DT)] = 0.0
    return y


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999999])
def test_add_shifted_matches_exact_delay(frac):
    n, nout = 400, 1200
    f0 = 10.0                                   # Ricker, content up to ~25 Hz
    t0 = (300 + frac) * DT
    tk = t0 + np.arange(n) * DT
    x = _pulse(tk, t0 + 0.2, f0)
    buf = np.zeros(nout)
    add_shifted(buf, x, t0 / DT)
    ref = _exact_on_grid(nout, t0, f0, n)
    err = np.max(np.abs(buf - ref)) / np.max(np.abs(ref))
    assert err < 2e-3, err                      # cubic Lagrange at 10 Hz Ricker, dt 2.5 ms


def test_whole_sample_is_bitwise_previous_placement():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    for mu in (0.0, 7.0, 123.0, -5.0, 7.0 + 1e-9):
        a = np.zeros(800)
        add_shifted(a, x, mu)
        m = int(round(mu))
        b = np.zeros(800)
        k0 = max(0, -m)
        b[m + k0:m + k0 + len(x) - k0] += x[k0:]
        assert np.array_equal(a, b)


def test_taps_sum_to_one_and_reduce_to_identity():
    for d in np.linspace(0.01, 0.99, 25):
        m0, h = lagrange4_taps(10.0 + d)
        assert m0 == 9 and abs(h.sum() - 1.0) < 1e-12
    assert lagrange4_taps(10.0)[1] is None


def test_station_tmin_offset():
    """A trace starting at tmin is on the grid; tmin > 0 is honoured."""
    tmin, tmax = 1.0, 3.0
    s = Station([0.0, 0.0, 0.0])
    t = tmin + np.arange(200) * DT
    z = np.sin(2 * np.pi * 3.0 * (t - tmin))
    s.add_to_response(z, z, z, t, tmin, tmax)
    zz, ee, nn, tt = s.get_response()
    assert abs(tt[0] - tmin) < 1e-12
    assert np.allclose(zz[:200], z, atol=0, rtol=0)


def test_neighbours_differ_continuously():
    """Two traces 0.1 sample apart must differ by ~0.1 sample of slope, not 0 or 1 sample."""
    n, nout, f0 = 400, 1200, 10.0
    out = []
    for frac in (0.40, 0.50):
        t0 = (300 + frac) * DT
        x = _pulse(t0 + np.arange(n) * DT, t0 + 0.2, f0)
        b = np.zeros(nout)
        add_shifted(b, x, t0 / DT)
        out.append(b)
    ref = [_exact_on_grid(nout, (300 + f) * DT, f0, n) for f in (0.40, 0.50)]
    d_num, d_ref = out[1] - out[0], ref[1] - ref[0]
    rel = np.linalg.norm(d_num - d_ref) / np.linalg.norm(d_ref)
    assert rel < 0.02, rel
