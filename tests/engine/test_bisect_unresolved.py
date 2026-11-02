"""Bisection of the distance nodes (run_inter scheme='tensor'): cells that still miss the value
tolerance when they cannot be split below dh_min are counted and reported with their error."""

import numpy as np

from shakermaker.run_inter_tensor import _bisect

BANDS = ((0.0, 1.0), (0.5, 1.0))


def _levels(fn, ncand=33, nf=40):
    x = np.arange(ncand)[:, None, None] * np.ones((1, 8, nf))
    f = np.linspace(0.0, 1.0, nf)[None, None, :]
    return [(fn(x, f) * (1.0 + 0j)).astype(np.complex64)]


def test_smooth_distance_variation_is_resolved():
    lev = _levels(lambda x, f: np.cos(0.05 * x * (1 + f)) + 2.0)
    sel, n, worst = _bisect(lev, [0, 8, 16, 24, 32], 3, 0.01, 0.05, BANDS)
    assert n == 0 and worst == 0.0 and len(sel) < 33


def test_fast_variation_is_reported():
    lev = _levels(lambda x, f: np.cos(2.9 * x * (1 + f)) + 0.1)
    sel, n, worst = _bisect(lev, [0, 8, 16, 24, 32], 3, 0.01, 0.05, BANDS)
    assert n > 0 and worst > 0.01
    assert len(sel) == 33                  # every candidate became a node, the floor is reached
