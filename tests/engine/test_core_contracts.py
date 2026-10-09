"""Contracts for the compiled FK core (shakermaker.core).

Each test pins a bug that was fixed or a property the engine relies on:

- subtrav must trace the first-arrival ray through the layers between source
  and receiver (it used to be off by one layer when they were not adjacent);
- subgreen must not return NaN at zero epicentral distance (0/0 in the
  Bessel terms);
- the OpenMP wavenumber loop must give bit-identical results for any number
  of threads;
- one call with several distances that share the wavenumber step returns,
  for each distance, exactly what a call with that distance alone returns
  (Stage 1 batches slots on this property).
"""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

core = pytest.importorskip("shakermaker.core")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Three layers over a half-space, velocity increasing with depth (km, km/s).
THICK = [1.0, 1.0, 1.0, 0.0]
VP = [3.0, 4.0, 5.0, 6.0]


def _direct_ray_time(x, layers):
    """First-arrival time of the direct ray through ``layers`` = [(h, v), ...].

    Solves for the ray parameter p with bisection; exact for x = 0.
    """
    hs = np.array([h for h, _ in layers])
    vs = np.array([v for _, v in layers])
    if x == 0.0:
        return float(np.sum(hs / vs))

    def offset(p):
        s = p * vs
        return np.sum(hs * s / np.sqrt(1.0 - s * s))

    lo, hi = 0.0, (1.0 / vs.max()) * (1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if offset(mid) < x:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    s = p * vs
    return float(np.sum(hs / (vs * np.sqrt(1.0 - s * s))))


@pytest.mark.parametrize("src_layer, x", [(2, 0.0), (3, 0.0), (4, 0.0), (4, 1.0)])
def test_subtrav_crosses_the_right_layers(src_layer, x):
    # Receiver at the surface (layer 1, 1-based); source at the top of
    # src_layer. The ray crosses layers 1 .. src_layer-1.
    tt0 = core.subtrav(len(THICK), VP, THICK, src_layer, 1, x)
    layers = list(zip(THICK[:src_layer - 1], VP[:src_layer - 1]))
    expected = _direct_ray_time(x, layers)
    assert tt0 == pytest.approx(expected, rel=1e-4), (
        f"subtrav={tt0:.6f} s, direct ray through layers 1..{src_layer - 1} "
        f"= {expected:.6f} s")


def _subgreen_loh1(x):
    # SCEC LOH.1 (elastic), source at 2 km (top of layer 3 after the split).
    d = [1.0, 1.0, 0.0]
    a = [4.0, 6.0, 6.0]
    b = [2.0, 3.464, 3.464]
    rho = [2.6, 2.7, 2.7]
    q = [10000.0, 10000.0, 10000.0]
    return core.subgreen(
        3, 3, 1, 2, 0, d, a, b, rho, q, q, 0.02, 512, 50, 1,
        2, 1, 1, 2, 0, 1, 0.1, 15.0, 0.9, x,
        0.0, 0.7853981633974483, 1.5707963267948966,
        0.0, 0.0, 0.0, x)


def test_subgreen_has_no_nan_at_zero_distance():
    tdata, z, e, n, t0 = _subgreen_loh1(0.0)
    for name, arr in (("tdata", tdata), ("z", z), ("e", e), ("n", n), ("t0", t0)):
        assert np.all(np.isfinite(arr)), f"non-finite values in {name} at x = 0"


_OMP_SNIPPET = (
    "import hashlib, sys;"
    "sys.path.insert(0, {root!r});"
    "from tests.engine.test_core_contracts import _subgreen_loh1;"
    "tdata, z, e, n, t0 = _subgreen_loh1(7.0);"
    "print(hashlib.sha256(tdata.tobytes() + t0.tobytes()).hexdigest())"
)


def _hash_with_threads(n):
    env = dict(os.environ, OMP_NUM_THREADS=str(n))
    out = subprocess.run([sys.executable, "-c", _OMP_SNIPPET.format(root=REPO_ROOT)],
                         env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_openmp_thread_count_does_not_change_results():
    assert _hash_with_threads(1) == _hash_with_threads(4)


def _subgreen_loh1_multi(xs):
    # Same model as _subgreen_loh1, several distances in one call.
    d = [1.0, 1.0, 0.0]
    a = [4.0, 6.0, 6.0]
    b = [2.0, 3.464, 3.464]
    rho = [2.6, 2.7, 2.7]
    q = [10000.0, 10000.0, 10000.0]
    xs = np.asarray(xs, dtype=np.float64)
    return core.subgreen(
        3, 3, 1, 2, 0, d, a, b, rho, q, q, 0.02, 512, 50, len(xs),
        2, 1, 1, 2, 0, 1, 0.1, 15.0, 0.9, xs,
        0.0, 0.7853981633974483, 1.5707963267948966,
        0.0, 0.0, 0.0, float(xs[0]))


def test_several_distances_per_call_match_one_call_each():
    # The wavenumber step is dk*pi/max(hs, x), with hs the finite thickness of
    # the model (2 km here). Distances up to hs share it, so the kernel is
    # evaluated once for all of them and only the Bessel terms differ.
    xs = [0.0, 0.3, 0.9, 1.4, 1.95]
    tdata, _, _, _, t0 = _subgreen_loh1_multi(xs)
    for i, x in enumerate(xs):
        one, _, _, _, t0_one = _subgreen_loh1_multi([x])
        assert np.array_equal(tdata[i], one[0]), f"tdata differs at x = {x}"
        assert t0[i] == t0_one[0], f"t0 differs at x = {x}"
