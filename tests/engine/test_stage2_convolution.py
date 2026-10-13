"""The Stage 2 convolution with each source time function resampled once
(SM_S2_CONV=fast) matches SourceTimeFunction.convolve on the samples that
add_to_response keeps, to single-precision FFT round-off."""

import numpy as np
import pytest

from shakermaker.shakermaker import _STFCache, _conv3
from shakermaker.stf_extensions.gaussian import Gaussian

@pytest.mark.parametrize("shift", [0.0, 0.37, 1.2345678])
def test_fast_convolution_matches_convolve(shift):
    rng = np.random.default_rng(1)
    dt, n = 0.01, 2048
    stf = Gaussian(t0=0.3, freq=20.0, M0=1.0, derivative=False)
    stf.dt = dt
    z, e, nn = (rng.standard_normal(n).astype(np.float32) for _ in range(3))
    t = np.arange(0, n * dt, dt) + shift          # the pair's own grid, as in run_fast
    dti = t[1] - t[0]
    keep = 700
    ref = np.vstack([stf.convolve(c, t)[:keep] for c in (z, e, nn)])
    out = _conv3(_STFCache(), stf, z, e, nn, dti, keep)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() <= 1e-6 * np.abs(ref).max()
