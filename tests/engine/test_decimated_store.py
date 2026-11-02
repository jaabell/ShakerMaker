"""Decimated Green's function database of run_inter scheme='tensor'.

With freq_cut the core leaves nothing above 1.4 freq_cut, so the database keeps one
sample every q dt and Stage 2 rebuilds the dt samples before using them. The motion
in the band must be the one of the full-rate database, the stored samples about q
times fewer, and the output dt unchanged. Also: run_inter sets its depth_tol on the
crust only while it runs (run_nearest keeps the CrustModel default)."""

import os

import h5py
import numpy as np
import pytest

from shakermaker.crustmodel import CrustModel
from shakermaker.run_inter_tensor import decimation, upsample
from tests.engine.test_run_inter import FK, ENV, _Capture, _model

FCUT, BAND = 8.0, 6.4
# nfft long enough for the stored samples plus the filter half-length (as in
# production, nfft dt >> tmax); with the short nfft of FK the core trace ends
# before them and run_inter keeps the full rate
FKD = dict(FK, nfft=4096)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    for k in ENV:
        os.environ.pop(k, None)
    out = {}
    for name, dec in (("full", False), ("dec", True)):
        m = _model()
        db = str(tmp_path_factory.mktemp(name) / "gf")
        w = _Capture("unused")
        m.run_inter(stage='all', h5_database_name=db, scheme='tensor', writer=w, showProgress=False,
                    tmin=0.0, tmax=2.0, store="f32", freq_cut=FCUT, decimate=dec, f_err=BAND, **FKD)
        out[name] = w.data
        out[name + "_tol"] = (m._crust.depth_tol, "depth_tol" in vars(m._crust))
        with h5py.File(db + "_gf.h5", "r") as hg, h5py.File(db + "_map.h5", "r") as hm:
            out[name + "_shape"] = hg["tdata"].shape
            out[name + "_q"] = int(hm["tensor"].attrs["decim"])
    return out


def _band(x, dt):
    f = np.fft.rfftfreq(2 * len(x), dt)
    return np.fft.irfft(np.fft.rfft(x, 2 * len(x)) * (f <= BAND), 2 * len(x))[:len(x)]


def test_same_motion_in_band(runs):
    a, b = runs["dec"], runs["full"]
    worst = peak = 0.0
    for i in b:
        assert np.array_equal(a[i][3], b[i][3]), "output time grid changed"
        for k in range(3):
            pa, pb = _band(a[i][k], FK["dt"]), _band(b[i][k], FK["dt"])
            worst = max(worst, np.abs(pa - pb).max()); peak = max(peak, np.abs(pb).max())
    assert worst / peak < 1e-6


def test_fewer_samples(runs):
    q = runs["dec_q"]
    assert runs["full_q"] == 1 and q == decimation(FK["dt"], FCUT)[0] >= 2
    assert runs["dec_shape"][:2] == runs["full_shape"][:2]
    assert runs["dec_shape"][2] < 1.5 * runs["full_shape"][2] / q


def test_upsample_band_limited_signal():
    dt, fcut = 0.0025, 25.0
    q, h = decimation(dt, fcut)
    t = np.arange(6000) * dt
    x = np.zeros_like(t)
    for f0 in (0.7, 6.3, 19.0, 31.0):                    # all below 1.4 fcut = 35 Hz
        x += np.sin(2 * np.pi * f0 * t + f0) * np.exp(-((t - 7.5) / 3.0) ** 2)
    y = upsample(x[::q], q, h, len(x))
    inner = slice(len(h), len(x) - len(h))
    assert q == 5 and np.abs(y - x)[inner].max() < 1e-5 * np.abs(x).max()


def test_short_nfft_keeps_full_rate(tmp_path):
    m = _model()
    db = str(tmp_path / "gf")
    m.run_inter(stage='all', h5_database_name=db, scheme='tensor', writer=_Capture("unused"),
                showProgress=False, tmin=0.0, tmax=2.0, store="f32", freq_cut=FCUT, f_err=BAND, **FK)
    with h5py.File(db + "_map.h5", "r") as hm:
        assert int(hm["tensor"].attrs["decim"]) == 1


def test_depth_tol_only_inside_run_inter(runs):
    assert runs["dec_tol"] == (CrustModel.depth_tol, False)
    assert CrustModel.depth_tol == 0.01
