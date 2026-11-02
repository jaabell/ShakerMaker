"""run_inter scheme='tensor' against run_nearest without reuse.

The model of test_run_inter (twelve sources at three depths, small DRM box) with a
finer wavenumber step (dk 0.05), so that the step itself, one per receiver depth
here and one per pair of depths in the reference, does not hide the interpolation
error. The iterative tensor grid must converge to the exact motion as the tolerance
tightens, keep the difference between neighbouring nodes within the tolerance, use
far fewer core calls than nodes, and give the same motion with the lossless zstd
database and nearly the same with ZFP."""

import os

import numpy as np
import pytest

from tests.engine.test_run_inter import FK as FK0, ENV, _Capture, _model, _rel, _rel_diff

FK = dict(FK0, dk=0.05)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    for k in ENV:
        os.environ.pop(k, None)
    out = {}
    m = _model()
    db = str(tmp_path_factory.mktemp("ref") / "gf")
    os.environ["SM_GF_XMAX"] = "common"
    m._crust.depth_tol = 1e-6          # the tolerance run_inter uses (its default depth_tol)
    m.run_nearest(stage=0, h5_database_name=db, delta_h=1e-5, delta_v_rec=1e-5, delta_v_src=1e-5,
                  showProgress=False, tmin=0.0, tmax=2.0, **FK)
    m.run_nearest(stage=1, h5_database_name=db, showProgress=False, tmin=0.0, tmax=2.0, **FK)
    os.environ.pop("SM_GF_XMAX", None)
    w = _Capture("unused")
    m.run_fast(h5_database_name=db, writer=w, writer_mode="progressive", showProgress=False,
               tmin=0.0, tmax=2.0, **FK)
    out["ref"] = w.data
    cases = {"loose_f32": dict(eps_v=1e-2, eps_s=5e-2, store="f32"),
             "loose_zstd": dict(eps_v=1e-2, eps_s=5e-2, store="zstd"),
             "loose_zfp": dict(eps_v=1e-2, eps_s=5e-2, store="zfp20"),
             "strict": dict(eps_v=1e-3, eps_s=5e-3, dh_min=0.005, dz_src_min=0.02, store="f32")}
    for name, kw in cases.items():
        m = _model()
        db = str(tmp_path_factory.mktemp(name) / "gf")
        w = _Capture("unused")
        m.run_inter(stage='all', h5_database_name=db, scheme='tensor', writer=w, showProgress=False,
                    tmin=0.0, tmax=2.0, **kw, **FK)
        out[name] = w.data
        import h5py
        with h5py.File(db + "_map.h5", "r") as hm:
            out[name + "_stats"] = dict(hm["tensor"].attrs)
    return out


def _pairs(r):
    return [(i, i + 1) for i in range(0, len(r) - 1, 3)]


def test_loose_tolerance(runs):
    assert _rel(runs["loose_f32"], runs["ref"]) < 5e-3
    assert _rel_diff(runs["loose_f32"], runs["ref"], _pairs(runs["ref"])) < 3e-2


def test_converges_with_tolerance(runs):
    e_loose = _rel(runs["loose_f32"], runs["ref"])
    e_strict = _rel(runs["strict"], runs["ref"])
    assert e_strict < 1e-3 and e_strict < 0.5 * e_loose


def test_fewer_calls_than_nodes(runs):
    st = runs["loose_f32_stats"]
    assert st["stat_calls"] < 0.2 * st["stat_computed"]


def test_zstd_is_lossless(runs):
    a, b = runs["loose_zstd"], runs["loose_f32"]
    for i in b:
        for k in range(3):
            assert np.array_equal(a[i][k], b[i][k])


def test_zfp_close(runs):
    assert _rel(runs["loose_zfp"], runs["loose_f32"]) < 1e-4
