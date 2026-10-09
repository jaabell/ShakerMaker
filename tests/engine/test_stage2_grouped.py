"""Grouped Stage 2 (SM_S2_GROUP=1) against the per-pair path.

The grouped path does the per-pair arithmetic in float64 and in another order,
so it matches the per-pair path (single-precision subfocal) to its round-off
and has exactly the same time grids. Its own variants (slot or source order,
compiled kernel or NumPy, any chunk size) only change the summation order.
The model has sources with different mechanisms and source time functions,
receivers of a small DRM box, and runs with tmin = 0 and tmin > 0."""

import os

import numpy as np
import pytest

import shakermaker.shakermaker as sm
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.sl_extensions import DRMBox
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.stf_extensions import Discrete

FK = dict(dt=0.01, nfft=256, dk=0.2, tb=40, smth=1, sigma=2)
ENV = ("SM_S2_GROUP", "SM_S2_ORDER", "SM_S2_FUSED", "SM_S2_CHUNK", "SM_S2_CHUNK_MB")


class _Capture(StationListWriter):
    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        self.data = {}

    def write_metadata(self, metadata):
        pass

    def write_station(self, station, index):
        z, e, n, t = station.get_response()
        self.data[index] = tuple(np.array(v) for v in (z, e, n, t))

    def close(self):
        pass


@pytest.fixture(scope="module")
def model_and_db(tmp_path_factory):
    crust = CrustModel(3)
    crust.add_layer(0.2, 1.32, 0.75, 2.4, 1000.0, 1000.0)
    crust.add_layer(0.8, 2.75, 1.57, 2.5, 1000.0, 1000.0)
    crust.add_layer(0.0, 5.50, 3.14, 2.5, 1000.0, 1000.0)
    rng = np.random.default_rng(3)
    src = []
    for i in range(4):
        for j in range(3):
            tr = 0.2 + 0.1 * rng.random()
            t = np.linspace(0, tr, int(tr / 0.005))
            svf = np.sin(np.pi * t / tr)
            svf /= np.trapz(svf, dx=0.005)
            src.append(PointSource([1.0 + 0.4 * i, 2.0 + 0.3 * j, 1.5 + 0.25 * j],
                                   [350 + 20 * rng.random(), 35 + 10 * rng.random(), 80 + 40 * rng.random()],
                                   tt=0.05 * i + 0.03 * j, stf=Discrete(svf * (0.5 + rng.random()), t)))
    box = DRMBox([0.3, 0.2, 0.0], [3, 3, 2], [0.005] * 3, {"name": "box"})
    model = sm.ShakerMaker(crust, FaultSource(src, {"name": "f"}), box)
    db = str(tmp_path_factory.mktemp("s2grouped") / "gf")
    for k in ENV:
        os.environ.pop(k, None)
    model.run_nearest(stage=0, h5_database_name=db, delta_h=0.004, delta_v_rec=0.005,
                      delta_v_src=0.2, showProgress=False, tmin=0.0, tmax=2.0, **FK)
    model.run_nearest(stage=1, h5_database_name=db, showProgress=False, tmin=0.0, tmax=2.0, **FK)
    return model, db


def _run(model, db, monkeypatch, tmin, **env):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    w = _Capture("unused")
    model.run_fast(h5_database_name=db, writer=w, writer_mode="progressive", showProgress=False,
                   tmin=tmin, tmax=2.0, **FK)
    return w.data


def _rel(a, b):
    worst = peak = 0.0
    for i in b:
        assert np.array_equal(a[i][3], b[i][3]), f"time grid of station {i}"
        for k in range(3):
            worst = max(worst, np.abs(a[i][k] - b[i][k]).max())
            peak = max(peak, np.abs(b[i][k]).max())
    return worst / peak


@pytest.mark.parametrize("tmin", [0.0, 0.25])
def test_grouped_matches_per_pair(model_and_db, monkeypatch, tmin):
    model, db = model_and_db
    ref = _run(model, db, monkeypatch, tmin)
    grp = _run(model, db, monkeypatch, tmin, SM_S2_GROUP=1)
    assert len(grp) == len(ref) == model._receivers.nstations
    assert _rel(grp, ref) < 1e-5


def test_grouped_variants_agree(model_and_db, monkeypatch):
    model, db = model_and_db
    base = _run(model, db, monkeypatch, 0.0, SM_S2_GROUP=1)
    for env in (dict(SM_S2_ORDER="source"), dict(SM_S2_CHUNK=7), dict(SM_S2_FUSED=0)):
        other = _run(model, db, monkeypatch, 0.0, SM_S2_GROUP=1, **env)
        assert _rel(other, base) < 1e-12, env
