"""run_inter against run_nearest without reuse (all tolerances 1 cm).

A small DRM box under twelve sources of different depths, mechanisms and source
time functions. run_nearest with 1 cm tolerances computes one Green's function
per pair (the exact reference), here with SM_GF_XMAX=common, the wavenumber step
run_inter uses, so that only the interpolation differs; run_inter computes them
on a 10 m distance grid and interpolates in reduced time. With the coarse dk of
this test, the step itself (common against per call) changes the motion by
~3e-3; the interpolation by ~1e-3. The difference between two neighbouring
nodes (what a DRM layer turns into forces) must be as close as the traces."""

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
ENV = ("SM_S2_GROUP", "SM_S2_ORDER", "SM_S2_FUSED", "SM_S2_CHUNK", "SM_S2_CHUNK_MB", "SM_GF_XMAX")


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


def _model():
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
    return sm.ShakerMaker(crust, FaultSource(src, {"name": "f"}), box)


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
    for order in (3, 1):
        m = _model()
        db = str(tmp_path_factory.mktemp(f"inter{order}") / "gf")
        w = _Capture("unused")
        m.run_inter(stage='all', h5_database_name=db, delta_v_src=1e-5, dh_grid=0.010, order=order,
                    writer=w, showProgress=False, tmin=0.0, tmax=2.0, **FK)
        out[order] = w.data
    return out


def _rel(a, b):
    worst = peak = 0.0
    for i in b:
        assert np.array_equal(a[i][3], b[i][3]), f"time grid of station {i}"
        for k in range(3):
            worst = max(worst, np.abs(a[i][k] - b[i][k]).max())
            peak = max(peak, np.abs(b[i][k]).max())
    return worst / peak


def _rel_diff(a, b, pairs):
    """Worst error of the difference of two stations, relative to its peak."""
    worst = 0.0
    for i, j in pairs:
        for k in range(3):
            da, db_ = a[i][k] - a[j][k], b[i][k] - b[j][k]
            worst = max(worst, np.abs(da - db_).max() / max(np.abs(db_).max(), 1e-300))
    return worst


def test_cubic_matches_exact(runs):
    assert _rel(runs[3], runs["ref"]) < 2e-3


def test_linear_matches_exact(runs):
    assert _rel(runs[1], runs["ref"]) < 5e-3


def test_neighbour_difference(runs):
    pairs = [(i, i + 1) for i in range(0, len(runs["ref"]) - 1, 3)]
    assert _rel_diff(runs[3], runs["ref"], pairs) < 2e-2
