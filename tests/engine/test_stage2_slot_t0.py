"""Stage 2 timing of a reused slot across the 10 m depth tolerance.

A DRM box 6.6 km from a source 2.86 km deep, under a soft 200 m top layer: receivers above
10 m are computed at the free surface (surface branch of subtrav), the rest as
buried receivers, and subtrav's first arrival differs between the two branches
by about 73 ms here. With delta_v_rec = 5 m the 10 m receivers reuse the slot of
a 7.5 m (surface) representative, so they must be timed with that depth, not
their own (SM_S2_T0=own reproduces the old shift). Also pins the DRMBox depth
rounding and the /t0 check of Stage 2.
"""

import os
import shutil

import h5py
import numpy as np
import pytest

import shakermaker.shakermaker as sm
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.sl_extensions import DRMBox
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.stf_extensions import Discrete

FK = dict(dt=0.01, nfft=512, dk=0.2, tb=40, smth=1, sigma=2)
TMAX = 4.0
ENV = ("SM_S2_GROUP", "SM_S2_SPLIT", "SM_S2_T0", "SM_S2_T0_CHECK")


class _Capture(StationListWriter):
    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        self.data = {}

    def write_metadata(self, metadata):
        pass

    def write_station(self, station, index):
        z, e, n, t = station.get_response()
        self.data[index] = np.array([z, e, n])

    def close(self):
        pass


@pytest.fixture(scope="module")
def model_and_db(tmp_path_factory):
    crust = CrustModel(4)
    crust.add_layer(0.2, 1.32, 0.75, 2.4, 1000.0, 1000.0)
    crust.add_layer(0.8, 2.75, 1.57, 2.5, 1000.0, 1000.0)
    crust.add_layer(14.5, 5.50, 3.14, 2.5, 1000.0, 1000.0)
    crust.add_layer(0.0, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    t = np.arange(0, 0.1, 0.005)
    svf = np.sin(np.pi * t / 0.1)
    svf /= np.trapz(svf, dx=0.005)
    src = PointSource([0.0, 0.0, 2.858], [358.0, 28.0, 88.0], tt=0.0, stf=Discrete(svf, t))
    box = DRMBox([6.59, 0.0, 0.0], [2, 2, 12], [0.0025] * 3, {"name": "box"})
    model = sm.ShakerMaker(crust, FaultSource([src], {"name": "f"}), box)
    db = str(tmp_path_factory.mktemp("s2slot") / "gf")
    for k in ENV:
        os.environ.pop(k, None)
    model.run_nearest(stage=0, h5_database_name=db, delta_h=0.04, delta_v_rec=0.005,
                      delta_v_src=0.2, showProgress=False, tmin=0.0, tmax=TMAX, **FK)
    model.run_nearest(stage=1, h5_database_name=db, showProgress=False, tmin=0.0, tmax=TMAX, **FK)
    return model, box, db


def _run(model, db, monkeypatch, **env):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    w = _Capture("unused")
    model.run_fast(h5_database_name=db, writer=w, writer_mode="progressive", showProgress=False,
                   tmin=0.0, tmax=TMAX, **FK)
    return w.data


def _columns(box, z_hi, z_lo):
    """(station at z_hi, station at z_lo) for every column with both depths."""
    xyz = np.array([box.get_station_by_id(i).x for i in range(box.nstations)])
    cols = {}
    for i, p in enumerate(xyz):
        cols.setdefault((round(p[0], 7), round(p[1], 7)), {})[round(p[2], 7)] = i
    return [(c[z_hi], c[z_lo]) for c in cols.values() if z_hi in c and z_lo in c]


def _xlag(a, b):
    """Lag in samples of b against a: the larger of the east and north lags."""
    lags = []
    for c in (1, 2):                    # east, north
        xc = np.correlate(b[c], a[c], "full")
        lags.append(int(np.argmax(np.abs(xc))) - (len(a[c]) - 1))
    return max(lags, key=abs)


def test_drmbox_faces_share_depth_levels(model_and_db):
    _, box, _ = model_and_db
    z = [box.get_station_by_id(i).x[2] for i in range(box.nstations)]
    assert 0.01 in z and 0.009999999999999998 not in z
    assert len(set(z)) == 14            # 0 to 32.5 m every 2.5 m, both faces on the same floats


@pytest.mark.parametrize("path", [{"SM_S2_GROUP": 0, "SM_S2_SPLIT": 0},
                                  {"SM_S2_GROUP": 1},
                                  {"SM_S2_GROUP": 0, "SM_S2_SPLIT": 1}],
                         ids=["per_station", "grouped", "split"])
def test_reused_slot_is_timed_with_representative_depth(model_and_db, monkeypatch, path):
    model, box, db = model_and_db
    pairs = _columns(box, 0.0075, 0.01)
    assert pairs
    slot = _run(model, db, monkeypatch, **path)
    own = _run(model, db, monkeypatch, SM_S2_T0="own", **path)
    lag_slot = [_xlag(slot[i], slot[j]) for i, j in pairs]
    lag_own = [_xlag(own[i], own[j]) for i, j in pairs]
    assert max(abs(l) for l in lag_slot) <= 1                 # 7.5 m and 10 m arrive together
    assert min(lag_own) <= -5                                  # the old timing: 10 m ~73 ms early


def test_t0_check_stops_on_a_database_from_another_core(model_and_db, monkeypatch, tmp_path):
    model, _, db = model_and_db
    bad = str(tmp_path / "gf")
    for s in ("_map.h5", "_gf.h5"):
        shutil.copy(db + s, bad + s)
    with h5py.File(bad + "_gf.h5", "r+") as f:
        f["/t0"][...] = f["/t0"][...] + 0.2
    _run(model, db, monkeypatch, SM_S2_T0_CHECK="strict")      # its own database passes
    with pytest.raises(RuntimeError, match="another t0"):
        _run(model, bad, monkeypatch, SM_S2_T0_CHECK="strict")
