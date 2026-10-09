"""Stage 0: the default grouping (hash table, rank 0 alone) writes the same
_map.h5 as the legacy greedy (SM_S0_LEGACY=1), dataset by dataset.

The cases put receivers every 2.5 m with a 5 m receiver-depth tolerance, so
some depth differences equal the tolerance exactly, and one case splits the
geometry into several blocks of stations."""

import h5py
import numpy as np
import pytest

import shakermaker.shakermaker as sm
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.sl_extensions import DRMBox

pytest.importorskip("numba")

DATASETS = ("pairs_to_compute", "dh_of_pairs", "dv_of_pairs", "zrec_of_pairs",
            "zsrc_of_pairs", "pair_to_slot", "delta_h", "delta_v_rec", "delta_v_src",
            "nstations", "nsources")


def _model(nx, nz, nel, h):
    crust = CrustModel(2)
    crust.add_layer(1.0, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    crust.add_layer(0.0, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    src = []
    for i in range(nx):
        for j in range(nz):
            s = (j + 0.5) / nz * 4.0
            src.append(PointSource([(i + 0.5) / nx * 8.0, 4.0 + s * np.cos(np.radians(40.0)),
                                    1.0 + s * np.sin(np.radians(40.0))], [0.0, 40.0, 90.0]))
    box = DRMBox([0.0, 0.0, 0.0], nel, [h, h, h], metadata={"name": "stage0"})
    return sm.ShakerMaker(crust, FaultSource(src, metadata={"name": "stage0"}), box)


def _maps_equal(a, b):
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert set(fa.keys()) == set(fb.keys())
        for k in DATASETS:
            x, y = np.asarray(fa[k][()]), np.asarray(fb[k][()])
            assert x.dtype == y.dtype and x.shape == y.shape, k
            assert np.array_equal(x, y), k


@pytest.mark.parametrize("tol", [(0.040, 0.005, 0.200), (0.040, 0.002, 0.200),
                                 (0.010, 0.002, 0.012)])
@pytest.mark.parametrize("block_pairs", [1 << 21, 600])
def test_default_map_equals_legacy(tmp_path, monkeypatch, tol, block_pairs):
    model = _model(12, 6, [6, 6, 6], 0.0025)
    monkeypatch.setattr(sm, "_S0_BLOCK_PAIRS", block_pairs)   # 600: 8 stations per block
    dh, dvr, dvs = tol
    monkeypatch.setenv("SM_S0_LEGACY", "1")
    model.gen_pairs(str(tmp_path / "legacy"), delta_h=dh, delta_v_rec=dvr,
                    delta_v_src=dvs, showProgress=False)
    monkeypatch.delenv("SM_S0_LEGACY")
    model.gen_pairs(str(tmp_path / "default"), delta_h=dh, delta_v_rec=dvr,
                    delta_v_src=dvs, showProgress=False)
    _maps_equal(tmp_path / "legacy_map.h5", tmp_path / "default_map.h5")
