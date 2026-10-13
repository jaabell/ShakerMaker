"""The compact Green's function layout (float32, only the smth*nfft non-zero
samples, attribute nt_full) reads back as exactly the array the core expects."""

import h5py
import numpy as np

from shakermaker.shakermaker import _read_tdata

def test_compact_layout_reads_back_exactly(tmp_path):
    rng = np.random.default_rng(0)
    nfft = 256
    full = np.zeros((2 * nfft, 9), dtype=np.float32)
    full[:nfft] = rng.standard_normal((nfft, 9)).astype(np.float32)
    path = tmp_path / "gf.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("old", data=full.astype(np.float64)[None])
        ds = f.create_dataset("new", data=full[:nfft][None])
        ds.attrs["nt_full"] = 2 * nfft
    with h5py.File(path, "r") as f:
        old = _read_tdata(f["old"], 0, None)
        new = _read_tdata(f["new"], 0, int(f["new"].attrs["nt_full"]))
    assert old.dtype == np.float64 and new.dtype == np.float32
    assert new.shape == old.shape == (2 * nfft, 9)
    # what the Fortran core receives (float32) is identical in both layouts
    assert np.array_equal(old.astype(np.float32), new)
