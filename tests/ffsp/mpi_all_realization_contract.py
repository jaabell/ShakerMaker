"""MPI contract executable for FFSP all-realization products."""

import os
from pathlib import Path
import tempfile

import numpy as np
from mpi4py import MPI

from tests.ffsp.test_all_realization_products import make_source


comm = MPI.COMM_WORLD
shared_dir = Path(tempfile.gettempdir()) / "shakermaker_ffsp_mpi_contract"
if comm.rank == 0:
    shared_dir.mkdir(exist_ok=True)
comm.Barrier()
os.chdir(shared_dir)

source = make_source(1, 4)
source.run()

if comm.rank == 0:
    assert source.all_realizations["stf_time"]["stf"].shape == (131072, 4)
    assert source.all_realizations["spectrum"]["moment_rate_synth"].shape == (65536, 4)
    assert source.all_realizations["spectrum_octave"]["logmean_synth"].shape == (16, 4)
    np.testing.assert_array_equal(source.all_realizations["realization_id"], [1, 2, 3, 4])
    best_index = int(np.argmin(source.source_stats["source_score"]["pdf"]))
    assert source.best_realization["realization_id"] == best_index + 1
    forbidden = {"calsvf.dat", "calsvf_tim.dat", "logsvf.dat"}
    assert forbidden.isdisjoint(path.name for path in shared_dir.iterdir())
    print("MPI_FFSP_CONTRACT_PASS")
