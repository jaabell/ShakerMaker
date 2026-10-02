"""Verify that MPI ranks without FFSP realizations remain safely idle."""

import numpy as np
from mpi4py import MPI

from tests.ffsp.test_all_realization_products import make_source


source = make_source(1, 1)
source.run()

if MPI.COMM_WORLD.rank == 0:
    assert source.all_realizations["stf_time"]["stf"].shape == (131072, 1)
    np.testing.assert_array_equal(source.all_realizations["realization_id"], [1])
    print("MPI_IDLE_RANK_CONTRACT_PASS")
