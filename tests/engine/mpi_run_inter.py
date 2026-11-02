"""MPI contract of run_inter: the same motion with any number of processes.

    mpirun -n 4 python -m tests.engine.mpi_run_inter

Rank 0 first runs the whole pipeline alone (static Stage 1, one call per grid
node); then all ranks run it together (dynamic Stage 1, every node of a pair of
depths in batched calls with the common wavenumber step; Stage 2 split by
sources and summed with one Reduce). Both must agree to the float32 round-off of
the database.
"""

import os
import shutil
import tempfile

import numpy as np

import shakermaker.shakermaker as sm
from shakermaker.shakermaker import comm, rank, nprocs
from tests.engine.test_run_inter import FK, _Capture, _model, _rel


def _run(db):
    w = _Capture("unused")
    _model().run_inter(stage='all', h5_database_name=db, delta_v_src=1e-5, dh_grid=0.01, order=3,
                       writer=w, showProgress=False, tmin=0.0, tmax=2.0, **FK)
    return w.data if rank == 0 else None


def main():
    assert nprocs >= 2, "run with mpirun -n 2 or more"
    work = comm.bcast(tempfile.mkdtemp(prefix="mpi_run_inter_") if rank == 0 else None, root=0)
    serial = None
    if rank == 0:
        saved = (sm.use_mpi, sm.nprocs)
        sm.use_mpi, sm.nprocs = False, 1            # rank 0 alone: static route, no MPI calls
        try:
            serial = _run(os.path.join(work, "serial"))
        finally:
            sm.use_mpi, sm.nprocs = saved
    comm.Barrier()
    par = _run(os.path.join(work, "mpi"))
    ok = True
    if rank == 0:
        err = _rel(par, serial)
        ok = err < 1e-5
        print(f"MPI_RUN_INTER_{'PASS' if ok else 'FAIL'} (rel {err:.2e})", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    ok = comm.bcast(ok, root=0)
    if not ok:
        comm.Abort(1)


if __name__ == "__main__":
    main()
