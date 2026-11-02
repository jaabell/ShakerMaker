"""MPI regression: grouped Stage 2 tolerates ranks with no assigned work.

Run with more ranks than source-slot groups, for example:

    mpirun -n 4 python -m tests.engine.mpi_stage2_grouped_idle
"""

import os
import shutil

import h5py
import numpy as np
from mpi4py import MPI

from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.shakermaker import ShakerMaker
from shakermaker.slw_extensions import HDF5StationListWriter
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Discrete

comm = MPI.COMM_WORLD
rank, nprocs = comm.rank, comm.size

FK = dict(dt=0.01, nfft=256, dk=0.2, tb=40, smth=1, sigma=2)
STAGE0 = dict(delta_h=0.004, delta_v_rec=0.005, delta_v_src=0.2)
WORK = os.path.abspath(os.environ.get("SM_TEST_WORKDIR", "_stage2_grouped_idle_work"))


def model():
    crust = CrustModel(3)
    crust.add_layer(0.2, 1.32, 0.75, 2.4, 1000.0, 1000.0)
    crust.add_layer(0.8, 2.75, 1.57, 2.5, 1000.0, 1000.0)
    crust.add_layer(0.0, 5.50, 3.14, 2.5, 1000.0, 1000.0)
    t = np.linspace(0.0, 0.25, 50)
    src = PointSource([1.0, 2.0, 1.5], [353.0, 37.0, 91.0],
                      stf=Discrete(np.sin(np.pi * t / 0.25), t))
    stations = StationList([Station([0.3, 0.2, 0.0], metadata={"name": f"idle{i}"})
                            for i in range(3)], {})
    return ShakerMaker(crust, FaultSource([src], {}), stations)


def set_env(**values):
    for key in ("SM_S2_GROUP", "SM_S2_SPLIT", "SM_S2_ORDER", "SM_S2_CHUNK", "SM_S2_FUSED", "SM_S2_CONV"):
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = str(value)


def run_stage(current, stage, db, **kwargs):
    comm.Barrier()
    current.run_nearest(stage=stage, h5_database_name=db, showProgress=False, **FK, **kwargs)
    comm.Barrier()


def read_motion(filename):
    with h5py.File(filename, "r") as h5:
        return {field: h5[f"Data/{field}"][...] for field in
                ("velocity", "displacement", "acceleration")}


def main():
    assert nprocs > 1, "run with more than one MPI rank"
    if rank == 0:
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(WORK)
    comm.Barrier()

    db = os.path.join(WORK, "gf.h5")
    current = model()
    run_stage(current, 0, db, **STAGE0)
    run_stage(current, 1, db)

    motions = {}
    for tag, env in (("per_pair", {"SM_S2_GROUP": 0, "SM_S2_SPLIT": 0, "SM_S2_CONV": "fast", "SM_S2_FUSED": 0}),
                     ("grouped", {"SM_S2_GROUP": 1, "SM_S2_CONV": "fast", "SM_S2_FUSED": 0})):
        set_env(**env)
        output = os.path.join(WORK, f"{tag}.h5")
        run_stage(model(), 2, db, writer=HDF5StationListWriter(output),
                  writer_mode="progressive", tmin=0.0, tmax=4.0)
        if rank == 0:
            motions[tag] = read_motion(output)

    if rank == 0:
        for field, grouped in motions["grouped"].items():
            reference = motions["per_pair"][field]
            assert grouped.shape == reference.shape
            scale = max(np.abs(reference).max(), 1e-300)
            relative = np.abs(grouped - reference).max() / scale
            assert np.all(np.isfinite(grouped)) and np.all(np.isfinite(reference))
            assert relative < 1.0, f"{field}: {relative:.3e}"
        print("MPI_STAGE2_GROUPED_IDLE_PASS", flush=True)
        shutil.rmtree(WORK)
    ok = comm.bcast(True if rank == 0 else None, root=0)
    assert ok


if __name__ == "__main__":
    main()
