"""MPI contract for the OP engine (run_nearest, stages 0-2).

Run from the repository root with at least 2 ranks, e.g.

    mpirun -n 4 python -m tests.engine.mpi_engine_contract

It checks properties that only show up under MPI:

1. Stage 1: the dynamic scheduler (default), the round-robin loop
   (SM_GF_STATIC=1) and the dynamic scheduler without rank 0 computing
   (SM_GF_RANK0_COMPUTE=0) write bit-identical Green's functions.
2. Stage 2: the source split (SM_S2_SPLIT=1) and the per-station loop
   (SM_S2_SPLIT=0) give the same motions (only the summation order differs),
   with receivers at depths that differ only by round-off (0.01 and
   0.03 - 0.02), which used to break the crust-model cache.
3. Stage 2 with the DRM writer in 'legacy' mode: a close() slower than 30 s
   (SM_TEST_SLOW_CLOSE seconds, default 35) still produces a complete file
   identical to a fast close.
4. No NaN anywhere.

Prints ENGINE_MPI_CONTRACT_PASS on rank 0 when everything holds. Work files go
to SM_TEST_WORKDIR (default ./_engine_contract_work, removed at the end); it
must be visible to every rank.
"""

import os
import shutil
import time

import h5py
import numpy as np
from mpi4py import MPI

from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.shakermaker import ShakerMaker
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter, HDF5StationListWriter
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian

comm = MPI.COMM_WORLD
rank, nprocs = comm.rank, comm.size

FK = dict(dt=0.02, nfft=512, dk=0.1, tb=50, smth=1)
STAGE0 = dict(delta_h=0.0025, delta_v_rec=0.0025, delta_v_src=0.2)
TMIN, TMAX = 0.0, 8.0
SLOW_CLOSE = float(os.environ.get("SM_TEST_SLOW_CLOSE", "35"))
ENV_KEYS = ("SM_GF_STATIC", "SM_GF_RANK0_COMPUTE", "SM_S2_SPLIT")

WORK = os.path.abspath(os.environ.get("SM_TEST_WORKDIR", "_engine_contract_work"))


def log(msg):
    if rank == 0:
        print(f"[engine-contract] {msg}", flush=True)


def stf():
    return Gaussian(t0=0.36, freq=1 / 0.06, M0=1000.0, derivative=False)


def fault_sources():
    # 12 point sources on a small dipping patch
    return FaultSource([PointSource([0.2 * i, 0.1 * j, 1.5 + 0.1 * j], [0.0, 90.0, 0.0], stf=stf())
                        for i in range(4) for j in range(3)], {})


def stations_with_roundoff_depths():
    # Three surface stations plus two whose depths differ only by round-off.
    z_a = 0.01
    z_b = 0.03 - 0.02          # 0.009999999999999998
    assert z_a != z_b and abs(z_a - z_b) < 1e-15
    xs = [([6.0, 8.0, 0.0], "a"), ([4.0, -3.0, 0.0], "b"), ([-5.0, 2.0, 0.0], "c"),
          ([6.0, 8.0, z_a], "deep_a"), ([6.0, 8.0, z_b], "deep_b")]
    return StationList([Station(x, metadata={"name": nm}) for x, nm in xs], {})


def with_env(**kv):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = str(v)


def run_stage(model, stage, db, **kw):
    comm.Barrier()
    model.run_nearest(stage=stage, h5_database_name=db, showProgress=False, **FK, **kw)
    comm.Barrier()


def read(path, groups, names):
    out = {}
    with h5py.File(path, "r") as f:
        for g in groups:
            for n in names:
                if f"{g}/{n}" in f:
                    out[f"{g}/{n}"] = f[f"{g}/{n}"][...]
    return out


def max_rel(a, b):
    scale = max(np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / scale)


class SlowCloseDRMWriter(DRMHDF5StationListWriter):
    def close(self):
        time.sleep(SLOW_CLOSE)
        super().close()


def main():
    assert nprocs >= 2, "run with mpirun -n 2 or more"
    if rank == 0:
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(WORK)
    comm.Barrier()
    p = lambda name: os.path.join(WORK, name)
    failures = []

    def check(cond, msg):
        if rank == 0:
            print(f"[engine-contract] {'ok  ' if cond else 'FAIL'} {msg}", flush=True)
            if not cond:
                failures.append(msg)

    # ---------------------------------------------------------------- 1 + 2
    model = ShakerMaker(SCEC_LOH_1(), fault_sources(), stations_with_roundoff_depths())
    gf = {}
    for tag, env in (("dynamic", {}), ("static", {"SM_GF_STATIC": 1}),
                     ("no_rank0", {"SM_GF_RANK0_COMPUTE": 0})):
        with_env(**env)
        db = p(f"gf_{tag}.h5")
        run_stage(model, 0, db, **STAGE0)
        run_stage(model, 1, db)
        if rank == 0:
            with h5py.File(db.replace(".h5", "_gf.h5"), "r") as f:
                gf[tag] = (f["/tdata"][...], f["/t0"][...])
    if rank == 0:
        base = gf["dynamic"]
        for tag in ("static", "no_rank0"):
            same = np.array_equal(base[0], gf[tag][0]) and np.array_equal(base[1], gf[tag][1])
            check(same, f"stage 1 '{tag}' GFs bit-identical to the dynamic scheduler")
        check(np.all(np.isfinite(base[0])), "stage 1 GFs have no NaN")

    motions = {}
    for tag, env in (("split", {"SM_S2_SPLIT": 1}), ("per_station", {"SM_S2_SPLIT": 0})):
        with_env(**env)
        out = p(f"motions_{tag}.h5")
        run_stage(model, 2, p("gf_dynamic.h5"), writer=HDF5StationListWriter(out),
                  writer_mode="progressive", tmin=TMIN, tmax=TMAX)
        if rank == 0:
            motions[tag] = read(out, ["Data"], ["velocity", "displacement", "acceleration"])
    if rank == 0:
        for k, v in motions["split"].items():
            ref = motions["per_station"][k]
            check(v.shape == ref.shape and max_rel(v, ref) < 1e-10,
                  f"stage 2 split vs per-station, {k}: max rel diff "
                  f"{max_rel(v, ref) if v.shape == ref.shape else 'shape mismatch'}")
            check(np.all(np.isfinite(v)), f"stage 2 {k} has no NaN")

    # ---------------------------------------------------------------- 3
    with_env()

    def drm_model():
        # A fresh model per Stage 2 run: in 'legacy' mode the stations keep
        # their response after the run, so reusing them would add to it.
        src = FaultSource([PointSource([0, 0, 2.0], [0.0, 90.0, 0.0], stf=stf())], {})
        drm = DRMBox([6.0, 8.0, 0.0], [4, 4, 2], [0.005, 0.005, 0.005], metadata={"name": "contract"})
        return ShakerMaker(SCEC_LOH_1(), src, drm)

    model = drm_model()
    db = p("gf_drm.h5")
    run_stage(model, 0, db, **STAGE0)
    run_stage(model, 1, db)
    drm_out = {}
    for tag, cls in (("fast", DRMHDF5StationListWriter), ("slow", SlowCloseDRMWriter)):
        out = p(f"drm_{tag}.h5drm")
        t1 = time.perf_counter()
        run_stage(drm_model(), 2, db, writer=cls(out), writer_mode="legacy", tmin=TMIN, tmax=TMAX)
        if rank == 0:
            log(f"stage 2 with {tag} writer close: {time.perf_counter() - t1:.1f} s")
            drm_out[tag] = read(out, ["DRM_Data", "DRM_QA_Data"],
                                ["velocity", "displacement", "acceleration"])
    if rank == 0:
        check(set(drm_out["slow"]) == set(drm_out["fast"]) and len(drm_out["fast"]) == 6,
              "slow-close .h5drm has every DRM dataset")
        for k, v in drm_out["slow"].items():
            ref = drm_out["fast"].get(k)
            check(ref is not None and np.array_equal(v, ref),
                  f"slow-close .h5drm {k} identical to the fast close")
            check(np.all(np.isfinite(v)), f".h5drm {k} has no NaN")

    comm.Barrier()
    if rank == 0:
        shutil.rmtree(WORK, ignore_errors=True)
        if failures:
            print(f"ENGINE_MPI_CONTRACT_FAIL ({len(failures)} checks failed)", flush=True)
        else:
            print("ENGINE_MPI_CONTRACT_PASS", flush=True)
    ok = comm.bcast(not failures if rank == 0 else None, root=0)
    if not ok:
        comm.Abort(1)


if __name__ == "__main__":
    main()
