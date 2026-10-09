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
5. Stage 2 without a writer returns on every rank instead of hanging.
6. 'legacy' and 'progressive' write the same time grid and the same values.
7. Running Stage 2 twice on the same model gives the same motions (the
   stations start from zero on every run).
8. Stage 1 batching (default): on a crust whose finite thickness exceeds the
   source-station distances, slots are grouped into multi-distance core calls
   and the Green's functions are bit-identical to one call per slot
   (SM_GF_BATCH=0).
9. The compact database (SM_GF_F32=1: float32, only the non-zero samples)
   gives bit-identical Stage 2 motions.
10. SM_S2_CONV=fast gives the same motions as the default convolution to
   1e-6 (single-precision FFT round-off), with both Stage 2 variants.
11. Stage 0 (default: hash-table grouping on rank 0 alone, the other ranks
   waiting) writes the same map as the legacy greedy (SM_S0_LEGACY=1), on the
   fault and on a DRM box with receivers every 5 m.
12. Grouped Stage 2 (SM_S2_GROUP=1, several chunks) gives the per-station
   path's motions to its single-precision round-off (1e-5), on the fault (few
   stations) and on the DRM box, with the same time grid.

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
from shakermaker.crustmodel import CrustModel
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
ENV_KEYS = ("SM_GF_STATIC", "SM_GF_RANK0_COMPUTE", "SM_S2_SPLIT", "SM_GF_BATCH", "SM_GF_F32",
            "SM_S2_CONV", "SM_S0_LEGACY", "SM_S2_GROUP", "SM_S2_ORDER", "SM_S2_CHUNK",
            "SM_S2_FUSED")

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

    # Stage 2 without a writer: every rank must leave (it used to hang ranks != 0).
    for tag, env in (("split", {"SM_S2_SPLIT": 1}), ("per_station", {"SM_S2_SPLIT": 0})):
        with_env(**env)
        t1 = time.perf_counter()
        run_stage(model, 2, p("gf_dynamic.h5"), writer=None, tmin=TMIN, tmax=TMAX)
        check(True, f"stage 2 without writer ({tag}) returns on every rank "
                    f"({time.perf_counter() - t1:.1f} s)")

    # Legacy and progressive share the time grid, and a model can be run twice.
    with_env()
    new_model = lambda: ShakerMaker(SCEC_LOH_1(), fault_sources(), stations_with_roundoff_depths())
    fields = ["velocity", "displacement", "acceleration"]

    def stage2(model_, mode, tmax, tag):
        out = p(f"modes_{tag}.h5")
        run_stage(model_, 2, p("gf_dynamic.h5"), writer=HDF5StationListWriter(out),
                  writer_mode=mode, tmin=TMIN, tmax=tmax)
        return read(out, ["Data"], fields) if rank == 0 else None

    def same(a, b):
        return all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in b)

    for tmax in (TMAX, 7.3):
        leg = stage2(new_model(), "legacy", tmax, f"legacy_{tmax}")
        pro = stage2(new_model(), "progressive", tmax, f"progressive_{tmax}")
        if rank == 0:
            check(same(leg, pro), f"legacy == progressive (shape and values), tmax={tmax}: "
                                  f"{[leg[k].shape for k in leg]} vs {[pro[k].shape for k in pro]}")
    model = new_model()
    runs = [stage2(model, mode, TMAX, f"reuse_{i}")
            for i, mode in enumerate(("legacy", "legacy", "progressive"))]
    if rank == 0:
        check(same(runs[1], runs[0]), "second legacy run on the same model == first")
        check(same(runs[2], runs[0]), "progressive run after legacy on the same model == first")

    # ---------------------------------------------------------------- 8
    # LOH.1 with an extra interface at 30 km (same half-space below): the finite
    # thickness then exceeds every distance and Stage 1 can group the slots.
    def deep_crust():
        c = CrustModel(3)
        c.add_layer(1.0, 4.0, 2.0, 2.6, 10000.0, 10000.0)
        c.add_layer(29.0, 6.0, 3.464, 2.7, 10000.0, 10000.0)
        c.add_layer(0.0, 6.0, 3.464, 2.7, 10000.0, 10000.0)
        return c

    calls = {"multi": 0}
    orig_multi = ShakerMaker._call_core_multi

    def counting_multi(self, *a, **k):
        calls["multi"] += 1
        return orig_multi(self, *a, **k)

    ShakerMaker._call_core_multi = counting_multi
    deep = ShakerMaker(deep_crust(), fault_sources(), stations_with_roundoff_depths())
    gfb = {}
    for tag, env in (("batched", {}), ("one_per_slot", {"SM_GF_BATCH": 0})):
        with_env(**env)
        calls["multi"] = 0
        db = p(f"gf_deep_{tag}.h5")
        run_stage(deep, 0, db, **STAGE0)
        run_stage(deep, 1, db)
        n_multi = comm.allreduce(calls["multi"], op=MPI.SUM)
        if rank == 0:
            with h5py.File(db.replace(".h5", "_gf.h5"), "r") as f:
                gfb[tag] = (f["/tdata"][...], f["/t0"][...], n_multi)
    ShakerMaker._call_core_multi = orig_multi
    if rank == 0:
        check(gfb["batched"][2] > 0 and gfb["one_per_slot"][2] == 0,
              f"stage 1 batching groups slots ({gfb['batched'][2]} multi-distance calls; "
              f"{gfb['one_per_slot'][2]} with SM_GF_BATCH=0)")
        check(np.array_equal(gfb["batched"][0], gfb["one_per_slot"][0])
              and np.array_equal(gfb["batched"][1], gfb["one_per_slot"][1]),
              "stage 1 batched GFs bit-identical to one core call per slot")

    # ---------------------------------------------------------------- 9
    with_env(SM_GF_F32=1)
    db32 = p("gf_f32.h5")
    run_stage(model, 0, db32, **STAGE0)
    run_stage(model, 1, db32)
    with_env()
    m64 = stage2(new_model(), "progressive", TMAX, "layout_f64")
    out = p("modes_layout_f32.h5")
    run_stage(new_model(), 2, db32, writer=HDF5StationListWriter(out),
              writer_mode="progressive", tmin=TMIN, tmax=TMAX)
    if rank == 0:
        with h5py.File(db32.replace(".h5", "_gf.h5"), "r") as f:
            ds = f["/tdata"]
            check(ds.dtype == np.float32 and int(ds.attrs.get("nt_full", 0)) == 2 * ds.shape[1],
                  f"compact database: float32, {ds.shape[1]} of {ds.attrs.get('nt_full')} samples")
        check(same(read(out, ["Data"], fields), m64), "compact database gives bit-identical motions")

    # ---------------------------------------------------------------- 10
    for tag, env in (("split", {"SM_S2_SPLIT": 1}), ("per_station", {"SM_S2_SPLIT": 0})):
        res = {}
        for conv in ("legacy", "fast"):
            with_env(SM_S2_CONV=conv, **env)
            out = p(f"conv_{tag}_{conv}.h5")
            run_stage(new_model(), 2, p("gf_dynamic.h5"), writer=HDF5StationListWriter(out),
                      writer_mode="progressive", tmin=TMIN, tmax=TMAX)
            if rank == 0:
                res[conv] = read(out, ["Data"], fields)
        if rank == 0:
            worst = max(max_rel(res["fast"][k], res["legacy"][k]) for k in res["legacy"])
            check(worst < 1e-6, f"SM_S2_CONV=fast vs default ({tag}): max rel diff {worst:.1e}")

    # ---------------------------------------------------------------- 3
    with_env()

    def drm_model():
        # A fresh model per Stage 2 run, so the two closes are independent.
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

    # ---------------------------------------------------------------- 12
    fields = ["velocity", "displacement", "acceleration"]
    for tag, grp, w_cls, mk in (
            ("fault", "Data", HDF5StationListWriter,
             lambda: ShakerMaker(SCEC_LOH_1(), fault_sources(), stations_with_roundoff_depths())),
            ("drm", "DRM_Data", DRMHDF5StationListWriter,
             lambda: ShakerMaker(SCEC_LOH_1(), fault_sources(),
                                 DRMBox([6.0, 8.0, 0.0], [4, 4, 2], [0.005] * 3, metadata={"name": "g"})))):
        with_env()
        db = p(f"gf_s2g_{tag}.h5")
        m = mk()
        run_stage(m, 0, db, **STAGE0)
        run_stage(m, 1, db)
        res = {}
        for mode, env in (("per_station", {"SM_S2_SPLIT": 0}), ("grouped", {"SM_S2_GROUP": 1, "SM_S2_CHUNK": 3})):
            with_env(**env)
            out = p(f"s2g_{tag}_{mode}.h5")
            run_stage(mk(), 2, db, writer=w_cls(out), writer_mode="progressive", tmin=TMIN, tmax=TMAX)
            if rank == 0:
                res[mode] = read(out, [grp], fields)
        if rank == 0:
            for k, v in res["grouped"].items():
                ref = res["per_station"][k]
                check(v.shape == ref.shape and max_rel(v, ref) < 1e-5,
                      f"stage 2 grouped vs per-station ({tag}), {k}: max rel diff "
                      f"{max_rel(v, ref) if v.shape == ref.shape else 'shape mismatch'}")
    with_env()

    # ---------------------------------------------------------------- 11
    s0_names = ["pairs_to_compute", "dh_of_pairs", "dv_of_pairs", "zrec_of_pairs",
                "zsrc_of_pairs", "pair_to_slot", "delta_h", "delta_v_rec", "delta_v_src",
                "nstations", "nsources"]
    box = DRMBox([6.0, 8.0, 0.0], [6, 6, 4], [0.005, 0.005, 0.005], metadata={"name": "s0"})
    for tag, m in (("fault", ShakerMaker(SCEC_LOH_1(), fault_sources(), stations_with_roundoff_depths())),
                   ("drm", ShakerMaker(SCEC_LOH_1(), fault_sources(), box))):
        maps = {}
        for mode, env in (("legacy", {"SM_S0_LEGACY": 1}), ("default", {})):
            with_env(**env)
            run_stage(m, 0, p(f"s0_{tag}_{mode}.h5"), **STAGE0)
            if rank == 0:
                maps[mode] = read(p(f"s0_{tag}_{mode}_map.h5"), [""], s0_names)
        if rank == 0:
            same = (len(maps["default"]) == len(s0_names) and
                    all(np.array_equal(maps["default"][k], maps["legacy"][k]) for k in maps["legacy"]))
            check(same, f"stage 0 default map identical to SM_S0_LEGACY=1 ({tag}, "
                        f"{len(maps['legacy']['/dh_of_pairs'])} slots)")
    with_env()

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
