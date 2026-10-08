# MPI and HPC — running ShakerMaker on a cluster safely

## What this is

ShakerMaker's OP pipeline (Stage 0/1/2, see `06_engine_run_modes.md`) and the legacy
`run()` path are both MPI-aware: they parallelize across `mpi4py` ranks with no code
changes needed beyond launching under `mpiexec`/`mpirun`. This file documents how MPI is
wired in, which stages actually parallelize what, the two real production hangs that have
hit this codebase on real clusters (one fixed, one still partially open), the environment
variables that matter on shared filesystems, and a real SLURM launch pattern.

Read this before submitting any multi-node job — a config mistake here doesn't crash
loudly, it **hangs** a job silently for hours until someone kills it.

## Source of truth

- `shakermaker/shakermaker.py:111-121` — MPI initialization (module-level `rank`,
  `nprocs`, `comm`, `use_mpi`).
- `shakermaker/shakermaker.py:83-109` — `_win_run` (Windows-only large-stack thread).
- `shakermaker/shakermaker.py:134-171` — `_dbg`, `_print_perf_stats` (the diagnostic
  instrumentation added after the Stage 2 hang investigation).
- `shakermaker/shakermaker.py:190-250` — `_wait_and_open_h5`, `_close_with_timeout`
  (cross-node filesystem hardening).
- `shakermaker/shakermaker.py:2455-2468` — `enable_mpi`, `mpi_is_master_process`,
  `mpi_rank`, `mpi_nprocs`.
- `examples/14_SFSI/DRM/run.sh`, `examples/14_SFSI/Surface/run.sh` — real SLURM launch
  scripts used in production.
- Commit `1093948` — "fix(engine): retry opening `_map.h5`/`_gf.h5` across a cross-node
  visibility lag" (this is `_wait_and_open_h5`).

## How MPI gets initialized

At import time (`shakermaker/shakermaker.py:111-121`):

```python
try:
    from mpi4py import MPI
    use_mpi = True
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
except (ImportError, RuntimeError):
    # RuntimeError covers mpi4py installed but no MPI runtime available
    use_mpi = False
    rank   = 0
    nprocs = 1
```

There is **no manual MPI setup required** — importing `shakermaker` and running the
script under `mpiexec -n N python script.py` is enough; every rank sees `rank`/`nprocs`
correctly. If `mpi4py` isn't installed (or no MPI runtime is available), ShakerMaker
degrades to a single serial process automatically (`use_mpi=False`, `rank=0`,
`nprocs=1`) — the same script runs fine with plain `python script.py`, just without
parallelism. `ShakerMaker.__init__` also stores `self._mpi_rank`/`self._mpi_nprocs` and
exposes them as `model.mpi_rank`/`model.mpi_nprocs`/`model.mpi_is_master_process()`.

## Which stages actually parallelize

| Call | MPI behavior |
|---|---|
| `run()` (legacy) | Pair-by-pair; parallelizes across (source, receiver) pairs. Full detail in `06_engine_run_modes.md`. |
| `gen_pairs()` (Stage 0) | Geometry computation (distances) is vectorised and distributed across ranks (`Gatherv` to rank 0); the greedy slot-finding itself runs **only on rank 0**, Numba-JIT-compiled if `numba` is installed (100-500x vs. plain Python — same algorithm, bit-for-bit identical slots either way). All ranks synchronise on a final `Barrier`. |
| `compute_gf()` (Stage 1) | Dynamic master-worker: rank 0 hands the next slot to whichever worker finishes first (longest slots first), workers send zlib-compressed slots, and rank 0 also computes in its main thread while a receiver thread does MPI and the HDF5 writes (`write_direct_chunk`). Needs the `threadsafe` core wrappers and `MPI_THREAD_MULTIPLE`. `SM_GF_STATIC=1` restores the round-robin loop; `SM_GF_RANK0_COMPUTE=0` keeps rank 0 as a pure receiver; `SM_GF_COSTFILE` orders slots by a previous run's `<gf_file>.slotcost.npy`. |
| `run_fast()` (Stage 2) | With fewer stations than ranks, every station's sources are split over all ranks; each rank sums its share on the station's output grid and an MPI `Reduce` adds the parts on rank 0, which writes the station. Otherwise (e.g. a DRM box) station `i` belongs to rank `i % nprocs`. `SM_S2_SPLIT=1`/`0` forces either mode. Split crust models are cached per exact depth pair. |

Launch any of them the same way: `mpiexec -n N python script.py` (or `mpirun` — see the
SLURM example below). No special flag is needed to "enable" MPI; it activates
automatically whenever `nprocs > 1`.

The FK kernel is also OpenMP-parallel. On 16-core / 32-thread nodes the measured best
layouts were 16 ranks x 2 threads per node up to 2 nodes and 8 x 4 from 4 nodes, with
`OMP_PLACES=threads OMP_PROC_BIND=close`. Measurements, launch lines and compiler flags:
`docs/web/guides/performance.md`.

## The `<root>_map.h5` / `<root>_gf.h5` split

The OP pipeline always writes **two separate HDF5 files** from one name you pass in:
`gen_pairs`/Stage 0 writes `<root>_map.h5` (lightweight — geometry + the
`pair_to_slot` index, loadable without touching any Green's Function data), and
`compute_gf`/Stage 1 appends `/tdata_dict` to `<root>_gf.h5` (the actual, much larger,
computed Green's Functions). You always pass the **root** (no `.h5` suffix, or the
methods strip it) to `run_nearest`/`gen_pairs`/`compute_gf`/`run_fast` — never the
suffixed filename directly. This split exists so Stage 2 (and any downstream tool) can
cheaply inspect/reuse the pairing map without opening the (potentially huge) GF file.

## Environment: `HDF5_USE_FILE_LOCKING=FALSE`

Set this **before** launching any OP-pipeline MPI job on a shared cluster filesystem
(NFS, Lustre, etc.):

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

With dozens to hundreds of MPI ranks opening the *same two* `.h5` files concurrently
(`<root>_map.h5`, `<root>_gf.h5`), HDF5's default file-locking behavior over a network
filesystem routinely causes spurious lock-acquisition failures that have nothing to do
with real data races (ShakerMaker's own read/write ordering — rank 0 writes, others read
after a `Barrier` — already prevents real concurrent-write corruption). Both real SLURM
scripts in this repo (`examples/14_SFSI/DRM/run.sh`, `Surface/run.sh`) set this
unconditionally before `mpirun`.

## Known hang #1 (fixed): Stage 2 (`run_fast`) hangs after computation finishes

Summary (fix: commit `ccbe2ad`, upstream PR #53):

- **Symptom**: the science finishes (`.h5` output is complete and valid), the log prints
  `"...done. Total time: ... s"`, but the SLURM job never releases its nodes — it hangs
  forever right before the "Performance statistics" block.
- **Root cause**: `_print_perf_stats()` calls `comm.Reduce(...)` unconditionally on every
  rank — a blocking MPI collective. If **any single rank** (out of possibly 100+) raises
  an unhandled exception anywhere before reaching that call — e.g. a transient I/O error
  opening the shared `.h5` files under heavy concurrent access — it used to die silently
  with no `comm.Abort()`, and every other rank would then wait at `comm.Reduce()` forever,
  since a hang (unlike a crash) leaves no trace and nothing to catch.
- **Fix**: `run_fast` now wraps every previously-unprotected block (file open/read
  preamble, the per-subfault loop body, the MPI send/recv blocks) in
  `try/except Exception: traceback.print_exc(); comm.Abort()`, so any real fault now kills
  the job cleanly and loudly within seconds instead of hanging silently. A final
  `comm.Barrier()` was also added to Stage 2, matching what Stage 1 already had.
- **Operational lesson** (not a code fix — a job-sizing one): with few stations Stage 2
  splits each station's sources over all ranks, but past one node it is limited by reading
  the GF database (one node with 16 ranks was the fastest on two finite-fault cases with
  3 stations and 4 096 or 32 768 subfaults). A Stage 2-only run does not need many nodes,
  and fewer ranks also means fewer that could hit a transient fault.

## Known hang #2 (partially open): a second, still-unexplained hang

`shakermaker/shakermaker.py:141-144` documents this directly, in the docstring of the
`_dbg()` diagnostic helper:

> "...this is for the *second*, still-unexplained hang that reproduces even with the
> `comm.Abort()` and `close()`-timeout fixes in place, so the culprit must be somewhere
> neither of those covers."

Two further hardening layers exist specifically to fight this:

- **`_close_with_timeout(f, timeout=30.0, label="")`** (`shakermaker.py:217-250`): HDF5
  file `close()` can itself block indefinitely under NFS close-to-open consistency — a
  rank stuck inside `close()` never reaches the `comm.Reduce()` calls, and being a blocked
  call rather than an exception, no `try/except` can catch it either. This helper runs
  `close()` in a daemon thread and gives up after `timeout` seconds, printing a
  `[WARNING]` and moving on rather than blocking the job forever. It deliberately leaks
  the stuck thread/handle rather than risk hanging the whole MPI collective that follows.
- **`SHAKERMAKER_PERF_STATS_DEBUG=1`** environment variable: turns on `_dbg()`, a
  per-rank, timestamped, `flush=True` print around every step of `_print_perf_stats`
  (before/after each `comm.Reduce` call). Set this on a job that reproduces the hang to
  get a per-rank timeline that pinpoints exactly which rank got stuck at which collective
  — the only way to diagnose a hang, since by definition it produces no exception.

**Practical takeaway**: the two fixes above reduce the blast radius (a stuck `close()`
now times out instead of hanging forever) but do not claim to fully close the root cause.
If Stage 2 (or any stage) hangs on your cluster after already having these fixes,
re-launch with `SHAKERMAKER_PERF_STATS_DEBUG=1` and capture the per-rank log before
escalating — that log is the only diagnostic signal this problem class produces.

## Cross-node filesystem visibility lag

`_wait_and_open_h5(path, mode, timeout=60.0, poll_interval=1.0)`
(`shakermaker.py:190-214`, from commit `1093948`): rank 0 creates and closes
`_map.h5`/`_gf.h5`, then a `comm.Barrier()` releases the other ranks — but the Barrier
only synchronises **MPI**, not the shared filesystem. On NFS (and similar) across compute
nodes, a rank on a *different node* than rank 0 can hit a transient `FileNotFoundError`
right after the barrier even though the file genuinely exists, simply because that node's
filesystem view hasn't caught up yet. `_wait_and_open_h5` polls for the file's existence
and retries the open (up to `timeout` seconds) instead of failing on the very first
attempt. This is used internally wherever a non-rank-0 process needs to open a file rank 0
just wrote — you don't call it directly from user code.

## Minimal working example — SLURM launcher

Adapted from `examples/14_SFSI/DRM/run.sh` (a real, previously-run production script).
**The venv path and the `mpirun` binary path are specific to that one cluster — do not
copy them verbatim onto a different machine; only the structure is the reusable part.**

```bash
#!/bin/bash
#SBATCH --job-name=my_shakermaker_job
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --output=log_my_shakermaker_job.log
pwd; hostname; date
SECONDS=0

source ~/path/to/your/venv/bin/activate

export HDF5_USE_FILE_LOCKING=FALSE
# Uncomment only if you need to diagnose a hang (see "Known hang #2" above):
# export SHAKERMAKER_PERF_STATS_DEBUG=1

mpirun python -s my_script.py

echo "Elapsed: $SECONDS seconds."
date
```

5 nodes × 16 tasks/node = 80 MPI ranks in the reference script — size this to your actual
station/subfault count (see the Stage 2 sizing note above; requesting more
ranks than you can actually use wastes allocation and adds failure surface for free).

## Known gotchas

- **Windows large-stack thread (`_win_run`)**: on Windows only, `run`, `compute_gf`,
  `run_fast`, and `run_nearest` automatically relaunch themselves inside a thread with a
  64 MB stack (`_WIN_STACK_SIZE`) before doing any real work, because the Fortran FK core
  needs more stack than Windows' default thread stack provides once `nfft` gets large.
  This is fully automatic — the guard checks
  `getattr(threading.current_thread(), '_sm_large_stack', False)` so it only relaunches
  once, not recursively. **Never bypass this manually** (e.g. by calling an internal
  method directly to "skip the overhead") — doing so risks a silent segfault with large
  `nfft`, not a clean error. This mechanism is a no-op on Linux/macOS (`_win_run` just
  calls the function directly if `sys.platform != 'win32'`).
- **Numba is optional, not required.** `gen_pairs` (Stage 0) tries
  `from numba import njit`; if unavailable, it falls back to a pure-Python
  implementation of the exact same greedy algorithm (correct, just 100-500x slower). No
  code change needed either way — just `pip install numba` for the speedup on large
  campaigns. A missing Numba install is never a correctness problem, only a speed one.
- **Plotting inside an MPI script is safe.** `ZENTPlot`, `StationPlot`, and `SourcePlot`
  (see `10_plotting.md`) all check MPI internally and only draw on rank 0 — you can call
  them unconditionally at the end of an MPI-launched script without wrapping them in an
  `if rank == 0:` guard yourself (though doing so anyway doesn't hurt).
- **`build_pair_to_slot_from_legacy_h5`** (see `06_engine_run_modes.md`) is itself
  MPI+KDTree parallel (each rank builds an identical `cKDTree`, queries its station
  slice, results are `Gatherv`'d to rank 0) — same launch pattern (`mpiexec -n N`)
  applies when migrating a large legacy database.

## Combines with

- `06_engine_run_modes.md` — the stage semantics this file assumes (Stage 0/1/2, `run()`
  vs `run_nearest`).
- `08_drm_workflow.md` and `RECIPES.md` — real DRM/HPC campaigns (e.g.
  `examples/14_SFSI/DRM/drm.py`) combine `PointCloudDRMReceiver` + the OP pipeline +
  this MPI/SLURM pattern together.
- `07_writers_persistence.md` — `writer_mode='progressive'` (the default almost
  everywhere) keeps peak per-rank RAM O(1) during Stage 2, which matters more the more
  ranks/stations a job has.

## See also

- `examples/14_SFSI/DRM/run.sh`, `examples/14_SFSI/Surface/run.sh` — real launch scripts.
- `examples/EXAMPLES_REFERENCE.md` §14_SFSI — documents that neither HPC script in that
  folder ships an output artifact in-repo but both are confirmed to have run to
  completion on a cluster.
