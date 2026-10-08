# Engine run modes — legacy `run()` vs. the OP pipeline (`gen_pairs`/`compute_gf`/`run_fast`/`run_nearest`)

## What this is

`ShakerMaker` offers two fundamentally different ways to turn a
`(CrustModel, FaultSource, StationList)` triple into seismograms:

1. **Legacy / direct — `run()`**: every (source, receiver) pair gets its own
   independent FK evaluation. Simple, no intermediate files, but cost grows
   with `nsources × nstations` even when many pairs are geometrically
   equivalent (same horizontal distance and depths, just translated).
2. **OP pipeline ("nearest method") — `gen_pairs` → `compute_gf` →
   `run_fast`, or the single-call wrapper `run_nearest(stage=...)`**: in a
   horizontally-layered crust, a Green's Function depends only on
   `(horizontal distance, source depth, receiver depth)`. The OP pipeline
   deduplicates (source, receiver) pairs that share those three numbers
   within a tolerance, computes each unique Green's Function exactly once,
   and reuses it for every pair that maps to the same "slot." This is what
   makes DRM boxes / surface grids with thousands to millions of receivers
   tractable.

Pick **legacy** for small models, debugging, and validation. Pick the **OP
pipeline** for anything with many receivers, anything you'll run on HPC/MPI,
or anything where you'll reuse the same Green's Functions across multiple
source realizations (e.g. FFSP ensembles — see `03_ffsp_source.md` §
"Ensemble eficiente").

## Source of truth

`shakermaker/shakermaker.py`:
- `run` — line 513
- `gen_pairs` (Stage 0) — line 724
- `compute_gf` (Stage 1) — line 1098
- `run_fast` (Stage 2) — line 1363
- `run_nearest` (orchestrator) — line 1767
- `build_pair_to_slot_from_legacy_h5` (migration helper) — line 1974

## 1. Legacy: `run(...)`

```python
run(self, dt=0.05, nfft=4096, tb=1000, smth=1, sigma=2, taper=0.9, wc1=1,
    wc2=2, pmin=0, pmax=1, dk=0.3, nx=1, kc=15.0, writer=None,
    verbose=False, debugMPI=False, tmin=0.0, tmax=100, showProgress=True,
    writer_mode='progressive')
```

Every (source, receiver) pair is computed independently — "useful for
debugging and validating results against the OP pipeline," per its own
docstring. No HDF5 Green's-Function database is created; results go
straight to each `Station`'s in-memory response (retrievable via
`station.get_response()`) and, optionally, to a `writer`.

Key parameters (shared with every run tier below):
- `dt` — simulation time step (s). Sets the usable frequency band.
- `nfft` — number of FFT time points (**must be a power of 2**).
- `dk` — wavenumber sample interval, in units of `pi/x` (0.1-0.4 typical).
  Coarser = faster but less resolved; see `05_check_parameters.md`.
- `tb` — number of samples of pre-arrival zero padding.
- `sigma` — damps the trace at rate `exp(-sigma*t)` to reduce wrap-around.
- `taper` — low-pass taper fraction (0-1); sets `f_max = (1-taper)*f_Nyq`.
- `pmin`/`pmax` — phase-velocity integration bounds, in `1/Vs`.
- `kc` — evanescent wavenumber cutoff (`kmax = kc/hs`); needs `kc > 10`.
- `nx` — number of distance ranges to compute (structural; always `1` in
  every example in this repo).
- `writer` — a `StationListWriter` instance (see `07_writers_persistence.md`)
  to persist output; if `None`, results only live in each `Station` object.
- `writer_mode` — `'progressive'` (default; writes/frees each station as it
  finishes, O(1) RAM) or `'legacy'` (accumulates everything in memory,
  flushes once at the end).
- `tmin`/`tmax` — output time window (s).

**Never call `run()` directly on Windows-thread-unsafe stacks** — it
self-relaunches in a 64 MB-stack thread automatically on `win32` for large
`nfft`; see `11_mpi_and_hpc.md`.

## 2. The OP pipeline — three explicit stages

### Stage 0 — `gen_pairs(h5_database_name, delta_h=0.04, delta_v_rec=0.002, delta_v_src=0.2, npairs_max=200000, showProgress=True)`

Scans every (station, source) pair, groups geometrically-equivalent ones
into "slots" within the given tolerances, and writes the flat index
`pair_to_slot[i_station*nsources + i_psource] = k` to
`<h5_database_name>` (root name — becomes `<root>_map.h5`).

- `delta_h` — horizontal distance tolerance (**km**) for grouping pairs
  into the same slot.
- `delta_v_rec` — receiver depth tolerance (km).
- `delta_v_src` — source depth tolerance (km).
- `npairs_max` — kept for API compatibility; **not used internally**.

Algorithm: MPI-parallel vectorised geometry computation across all ranks,
then a Numba `@njit`-compiled greedy slot-assignment on rank 0 (falls back
to plain Python automatically if Numba isn't installed — 100-500× slower
but bit-identical results). This is cheap enough to run alone just to
inspect the dedup ratio before spending any real FK compute — see
`examples/06_nearest_method/notebooks/nearest_explained.ipynb`.

Writes to `<root>_map.h5`: `/pairs_to_compute` (n_slots,2),
`/dh_of_pairs`, `/dv_of_pairs`, `/zrec_of_pairs`, `/zsrc_of_pairs`
(n_slots,), `/pair_to_slot` (nsta*nsrc,), `/delta_h`, `/delta_v_rec`,
`/delta_v_src`, `/nstations`, `/nsources`.

**Picking tolerances**: tighter tolerances (smaller `delta_h` etc.) mean
less reuse (more unique slots, closer to legacy-run cost) but more
geometric fidelity; looser tolerances mean more reuse (faster) at the cost
of approximating nearby pairs with one shared Green's Function. There is no
universal default — size them to your problem: `examples/08_drm/drm_loh1.py`
uses `delta_h=40m, delta_v_rec=5m, delta_v_src=200m` (defined via
`_m = 0.001` km-per-metre) for a benchmark-accuracy DRM box. **A tolerance
many orders of magnitude smaller than your coordinate precision (e.g.
`1e-15` km) is not "extra safe" — it effectively disables reuse and can
signal a units bug; verify the arithmetic before trusting it.**

### Stage 1 — `compute_gf(h5_database_name, dt=0.05, nfft=4096, tb=1000, smth=1, sigma=2, taper=0.9, wc1=1, wc2=2, pmin=0, pmax=1, dk=0.3, nx=1, kc=15.0, verbose=False, debugMPI=False, showProgress=True)`

Computes the FK kernel (`tdata`) for every unique slot from Stage 0.
MPI-parallel: rank 0 coordinates and writes, worker ranks compute and send.
Appends `/tdata_dict` to `<root>_gf.h5` (a **second** file, sibling to
`<root>_map.h5` — read Stage 0's map first). Same FK parameters as `run()`.

### Stage 2 — `run_fast(h5_database_name, dt=..., ..., writer=None, writer_mode='progressive', tmin=0., tmax=100, ...)`

For each (station, source) pair, looks up the precomputed `tdata` via the
`pair_to_slot` index (O(1), no FK integration), convolves with each
source's STF, and accumulates the station response. **Requires
`/pair_to_slot` to already exist** in the database (written by Stage 0, or
retrofitted via `build_pair_to_slot_from_legacy_h5`, below).

MPI: single unified pass — every rank iterates all stations in canonical
order; owner of station `i` is `i % nprocs`; each station is computed,
sent/received, written, and cleared immediately (O(1) RAM per rank
regardless of station count). See `11_mpi_and_hpc.md` for the historical
hang bug this loop's error handling was hardened against.

### The orchestrator — `run_nearest(stage='all', h5_database_name=None, ...)`

Single entry point that runs stage `0`, `1`, `2`, `'0_1'` (0 then 1), or
`'all'` (0, 1, and 2) in one call — accepts the union of Stage 0's and
Stage 2's parameters. `h5_database_name` is **required**. This is the
recommended entry point for most scripts; call the stages **separately**
(via `gen_pairs`/`compute_gf`/`run_fast` directly, or `stage='0_1'` then
`stage=2`) when you're on HPC and want Stage 1-2 (the MPI-heavy,
compute-bound stages) scheduled with different resources than Stage 0, or
when — as with an FFSP realization ensemble — you want to compute Stage 0-1
**once** and re-run only Stage 2 per realization (see `03_ffsp_source.md`).

```python
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)  # always first
model.run_nearest(stage='all', h5_database_name='sim_001.h5',
                   dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax,
                   writer=my_writer, writer_mode='progressive')
```
Produces `sim_001_map.h5` + `sim_001_gf.h5` — pass the **root**, not either
suffixed filename. Launch with `mpiexec -n N python script.py` for
Stage 1-2 to actually parallelize (see `11_mpi_and_hpc.md`).

## 3. Migrating an old-format database — `build_pair_to_slot_from_legacy_h5`

```python
build_pair_to_slot_from_legacy_h5(self, h5_database_name,
                                   delta_h=None, delta_v_rec=None,
                                   delta_v_src=None, showProgress=True)
```

Older ("JAA/PXP") Green's-Function databases hold `/dh_of_pairs`,
`/zrec_of_pairs`, `/zsrc_of_pairs`, `/tdata_dict` but not the
`/pair_to_slot` index `run_fast`/`run_nearest(stage=2)` need. This method
adds `/pair_to_slot`, `/nstations`, `/nsources` **into the same file, in
place** (per its docstring) using an MPI + KDTree nearest-neighbor match
against the stored slot geometry, normalised by tolerance. After the call,
the same file works directly with `run_fast`/`run_nearest(stage=2)` — no
Green's-Function recomputation needed. If `delta_h`/`delta_v_rec`/
`delta_v_src` are left `None`, the tolerances **stored in the file** (the
original run's values) are reused.

## Gotchas

- **`nfft` must be a power of 2** in every tier — `check_parameters`
  (`05_check_parameters.md`) flags this as a hard error if violated.
- **Always pass the database root name**, not `<root>_map.h5` or
  `<root>_gf.h5` explicitly, to `gen_pairs`/`compute_gf`/`run_fast`/
  `run_nearest` — they append the suffixes themselves.
- **Stage 2 needs Stage 0's `/pair_to_slot`** to already exist in the
  `_map.h5` file — running `run_fast`/`run_nearest(stage=2)` cold, without
  ever running Stage 0 (or the legacy-migration helper), will fail.
- **Numba absence silently degrades Stage 0**, not fails it — 100-500×
  slower, same result. Install `numba` for any non-trivial receiver count.
- **`writer_mode='progressive'` is the default and the right choice for
  large campaigns** — `'legacy'` mode buffers every station in memory
  before writing, which defeats the whole point of O(1)-RAM staged
  processing on a big DRM box or surface grid.
- **A `_m`-style tolerance constant many orders of magnitude smaller than
  your coordinate precision is a red flag, not a safety margin** — see the
  Stage 0 tolerance note above; this exact mistake was found in two example
  scripts under `examples/14_SFSI/` during review (an extra unintended
  `/1e12` factor collapsing `delta_h` to `~1e-15` km).

## Combines with

- `05_check_parameters.md` — always run first, with the exact same
  `dt/nfft/dk/tb/tmax` you're about to pass to whichever tier you picked.
- `04_receivers.md` — any receiver type (`Station`, `DRMBox`, `SurfaceGrid`,
  `PointCloudDRMReceiver`) works with either tier; the OP pipeline is what
  makes the DRM/SurfaceGrid receiver types practical at real scale.
- `07_writers_persistence.md` — `writer`/`writer_mode` are shared parameters
  across `run()`, `run_fast()`, and `run_nearest()`.
- `03_ffsp_source.md` — the "ensemble efficient" pattern (Stage 0-1 once,
  Stage 2 per realization) is the main reason to use the OP pipeline
  explicitly staged rather than via `run_nearest(stage='all')`.
- `11_mpi_and_hpc.md` — which stages are MPI-parallel, how to launch, and
  the historical Stage 2 hang bug and its fix.
- `RECIPES.md` — worked minimal templates for each receiver × run-tier
  combination.

## See also

- `examples/05_engine_direct/` — legacy `run()` walkthrough.
- `examples/06_nearest_method/` — `run_nearest(stage='all')` vs. explicit
  `stage_by_stage.py`, plus `legacy_migration.py` and the
  `nearest_explained.ipynb` visual proof of Stage-0 dedup.
- `examples/08_drm/drm_loh1.py` — full OP pipeline against a real `DRMBox`.
