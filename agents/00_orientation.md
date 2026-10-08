# Orientation — how to configure and run ShakerMaker

## What this is

ShakerMaker synthesizes 3-component (Z, E, N) seismograms in a 1-D layered
viscoelastic half-space using the **Frequency-Wavenumber (FK)** method. A
compiled Fortran77/90 core (fork of L. Zhu's SLU code, `shakermaker/core/`,
bound via f2py) does the physics; a Python object layer on top handles crust
models, sources (point, finite-fault, stochastic FFSP), receivers (single
stations, DRM boxes, surface grids, imported FEM point clouds), source-time
functions, output writers, MPI parallelism, and export to **SW4** and
**H5DRM** (OpenSees `H5DRMLoadPattern`).

This `agents/` folder is a from-source-code knowledge base, one file per
functionality, so any agent (human-directed or autonomous) can configure and
run any combination of ShakerMaker's capabilities without re-deriving it from
scratch each time. It complements — and does not replace — the actual source,
which remains the ground truth; every claim in these files is grounded in a
specific file/line and should be re-verified if the code changes.

## Package layout

```
shakermaker/
  shakermaker.py        <- class ShakerMaker: THE engine. See 05_check_parameters.md,
                            06_engine_run_modes.md, 08_drm_workflow.md, 09_sw4_export.md.
  crustmodel.py          <- CrustModel. See 01_crust_model.md.
  pointsource.py          <- PointSource. See 02_sources_and_stf.md.
  faultsource.py          <- FaultSource. See 02_sources_and_stf.md.
  ffspsource.py           <- FFSPSource (2701 lines). See 03_ffsp_source.md.
  station.py               <- Station, StationObserver. See 04_receivers.md.
  stationlist.py            <- StationList. See 04_receivers.md.
  sourcetimefunction.py      <- SourceTimeFunction base class. See 02_sources_and_stf.md.
  stf_extensions/              <- Dirac, Discrete, Brune, Gaussian, SRF2. See 02_sources_and_stf.md.
  sl_extensions/                 <- DRMBox, SurfaceGrid, PointCloudDRMReceiver. See 04_receivers.md.
  slw_extensions/                  <- HDF5StationListWriter, DRMHDF5StationListWriter. See 07_writers_persistence.md.
  cm_library/                        <- SCEC_LOH_1/3, AbellThesis, SOCal_LF presets. See 01_crust_model.md.
  crust1/                              <- CRUST 1.0 global lookup. See 01_crust_model.md.
  sw4_exporter/                          <- Export to SW4 + rebuild .h5drm after. See 09_sw4_export.md.
  tools/plotting.py                        <- ZENTPlot, StationPlot, SourcePlot. See 10_plotting.md.
  core/                                      <- FK Fortran (f2py). Compiled, do not edit by hand.
  ffsp/                                        <- FFSP Fortran (f2py) -> ffsp_core. Compiled.
```

## Coordinate system and units — get these wrong and everything else fails silently

- **Position** `x = [x, y, z]`: **km**. `x = North, y = East, z = Down`
  (positive down). This is the ShakerMaker convention everywhere *except*
  inside FFSP's raw kernel output (metres) — see the gotchas below.
  Positions vs motion, and the order of components in each output, are in
  `12_coordinates_and_conventions.md`.
- **Angles** (`strike, dip, rake`): **degrees** at the API boundary
  (`PointSource.__init__`), but stored internally as **radians** — the
  `.angles` property returns radians, not what you passed in.
- **Velocities**: km/s. **Density**: g/cm³. **Time**: s. **Frequency**: Hz.
  **Attenuation**: dimensionless quality factors `Qp`/`Qs` (named `qp`/`qs`
  in `CrustModel.add_layer`, **not** `gp`/`gs`).
- **FFSP is the one exception to the km convention**: `FFSPSource.
  get_subfaults()`/`get_realization(i)` return `x, y, z` in **METERS**
  (`shakermaker/ffsp/ffsp_wrapper.f90:294-296` multiplies by 1000 internally).
  Divide by `1e3` before handing them to `PointSource`. See
  `03_ffsp_source.md` — this is the single most-repeated unit bug across the
  whole codebase and its example history.
- **SW4 uses the same frame as ShakerMaker**: `x = North, y = East,
  z = down` (SW4's default `az = 0`), in metres, in a local cartesian box
  with the origin at a corner. ShakerMaker's exporter (`09_sw4_export.md`)
  transforms with a pure **translation** (km → m, no rotation).
- **Output motion order differs by format** (same signs everywhere):
  `get_response()` → (down, East, North); `.h5`/`.h5drm` rows → (East,
  North, down); positions (`xyz`) → (North, East, depth). Full table, the
  reason, and the OpenSees matrix `T` in `12_coordinates_and_conventions.md`.

## The three "run tiers" (pick one before doing anything else)

1. **Legacy / direct** — `ShakerMaker.run(...)`: every (source, receiver)
   pair is computed independently, no reuse. Simple, correct, but O(pairs)
   in FK evaluations. Good for debugging, small models, validation against
   the OP pipeline. See `06_engine_run_modes.md`.
2. **OP pipeline (staged)** — `gen_pairs` (Stage 0) → `compute_gf`
   (Stage 1) → `run_fast` (Stage 2), or the convenience wrapper
   `run_nearest(stage=...)`. Deduplicates geometrically-equivalent
   (source, receiver) pairs so each unique Green's Function is computed
   once and reused — this is what makes large DRM/SurfaceGrid campaigns
   (hundreds to millions of receivers) tractable. See `06_engine_run_modes.md`.
3. **Raw kernel** — `shakermaker.core.subgreen(...)` directly, bypassing
   the whole object model. Only for exploring the FK kernel itself
   (`examples/05_engine_direct/core_subgreen.py`,
   `examples/legacy_examples/example5-exploregreen.py`). Not documented
   further here — see `references/api.md` in the `shakemaker-skill` (or
   read `core.pyf`) if you need this.

**Always call `check_parameters(...)` before any real run** — it's a cheap,
pure-arithmetic pre-check (no FK evaluation) that tells you whether your
`dt`/`nfft`/`dk`/`tb`/`tmax` choice is physically sound for your model
geometry. See `05_check_parameters.md`. It is **advisory only by design** —
it never blocks execution, it only prints a report.

## General gotchas (apply almost everywhere)

- **MPI plotting**: `ZENTPlot`, `StationPlot`, `SourcePlot` all check MPI
  internally and only draw on rank 0 — safe to call unconditionally inside
  an MPI script.
- **Windows large-stack relaunch**: `run`/`compute_gf`/`run_fast`/
  `run_nearest` auto-relaunch themselves in a 64 MB-stack thread on Windows
  so the Fortran FK core doesn't segfault on large `nfft`. Don't bypass this.
  See `11_mpi_and_hpc.md`.
- **Numba is optional but strongly recommended**: `gen_pairs` (Stage 0) uses
  a Numba `@njit`-compiled greedy algorithm if available (100-500× faster),
  with an automatic pure-Python fallback if not installed.
- **HDF5 double-file convention**: the OP pipeline writes `<root>_map.h5` +
  `<root>_gf.h5` — always pass the **root** name (no `.h5` suffix needed,
  it's stripped/added consistently) to `gen_pairs`/`compute_gf`/`run_fast`/
  `run_nearest`.
- **`HDF5_USE_FILE_LOCKING=FALSE`**: set this environment variable before
  MPI runs on shared cluster filesystems — see `11_mpi_and_hpc.md`.
- **SciPy `trapz`/`trapezoid` shim: no longer needed.** Several older
  notebooks patch `scipy.integrate.trapz = scipy.integrate.trapezoid`
  before importing `shakermaker.tools.plotting`, working around
  `SourcePlot` importing the SciPy ≥ 1.14-removed name `trapz`. This was
  fixed upstream in commit `baddb87` — `plotting.py:233` now imports
  `trapezoid` natively. Do not add this shim to new code; see
  `10_plotting.md`.

## How to use this folder to generate a new example

1. Identify which **receiver type** you need (`04_receivers.md`):
   single `Station`, `DRMBox`, `SurfaceGrid`, or `PointCloudDRMReceiver`.
2. Identify which **source type** you need (`02_sources_and_stf.md` for
   point/finite-fault sources with a standard STF, or `03_ffsp_source.md`
   for a stochastic finite-fault via FFSP).
3. Identify which **run tier** fits your problem size (`06_engine_run_modes.md`)
   — legacy for small/debug, OP/`run_nearest` for anything with many
   receivers or that needs to reuse Green's Functions across realizations.
4. Identify which **writer** you need (`07_writers_persistence.md`):
   plain `HDF5StationListWriter`, `DRMHDF5StationListWriter` for OpenSees
   DRM, or none (in-memory `station.get_response()` / `.save()` to `.npz`).
5. Call `check_parameters(...)` first (`05_check_parameters.md`).
6. If exporting to SW4 instead of running the FK engine directly, see
   `09_sw4_export.md`.
7. **`RECIPES.md`** has worked, minimal, end-to-end templates for the most
   common combinations of the above — check there first before assembling
   one from scratch.

## See also

- `examples/EXAMPLES_REFERENCE.md` — a full audit of every script/notebook
  under `examples/`: what each one does, needs, and produces. Use it to find
  a close prior-art example before writing a new one from scratch.
