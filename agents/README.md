# `agents/` — ShakerMaker knowledge base for autonomous configuration

This folder gives any agent (Claude, Codex, or otherwise) working in this
repository everything needed to **configure and run ShakerMaker correctly**
without re-reading the whole source tree from scratch each time — organized
by functionality, grounded in the actual source code (file/line citations
throughout), and cross-linked so that any combination of features can be
composed on demand.

It lives in the repo (not in a tool-specific config directory) so it travels
with the code and is available to whatever agent or human opens this
checkout.

**Nothing here should be trusted blindly if the underlying code has
changed since it was written** — each file cites the exact source file/line
it's grounded in; re-verify against that when in doubt.

## Start here

- **New to this repo / first time configuring anything** → read
  [`00_orientation.md`](00_orientation.md) first. It covers the package
  layout, the coordinate/units convention (the #1 source of silent bugs),
  the three run tiers, and a step-by-step "how to use this folder."
- **Want a working script for a specific combination of features right
  now** → go straight to [`RECIPES.md`](RECIPES.md) — a compatibility
  matrix plus ~10 worked, minimal, complete templates (single station,
  DRM box, FFSP ensemble, SW4 export, etc.).
- **Need the exact signature/units/gotchas of one specific piece** → pick
  the matching file below.

## Files, by functionality

| File | Covers |
|---|---|
| [`00_orientation.md`](00_orientation.md) | Package map, units/coordinates, run tiers, general gotchas — read first |
| [`01_crust_model.md`](01_crust_model.md) | `CrustModel`, `cm_library` presets (LOH.1/LOH.3, AbellThesis, SOCal_LF), `Crust1` (CRUST 1.0 lookup) |
| [`02_sources_and_stf.md`](02_sources_and_stf.md) | `PointSource`, `FaultSource`, the 5 source-time functions (`Dirac`, `Discrete`, `Brune`, `Gaussian`, `SRF2`) |
| [`03_ffsp_source.md`](03_ffsp_source.md) | `FFSPSource` — stochastic finite-fault: constructor, realizations, persistence, the 9 plot methods, the manual FFSP→`FaultSource` bridge |
| [`04_receivers.md`](04_receivers.md) | `Station`/`StationList`, `DRMBox`, `SurfaceGrid`, `PointCloudDRMReceiver` |
| [`05_check_parameters.md`](05_check_parameters.md) | `check_parameters(...)` deep dive — every formula behind `dt`/`nfft`/`dk`/`tb`/`tmax`, advisory-only by design |
| [`06_engine_run_modes.md`](06_engine_run_modes.md) | Legacy `run()` vs. the OP pipeline (`gen_pairs`/`compute_gf`/`run_fast`/`run_nearest`), legacy-DB migration |
| [`07_writers_persistence.md`](07_writers_persistence.md) | `HDF5StationListWriter`, `DRMHDF5StationListWriter`, `export_drm_geometry`, `Station.save`/`.load` |
| [`08_drm_workflow.md`](08_drm_workflow.md) | How the DRM pieces fit together end to end: sizing, QA station, known bugs |
| [`09_sw4_export.md`](09_sw4_export.md) | Export to SW4, coordinate convention, running SW4 externally, rebuilding `.h5drm` after |
| [`10_plotting.md`](10_plotting.md) | `ZENTPlot`, `StationPlot`, `SourcePlot` |
| [`11_mpi_and_hpc.md`](11_mpi_and_hpc.md) | MPI parallelism, SLURM patterns, known hangs (one fixed, one still partially open — documented in the source itself) |
| [`12_coordinates_and_conventions.md`](12_coordinates_and_conventions.md) | Positions vs motion, component order and signs of every output, SW4 frame, the OpenSees `H5DRMLoadPattern` matrix `T` |
| [`RECIPES.md`](RECIPES.md) | Compatibility matrix + worked combination templates — the generative layer |

Every file follows the same shape: **What this is → Source of truth →
API reference → Minimal working example → Known gotchas → Combines with →
See also**.

## Relationship to other documentation in this repo

- **`examples/EXAMPLES_REFERENCE.md`** — an exhaustive, file-by-file audit
  of every script/notebook under `examples/` (purpose, inputs, outputs).
  Use it to check whether a working example for what you need already
  exists before writing a new one from a `RECIPES.md` template.
- **`docs/web/`** — the MkDocs user-facing documentation site. `agents/`
  is denser, source-grounded, and organized for machine consumption; the
  MkDocs site is prose aimed at a human reader learning the tool.
- **`.claude/`, `.codex*/`** — tool-specific configuration directories
  (Claude Code / Codex sessions). `agents/` is deliberately tool-agnostic
  and lives in the repo itself so it isn't tied to any one assistant's
  local config.

## Keeping this up to date

These files are a snapshot grounded in the source as of when each was
written (dated by the git history of this folder). If you change a
signature, default, or documented behavior in `shakermaker/`, update the
corresponding `agents/*.md` file in the same change — treat it like a
docstring that happens to live in its own file.
