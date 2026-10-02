# Examples

The [`examples/`](https://github.com/ppalacios92/ShakerMaker/tree/master/examples)
tree is organised by **topic**, one folder per concept. Each script is a small,
self-contained input → result and is the fastest way to see which inputs produce
which output. Most scripts end in a `print("PASS")` so they double as smoke
tests.

!!! tip "Run everything at once"
    [`examples/run_all_smoke.py`](https://github.com/ppalacios92/ShakerMaker/blob/master/examples/run_all_smoke.py)
    walks every example `.py`, runs it, and reports **PASS / SKIP / FAIL**.
    Scripts that need an optional dependency (`h5py`, MPI) or an external data
    file print `SKIP` instead of failing. Pass `--full` to also run the slow
    validation cases.

    ```bash
    cd examples
    python run_all_smoke.py          # fast subset
    python run_all_smoke.py --full   # include 12_validation
    ```

Every folder also ships a `notebooks/` subdirectory. The notebooks re-run the
same recipes interactively and save `.png` previews (crust profiles, source
geometry, STFs, seismograms, …) next to the `.ipynb`, so you can browse the
expected output without executing anything.

---

## 01 · CrustModel

Build and inspect the layered velocity model that underpins every FK run.

| Script | What it shows |
|---|---|
| `crustmodel_build.py` | `CrustModel(nlayers)`, `add_layer(thickness, vp, vs, rho, Qp, Qs)`, `modify_layer`, `properties_at_depths`, and the `SCEC_LOH_1/3` library presets |
| `crust1_sites.py` | the bundled CRUST1.0 reader (`from shakermaker.crust1 import Crust1`): crustal profile at a given lat/lon plus a ready-to-paste `CrustModel` snippet |

Notebook: `notebooks/crustmodel.ipynb` (layer table + velocity-profile plot).

## 02 · Sources

Assemble the moment-tensor sources that drive the wavefield.

| Script | What it shows |
|---|---|
| `pointsource.py` | a single `PointSource([x,y,z], [strike,dip,rake], stf=...)` wrapped in a `FaultSource` |
| `faultsource_srf2.py` | a 2×5 grid of subfaults with per-subfault `SRF2` STFs and randomised rake/slip |

Notebook: `notebooks/sources.ipynb` (3-D source geometry).

## 03 · Source time functions

The full STF gallery.

| Script | What it shows |
|---|---|
| `stf_gallery.py` | instantiates all five STFs — `Dirac`, `Discrete`, `Brune`, `Gaussian`, `SRF2` — and checks each generates data |

Notebook: `notebooks/stf_gallery.ipynb` (time-domain + spectra for each STF).

## 04 · Receivers

Every receiver-array layout ShakerMaker ships.

| Script | Layout |
|---|---|
| `single_station.py` | one `Station` inside a `StationList` |
| `drmbox.py` | a `DRMBox(center, [nx,ny,nz], [hx,hy,hz])` boundary box |
| `surface_grid.py` | `SurfaceGrid` in `plane`, `hollow` and `filled` modes |
| `pointcloud_drm.py` | `PointCloudDRMReceiver` built from a FEM node TSV (mm → km transform) |

Notebook: `notebooks/receivers_geometry.ipynb` (DRM box, surface grid, point-cloud previews).

## 05 · The engine (direct)

The classic per-station FK pipeline.

| Script | What it shows |
|---|---|
| `run_simple.py` | a full `ShakerMaker(crust, fault, stations).run(...)` on three stations |
| `check_parameters.py` | the pure-arithmetic `check_parameters(dt, nfft, dk, tb, tmax)` pre-run report (no FK run) |
| `core_subgreen.py` | a direct call to the low-level FK kernel `shakermaker.core.subgreen` |

Notebook: `notebooks/engine_direct.ipynb` (crust, source, and the resulting seismogram).

## 06 · The nearest method (fast pipeline)

The Green's-function-reuse pipeline that computes one GF per unique
source–receiver distance and reuses it everywhere.

| Script | What it shows |
|---|---|
| `nearest_all.py` | `run_nearest(..., stage='all')` on a small surface grid |
| `stage_by_stage.py` | the same run split into `gen_pairs → compute_gf → run_fast` |
| `legacy_migration.py` | upgrading a legacy GF database to the new format (SKIPs if no legacy DB is present) |

Notebook: `notebooks/nearest_explained.ipynb` (calc-vs-reuse comparison).

## 07 · Writers & I/O

Persisting results.

| Script | What it shows |
|---|---|
| `hdf5_writer.py` | `HDF5StationListWriter` (legacy + progressive modes) |
| `drm_writer.py` | `DRMHDF5StationListWriter` → a `.h5drm` from a `DRMBox` |
| `save_load_station.py` | save one `Station` to `.npz`, reload and compare |
| `explore_h5_output.py` | walk any `.h5`/`.h5drm` and print every group/dataset name and shape |

Notebook: `notebooks/writers.ipynb` (round-trips a station through HDF5).

## 08 · DRM

Domain-Reduction-Method boxes for finite-element coupling.

| Script | What it shows |
|---|---|
| `drm_vs_direct.py` | the same point computed directly and inside a DRM box, overlaid |
| `export_drm_geometry.py` | export DRM geometry only (no FK run) for a `DRMBox` and a `SurfaceGrid` |

Notebook: `notebooks/drm.ipynb` (DRM box geometry).

## 09 · SW4 export

Bridging ShakerMaker output into the SW4 finite-difference solver.

| Script | What it shows |
|---|---|
| `export_sw4.py` | export a single-point-source model to SW4 (no topography), with UTM → local-km framing |
| `export_sw4_topo.py` | the same export with cartesian topography (SKIPs if the topo file is absent) |
| `package_h5_roundtrip.py` | export the compact SW4 package HDF5, then `unpack_sw4_package_h5` back into an SW4 file tree |
| `build_h5drm_from_sw4.py` | build an `.h5drm` from an SW4 case (needs SW4 result `.txt` files) |
| `build_h5drm_from_sw4_case.py` | the reusable reference builder used by the script above (SW4-local-km ↔ ShakerMaker/UTM-km conversion) |

Notebook: `notebooks/sw4_export.ipynb` (SW4 geometry map + 3-D view).

## 10 · FFSP

The stochastic finite-fault-source-process generator.

| Script | What it shows |
|---|---|
| `ffsp_run.py` | build a small/fast `FFSPSource` (Mw 6.0, single realization) and `run()` it |
| `ffsp_io.py` | write the realization to HDF5 + legacy format, load it back and check |

Notebooks: `notebooks/ffsp.ipynb` (slip distribution + source-time function) and the
original `notebooks/example_FFSP.ipynb`.

## 11 · Plotting

The plotting helpers in `shakermaker.tools.plotting`.

| Script | What it shows |
|---|---|
| `plotting_tools.py` | build a small `FaultSource` and save a `SourcePlot` (no FK run) |

Notebook: `notebooks/plotting_tools.ipynb` (`SourcePlot`, station plot, `ZENTPlot`).

## 12 · Validation

The SCEC LOH.1 benchmark.

| Script | What it shows |
|---|---|
| `LOH1.py` | the LOH.1 run for a single receiver at (6, 8, 0) |
| `LOH1_check.py` | compare the result against the Prose reference solution in `data/LOH.1_prose3` (SKIPs until `LOH1.py` has been run) |

Notebooks: `notebooks/LOH1_validation.ipynb` and `notebooks/example_LOH1_gf.ipynb`.

---

## 13 · ShakerMaker vs SW4

Cross-validate the FK engine against a full SW4 finite-difference run on the
same model (4-layer crust, 100 FFSP sources, 2 stations).

| Script | What it shows |
|---|---|
| `shaker_vs_sw4.py` | rebuild the model from `data/model_summary.h5`, run FK, read + band-pass (ObsPy) the SW4 output, overlay the two |

Resources in `data/` (`model_summary.h5`, `shakermaker2sw4.in`, `sf0000{1,2}.txt`)
are copied from a real SW4 run. Needs `obspy`. See the
[exercise](../exercises/13_shakermaker_sw4.md) for the full walk-through.

---

## Legacy examples

[`examples/legacy_examples/`](https://github.com/ppalacios92/ShakerMaker/tree/master/examples/legacy_examples)
holds José Abell's original upstream examples (`example0_readme_example.py`,
`example1_simple.py`, `example2_drm.py`, `example3-save-station.py`,
`example4-load-station.py`, `example5-exploregreen.py`). They are kept
**unmodified** as a regression reference and are skipped by `run_all_smoke.py`.

---

## Generating the figures

Every figure in this documentation is produced by a small, comment-free
script under `docs/web/examples/scripts/`. They double as **minimal,
copy-pasteable recipes** for the most common tasks, and as a regression
check that the API still behaves. All are fast (the heaviest is a reduced
FFSP); none needs MPI.

| Script | Produces | Demonstrates |
|---|---|---|
| `gen_quick_example.py` | `example_0_quick_example` | the README quick start (LOH.1 crust + Gaussian point source) |
| `gen_crust_profiles.py` | `crust_loh1`, `crust_basin` | `CrustModel.plot_profile()` |
| `gen_stf_gallery.py` | `stf_overview` | STF time domain + spectrum |
| `gen_station_geometry.py` | `geom_surface_grid`, `geom_drmbox`, `geom_hollow_box` | `SurfaceGrid` modes, `DRMBox`, `StationPlot` |
| `gen_receiver_clouds.py` | `receiver_clouds` | every receiver geometry, internal/external coloured |
| `gen_seismogram.py` | `seismogram_velocity/displacement/acceleration` | a full FK run + `ZENTPlot` |
| `gen_ffsp.py` | `ffsp_slip`, `ffsp_rise_time` | a small stochastic `FFSPSource` |

Run one (or all):

```bash
cd docs/web/examples/scripts
python gen_seismogram.py
# or regenerate everything:
python gen_all.py
```

The scripts save straight into `docs/web/assets/images/`, so the docs pick up
the refreshed figures on the next build.
