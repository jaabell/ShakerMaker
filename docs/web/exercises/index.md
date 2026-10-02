# Exercises

A graded path through ShakerMaker. Each exercise is a **complete, runnable
script** plus the result you should see and how to read it. They build on
each other, do them in order the first time.

Every exercise maps to a folder under [`examples/`](../examples/index.md), so
the script and the figures you see here are the ones that actually ran.

| # | Exercise | You learn | Example folder |
|---|---|---|---|
| 1 | [First run & the four arrivals](01_first_run.md) | the full pipeline; reading a seismogram | `05_engine_direct` |
| 2 | [Numerical convergence](02_convergence.md) | what `dk` and `nfft` actually control | `05_engine_direct` |
| 3 | [A sedimentary basin](03_basin.md) | how layering shapes motion; resonance | `01_crustmodel` |
| 4 | [Source time functions compared](04_stf.md) | choosing an STF for a target band | `03_stf` |
| 5 | [DRM box → H5DRM](05_drm.md) | boundary motions for OpenSees | `08_drm` |
| 6 | [FFSP stochastic rupture](06_ffsp.md) | an ensemble of admissible ruptures | `10_ffsp` |
| 7 | [Receiver geometries & the fast pipeline](07_receivers_pipeline.md) | array layouts; Green's-function reuse | `04_receivers`, `06_nearest_method` |
| 8 | [Sources: point & finite faults](08_sources.md) | mechanisms; building a finite fault | `02_sources` |
| 9 | [Saving results & writers](09_writers.md) | `.npz`, HDF5, progressive mode | `07_writers` |
| 10 | [Exporting to SW4](10_sw4_export.md) | hand a model to SW4 and back | `09_sw4_export` |
| 11 | [Plotting & visualisation](11_plotting.md) | `ZENTPlot`, `StationPlot`, `SourcePlot` | `11_plotting` |
| 12 | [Validation: SCEC LOH.1](12_validation.md) | the benchmark that proves it's correct | `12_validation` |
| 13 | [ShakerMaker vs SW4](13_shakermaker_sw4.md) | cross-validation against finite-difference | `13_shakermaker_sw4` |

!!! tip "How to run"
    Each script is self-contained, copy it into a `.py` file and run with
    `python exercise.py`. On a workstation the FK examples finish in seconds;
    the FFSP, DRM, and fast-pipeline ones take longer. For large jobs, launch
    under MPI: `mpirun -n 8 python exercise.py`.
