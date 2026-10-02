# Writers

<!-- Auto-generated from docstrings. -->

Writers stream simulation output to disk as the engine computes it. Pass an
instance via the `writer=` argument of `ShakerMaker.run()` or
`ShakerMaker.run_nearest()` so each station is flushed and freed as soon as
it is finished, keeping memory use bounded for large receiver sets.

`StationListWriter` is the abstract base; `HDF5StationListWriter` produces a
plain HDF5 file of station traces, and `DRMHDF5StationListWriter` produces the
DRM-compatible `.h5drm` format. See [Outputs & writers](../guides/outputs.md)
for the workflow.

::: shakermaker.stationlistwriter.StationListWriter

::: shakermaker.slw_extensions.HDF5StationListWriter

::: shakermaker.slw_extensions.DRMHDF5StationListWriter
