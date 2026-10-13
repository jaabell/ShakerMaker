# Plotting

## What this is

`shakermaker/tools/plotting.py` holds the three generic, cross-cutting plot helpers that
work on any model regardless of which functionality produced it: `ZENTPlot` (a station's
time-domain response), `StationPlot` (receiver geometry), `SourcePlot` (source geometry,
colored by a source property). Every other plotting capability in the package
(`CrustModel.plot`/`plot_profile`, `Crust1`'s 6 map/profile plots, `FFSPSource`'s 9
`plot_*` methods) lives on its own class and is documented in that functionality's file —
see "Combines with" below, not repeated here.

## Source of truth

`shakermaker/tools/plotting.py` (336 lines) — read in full for this doc.

## Full API reference

### `ZENTPlot(station, fig=0, show=False, xlim=[], label=[], integrate=0, differentiate=0, savefigname="", linestyle="-", linewidth=2)`

Plots a station's 3-component response as a 3-row figure (Z, E, N), sharing x/y axes.

- `station`: a `shakermaker.station.Station` instance (asserted).
- `integrate` / `differentiate`: **not** a repeat count despite the docstring's wording
  ("Show integral ... `integrate` times"). The implementation only checks
  `integrate > 0` / `differentiate > 0` as a boolean — any positive value does exactly
  **one** cumulative-trapezoidal integration (`scipy.integrate.cumulative_trapezoid`) or
  one first-order finite-difference derivative. Passing `integrate=2` behaves identically
  to `integrate=1`; there is no way to get a second integral/derivative from this
  function directly. Passing both `integrate>0` and `differentiate>0` together prints a
  message and returns `0` (no plot).
- Y-axis labels switch automatically: velocity (`$\dot u$`) when neither flag is set,
  displacement (`$u$`) when integrating, acceleration (`$\ddot u$`) when differentiating
  — i.e. `Station.get_response()` is treated as **velocity** by this function.
- `savefigname`: if non-empty, calls `plt.savefig(savefigname)` after the `show` branch.
  Can be combined with `show=True`.
- MPI-safe: `if rank == 0` gates the whole body; other ranks return `0` immediately.

### `StationPlot(stations, fig=0, show=False, autoscale=False)`

3-D scatter of every station in `stations` (a `StationList` or anything iterable of
`Station`-like objects with `.x`). Plotted as `(y, x, -z)` — i.e. axes are labeled
"(Y) Easting", "(X) Northing", "(Z) Depth", and `ax.invert_zaxis()` is called so depth
increases downward visually. `autoscale=True` recentres all three axes to the same range
(a manual equal-aspect trick, independent of `set_axes_equal`). MPI-safe:
`if rank > 0: return 0` gates the whole body (same effect as `ZENTPlot`'s check, opposite
polarity).

### `SourcePlot(sources, fig=0, show=False, autoscale=False, colorby="maxstf", colorbar=False, axes_equal=True)`

3-D scatter of every `PointSource` in a `FaultSource` (or any iterable of sources),
colored by one scalar property per source. Same axis convention/inversion as
`StationPlot`.

`colorby` accepts **six** values, not five — the docstring only lists five
(`"maxstf"|"strike"|"dip"|"rake"|"tt"`) but the implementation's `case`/`clabel` dicts
also define **`"slip"`** (verified directly in source, `case==5`):

| `colorby` | Computed as |
|---|---|
| `"maxstf"` | `stf.data.max()` after forcing `stf.dt = 0.01` |
| `"strike"` | `angles[0]` in degrees |
| `"dip"` | `angles[1]` in degrees |
| `"rake"` | `angles[2]` in degrees |
| `"tt"` | `src.tt` (trigger time) |
| `"slip"` *(undocumented)* | `scipy.integrate.trapezoid(stf.data, stf.t)` after forcing `stf.dt = 0.01` |

**Important**: `SourcePlot` unconditionally sets `stf.dt = 0.01` on every source's STF
before evaluating any `colorby` case — you do **not** need to pre-set `.dt` yourself
before calling this function for it to work; it always uses `0.01` internally regardless
of what the STF's `.dt` was set to beforehand. (Some example notebooks pre-set `.dt` for
other reasons — e.g. to also plot the STF manually right after — not because `SourcePlot`
requires it.)

`axes_equal=True` (default) calls `set_axes_equal(ax)`; `autoscale=True` does the
separate manual recentring trick instead (both can technically be combined; `axes_equal`
is applied after `autoscale`).

### `set_axes_equal(ax)`

Standalone 3-D helper: makes a Matplotlib 3-D axis's x/y/z ranges equal so spheres look
like spheres — Matplotlib's own `ax.set_aspect('equal')` doesn't work for 3-D. Used
internally by `SourcePlot`; safe to call on any 3-D axis directly.

## Minimal working example

```python
from shakermaker.tools.plotting import ZENTPlot, StationPlot, SourcePlot

# after model.run(...) has populated the station's response:
ZENTPlot(station, xlim=[0, 30], show=True)
ZENTPlot(station, integrate=1, savefigname="displacement.png")   # one integration -> displacement

StationPlot(stations, show=True, autoscale=True)

SourcePlot(fault, colorby="slip", colorbar=True, show=True)
```

## Known gotchas

- **The SciPy `trapz`→`trapezoid` shim seen in some example notebooks is no longer
  needed.** `SourcePlot` used to do `from scipy.integrate import trapz` (removed in
  SciPy ≥1.14); this was fixed in commit `baddb87` (2026-06-07) to
  `from scipy.integrate import trapezoid` directly. The current source imports
  `trapezoid` natively — confirmed by reading `shakermaker/tools/plotting.py` directly,
  no `trapz` reference remains anywhere in the file. Any notebook that still does
  `scipy.integrate.trapz = scipy.integrate.trapezoid` before importing this module is
  doing unnecessary (harmless) defensive patching left over from before the fix landed —
  **not** a currently-required workaround. (`examples/EXAMPLES_REFERENCE.md`'s notes on
  `02_sources/notebooks/sources.ipynb` and `11_plotting/notebooks/plotting_tools.ipynb`
  describe this as a live gotcha; that should be read as "was needed against an older
  version of this file," not "is needed against the current one.")
- **`integrate`/`differentiate` are booleans in disguise**, not repeat counts — see above.
- **`colorby="slip"` is real but undocumented** in `SourcePlot`'s own docstring.
- All three functions check MPI rank internally and only draw on rank 0 — safe to call
  unconditionally inside an MPI-parallel script without wrapping in your own rank check.
- `ZENTPlot` treats the station's stored response as **velocity** for labeling purposes
  (Z/E/N with dots for velocity, no dots for displacement after one integration, double
  dots for acceleration after one differentiation) — this matches how the FK engine
  actually populates `Station` responses (ground velocity), but is worth knowing if you
  feed it something else.

## Combines with

- `04_receivers.md` — any `StationList`/`DRMBox`/`SurfaceGrid`/`PointCloudDRMReceiver`
  works directly with `StationPlot` (they all subclass `StationList`).
- `02_sources_and_stf.md` — `SourcePlot` needs each `PointSource`'s `.stf` to be a valid
  `SourceTimeFunction` instance; it forces `.dt=0.01` itself, so no pre-setup required.
- `01_crust_model.md` — `CrustModel.plot()`/`plot_profile()` and `Crust1`'s 6 plot
  methods are separate, on those classes directly (not in this file).
- `03_ffsp_source.md` — `FFSPSource`'s 9 `plot_*` methods are separate, on that class
  directly (not in this file); most call `plt.show()` internally except
  `plot_spectral_comparison`.

## See also

- `examples/11_plotting/` (full script/notebook breakdown in
  `examples/EXAMPLES_REFERENCE.md`) — the canonical demo of all three functions together,
  including a real minimal FK run so `ZENTPlot` has data to show.
- `examples/02_sources/notebooks/sources.ipynb` — `SourcePlot(fault, colorby="slip", ...)`
  on an `SRF2`-driven subfault grid.
