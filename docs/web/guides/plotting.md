# Plotting & visualisation

The plotting helpers live in `shakermaker.tools.plotting`. They turn stations
and sources into publication-ready figures with one call.

## Seismograms: `ZENTPlot`

The workhorse: a three-component (Z, E, N) seismogram overlay.

```python
from shakermaker.tools.plotting import ZENTPlot

ZENTPlot(sta, xlim=[0, 60], show=True)
```

`ZENTPlot(station, fig=0, show=False, xlim=[], label=[], integrate=0, differentiate=0, savefigname="", linestyle="-", linewidth=2)`:

| Argument | Effect |
|---|---|
| `xlim` | time-axis window |
| `integrate=1` | velocity → **displacement** |
| `differentiate=1` | velocity → **acceleration** |
| `fig` | reuse a figure handle to **overlay** traces |
| `label` | legend label |
| `savefigname` | save the figure to a file |

### Overlaying stations or runs

Pass the same `fig` to stack traces, the pattern shown in
`examples/11_plotting/plotting_tools.py`:

```python
import matplotlib.pyplot as plt
fig = plt.figure(1)
for s in stations:
    ZENTPlot(s, fig=fig, show=False, xlim=[0, 15], label=s.metadata["name"])
plt.legend(); plt.show()
```

## Geometry: `StationPlot` and `SourcePlot`

```python
from shakermaker.tools.plotting import StationPlot, SourcePlot

StationPlot(stations, show=True)     # receiver layout (works on a DRMBox too)
SourcePlot(fault, show=True, colorby="maxstf", colorbar=True)
```

| Function | Shows | Key options |
|---|---|---|
| `StationPlot(stations, fig=0, show=False, autoscale=False)` | receiver geometry | `autoscale` |
| `SourcePlot(sources, fig=0, show=False, colorby="maxstf", colorbar=False, axes_equal=True)` | source / subfault geometry | `colorby` (e.g. slip), `colorbar` |

`SourcePlot` colours each subfault by a field (`colorby`), making it the
natural way to view a finite-fault slip distribution.

## FFSP figures

The FFSP plots live on `FFSPSource` itself (slip maps, rupture snapshots,
ensemble metrics), see [FFSP → Visualisation](ffsp.md#result-run-inspect).

## Conventions

- Components are **Z** (vertical, up), **E** (east), **N** (north).
- The default trace is **velocity**; use `integrate` / `differentiate` for
  displacement / acceleration.

## Reference

[Plotting API →](../api/plotting.md)
