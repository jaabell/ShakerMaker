# Sources and Source-Time-Functions

## What this is

A ShakerMaker model needs a seismic source. The atomic unit is `PointSource`
(position + focal mechanism + source-time function). A `FaultSource` is
simply a collection of `PointSource`s — this is how both simple point
sources and hand-built finite-fault subfault grids are constructed (or as a
bridge from `FFSPSource`, see `03_ffsp_source.md`).

Each `PointSource` carries a `SourceTimeFunction` (STF): the time-domain
shape that gets convolved with the Green's Function computed by the FK
kernel before being summed into the station response. ShakerMaker ships 5
ready-made STFs: `Dirac`, `Discrete`, `Brune`, `Gaussian`, `SRF2`.

## Source of truth

- `shakermaker/pointsource.py` — class `PointSource` (61 lines)
- `shakermaker/faultsource.py` — class `FaultSource` (60 lines)
- `shakermaker/sourcetimefunction.py` — abstract base class `SourceTimeFunction` (88 lines)
- `shakermaker/stf_extensions/dirac.py`, `discrete.py`, `brune.py`, `gaussian.py`, `srf2.py`

## Full API reference

### `PointSource(x, angles, stf=Dirac(), tt=0)`

```python
PointSource.__init__(self, x, angles, stf=Dirac(), tt=0)
```

| Parameter | Type | Unit | Meaning |
|---|---|---|---|
| `x` | list or np.array shape (3,) | **km** | source position `[x, y, z]` |
| `angles` | list or np.array shape (3,) | **degrees** at construction | `[strike, dip, rake]` |
| `stf` | `SourceTimeFunction` | — | time function to convolve (default `Dirac()`) |
| `tt` | float | s | trigger time / source onset time |

Properties: `.x`, `.angles`, `.tt`, `.stf`.

**⚠️ Real gotcha (a docstring bug, not a behavioral one — but it still trips
people up):** the constructor converts `angles` to radians internally
(`shakermaker/pointsource.py:39`: `self._angles = np.pi*angles/180`), and
the `.angles` property **returns those radians**, not the original degrees
— even though the property's own docstring literally says "in degrees"
(`pointsource.py:50`). If you write code that reads `psource.angles`
expecting degrees back, you'll get radians instead. **Construction** does
take degrees (the convention used everywhere in this repo); only the later
read-back is in radians.

### `FaultSource(sources, metadata)`

```python
FaultSource.__init__(self, sources: List[PointSource], metadata: Dict)
```

- `sources`: list of `PointSource` (minimum 1 element, no upper bound — use
  N for a hand-built or FFSP-bridged finite fault).
- `metadata`: dict, **required** (no default) — typically `{"name": "..."}`.
- Iterable (`for src in fault: ...`).
- `.nsources` (property, int), `.metadata` (property).
- `.get_source_by_id(id)` → `PointSource`, raises `IndexError` if `id` is
  out of range.

### `SourceTimeFunction` (abstract base class)

Every STF inherits from this. Shared contract:

- `.dt` — property with a **setter**. Assigning `stf.dt = value` internally
  triggers `_generate_data()` (regenerates `.data`/`.t`). **No STF has
  sampled data until `.dt` is explicitly assigned** (except `Dirac`, which
  is trivial, and `SRF2`, which takes `dt` directly in its constructor —
  see below).
- `.data` / `.t` — the sampled STF arrays; accessing them also lazily
  triggers `_generate_data()` if not yet generated.
- `.convolve(val, t, debug=False)` — convolves `val` (sampled at `t`) with
  this STF via FFT (`scipy.signal.fftconvolve`), first resampling the STF
  to `dt = t[1]-t[0]`. If `len(self.data) == 1` (the `Dirac` case), it's
  just a scaling, not a real convolution. This is what the engine calls
  internally after computing the raw Green's Function.

### The 5 STFs

#### `Dirac()`
Unit impulse at `t=0` (`data=[1.0]`, `t=[0.0]`). No parameters. Use it when
you want the raw Green's Function with no source shape (e.g. to inspect the
FK kernel directly).

#### `Discrete(data, t)`
```python
Discrete(data, t)   # data: array Nt, t: array Nt (must start and end at 0)
```
An arbitrary, user-sampled STF. If `t` isn't uniformly spaced, it's
resampled to the simulation's `dt` before convolution (`interp1d`,
`bounds_error=False`, clamps at the edges). Useful for injecting your own
waveform (measured, external synthetic, etc.) without needing a parametric
model.

#### `Brune(slip=1.0, f0=0.0, t0=0.0, dsigma=0.0, M0=1.0, Vs=0.0, smoothed=False)`
The classic Brune slip-rate function:
`f_s(t) = slip · ω0² · (t-t0) · exp(-ω0(t-t0))` for `t ≥ t0`, with
`ω0 = 2π·f0`.

Two ways to specify the corner frequency:
- **(i)** give `f0` directly, or
- **(ii)** give `dsigma` (stress drop, bar), `M0` (seismic moment, dyne-cm)
  and `Vs` (local shear velocity, km/s) — the constructor derives
  `f0 = 4.9e6 · (Vs/1000) · ((dsigma·10/1e6)/(M0·1e7))^(1/3)`.

`smoothed=True` uses a smoothed version (cumulative integral, no sharp peak
at `t0`) — useful when the classic version's discontinuous derivative
introduces numerical artifacts in the convolution. `t0` shifts the STF on
its own time axis; the class docstring recommends using `PointSource`'s
`tt` for the rupture's trigger time instead, and leaving `t0=0` on the STF.

**Requires `.dt` to be set** before generating data (`assert self._dt > 0`
in `_generate_data`).

#### `Gaussian(t0=0.36, freq=16.6667, M0=1.0, derivative=False)`
```
g(t) = (freq/√(2π)) · exp(-½·(freq·(t-t0))²)
```
or its time derivative if `derivative=True`, scaled by `M0`.

**The LOH.1 convention used across almost the entire repo** (SCEC LOH.1
benchmark and most examples): define a width `sigma` (s) and derive:

```python
sigma = 0.06
t0    = 6 * sigma        # -> 0.36 (the class's literal default)
freq  = 1 / sigma        # -> 16.6667 (the class's literal default)
stf   = Gaussian(t0=t0, freq=freq, M0=M0, derivative=False)
```

The class defaults (`t0=0.36, freq=16.6667`) are **exactly** this
convention with `sigma=0.06`. If you change `sigma`, recompute both.

**Requires `.dt` to be set** before generating data.

#### `SRF2(Tr, Tp, Te, dt, slip, a, b)`
A 3-branch slip-rate function over `[0, Tr)`:
- rising sine ramp for `t < Tp`,
- plateau `sqrt(a + b/t²)` for `Tp ≤ t < Te`,
- decaying sine tail for `t ≥ Te`.

Normalized to unit area (`np.trapz`) and then scaled by `slip`. Typical
values seen in the repo: `a=1.0, b=1.0` (simple-source examples) or
`a=1.0, b=100.0` (the FFSP bridge, see `03_ffsp_source.md`).

**Key difference from Brune/Gaussian: `dt` is passed directly in the
constructor**, not assigned afterward via `.dt`. Internally,
`SourceTimeFunction.__init__` stores `self._dt = dt` without going through
the setter (the one that triggers `_generate_data()`), so data is still
generated lazily on first access to `.data`/`.t`, but **the extra
`stf.dt = dt` step that Brune and Gaussian require is not needed**.

This is the STF needed for the manual FFSP → `FaultSource` bridge (one
instance per subfault, with `Tr`/`Tp`/`Te` derived from each subfault's
`rise_time`) — see `03_ffsp_source.md`.

## Minimal working example

Simple point source with the LOH.1 convention:

```python
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian

sigma = 0.06
stf = Gaussian(t0=6 * sigma, freq=1 / sigma, M0=1e18 / 5e14 / 2, derivative=False)
src = PointSource([0, 0, 4], [90, 90, 0], stf=stf)   # km ; strike/dip/rake in degrees
fault = FaultSource([src], metadata={"name": "mainshock"})
```

Hand-built multi-subfault grid (pattern from
`examples/02_sources/faultsource_srf2.py`):

```python
import numpy as np
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.srf2 import SRF2

rng = np.random.default_rng(0)   # fixed seed -> reproducible
sources = []
for i in range(2):        # along-dip
    for j in range(5):    # along-strike
        rake = 0.0 + rng.uniform(-10, 10)
        slip = rng.uniform(0.5, 1.5)
        stf = SRF2(Tr=2.0, Tp=0.1, Te=1.5, dt=0.01, slip=slip, a=1.0, b=1.0)
        sources.append(PointSource(
            [0 + j * 1.0, 0 + i * 1.0, 3.0],   # km, grid around the hypocenter
            [0.0, 90.0, rake],
            stf=stf,
        ))
fault = FaultSource(sources, metadata={"name": "grid_2x5"})
```

## Known gotchas

- **`PointSource.angles` returns radians, not degrees**, despite what its
  docstring says — built in degrees, read back in radians
  (`pointsource.py:39,50`).
- **`Brune` and `Gaussian` have no data until `.dt` is assigned**
  (`stf.dt = dt`, or let the engine do it internally when running) —
  calling `.data`/`.t` before that triggers `_generate_data()` under the
  `dt > 0` assertion, which fails with `AssertionError` if `dt` is still at
  its default `-1`.
- **`SRF2` takes `dt` in its constructor**, no later `.dt = ...` step is
  needed.
- **Before plotting a source with `SourcePlot`** (see `10_plotting.md`),
  set `.dt` on the STF of every `PointSource` in the fault
  (`for s in fault: s.stf.dt = dt`) — otherwise the default STF-based
  coloring (`colorby="maxstf"`) has no sampled data to show. Seen in
  `examples/05_engine_direct/notebooks/engine_direct.ipynb` and
  `examples/11_plotting/notebooks/plotting_tools.ipynb`.
- `FaultSource.metadata` has no default — you must pass at least `{}` or
  `{"name": "..."}` explicitly.

## Combines with

- `01_crust_model.md` — every source needs a `CrustModel` to compute the
  Green's Function against.
- `03_ffsp_source.md` — the FFSP → `FaultSource` bridge reuses
  `PointSource`/`FaultSource` as-is, plus `SRF2` as the per-subfault STF.
- `04_receivers.md` — the source combines with a receiver list to form the
  model `ShakerMaker(crust, fault, receivers)`.
- `05_check_parameters.md` and `06_engine_run_modes.md` — once the source
  is built, the next step is validating parameters and running the engine.
- `10_plotting.md` — `SourcePlot` visualizes source geometry, colored by
  slip or STF.

## See also

- `examples/02_sources/pointsource.py`, `examples/02_sources/faultsource_srf2.py`,
  `examples/02_sources/notebooks/sources.ipynb`
- `examples/03_stf/stf_gallery.py`, `examples/03_stf/notebooks/stf_gallery.ipynb`
  (visual gallery of all 5 STFs)
- `examples/EXAMPLES_REFERENCE.md`, sections "02. Sources" and "03. Source
  Time Functions"
