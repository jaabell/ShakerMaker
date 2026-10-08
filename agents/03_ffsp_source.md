# FFSP — Stochastic Finite-Fault Source

## What this is

`FFSPSource` generates a stochastic finite-fault source (random fields of
slip, rise time, peak time, rupture velocity, perturbed rake/dip) over a
grid of subfaults, using the FFSP Fortran kernel (Liu et al. 2006, modified
by Chen Ji 2020). This is the package's realistic-rupture generator — not a
simple `PointSource` nor a hand-rolled `SRF2`: it produces **complete
per-subfault fields** plus quality metrics, a synthetic spectrum, and a
comparison against a double-corner-frequency (DCF) target model, for one or
several realizations of the random process.

**Hard rule: do not reimplement any of this.** The class already provides
generation, realization selection, HDF5 I/O, native FFSP-format I/O, and
**nine** plotting methods. The only thing you do have to write by hand is
the bridge to `FaultSource` (see "The one bridge you do have to write,"
below) — deliberately not packaged.

## Source of truth

- `shakermaker/ffspsource.py` (2701 lines) — the single source of truth for
  the class.
- `shakermaker/ffsp/` — Fortran kernel (`ffsp_wrapper.f90`, `s2m.f90`,
  `slip_rate.f90`, `spfield_n.f90`, `ffsp_dcf_v2.f90`, `dcf_subs_1.f90`,
  `ffsp.pyf`), compiled to `ffsp_core` (f2py extension).
- `tests/ffsp/test_all_realization_products.py`, `tests/ffsp/mpi_*_contract.py`
  — verified contracts.

## Full API reference

### Constructor (29 arguments, all required except the last two)

`shakermaker/ffspsource.py:122`

```python
from shakermaker.ffspsource import FFSPSource

source = FFSPSource(
    id_sf_type=8,                    # kernel slip-rate function type
    freq_min=0.01, freq_max=24.0,    # Hz, synthesis band
    fault_length=30.0,               # km, along strike
    fault_width=16.0,                # km, down-dip
    x_hypc=15.0, y_hypc=8.0,         # km, hypocenter IN THE FAULT PLANE (along-strike, along-dip)
    depth_hypc=8.0,                  # km, hypocenter depth
    xref_hypc=0.0, yref_hypc=0.0,    # km, hypocenter reference origin
    magnitude=6.0,                   # Mw
    fc_main_1=0.09, fc_main_2=3.0,   # Hz, double-corner spectrum corners
    rv_avg=3.0,                      # km/s, average rupture velocity
    ratio_rise=0.3,                  # -, rise-time ratio
    strike=358.0, dip=40.0, rake=113.0,   # degrees
    pdip_max=15.0, prake_max=30.0,   # degrees, max dip/rake perturbation
    nsubx=16, nsuby=8,               # subfaults along-strike / down-dip
    nb_taper_trbl=[5, 5, 5, 5],      # taper zones [top, right, bottom, left]
    seeds=[52, 448, 4446],           # 3 seeds
    id_ran1=1, id_ran2=1,            # realization range [start, end], 1-based
    angle_north_to_x=0.0,            # degrees, north -> x rotation
    is_moment=3,                     # kernel result flag
    crust_model=crust,               # CrustModel (required, type-checked)
    output_name="FFSP_OUTPUT",       # optional, file prefix
    verbose=True,                    # optional
)
```

**Pre-Fortran validation**: `_validate_ffsp_kernel_contract(params)` (`:67`)
rejects inputs that would `STOP` the Fortran code or overrun f2py buffers.
Checks positivity, `freq_min < freq_max`, `id_ran2 >= id_ran1`, hypocenter
inside the grid, `ntime <= 131072`, and the frequency band falling inside
the octave range. If it raises `ValueError`, that's a bad input, not a
framework bug.

**Grid resolution rule**: `dx, dy <= Vs_min / (5 * fmax)` (subfault
dimensions relative to the frequency band and the crust's slowest shear
velocity).

### ⚠️ Critical geometric gotcha — a fault that breaks the surface

The plane's top edge sits at `z_top = depth_hypc - y_hypc * sin(dip)`.
**`depth_hypc >= y_hypc * sin(dip)` must hold**; otherwise subfaults end up
with `z < 0` (in the air), and the kernel marks them with
`rupture_time = 1e10` (sentinel) and `slip = NaN`. The run is then
**physically compromised**, not just cosmetically off.

Classic symptom: `plot_spacial_distribution(contour_field='rupture_time',
contour_interval=0.5)` raises `MemoryError: Unable to allocate 149 GiB`
because it does `np.arange(0, max+interval, interval)` with `max=1e10`.
**Fix the input (`depth_hypc`/`y_hypc`/`dip`), never the plot.**

### Life cycle

| Method | Line | What it does |
|---|---|---|
| `run()` | `:228` | Runs the Fortran kernel (`ffsp_core.ffsp_run_wrapper`), MPI-aware. Returns the best realization's dict (rank 0 only under MPI; other ranks return `None`). |
| `get_realization(index)` | `:550` | Dict for realization `index` (**0-based** — do not confuse with `id_ran1..id_ran2`, which is 1-based). |
| `set_active_realization(index)` | `:615` | Sets `self.subfaults` to that realization; the plots use the active one. |
| `get_subfaults()` | `:629` | Returns the active realization. Raises if none is set. |

Attributes available after `run()`:

- `self.all_realizations` — dict with **every** realization (2-D arrays
  `[nsubfaults, n_realizations]`)
- `self.best_realization` — the one with the lowest `pdf` (`argmin`)
- `self.subfaults` — the active one (starts as the best)
- `self.active_realization` — `'best'` or the integer index
- `self.source_stats` — ensemble metrics used by the plots
- `self.params` — dict of every input. **Note**: after `run()`,
  `fc_main_1`/`fc_main_2` get **overwritten** with the kernel's effective
  values; the originally-requested values remain in
  `fc_main_1_requested` / `fc_main_2_requested`.
- `self.dx`, `self.dy`, `self.area` — subfault dimensions (km, km, km²)

Under MPI: `run()` distributes `id_ran1..id_ran2` across ranks and gathers
to rank 0. Idle ranks (more ranks than realizations) return `None` cleanly
(see `11_mpi_and_hpc.md`).

### Data structure — what it actually returns

`get_realization(i)` / `get_subfaults()` → 1-D arrays of length
`nsubx*nsuby`:

| Key | Type | **Unit** |
|---|---|---|
| `x`, `y`, `z` | 1-D array | **METERS** (see "The one bridge..." section) |
| `slip` | 1-D array | m |
| `rupture_time` | 1-D array | s (onset) |
| `rise_time`, `peak_time` | 1-D array | s |
| `strike`, `dip`, `rake` | 1-D array | degrees |
| `nseg`, `npts` | int | — |
| `realization_id` | int | kernel's 1-based id |
| `metrics` | dict | `ave_tr`, `ave_tp`, `ave_vr`, `err_spectra`, `pdf` (scalars) |
| `stf_time` | dict | `time` (s), `stf` (1/s, **unit area**) |
| `spectrum` | dict | `freq`, `moment_rate_synth`, `moment_rate_dcf` |
| `spectrum_octave` | dict | `freq_center`, `logmean_synth`, `logmean_dcf` |

`all_realizations` has the same schema but 2-D, `[:, i_realization]`, plus
`n_realizations`, `realization_id` (array), and the sub-dicts with a second
dimension = realization.

**STF and scalar moment**: the stored STF is **normalized to unit area**
(unit 1/s, not N·m/s). For physical moment, multiply by `M0`, computed with
the kernel's own constant:

```python
from shakermaker.ffspsource import FFSP_MAGNITUDE_CONSTANT   # = 9.05
M0 = 10 ** (1.5 * magnitude + FFSP_MAGNITUDE_CONSTANT)       # N*m
```

`9.05` instead of the SI Hanks & Kanamori `9.1` is a **kernel convention,
not a bug** (`ffspsource.py:39-43`). Do not "fix" it.

### Persistence — already exists, don't hand-write readers

| Method | Line | What it does |
|---|---|---|
| `write_hdf5(filename)` | `:638` | Every realization + params, gzip+shuffle. Appends `.h5` if missing. |
| `load_hdf5(filename)` | `:701` | Loads in-place onto the instance. |
| `FFSPSource.from_hdf5(filename)` | `:790` (classmethod) | **Rebuilds the whole object** from the `.h5`. The normal reload path. |
| `write_ffsp_format(output_dir, output_name=None)` | `:892` | Native FFSP text format + `ffsp.inp` + `source_model.params`. |
| `load_ffsp_format(input_dir, output_name="FFSP_OUTPUT")` | `:1122` | Loads in-place from the text tree. |
| `FFSPSource.from_ffsp_format(input_dir, output_name, verbose)` | `:1369` (classmethod) | Rebuilds from text; tries `_load_from_params_file` and falls back to `_load_from_fortran_files`. |

HDF5 schema: `FFSP_HDF5_SCHEMA_VERSION = "2.0"`.

`from_ffsp_format` also reads output from the **original** Fortran
executable (verified against a tree produced by the reference binary, not
by ShakerMaker): it detects the realization count, the `nsubx x nsuby`
grid, parses `ffsp.inp`, and builds the `CrustModel` from whatever `.vel`
file it finds. So comparing "original kernel vs. our kernel" needs no
hand-written parser at all — both sides load as `FFSPSource` objects.

**Difference in the legacy path**: `all_realizations` loaded via
`from_ffsp_format` is **missing** the `metrics` key (it only carries
`x,y,z,slip,rupture_time,rise_time,peak_time,strike,dip,rake,nseg,npts,
n_realizations`), and `subfaults`/`best_realization` are missing
`spectrum`, `spectrum_octave`, and `stf_time`.

| Want | HDF5 (`from_hdf5`) | Legacy (`from_ffsp_format`) |
|---|---|---|
| quality metrics | `all_realizations['metrics']` **or** `source_stats['source_score']` | **only** `source_stats['source_score']` |
| STF | `get_realization(i)['stf_time']` | **only** `source_stats['stf_time']` |
| spectrum | `get_realization(i)['spectrum']` | not available |

→ `source_stats['source_score']` is the uniform path that works for both
formats (same keys `ave_tr, ave_tp, ave_vr, err_spectra, pdf,
n_realizations`) — use it when comparing objects of different origin.
That's also why `plot_spectral_comparison` and `plot_source_time_function`
print "No source statistics available" and bail out when the object came
from the legacy format — not a bug.

Round trip verified in `examples/10_ffsp/ffsp_io.py` and
`tests/ffsp/test_all_realization_products.py`.

### Plotting — nine methods, use them

All import matplotlib lazily and operate on the **active** realization
(except `plot_histogram`, `plot_quality_metrics`, `plot_temporal_metrics`,
which sweep the whole ensemble).

**⚠️ All of them call `plt.show()` internally except `plot_spectral_comparison`.**
Consequences:
- In a notebook (inline backend), adding artists after the call
  (`plt.suptitle`, `ax.axhline`, `ax.legend`) has no effect — the figure has
  already been rendered. Print an identifying line **before** the call if
  you need to distinguish models.
- With the `Agg` backend, the figure stays alive, so
  `plt.gcf().savefig(...)` does work right after the call (the pattern used
  in `docs/web/examples/scripts/gen_ffsp.py` and
  `examples/10_ffsp/notebooks/ffsp.ipynb`).
- `plot_spectral_comparison` is the only one that does **not** call
  `show()`: call/save it separately.

| Method | Full signature |
|---|---|
| `plot_histogram` | `(field='slip', bins=50, figsize=(7,5))` — `field` ∈ `x,y,z,slip,rupture_time,rise_time,peak_time,strike,dip,rake` |
| `plot_spacial_distribution` | `(figsize=(10,8), field='rise_time', rotate=False, cmap='coolwarm', show_contours=True, contour_field='rupture_time', show_hypocenter=True, contour_interval=None, contour_color='blue', internal_ref=None, external_coord=None, save_fig=False, model_name='model', image_type='png')` |
| `plot_rupture_snapshot` | `(time_snapshot, figsize=(10,8), field='slip', cmap='YlOrRd', show_rupture_front=True, internal_ref=None, external_coord=None, save_fig=False, model_name='model', image_type='png')` |
| `plot_quality_metrics` | `(figsize=(14,5))` — PDF + spectral error per realization |
| `plot_temporal_metrics` | `(figsize=(15,5))` — rise time, peak time, rupture velocity |
| `plot_spectral_comparison` | `(figsize=(14,6))` — moment-rate vs. target DCF + octaves |
| `plot_source_time_function` | `(figsize=(10,6), xlim=None, save_fig=False, model_name='source', image_type='png')` |
| `plot_crust_layers` | `(figsize=(6,4))` |
| `create_animation` | `(field='slip', figsize=(10,8), cmap='cmap_white', show_contours=True, contour_field='rupture_time', show_hypocenter=True, contour_interval=None, contour_color='black', internal_ref=None, external_coord=None, rotate=False, n_frames=50, fps=10, dpi=100, output_dir='animation_frames', output_video='rupture_animation.mp4', ffmpeg_path=None)` — requires `ffmpeg` |

`internal_ref=[x,y]` (local FFSP coords, km) + `external_coord=[x,y]`
(ShakerMaker coords, km) relocate the grid when plotting. `rotate=True`
swaps the strike/dip axes. None of them mask the `1e10`/`NaN` sentinels —
if the input breaks the surface (see the geometric gotcha above), they
blow up.

## The one bridge you do have to write: FFSP → `FaultSource`

**There is no `to_faultsource()`** (deliberate, see
`docs/web/guides/ffsp.md:125`). You build one `PointSource` per subfault by
hand.

### ⚠️ Units — the mistake that repeats across the whole repo

The kernel returns `x, y, z` in **METERS**
(`shakermaker/ffsp/ffsp_wrapper.f90:294-296` multiplies by 1000).
`PointSource` expects **KM**. Divide by `1e3`.

> `docs/web/guides/ffsp.md:136` says "(km)" and passes `sf["x"][i]` raw to
> `PointSource` — **that is WRONG**, it places the sources 1000× too far
> away. The notebook `examples/10_ffsp/notebooks/example_FFSP.ipynb`
> (cell 8) does it right: `x[i] / 1e3`. Follow the notebook, not the guide.

### Canonical pattern (STF already provided by the package)

```python
import numpy as np
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import SRF2   # or Brune, or Discrete

source.run()
source.set_active_realization(0)
sf = source.get_subfaults()

dt = 0.01
sources = []
for i in range(len(sf["x"])):
    Tr = float(sf["rise_time"][i])
    Tp = 0.15 * Tr            # peak time; or float(sf["peak_time"][i])
    Te = 0.7 * Tr             # start of the decaying tail
    stf = SRF2(Tr=Tr, Tp=Tp, Te=Te, dt=dt,
               slip=float(sf["slip"][i]), a=1.0, b=100.0)
    sources.append(PointSource(
        [sf["x"][i] / 1e3, sf["y"][i] / 1e3, sf["z"][i] / 1e3],   # m -> km
        [float(sf["strike"][i]), float(sf["dip"][i]), float(sf["rake"][i])],
        stf=stf,
        tt=float(sf["rupture_time"][i]),
    ))

fault = FaultSource(sources, metadata={"name": "ffsp_rlz_0"})
```

`SRF2` (`shakermaker/stf_extensions/srf2.py`) **already implements** exactly
the function that used to be hand-written in older notebooks: rising sine
branch for `t<Tp`, plateau `sqrt(a + b/t²)` for `Tp<=t<Te`, decaying sine
tail for `t>=Te`, normalized to unit area and scaled by `slip`. **Do not
rewrite it.** The reference notebook's values are `a=1.0, b=100.0` (note
this differs from `a=1.0, b=1.0`, used in simple non-FFSP source examples —
see `02_sources_and_stf.md`).

### Capping the grid size when building the bridge — the `MINSLIP` pattern

A typical FFSP grid (e.g. `nsubx=256, nsuby=128` = 32768 subfaults) is far
too large to turn every subfault into a `PointSource` and run the
**legacy** engine (`model.run()`, pair-by-pair, no Green's-Function reuse)
— it would be prohibitively slow. The pattern used in
`examples/10_ffsp/notebooks/example_FFSP.ipynb` is to filter by a slip
threshold before building the `PointSource` loop:

```python
MINSLIP = 1.9217   # example: trims down to a small subset of subfaults
for i in range(len(sf["x"])):
    if sf["slip"][i] < MINSLIP:
        continue
    # ... build PointSource as above ...
```

**`MINSLIP` is a practical runtime limiter, not a physically-derived
value** — its only purpose is to keep a small subset of the highest-slip
subfaults so the example runs in reasonable time on the legacy engine.
Choose the value by inspecting the distribution of `sf["slip"]` (e.g. a
high percentile) based on how many subfaults you want to keep. If you use
the OP pipeline (`run_nearest`, see `06_engine_run_modes.md`) instead of
the legacy engine, this filtering is no longer needed for the example to be
viable — the OP pipeline reuses Green's Functions across geometrically
nearby subfaults and scales far better to thousands of subfaults.

### Efficient ensemble (multiple realizations)

Every realization shares fault and receiver geometry, so the FK Green's
Functions are common: run Stage 0-1 **once** and only Stage 2 per
realization (see `06_engine_run_modes.md` for what the stages mean):

```python
model = ShakerMaker(crust, fault, stations)
model.run_nearest(stage='0_1', h5_database_name='gf.h5', dt=dt, nfft=4096, dk=0.1)
for k in range(n_rlz):
    source.set_active_realization(k)
    fault_k = build_fault(source.get_subfaults())      # same pattern as above
    ShakerMaker(crust, fault_k, stations).run_nearest(
        stage=2, h5_database_name='gf.h5', dt=dt, nfft=4096, writer=writer_k)
```

## Known gotchas

- **`x, y, z` from `get_subfaults()`/`get_realization()` come in METERS**,
  not km — divide by `1e3` before passing them to `PointSource`. This is
  the most-repeated gotcha in the whole repo; the guide at
  `docs/web/guides/ffsp.md:136` has the example WRONG, follow the notebook
  instead.
- **Geometric gotcha**: `depth_hypc >= y_hypc * sin(dip)` must hold, or
  subfaults end up above the surface with `rupture_time=1e10`/`slip=NaN` —
  fix the input, not the plot that blows up downstream.
- **`fc_main_1`/`fc_main_2` get overwritten after `run()`** with the
  kernel's effective values — the requested ones remain in
  `fc_main_1_requested`/`fc_main_2_requested`.
- **`realization_id` is 1-based, the index into `get_realization(i)` is
  0-based** — don't confuse `id_ran1..id_ran2` (the requested range,
  1-based) with the access index.
- **`9.05` in `FFSP_MAGNITUDE_CONSTANT` is intentional**, not the `9.1`
  from Hanks & Kanamori — a kernel-specific convention, do not "fix" it.
- **Objects loaded via `from_ffsp_format` (legacy) lack `metrics`,
  `stf_time`, and `spectrum`** in `all_realizations`/`subfaults` — use
  `source_stats['source_score']` as the uniform path when comparing against
  an HDF5-loaded object.
- **Every `plot_*` calls `plt.show()` except `plot_spectral_comparison`** —
  in notebooks you can't decorate the figure after the call; with the
  `Agg` backend you can still save it afterward
  (`plt.gcf().savefig(...)`).
- **`MINSLIP` (or any similar slip filter) is only a runtime limiter** for
  the legacy engine — it has no physical justification, and isn't needed
  with the OP pipeline.

## Combines with

- `01_crust_model.md` — `FFSPSource` requires a valid `CrustModel`
  (`crust_model=crust`, type-checked).
- `02_sources_and_stf.md` — the bridge to `FaultSource` reuses
  `PointSource`/`FaultSource`/`SRF2` from that file.
- `06_engine_run_modes.md` — the "efficient ensemble" pattern (Stage 0-1
  once, Stage 2 per realization) depends on understanding the OP pipeline.
- `11_mpi_and_hpc.md` — `run()` distributes realizations across MPI ranks
  when there's more than one (`id_ran2 > id_ran1`).
- `10_plotting.md` — `FFSPSource`'s 9 `plot_*` methods are self-contained;
  they do not go through `tools/plotting.py`.
- `RECIPES.md` — the "FFSP → finite fault → OP run" recipe combines
  everything above.

## See also

- `examples/10_ffsp/ffsp_run.py`, `examples/10_ffsp/ffsp_io.py` — minimal
  smoke tests.
- `examples/10_ffsp/notebooks/example_FFSP.ipynb` — **the correct
  FFSP→FaultSource→ShakerMaker bridge**, including the unit conversion.
- `examples/10_ffsp/notebooks/ffsp_all_realization_products.ipynb` — full
  per-realization products, HDF5 round trip.
- `docs/web/guides/ffsp.md` — user guide (⚠️ units bug on line 136, don't
  follow that specific example).
- `docs/web/background/finite_fault.md` — theory: the random fields, the
  PDF scoring.
- `tests/ffsp/test_all_realization_products.py`, `tests/ffsp/mpi_*_contract.py`
  — verified contracts.
- `examples/EXAMPLES_REFERENCE.md`, section "10. FFSP".

**Mandatory citation**: `FFSP_CITATION` — Pengcheng Liu (c) 2005,
modifications by Chen Ji (2020), based on Liu et al. (2006). Printed
automatically when `verbose=True`.
