# Interpolated Green's functions: `run_inter`

`run_inter(scheme='tensor')` is the pipeline for **DRM boxes** and any dense
receiver set where neighbouring receivers must stay coherent with each other.
It replaces the reuse of `run_nearest` by interpolation in distance: receiver
and source depths are computed exactly and the horizontal distance is
interpolated with 4-point Lagrange weights in reduced time. The method, its
validation and its limits are explained in
[Interpolated Green's functions](../background/gf_interpolation.md).

## When to use which

| Case | Use |
|---|---|
| DRM box (two layers of nodes one element apart) | **`run_inter(scheme='tensor')`** |
| dense receiver grid whose differences matter (gradients, strains, arrays) | **`run_inter(scheme='tensor')`** |
| a few stations, validation, debugging | `run()` |
| sparse surface stations, large tolerances acceptable | `run_nearest()` |
| reusing a database built with `run_nearest` | `run_nearest()` (the databases are not interchangeable) |

## Minimal example

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox

box = DRMBox([x0, y0, 0.0], [42, 42, 12], [0.0025, 0.0025, 0.0025], metadata={"name": "box"})
model = ShakerMaker(crust, fault, box)

band = 20.0                                  # Hz, band of use
model.run_inter(
    stage="all", h5_database_name="./gf/gf", scheme="tensor",
    f_err=band, freq_cut=1.25 * band, dh_min=0.020 * 20.0 / band,
    dt=0.0025, nfft=65536, tb=800, dk=0.1, kc=30, sigma=2, taper=0.9, pmax=1,
    tmin=0.0, tmax=65.5,
    writer=DRMHDF5StationListWriter("./box.h5drm"), writer_mode="progressive",
)
```

```bash
mpirun -n 64 python run_box.py
```

Rank 0 coordinates Stage 1 and writes the database; the other ranks compute
core calls. In Stage 2 the sources are split among all ranks. The database is
`<h5_database_name>_gf.h5` and `_map.h5`.

## Parameters

| Parameter | Recommended | Meaning |
|---|---|---|
| `scheme` | `'tensor'` | the validated scheme (`'grid'`, a fixed distance grid per pair of depths, and `src_levels='interp'` are experimental) |
| `f_err` | the band of use (Hz) | band in which the interpolation error is measured |
| `freq_cut` | 1.25 × band | the core stops at 1.4 × `freq_cut`; nothing changes below `freq_cut` |
| `dh_min` | 0.020 × 20 / band (km): 20 m at 20 Hz, 27 m at 15 Hz, 40 m at 10 Hz, 80 m at 5 Hz | spacing of the candidate distances; smaller in the near field (below) |
| `dh0` | default (2 × `dh_min`) | initial node spacing of the bisection; starting coarser accepts cells with a larger interior error |
| `eps_v`, `eps_s` | 0.01, 0.05 (default) | value and slope tolerances of a distance cell |
| `eps_bands` | default (whole band and upper half) | frequency ranges where the tolerances hold |
| `store` | `'zfp20'` (default) | database compression: `'zfp<bits>'` (lossy, ~1e-4 at 20 bits), `'zstd'` (lossless), `'f32'` |
| `decimate` | `True` (default) | with `freq_cut`, one sample every *q* in the database, rebuilt in Stage 2 (~4 times smaller) |
| `depth_tol` | 1e-6 km (default) | depth tolerance of the crust while `run_inter` runs (restored afterwards) |
| `delta_v_rec` | 1e-5 km (default) | receiver depths closer than this share a level |
| core parameters | as `run_nearest` | `dt`, `nfft`, `tb`, `dk`, `kc`, `sigma`, `taper`, `pmin`, `pmax`, `smth` |
| `stage` | `'all'`, `1` or `2` | `1` builds the database; `2` convolves with an existing one |

**Time step.** `dt` is the time step of the output and is set by the model
that reads it (for a finite element model, its stability limit). For a 20 Hz
band, `dt` = 0.005 s is enough for the FK solution; if the finite element model
runs at 0.0025 s and interpolates the input linearly in time, use 0.0025 s to
avoid a few percent of error near 20 Hz.

**Window.** Keep `nfft × dt` well above `tmax` (wrap-around) and `tb × dt` of a
second or two before the first arrival. The database stores only the samples
up to `tmax`.

## One database, many ruptures

The database depends on the receiver positions, the source **positions** and
the crust, not on slip, rupture times or source time functions. Build it once
(`stage=1`) and run `stage=2` for every rupture with the same subfault
positions:

```python
model.run_inter(stage=1, h5_database_name=db, scheme="tensor", **params)        # once
for rupture in ruptures:                                                         # many
    model_r = ShakerMaker(crust, rupture_fault, box)
    model_r.run_inter(stage=2, h5_database_name=db, scheme="tensor",
                      writer=DRMHDF5StationListWriter(out[rupture]), **params)
```

On a cluster, copy the database to the node-local disk before Stage 2: every
rank reads it many times.

## Reading the output of Stage 1

```text
[Stage 1 tensor] 50/1792 levels, 5694 nodes, 79 s
...
[Stage 1 tensor] 172683 nodes stored (343224 computed, 6216 calls, 1792 levels), 10734.9 MB (zfp20), 2020 s
```

- *levels*: one per (receiver depth, source depth); progress is linear;
- *nodes stored* against *computed*: how many candidates the bisection kept;
- a **warning** means some cells could not meet `eps_v` with `dh_min`:

```text
[Stage 1 tensor] WARNING: 84 distance cells miss eps_v at their midpoint and cannot
be split below dh_min = 20.0 m (largest value error 173.3 %, tolerance 1.0 %).
```

| Largest value error | Meaning | Action |
|---|---|---|
| no warning | every tested cell met the tolerances | none |
| a few percent | tolerance slightly exceeded in a few cells | usually acceptable; check the DRM layer difference if in doubt |
| tens of percent or more | the interpolation failed there (near field) | reduce `dh_min` (e.g. 5 m) and rebuild the database |

## How not to use it

- **Near field with the default `dh_min`.** A source in a slow shallow layer
  with receivers at a similar depth and less than ~1 km away needs nodes closer
  than $0.1 / (f_\text{max} \Delta p)$, with $\Delta p \approx 1/V_s - 1/V_p$
  of that layer (about 8 m at 20 Hz for $V_s$ = 0.75 km/s). Watch the warning.
- **A band above `freq_cut`.** Content above `freq_cut` is tapered and above
  1.4 × `freq_cut` it is zero. Do not use the output above the band of use.
- **A coarser `dh0`.** Starting the bisection at 8 × `dh_min` lets cells with a
  large interior error pass (4.3 % instead of 0.9 % at 5–10 Hz in a 10 Hz
  band). Keep the default.
- **Interpolating the source depth** (`src_levels='interp'`). Above ~5 Hz the
  surface-reflected phases make it fail (10 % at 20 Hz with 80 m rows).
- **Mixing databases.** A `run_nearest` database cannot be read by
  `run_inter`, and a database built with one `dt`, `nfft`, band or receiver set
  is only valid for that one.
- **Expecting bit-identical results with older runs.** Every contribution is
  now placed at its real time, fraction of a sample included, also in
  `run_nearest`.

## Cost and size

A DRM box of 8 179 nodes driven by 32 768 point sources at 2.9–13 km (1 792
levels), dt 0.005 s, 10 Hz band, 64 MPI processes:

| Stage | Time | Size / memory |
|---|---|---|
| Stage 1 | 34 min | database 10.7 GB (172 683 nodes); memory constant (~7 GB with 8 processes) |
| Stage 2 | 38 min | `.h5drm` 7.7 GB; ~2.3 GB per process |

Per stored Green's function: ~118 KB with `nfft` 65 536, dt 0.0025 s, 20 Hz,
zfp20 and decimation (2.4 MB without trimming, compression and decimation).
The `.h5drm` output does not depend on the method; for many ruptures it is the
largest part of the disk.

## Reference

- Method and validation: [Interpolated Green's functions](../background/gf_interpolation.md)
- Example: `examples/16_run_inter/`
- API: [`ShakerMaker.run_inter`](../api/shakermaker.md)
