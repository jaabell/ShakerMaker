# `run_inter` — Green's functions interpolated in distance (coherent DRM boxes)

## What this is

`ShakerMaker.run_inter(...)` is the third pipeline of the engine, next to the
legacy `run()` and the OP pipeline `run_nearest()` (`06_engine_run_modes.md`).
It exists because **reuse by slot breaks DRM boxes**: `run_nearest` gives every
pair inside a slot (`delta_h`, `delta_v_rec`, `delta_v_src`) the same
waveform, so two neighbouring box nodes either get identical traces or a jump
at the slot edge. The DRM turns the difference between its two layers of nodes
into effective forces, so the staircase becomes spurious high-frequency input
(a free-field FE model fed with such a box amplified 4.5–7 Hz by 7–12 times
and let 32 % of the energy leave the box). Lowering the tolerances does not
fix it.

`run_inter(scheme='tensor')`:
- computes **receiver and source depths exactly** (one level per pair of
  depths; the source depth cannot be interpolated above ~5 Hz because
  surface-reflected phases move the other way);
- **interpolates the horizontal distance** with 4-point (cubic) Lagrange
  weights in **reduced time** (every node trace aligned at its own first
  arrival), and places each pair at its own first arrival, fraction of a
  sample included;
- chooses the distance nodes by **bisection** on candidates computed every
  `dh_min` in shared-kernel core calls (each extra distance ≈ 5 % of a call);
- streams Stage 1 level by level (memory constant), stores a trimmed,
  ZFP-compressed and decimated database, and uses a per-receiver Stage 2
  kernel.

Use it for **DRM boxes** and any dense receiver set whose neighbour
differences matter. Use `run()` for a few stations and `run_nearest()` for
sparse stations where reuse is acceptable.

## Source of truth

- `shakermaker/shakermaker.py` — `ShakerMaker.run_inter` (entry point; sets
  `depth_tol` on the crust while it runs, `dh0 = 2*dh_min` when `None`).
- `shakermaker/run_inter_tensor.py` — `stage01_tensor` (Stage 0+1; with exact
  source depths it calls `_stage01_exact_streamed`), `_bisect` (node
  selection; returns unresolved cells), `decimation` / `upsample` (decimated
  database), `stencil4` (vectorised weights), `_kernel2` (Stage 2 kernel),
  `stage2_tensor` (Stage 2).
- `shakermaker/run_inter.py` — the experimental `'grid'` scheme
  (`stage0_inter`, `stage1_inter`, `stage2_inter`).
- `shakermaker/station.py` — `lagrange4_taps`, `add_shifted` (placement with a
  fraction of a sample; used by every Stage 2 path, `run_nearest` included).
- `shakermaker/core/subfk.f`, `subgreen.f`, `core.pyf` — optional `fcutin`
  (frequency cut) and `xmaxin` (common wavenumber step) of `core.subgreen`.

## Signature (tensor scheme, the relevant arguments)

```python
model.run_inter(
    stage='all',                 # 'all' | 0/1 (database) | 2 (convolution)
    h5_database_name=...,        # base name: writes <name>_gf.h5 and <name>_map.h5
    scheme='tensor',             # the validated scheme ('grid' is experimental)
    f_err=20.0,                  # Hz, band where the interpolation error is measured
    freq_cut=25.0,               # Hz, core stops at 1.4*freq_cut (recommended 1.25*band)
    dh_min=0.020,                # km, candidate spacing (0.020*20/band)
    dh0=None,                    # km, initial node spacing (None -> 2*dh_min)
    eps_v=0.01, eps_s=0.05,      # value / slope tolerances of a distance cell
    eps_bands=((0.0, 1.0), (0.5, 1.0)),   # fractions of f_err where they hold
    store='zfp20',               # 'zfp<bits>' | 'zstd' | 'f32'
    decimate=True,               # with freq_cut: one sample every q dt in the database
    depth_tol=1e-6,              # km, crust depth tolerance while run_inter runs
    src_levels='exact',          # do not use 'interp' above ~5 Hz
    delta_v_rec=1e-5,            # km, receivers closer in depth share a level
    dt=..., nfft=..., tb=..., dk=..., kc=..., sigma=2, taper=0.9, pmin=0, pmax=1, smth=1,
    tmin=0., tmax=...,
    writer=..., writer_mode='progressive', showProgress=True,
)
```

## Recommended values by band

| Band (Hz) | `f_err` | `freq_cut` | `dh_min` | `dh0` (default) | Core speed-up |
|---|---|---|---|---|---|
| 20 | 20 | 25 | 0.020 km | 0.040 km | 33× |
| 15 | 15 | 18.75 | 0.027 km | 0.053 km | 57× |
| 10 | 10 | 12.5 | 0.040 km | 0.080 km | 122× |
| 5 | 5 | 6.25 | 0.080 km | 0.160 km | 292× |

`dt` is the output time step (set by the consumer, e.g. the FE stability
limit); 0.005 s is enough for a 20 Hz band. Keep `nfft*dt` well above `tmax`.

## Minimal working example

`examples/16_run_inter/drm_coherence.py`: a 2.5 m DRM box (531 nodes) and a
2 × 2 patch of point sources in a 4-layer crust; reference with one Green's
function per pair, `run_nearest` with reuse (40 m / 5 m / 200 m) and
`run_inter`. Error of the difference between the two DRM layers: reuse
35–50 %, `run_inter` 0.1–0.4 % (0.1–10 Hz). `run_inter` alone takes ~4 s on
one process; the reference dominates the runtime.

Production pattern — one database, many ruptures with the same subfault
positions:

```python
model.run_inter(stage=1, h5_database_name=db, scheme='tensor', **params)       # once
for name, fault in ruptures.items():
    m = ShakerMaker(crust, fault, box)
    m.run_inter(stage=2, h5_database_name=db, scheme='tensor',
                writer=DRMHDF5StationListWriter(f"{name}.h5drm"), **params)
```

Run with MPI (`mpirun -n N python script.py`): rank 0 coordinates Stage 1 and
writes the database, the other ranks compute core calls; Stage 2 splits the
sources over all ranks. On a cluster, copy the database to node-local disk
before Stage 2.

## Reading Stage 1 output

- `[Stage 1 tensor] k/N levels, M nodes, T s` — progress, linear in levels.
- `... nodes stored (C computed, K calls, N levels), S MB (zfp20)` — summary.
- `WARNING: n distance cells miss eps_v ... below dh_min = X m (largest value
  error E %)` — cells that still miss the value tolerance at the finest
  spacing. **E of tens of percent or more = interpolation failed there (near
  field): reduce `dh_min` and rebuild.** E of a few percent = tolerance
  slightly exceeded.

## Gotchas

- **Near field.** Reduced time aligns the first arrival only; later phases
  move by Δp·Δx between nodes. Rule: `f_max * Δp * dh_min ≲ 0.1`, with Δp ≈
  1/Vs − 1/Vp of the source layer when the source is shallow and the box is
  close (< ~1 km). Example: source 100 m deep in a Vs 0.75 km/s layer, box
  0.5 km away, 20 Hz → needs ~8 m; `dh_min` 20 m gave 24 % error at 15–20 Hz,
  5 m gave 0.3 %. Far field (sources at km depth, boxes several km away):
  defaults are fine.
- **Do not use the output above the band.** Above `freq_cut` it is tapered,
  above 1.4·`freq_cut` it is zero.
- **Keep `dh0` at its default.** Starting at 8·`dh_min` let cells with interior
  error pass (4.3 % vs 0.9 % at 5–10 Hz, 10 Hz band).
- **Databases are not interchangeable**: `run_nearest` and `run_inter`
  databases differ, and a `run_inter` database is tied to its `dt`, `nfft`,
  band, receiver set and source positions (not to slip or rupture times).
- **Placement changed for every Stage 2 path**: contributions are placed at
  their real time (fraction of a sample, 4 Lagrange taps); results are not
  bit-identical with older runs, also for `run_nearest`.
- **Database folder**: `run_inter` creates it; `run_nearest` does not.
- **The core must be rebuilt** after updating (Fortran `subfk.f`,
  `subgreen.f`, `core.pyf` changed).
- **`depth_tol`**: `CrustModel.depth_tol` stays 0.01 km for `run`/
  `run_nearest`; `run_inter` sets 1e-6 km on the crust only while it runs.

## Validation summary

- 2.5 m DRM box, 4-layer crust (200 m at Vs 0.75 km/s), 256 point sources at
  7–8 km: layer-difference error 0.20 % (P95 0.49 %) at 15–20 Hz vs 457–1 631 %
  with reuse.
- Five-layer crust (phase crossings), 20 Hz: 0.58 % (P95 1.4 %).
- Free-field FE model: Fourier ratio 1.01–1.02 at 4.5–7 Hz, 0.005 % energy
  leaving the box (reuse: 7–12 and 32 %).
- Against SW4 (8 179-node box, 32 768 point sources, 6.5–24 km): displacement
  correlation 0.975–0.988, response spectra 0.5–5 s median ratio 0.97–0.99.
- Cost of that box (dt 0.005 s, 10 Hz, 64 processes): Stage 1 34 min
  (10.7 GB database), Stage 2 38 min (7.7 GB output).

## Combines with

- `04_receivers.md` — `DRMBox`, `PointCloudDRMReceiver` (receiver depths become levels).
- `07_writers_persistence.md` — `DRMHDF5StationListWriter` (output unchanged).
- `08_drm_workflow.md` — the DRM campaign; use `run_inter` instead of `run_nearest`.
- `03_ffsp_source.md` — FFSP ruptures with the same subfault positions share one database.
- `11_mpi_and_hpc.md` — MPI launch, node-local copy of the database.

## See also

- `docs/web/background/gf_interpolation.md` — the method with figures.
- `docs/web/guides/run_inter.md` — the user guide (how to use, how not to).
- `examples/16_run_inter/` — runnable comparison.
- `tests/engine/test_run_inter_tensor.py`, `test_decimated_store.py`,
  `test_bisect_unresolved.py`, `test_freq_cut.py`, `mpi_run_inter_tensor.py`.
