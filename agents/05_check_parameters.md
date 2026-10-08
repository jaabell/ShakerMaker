# check_parameters — deciding dt/nfft/dk/tb/tmax before you spend compute

## What this is

`ShakerMaker.check_parameters(...)` is a **pure-arithmetic pre-run sanity
check** (no FK evaluation, no Fortran call) that tells you whether your
choice of `dt`, `nfft`, `dk`, `tb`, `tmax` is physically sound for the
specific crust + source + receiver geometry you built. It mirrors the exact
formulas the Fortran core (`fk.f`/`subfk.f`) uses internally, and every line
of its report is annotated with the source line it corresponds to
(`[fk.f:72]`, etc.).

Call it **immediately after building the model, before every real run** —
it costs nothing and catches the two most common failure modes: (1) a
`tmax`/`nfft` combination that silently truncates or wraps around the
signal, and (2) a `dk` too coarse for the wavenumber integral, which
corrupts the whole seismogram without raising any error.

## Source of truth

`shakermaker/shakermaker.py:287-511` (`ShakerMaker.check_parameters`).

## The model organized around two numbers you actually choose

Per its own docstring: *"Pre-run parameter check, organised around the two
numbers YOU pick: `dt` (frequency band) and `tmax` (output window).
Everything else (`nfft`, `dk`, `tb`) is *derived* from those plus the model
geometry."* Call it right after building the model:

```python
model = ShakerMaker(crust, fault, stations)
model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
```

## Full signature

```python
check_parameters(self, dt, nfft, dk, tb, tmax=100., tmin=0.,
                 sigma=2, smth=1, taper=0.9, wc1=1, wc2=2,
                 pmin=0, pmax=1, nx=1, kc=15.0,
                 n_per_wavelength=10, courant=1.0,
                 fem_fmax=None, coda=10.0)
```

- `dt`, `nfft`, `dk`, `tb`, `tmax`, `tmin`, `sigma`, `smth`, `taper`, `wc1`,
  `wc2`, `pmin`, `pmax`, `nx`, `kc` — same meaning/units as `run()`/
  `run_nearest()` (see `06_engine_run_modes.md`); pass here exactly what
  you intend to pass to the real run.
- `n_per_wavelength=10` — points per shortest wavelength used **only** for
  the FEM-mesh sizing advice printed in the report (not an FK parameter).
- `courant=1.0` — Courant number `C` for the FEM CFL step estimate
  (`dt_fem = C*dx/Vp_surf`), again advisory only.
- `fem_fmax=None` — target frequency for FEM mesh sizing; defaults to the
  FK usable band `f_max` so the advised mesh matches the motion that will
  drive it.
- `coda=10.0` — fixed margin (s) added when estimating the end of the
  useful signal window; not an FK parameter, only affects the
  recommendation, not the hard pass/fail gates.

## What it actually computes (the formulas behind the report)

Geometry is read straight from the model (`src`, `rec` positions in km,
`tt` = source rupture-time offsets):

- `hs` = **source-receiver vertical separation** (`dz`, clamped `>=1e-6`),
  **not** total crust thickness — this is what the Fortran core (`fk.f:48`)
  actually uses for `dk` resolution and `kc`.
- `r` = horizontal distance, `slant` = 3-D distance.
- `Vp_max` = fastest Vp in the crust, `Vs_min` = slowest Vs (and its layer's
  `Vp_surf`, used for the FEM CFL estimate because the small elements live
  in that soft surface layer, not in the fast layer).
- `Vray = 0.92 * Vs_min` — a Rayleigh-wave-speed proxy for the slow surface
  tail.

**`dt` → frequency band and implied FEM mesh** (informational, not gated):
```
f_Nyq        = 1/(2*dt)                                    [fk.f:72]
f_max usable = (1-taper)*f_Nyq                              [fk.f:74]
lambda_min   = Vs_min / f_max                                (m)
dx (FEM)     <= lambda_min / n_per_wavelength                (m)
dt (FEM CFL) <= courant * dx / Vp_surf
```

**`nfft` → must hold the signal without wrap-around** (hard gate):
```
T_sig  = (last_arrival - first_arrival) + tb*dt   (physics-driven length)
T_tmax = (tmax - first_arrival) + tb*dt           (tmax-driven length)
T_need = max(T_sig, T_tmax)
nfft_recommended = next_power_of_2(T_need / dt)
PASS if: nfft is a power of 2 AND nfft*dt >= T_sig
```

**`dk` → wavenumber-integral resolution** (hard gate):
```
xmax = max(r, hs)
N    = kc * xmax / (pi * hs * dk)     # points to the evanescent cutoff, need >= 10   [fk.f:115]
L    = 2 * xmax / dk                  # spatial period of the periodic (ghost) source [fk.f:107]
PASS if: dk < 0.5 AND N >= 10
```
Also reports when the nearest **ghost-source image** arrival falls relative
to `tmax` — you want the image arrival *after* `tmax`, or the periodic
repetition of the source contaminates your record.

**`tb` → pre-arrival zero-padding** (hard gate):
```
tb_min = ceil(1/dt)                      # at least 1s of pre-roll
tb_max = (T_valid - window_length) / dt  # can't push the coda past the record
PASS if: tb_min <= tb <= tb_max
```

**`tmax`/`tmin` → coherent with the record window** (hard gate):
```
end_of_record = first_arrival - tb*dt + nfft*dt      [fk.f:72]
PASS if: 0 <= tmin < tmax <= end_of_record
```
If `tmax` is below `end_of_record` but also below the full physical signal
length (`last_arrival`), it's flagged `[OK, cuts coda]` — not a hard error,
but you're clipping the surface-wave tail on purpose or by oversight.

## Reading the RESULT

The report separates, explicitly:
- **Hard checks** (`nfft_ok`, `dk_ok`, `tb_ok`, `tmax_ok`) — errors that
  would corrupt the physical result (wrap-around, undersampled Bessel
  integral, clipped record). Printed as `RESULT: N error(s) -- fix before
  running` with one line per failing check and its recommended fix.
- **Recommended changes** — it runs fine, but a better value exists (e.g.
  raising `tmax` to `tmax_reco` so the coda isn't clipped). Printed
  separately, never counted as an "error."

It also prints a ready-to-paste `model.run(...)` call with every parameter
annotated as `YOU SET`, `DERIVED`, `RECOMMEND`, or `default`, each tagged
with its Fortran source line.

## Return value

```python
{
    "passed": bool,           # AND of all 4 hard checks
    "recommended": {
        "dk": float, "tb": int, "nfft": int, "tmax": float
    }
}
```

## Gotchas — the most important one first

- **`check_parameters` is advisory-only by design and never blocks
  execution.** Its own docstring: *"RESULT separates hard checks (errors
  that corrupt the result) from recommended changes (it runs, but a better
  value exists)."* Nothing in `run()`/`run_nearest()`/etc. reads its return
  value or refuses to proceed. Multiple notebooks in `examples/12_validation/`
  capture `check_parameters` reporting hard errors (e.g. `tmax` beyond the
  valid record end) and then run anyway with the un-recommended values —
  this is expected behavior, not a bug, and the resulting seismogram is only
  actually corrupted **past** `end_of_record`; if your downstream analysis
  only looks at an earlier time window, an unheeded warning may be harmless.
  **You are responsible for reading the report and deciding**, this method
  will not stop you from running with bad parameters.
- `hs` is the source-receiver **vertical** separation, not total crust
  thickness — easy to misread when eyeballing the report.
- The FEM-mesh sizing block (`n_per_wavelength`, `courant`, `fem_fmax`) is
  pure advice for anyone about to drive a finite-element model with this
  motion (e.g. via DRM/H5DRM) — it has no effect on the FK run itself.
- `tb` has both a floor (`tb_min`, at least 1s of pre-roll) and a ceiling
  (`tb_max`, don't push the coda out of a fixed-length record) — increasing
  `tb` blindly to "be safe" can itself fail the check if it exceeds `tb_max`.

## Combines with

- `06_engine_run_modes.md` — call `check_parameters` with the exact same
  `dt/nfft/dk/tb/tmax` you're about to pass to `run()`, `compute_gf()`,
  `run_fast()`, or `run_nearest()`.
- `04_receivers.md` / `02_sources_and_stf.md` — the geometry it reads
  (`hs`, `r`, `Vs_min`, etc.) comes directly from whatever `CrustModel`,
  `FaultSource`, and receiver `StationList` you already built — build the
  full model object graph first, then call this.
- `RECIPES.md` — every worked recipe should include a `check_parameters`
  call before the actual run.

## See also

- `examples/05_engine_direct/check_parameters.py`,
  `examples/05_engine_direct/notebooks/engine_direct.ipynb` — minimal demo.
- `examples/12_validation/notebooks/LOH1_validation.ipynb`,
  `LOH1_greens_functions.ipynb` — real examples of the advisory-only
  behavior (hard errors reported, run proceeds anyway).
