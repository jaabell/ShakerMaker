# Coordinates and sign conventions — positions, motion, SW4, and the OpenSees matrix T

## What this is

The single operational reference for axes, component order, and signs across
every ShakerMaker output, SW4, and the OpenSees `.h5drm` consumer. The
narrative version with figures is `docs/web/background/conventions.md` and
the "Using the .h5drm in OpenSees" section of `docs/web/guides/drm.md`; the
figures live in `docs/web/assets/images/conventions/` (regenerate with
`make_figures.py` there).

Everything below was checked against SW4 and the SCEC LOH.1 analytical
solution by reading the stored rows without relabelling and deducing, by
correlation, which physical axis each row carries
(`verification_loh1_formats_vs_prose.png`, `verification_ring_vs_sw4.png`).

## Rule 1 — one frame for ShakerMaker and SW4

| | x | y | z | units | origin |
|---|---|---|---|---|---|
| ShakerMaker | North | East | depth, positive **down** | km | user's choice (e.g. epicentre) |
| SW4 (`grid az=0`, the default) | North | East | depth, positive **down** | m | corner of the domain |

Same NED frame (`x × y = z`). The SW4 exporter applies a **pure translation**
(km → m, shift to the corner) — `shakermaker/sw4_exporter/coordinates.py`.
SW4's own source confirms the default: `sw4/src/EW.C` `mGeoAz(0.0) // x=North, y=East`.

## Rule 2 — positions vs motion

- **Positions** (where a point is) are always stored as given:
  **(North, East, depth)**, km. `Station.x`, `PointSource.x`, `.npz` `_x`,
  `.h5` `Data/xyz`, `.h5drm` `DRM_Data/xyz`, `DRM_QA_Data/xyz`,
  `DRM_Metadata/drmbox_x0` and box bounds. No writer transforms them.
- **Motion** (how it moves) keeps the same signs everywhere — positive North,
  East, **down** — but the **order** depends on the output:

| Output | order | where it is decided |
|---|---|---|
| `z, e, n, t = sta.get_response()` | (down, East, North) | `shakermaker/station.py` `get_response` |
| `.npz` (`Station.save`/`load`) | stored by name (`_z`, `_e`, `_n`); `get_response()` gives (down, East, North) | `station.py` `save`/`load` |
| Station Green's functions (`metadata={"save_gf": True}`) | `(z, e, n, t, tdata, t0)` = (down, East, North), no STF | `station.py` `add_greens_function` |
| `tdata` (9 fundamental Green's functions, OP `_gf.h5` `/tdata`) | internal cylindrical frame of the core; **not physical components** — use only through `subfocal`/`subgreen2` | `shakermaker/core/subfk.f`, `subfocal.f` |
| `.h5` `Data/{velocity,displacement,acceleration}` rows `3i,3i+1,3i+2` | **(East, North, down)** | `slw_extensions/hdf5stationlistwriter.py` |
| `.h5drm` `DRM_Data/...` and `DRM_QA_Data/...` rows `3i,3i+1,3i+2` | **(East, North, down)** | `slw_extensions/drmhdf5stationlistwriter.py` (`ve, vn, vz`) |
| SW4 `rec`, `nsew=0` (exporter default) | (North, East, down) | `sw4/src/TimeSeries.C` |
| SW4 `rec`, `nsew=1` | (East, North, **up**) | SW4 flips z |

In ShakerMaker axes, the `.h5`/`.h5drm` motion rows are `(u_y, u_x, u_z)`:
the horizontals are swapped relative to the positions, no sign is changed.
Velocity, displacement and acceleration share the same order (the writers
integrate/differentiate each component separately).

To compare with SW4 (`nsew=0`): North = SW4 X = `.h5` row 1 = `n`;
East = SW4 Y = row 0 = `e`; down = SW4 Z = row 2 = `z`. No sign changes.

## Rule 3 — why the `.h5drm` motion is reordered and the positions are not

The `.h5drm` is read by OpenSees' `H5DRMLoadPattern`
(`SRC/domain/pattern/drm/H5DRMLoadPattern.cpp`), which:

1. **Places nodes with the positions**:
   `x_model = T · ((x_file − drmbox_x0) · crd_scale) + x0`. `T` is used only
   here (the coordinate lambda in `do_intitialization`).
2. **Applies the motion unrotated**: it reads `displacement` and
   `acceleration` (never `velocity`), copies row 0 → DOF 1, row 1 → DOF 2,
   row 2 → DOF 3, and **flips row 2 on read** (`d[2] = -d[2]`, `a[2] = -a[2]`),
   then assembles the DRM effective forces component by component.

So the positions stay in ShakerMaker's frame (the consumer maps them with
`T`, which it needs anyway for km → m and the origin), while the motion is
written already in the order of a model built in ENU (X = East, Y = North,
Z = up), with the vertical flip done by the load pattern.

## Rule 4 — the matrix T for a Z-up model

```
               North  East  depth     <- file axes (columns)
model X     [    0     1     0  ]     X = East
model Y     [    1     0     0  ]     Y = North
model Z     [    0     0    -1  ]     Z = up
```

STKO: **Local X = (0, 1, 0)**, **Local Y = (1, 0, 0)**; STKO computes
Local Z = X × Y = (0, 0, −1) and writes the full 9 values with
`do_transform = 1`
(`external_solvers_STKO/opensees/analysis_steps/Patterns/addPattern/H5DRM.py`).
With it: DOF X ← East, DOF Y ← North, DOF Z ← down flipped = up. All three
match.

Command, always with all 18 values:

```tcl
pattern H5DRM 1 "motions.h5drm" 1.0 1000.0 1.0e-3 1   0 1 0   1 0 0   0 0 -1   0.0 0.0 0.0
```

```python
ops.pattern('H5DRM', 1, 'motions.h5drm', 1.0, 1000.0, 1.0e-3, 1,
            0, 1, 0,  1, 0, 0,  0, 0, -1,  0.0, 0.0, 0.0)
```

Optional arguments are parsed only when more arguments follow them; stopping
at `do_transform` silently leaves the transformation and `crd_scale` off.

## Common mistakes

- `T` = identity → model X = North receives East, Y = East receives North,
  and the flipped vertical lands on a Z-down axis: everything wrong.
- Flipping row 2 of the file by hand → the load pattern flips it again.
- Reading `.h5`/`.h5drm` rows as (North, East, …) because `xyz` is
  (North, East, depth) → horizontals swapped.
- Unpacking `get_response()` as `e, n, z` → it is `z, e, n`.

## How to verify a new OpenSees model

Record an interior node near the QA station and correlate with the file's QA
rows: `u_X` vs row 0 positive, `u_Y` vs row 1 positive, `u_Z` vs row 2
**negative** (model Z up, file row 2 down). Use a source/QA geometry where
North and East differ clearly, or a horizontal swap goes unnoticed.
