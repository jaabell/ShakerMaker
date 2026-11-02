# 18 - The five ShakerMaker engines against SW4

Uses the SW4 reference cases of `17_sw4_reference` (two media: SCEC LOH.1, and a layered 1D crust with
200 m at Vs 0.75, 800 m at 1.57, 14.5 km at 3.14 km/s over a half-space at 4.0 km/s; three sources: a deep
point source, a shallow point source at 0.5 km and a 4 x 4 km fault of 256 subfaults; ~100 receivers per
medium on SW4 grid nodes, including 24 nodes of a DRM box) and compares, case by case, every engine:

| Engine | What it is |
|---|---|
| `run` | direct engine, one Green's function per pair (not for the 256-subfault cases: hours) |
| `exact` | `run_nearest` without reuse (tolerances 1 cm), one wavenumber step per pair of depths |
| `nearest` | `run_nearest` with reuse tolerances of 40 m, 5 m and 200 m |
| `inter_grid` | `run_inter`, grid scheme, cubic interpolation on a 40 m distance grid |
| `inter_tensor` | `run_inter`, tensor scheme (frequency cut, ZFP, decimated database) |

`data/sm/` holds the results of `run` and `inter_grid`; the others are read from
`../17_sw4_reference/data/sm/`. Regenerate any of them with `../17_sw4_reference/run_shakermaker.py`.

```bash
python compare.py          # ~25 s; writes _output/comparison_{sw4,engines,drm}.csv and prints a summary
```

What it prints (0.1-2 Hz):

- against SW4, median correlation (best lag within 0.1 s) 1.000 for every engine in LOH.1, 0.997-0.998
  (deep point), 0.980-0.981 (shallow point) and 0.991 (fault) in the layered crust; peak ratios 0.98-1.01;
- each engine against `run`;
- the difference between the two layers of the DRM box faces against `run`: `nearest` 28-77 % (median
  normalised RMS), `inter_tensor` 1-3 %, `inter_grid` below 1.4 %: reuse breaks the coherence of the box,
  interpolation keeps it.

The layered 256-subfault case has no `inter_grid` or `inter_tensor` result yet; `compare.py` skips missing
results.
