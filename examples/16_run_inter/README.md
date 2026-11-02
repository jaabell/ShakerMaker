# 16 - DRM box: reused vs interpolated Green's functions

`drm_coherence.py` drives a small DRM box (2.5 m elements, 531 nodes) with a
2 x 2 patch of point sources at about 2 km depth in a 4-layer crust (200 m at
Vs 0.75 km/s, 800 m at 1.57 km/s, 14.5 km at 3.14 km/s, half-space) and
compares three engines:

| Run | Engine | Green's functions |
|---|---|---|
| reference | `run_nearest`, tolerances 1 cm, one wavenumber step per pair of depths | one per pair |
| reuse | `run_nearest`, tolerances 40 m / 5 m / 200 m | one per slot |
| inter | `run_inter(scheme='tensor')`, 10 Hz band | interpolated in distance |

The metric is what the DRM uses: the difference between every interior node
and its nearest exterior node, compared with the reference, per band
(median relative error).

| Band | reuse | inter |
|---|---|---|
| 0.1-2.5 Hz | 48.6 % | 0.44 % |
| 2.5-5 Hz | 39.9 % | 0.14 % |
| 5-7.5 Hz | 34.7 % | 0.10 % |
| 7.5-10 Hz | 49.8 % | 0.20 % |

```bash
python drm_coherence.py              # ~10 min on one process (the reference dominates)
mpirun -n 4 python drm_coherence.py     # ~3 min
```

Writes `run_inter_drm_coherence.png` and prints PASS when `run_inter` stays
below 5 % and reuse above it. `run_inter` alone takes a few seconds; the
method is described in `docs/web/background/gf_interpolation.md` and its use
in `docs/web/guides/run_inter.md`.
