# 17 - SW4 reference cases: ShakerMaker against finite differences

Six cases computed once with **SW4** (4th-order finite differences, elastic) and with ShakerMaker, stored in `data/`.
The smoke tests and the notebook only read them, so ShakerMaker is checked against an independent solution on every
run without running SW4.

| Case | Medium | Source |
|---|---|---|
| `loh1_p1`, `loh1_p2`, `loh1_f1` | SCEC LOH.1: 1 km at Vs 2 km/s over a half-space at 3.464 km/s | point at 2 km, point at 0.5 km, 16 x 16 = 256 subfaults |
| `layered_p1`, `layered_p2`, `layered_f1` | layered 1D crust: 200 m at Vs 0.75, 800 m at 1.57, 14.5 km at 3.14 km/s, half-space at 4.0 | point at 6 km, point at 0.5 km, 256 subfaults |

- Mechanism strike 30, dip 60, rake 45 (oblique, so no component sits on a nodal plane everywhere); unit-area
  Gaussian, sigma 0.06 s (LOH.1) and 0.1 s (layered crust); point sources 1e16 N m.
- Fault: 4 x 4 km, 250 m subfaults, top at 0.3 km, hypocentre at the centre of the base (3.76 km), uniform moment
  (3.5e16 N m in total), rupture at 0.8 Vs of the hypocentre layer; each subfault is the same Gaussian delayed by
  its rupture time, in both codes (`data/fault_<medium>.csv`).
- About 100 receivers per medium (`data/receivers_<medium>.csv`), all on SW4 grid nodes (SW4 records at the nearest
  node): a surface ring at 1, 4 and 8 km every 30 deg, three verticals down to 8 km (both sides of every interface,
  above and below every source), the epicentre and 24 nodes of a DRM box (two layers of its faces).
- SW4 (`sw4/*.in`, run logs in `sw4/logs/`): LOH.1 24 x 24 x 12 km, h 100 m refined to 50 m in the first km
  (11.3 M points, 12 s, 8-10 min on one 16-rank node); layered crust 24 x 24 x 20 km, h 100 / 50 / 25 m
  (26.6 M points, 15 s, 46-60 min). Elastic: no attenuation in SW4, Q = 1000 in ShakerMaker.

## Files

| File | What it does | Time (one process) |
|---|---|---|
| `smoke_vs_sw4.py` | LOH.1 point source, 10 receivers, `run()` against the stored SW4 in 0.1-2 Hz, absolute amplitude | ~6 s |
| `smoke_engines.py` | small DRM box, `nearest`, `run_inter` grid and tensor against the exact reference: traces and neighbour differences | ~21 s |
| `smoke_source_depth.py` | receivers at and away from the source depth against SW4; the one at the source depth is a known issue | ~3 s |
| `notebooks/sw4_reference.ipynb` | reads `data/`: model, traces, agreement per band, engines against the exact reference, findings | ~8 s |
| `make_sw4_inputs.py` | writes the decks and the tables (`--write`; SKIP otherwise) | — |
| `run_shakermaker.py` | regenerates `data/sm/<case>_<engine>.npz` (`<case> <engine>`, MPI; SKIP otherwise) | 10 s-30 min per engine on a 32-core node |

`data/sw4/<case>.npz`: `t`, `names`, `vel` (receivers, 3, samples) in m/s, components north, east, down.
`data/sm/<case>_<engine>.npz`: `t`, `names`, `zen` (receivers, 3, samples) in ShakerMaker units, engines `exact`,
`nearest` and `inter_tensor` (error band `f_err` 2 Hz, `freq_cut` 2.5 Hz).

## Conventions

SW4 records velocity; with `Gaussian(derivative=False)` the ShakerMaker output is its time integral, so the SW4
velocity is integrated (cumulative trapezoid; no baseline correction). One ShakerMaker moment unit is 1e15 N m in
SW4: amplitudes are compared without any fitted factor.

## Result

| | Median correlation with SW4 | Peak ratio |
|---|---|---|
| LOH.1, three sources, three bands (0.1-0.5, 0.5-1, 1-2 Hz) | 0.9998-1.0000 | 0.99-1.01 |
| layered crust, deep source | 0.997-1.000 | 0.99-1.00 |
| layered crust, shallow source and fault, below 1 Hz | 0.978-0.999 | 0.97-0.99 |
| layered crust, shallow source and fault, 1-2 Hz | 0.91-0.92 (lag up to 140 ms at 8 km) | 0.98 |

| Engine against the exact reference (low-pass 2 Hz) | Traces | DRM layer difference |
|---|---|---|
| `nearest` | 0-3 % | 28-77 % |
| `run_inter(scheme='tensor')` | 0.2-1.7 % | 0.5-3.3 % |

Two open findings, shown in the notebook:

1. **Receivers at or a few metres from a source depth** depart from SW4 with every engine (correlation 0.2-0.7
   at zero separation), while the same vertical 0.9 km above and 1 km below matches it. In the fault cases a row
   of subfaults 6-8 m from a receiver depth (receivers at 0.4 and 3.0 km) does the same: those few receivers
   differ from the exact engine by tens of percent or more with every engine, and Stage 1 of `run_inter` warns
   about cells it cannot resolve there.
2. **Shallow source under the thin soft layer, 1-2 Hz**: ShakerMaker lags SW4 by a time that grows with distance,
   attributed to the effective interface of SW4, half a cell deeper than the nominal one.
