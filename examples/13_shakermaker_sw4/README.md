# 13 · ShakerMaker vs SW4

Cross-validate the FK engine against the **SW4** finite-difference solver on the
*same* model: a 4-layer crust, 100 FFSP point sources, and 2 surface stations.

Both solvers were driven from one ShakerMaker export, so the source and geometry
are identical — any difference is purely method (FK semi-analytic vs SW4
finite-difference).

## Run

```bash
python shaker_vs_sw4.py          # heavy FK run; for real use launch under MPI:
mpiexec -n 8 python shaker_vs_sw4.py
```

It rebuilds the model from `data/model_summary.h5`, runs the FK engine, reads
the SW4 output (`data/sf0000*.txt`), band-passes SW4 with ObsPy (0.25–15 Hz),
and writes `compare_sf0000*.png` overlaying the two solutions.

## Files

| Path | What |
|------|------|
| `shaker_vs_sw4.py` | the runnable comparison script |
| `data/model_summary.h5` | compact SW4 export package (crust + 100 sources + 2 stations) |
| `data/shakermaker2sw4.in` | SW4 input (station coordinates) |
| `data/sf0000{1,2}.txt` | SW4 velocity time series for the two stations |
| `notebooks/shaker_vs_sw4.ipynb` | the same workflow as a notebook, step by step |
| `notebooks/*.png` | the figures the notebook/script produce |

## Dependencies

`h5py`, `numpy`, `matplotlib`, and **`obspy`** (for the band-pass filter):
`pip install obspy`.
