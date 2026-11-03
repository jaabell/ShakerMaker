# 13 · ShakerMaker vs SW4

Cross-validate the FK engine against the **SW4** finite-difference solver on the
*same* model: a 4-layer crust, 100 FFSP point sources, and 2 surface stations.

Both solvers were driven from one ShakerMaker export, so the source and geometry
are identical — any difference is purely method (FK semi-analytic vs SW4
finite-difference).

## Run

```bash
mpiexec -n 16 python shaker_vs_sw4.py     # ~30 min; ~80 min with 6 processes; ~6.5 h on one
```

The FK run has 200 core calls (100 sources x 2 stations, `nfft=32768`), about 2 min each; under MPI rank 0
coordinates and the other ranks compute. The script rebuilds the model from `data/model_summary.h5`, runs the FK
engine, reads the SW4 output (`data/sf0000*.txt`), band-passes both solutions with ObsPy (0.25-15 Hz), checks the
correlation and the peak ratio with SW4 on every component (it fails outside the limits in its header) and writes
to `_output/` (ignored by git): `compare_sf0000*.png` and `shakermaker_sf0000*.npz` (both station responses).
`run_all_smoke.py --full` runs it under MPI (`--np`, default 6).

The notebook reads the responses saved by the script, so run the script first.

## Files

| Path | What |
|------|------|
| `shaker_vs_sw4.py` | the runnable comparison script |
| `data/model_summary.h5` | compact SW4 export package (crust + 100 sources + 2 stations) |
| `data/shakermaker2sw4.in` | SW4 input (station coordinates) |
| `data/sf0000{1,2}.txt` | SW4 velocity time series for the two stations |
| `notebooks/shaker_vs_sw4.ipynb` | the same workflow step by step, on the responses saved by the script |
| `notebooks/*.png` | the figures the notebook/script produce |

## Dependencies

`h5py`, `numpy`, `matplotlib`, and **`obspy`** (for the band-pass filter):
`pip install obspy`.
