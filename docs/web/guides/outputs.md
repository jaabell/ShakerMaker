# Outputs & writers

A run produces three-component motion at every receiver. Where that motion
goes depends on how you ask for it — there are four routes, from a single
in-memory array to a coupling-ready HDF5 file an external solver reads.

| Route | What you get | Best for |
|---|---|---|
| **In memory** | NumPy arrays on each `Station` | a handful of stations, interactive work |
| **`.npz` archive** | one file per station | save a run, reload and plot later |
| **HDF5** (`HDF5StationListWriter`) | one file, all stations | many stations, post-processing |
| **H5DRM** (`DRMHDF5StationListWriter`) | DRM-layout HDF5 | OpenSees `H5DRMLoadPattern`, SW4 coupling |

---

## In-memory results

After `run()`, each `Station` holds its three velocity components and the time
vector:

```python
z, e, n, t = sta.get_response()      # NumPy arrays, length nfft (× smth)
```

The components are **Z (vertical), E (East), N (North)**. The natural FK output
is **velocity**, and its units follow the units in which the STF source was
defined — the Green's functions are scaled by the source, so the output carries
whatever moment/amplitude units you used for the STF. Integrate once for
displacement, differentiate once for acceleration (both built into
[plotting](plotting.md), and precomputed by the HDF5 writers below).

In a writer-driven run you can free a station's arrays right after it is written
with `sta.clear_response()` — that is what keeps memory flat on large DRM jobs.

---

## Per-station NumPy archive (`.npz`)

The simplest persistent format, one file per station:

```python
sta.save("STA01.npz")
sta2 = Station(); sta2.load("STA01.npz")
```

The archive stores the location `_x`, the `_metadata` dict, the `_internal`
flag, and — once the station has a response — the three components `_z, _e, _n`,
the time vector `_t`, and `_dt, _tmin, _tmax`. Good for one-off explorations,
unit tests, and "run once, plot many times" (see
[`examples/07_writers/save_load_station.py`](../examples/index.md#07-writers-io)).

---

## Aggregate HDF5 (`slw_extensions`)

For more than a few stations, pass a `writer=` to `run()` (or `run_fast()`) so
results stream to a **single file** as they are computed:

```python
from shakermaker.slw_extensions import HDF5StationListWriter

writer = HDF5StationListWriter("motions.h5")
model.run(..., writer=writer, writer_mode="progressive")
```

There are two writers, sharing the same file philosophy:

| Writer | File | Use case |
|---|---|---|
| `HDF5StationListWriter` | generic HDF5 | a station-list archive for your own post-processing |
| `DRMHDF5StationListWriter` | HDF5 in the **DRM layout** | DRM boundary motions for OpenSees / SW4 |

### What ends up in the file

`HDF5StationListWriter` writes two top-level groups:

```
/Data
    xyz            (nstations, 3)        receiver coordinates [x, y, z] km
    internal       (nstations,)  bool    interior vs boundary (DRM)
    data_location  (nstations,)  int32   row offset of each station = 3·index
    velocity       (3·nstations, nt)     the recorded motion
    acceleration   (3·nstations, nt)     d/dt of velocity   (precomputed)
    displacement   (3·nstations, nt)     ∫ velocity dt      (precomputed)
/Metadata
    dt, tstart, tend                     the output time grid
    …                                    any metadata you passed to the writer
```

Each station occupies **three consecutive rows** in the signal datasets,
ordered **E, N, Z** — station `i` starts at row `3·i` (that is exactly what
`data_location[i]` records). Velocity is computed by the FK engine;
acceleration and displacement are differentiated/integrated **at write time**,
so the file carries all three ready to use.

`DRMHDF5StationListWriter` uses the same idea with DRM-specific groups:

```
/DRM_Data        xyz, internal, data_location, velocity, displacement, acceleration
/DRM_QA_Data     the single QA control station (xyz + signals)
/DRM_Metadata    dt, tstart, tend, the box geometry, …
```

The **QA station** (the one whose metadata `name == "QA"`, added automatically
by `DRMBox` / `PointCloudDRMReceiver` at the box centre) is split off into
`/DRM_QA_Data`; every boundary node goes to `/DRM_Data`. This is the file
OpenSees `H5DRMLoadPattern` reads directly — see the [DRM guide](drm.md).

To peek inside any `.h5` / `.h5drm` and list its groups and shapes, use
[`examples/07_writers/explore_h5_output.py`](../examples/index.md#07-writers-io).

### Writer modes: `legacy` vs `progressive`

Both writers accept `writer_mode=` and behave very differently in memory:

| | `legacy` (default) | `progressive` |
|---|---|---|
| When data is written | all at once in `close()` | each station, immediately |
| Memory | holds every response until the end | **O(1)** — one station at a time |
| Time grid | inferred from the responses | you give `tmin`, `tmax`, `dt` up front |
| Crash safety | lose everything if it dies | file flushed after every station |
| Use it for | small runs, legacy workflows | large DRM runs (thousands of nodes) |

```python
writer = DRMHDF5StationListWriter("motions.h5drm")
model.run(dt=0.025, nfft=2048, tb=1000, dk=0.1, tmax=30,
          writer=writer, writer_mode="progressive")
```

Progressive mode pre-allocates the datasets to `nt = len(arange(tmin, tmax, dt))`
samples, interpolates each station onto that grid, and flushes — which is why it
needs the time window in `initialize()`.

### Writing a custom format

Every writer subclasses `StationListWriter`, so a new format only has to
implement this contract:

| Member | Purpose |
|---|---|
| `initialize(station_list, num_samples, …)` | open the file, allocate datasets |
| `write_metadata(metadata)` | store run metadata |
| `write_station(station, index)` | write one station's response |
| `write(station_list, num_samples)` | write the whole list at once |
| `close()` | flush and close |

It also exposes `filename` and an optional `transform_function` (a coordinate
transform applied to receiver positions on write).

---

## SW4 export

ShakerMaker can hand sources and motions to the **SW4** finite-difference
solver, the other half of the coupling story:

| Method | Writes |
|---|---|
| `model.export_sw4(path=...)` | sources + motions packaged for SW4 |
| `model.export_sw4_topo(path=...)` | the same, with topography |

The reverse path — building an `.h5drm` *from* a finished SW4 run — is
[`examples/09_sw4_export/build_h5drm_from_sw4_case.py`](../examples/index.md#09-sw4-export).
Full details in the [SW4 export guide](sw4_export.md).

---

## Reference

[Plotting & visualisation](plotting.md) · [DRM guide](drm.md) ·
[ShakerMaker engine API](../api/shakermaker.md)
