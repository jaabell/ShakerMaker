# Writers & Persistence

## What this is

How ShakerMaker gets results out of memory and onto disk. There are three
independent paths, each for a distinct purpose:

1. **`StationListWriter` (base interface) + its two implementations** — the
   normal path, wired into the engine (`run()`, `run_fast()`,
   `run_nearest()`) via the `writer=` kwarg. Writes **all** stations of a
   `StationList` into a single HDF5 file, component by component (velocity,
   displacement, acceleration).
   - `HDF5StationListWriter` → generic `.h5`/`.hdf5`, for any `StationList`.
   - `DRMHDF5StationListWriter` → `.h5drm`, specialized for DRM receivers
     (`DRMBox`, `SurfaceGrid`, `PointCloudDRMReceiver`), consumable by
     OpenSees' `H5DRMLoadPattern`.
2. **`ShakerMaker.export_drm_geometry(filename)`** — exports **only**
   geometry for a DRM receiver, without running the FK engine. For quickly
   inspecting the mesh in STKO before spending real compute.
3. **`Station.save(npzfilename)` / `Station.load(npzfilename)`** — native
   serialization of a **single** `Station` (not a full `StationList`) to
   `.npz`, with no h5py dependency. For quick debugging or comparing one
   station in isolation.

## Source of truth

- `shakermaker/stationlistwriter.py` — abstract base class
  `StationListWriter`.
- `shakermaker/slw_extensions/hdf5stationlistwriter.py` —
  `HDF5StationListWriter`.
- `shakermaker/slw_extensions/drmhdf5stationlistwriter.py` —
  `DRMHDF5StationListWriter` (inherits from `HDF5StationListWriter`).
- `shakermaker/station.py:193-254` — `Station.save`/`Station.load`.
- `shakermaker/shakermaker.py:2206-2276` — `ShakerMaker.export_drm_geometry`.

## Full API reference

### `StationListWriter` (base, `shakermaker/stationlistwriter.py`)

```python
class StationListWriter(metaclass=abc.ABCMeta):
    def __init__(self, filename, transform_function=None)

    def write(self, station_list, num_samples):
        # orchestrates: initialize() -> write_metadata() -> write_station()
        # per station -> close(). This is the high-level method; in
        # practice the engine (run/run_fast/run_nearest) calls
        # initialize/write_metadata/write_station/close directly, not
        # write().

    # Abstract, each subclass implements these:
    def initialize(self, station_list, num_samples): ...
    def write_station(self, station, index): ...
    def write_metadata(self, metadata): ...
    def close(self): ...

    # properties: filename, transform_function (settable)
```

Wired into the engine by passing an already-constructed instance to the
`writer=` kwarg of `ShakerMaker.run()`, `run_fast()`, or `run_nearest()`.
The engine decides internally when to call
`initialize`/`write_metadata`/`write_station`/`close` — `write()` itself is
never called manually in the normal flow.

### `HDF5StationListWriter` (`shakermaker/slw_extensions/hdf5stationlistwriter.py`)

```python
class HDF5StationListWriter(StationListWriter):
    def __init__(self, filename)

    def initialize(self, station_list, num_samples,
                    tmin=None, tmax=None, dt=None, writer_mode='legacy')
    def write_metadata(self, metadata)
    def write_station(self, station, index)
    def close(self)
```

**Two modes, identical on-disk schema, different write timing**:

| | `writer_mode='legacy'` (default) | `writer_mode='progressive'` |
|---|---|---|
| When each station is written | Accumulated in an in-RAM dict; everything is interpolated/integrated/written only at `close()` | Immediately on `write_station()`, followed by `h5file.flush()` |
| RAM | O(nstations) — grows with station count | O(1) per station — critical for DRM with thousands of stations |
| Extra `initialize()` args required | None (uses `num_samples`) | `tmin`, `tmax`, `dt` — missing any raises an explicit `ValueError` |
| Crash resilience | If the job dies before `close()`, everything is lost | Each already-written station survives a later crash |
| Final time grid | `t_final = arange(tstart_real, tend_real + dt/2, dt)` — built from what actually arrived | `t_final = arange(tmin, tmax, dt)` — fixed, defined up front |

**On-disk schema (identical in both modes)**:

```
/Data/xyz              (nstations, 3)   float64   -- coordinates [x,y,z] km
/Data/internal         (nstations,)     bool       -- DRM interior/exterior flag
/Data/data_location    (nstations,)     int32      -- arange(nstations)*3, row offset
/Data/velocity         (3*nstations, nsamples) float64   -- E,N,Z rows per station
/Data/acceleration     (3*nstations, nsamples) float64
/Data/displacement     (3*nstations, nsamples) float64
/Metadata/dt           scalar
/Metadata/tstart       scalar
/Metadata/tend         scalar
/Metadata/<key>        any other field from the StationList/FaultSource metadata
```

Row layout for station index `i`: `row = 3*i` → `row`=E, `row+1`=N,
`row+2`=Z (vertical, positive **down**). This matches
`Data/data_location[i] == 3*i`. Note that `xyz` is (North, East, depth) while
the motion rows are (East, North, down) — on purpose; see
`12_coordinates_and_conventions.md`.

`acceleration` and `displacement` are derived from `velocity` inside the
writer itself (not by the engine): backward finite difference for
acceleration (`a[1:] = (v[1:]-v[:-1])/dt`, `a[0]=0`), cumulative
trapezoidal integration (`scipy.integrate.cumulative_trapezoid`,
`initial=0`) for displacement. Any `NaN` in the raw velocity propagates
automatically to both.

### `DRMHDF5StationListWriter` (`shakermaker/slw_extensions/drmhdf5stationlistwriter.py`)

Inherits from `HDF5StationListWriter`, same legacy/progressive pattern,
but:

```python
class DRMHDF5StationListWriter(HDF5StationListWriter):
    def __init__(self, filename)

    def initialize(self, station_list, num_samples,
                    tmin=None, tmax=None, dt=None, writer_mode='progressive')
    def write_metadata(self, metadata)
    def write_station(self, station, index)
    def close(self)
```

- `station_list` **must be** `DRMBox`, `SurfaceGrid`, or
  `PointCloudDRMReceiver` — otherwise an explicit `AssertionError` in
  `initialize()`.
- `writer_mode` defaults to `'progressive'` here (unlike the generic
  writer, whose default is `'legacy'`) — reflecting that the typical DRM
  use case involves large meshes (8000+ stations), where `'legacy'` would
  be RAM-prohibitive.
- `self.nstations = station_list.nstations - 1` — **excludes the QA
  station** (always the last in the list) from the `/DRM_Data` count.
- The QA station is identified by
  `station.metadata.get("name") == "QA"` — any DRM receiver (`DRMBox`,
  `SurfaceGrid`, `PointCloudDRMReceiver`) must automatically append that
  station with exactly that name as the last one in its list for the
  writer to route it correctly.

**On-disk schema**:

```
/DRM_Data/xyz              (nstations-1, 3)   float64
/DRM_Data/internal         (nstations-1,)     bool
/DRM_Data/data_location    (nstations-1,)     int32
/DRM_Data/velocity         (3*(nstations-1), nsamples) float64
/DRM_Data/acceleration     (3*(nstations-1), nsamples) float64
/DRM_Data/displacement     (3*(nstations-1), nsamples) float64

/DRM_QA_Data/xyz              (1, 3)   float64
/DRM_QA_Data/velocity         (3, nsamples) float64
/DRM_QA_Data/acceleration     (3, nsamples) float64
/DRM_QA_Data/displacement     (3, nsamples) float64

/DRM_Metadata/dt, tstart, tend, nt, writer_mode, created_by, program_used, created_on, <model metadata>
```

**What the `DRM_Data` / `DRM_QA_Data` split is for**: `DRM_Data` is what
OpenSees' `H5DRMLoadPattern` actually consumes to excite the boundary of an
FE model (Domain Reduction Method, Bielak). `DRM_QA_Data` is a single
reference station at the box center, meant for **visual verification** —
compared against a direct calculation at that same point (see
`08_drm_workflow.md`) to confirm the DRM synthesis didn't introduce error.
It isn't required for OpenSees to run; it's a quality-control aid.

`write_metadata` in this class also automatically adds `created_by`,
`program_used` (ShakerMaker version), and `created_on` (timestamp) to the
`/DRM_Metadata` group — no need to pass them yourself.

### `ShakerMaker.export_drm_geometry(filename="drm_geometry.h5drm")`

```python
model.export_drm_geometry("drm_geometry.h5drm")
```

- **Does not run the FK engine.** Writes only geometry — for that reason it
  requires the model's receiver to be `DRMBox`, `SurfaceGrid`, or
  `PointCloudDRMReceiver`; otherwise `TypeError`.
- Writes the same group schema (`DRM_Data`, `DRM_QA_Data`, plus
  `DRM_Metadata` with only `dt=0.0005`, `tstart=0.0`) as the real writer,
  but with **synthetic data**: a 2-sample linear ramp `[0.0, 10.0]`
  repeated for every component/station — not real physics. Only useful for
  reviewing mesh/geometry (e.g. in STKO) before investing real compute.
- Only rank 0 writes under MPI (`if rank != 0: return filename`).

### `Station.save(npzfilename)` / `Station.load(npzfilename)`

```python
station.save("station_results.npz")     # after a run
s2 = Station()
s2.load("station_results.npz")          # reconstructs the SAME station
```

- Serializes a **single** complete `Station`: position (`_x`), metadata,
  internal flags, and — if it already has a response (`_initialized`) —
  also `_z, _e, _n, _t, _dt, _tmin, _tmax`. Uses
  `np.savez`/`np.load(allow_pickle=True)` — no h5py dependency.
- No equivalent for a full `StationList` — deliberately single-station.
  For several stations at once, use `HDF5StationListWriter`.
- Useful for: saving the result of ONE station for later comparison
  without re-running the engine (a pattern used in several validation
  notebooks — see `examples/12_validation/`), or for quick debugging
  without depending on h5py.

## Minimal working example

### Generic writer, progressive mode (recommended for production)

```python
from shakermaker.shakermaker import ShakerMaker
from shakermaker.slw_extensions import HDF5StationListWriter

writer = HDF5StationListWriter("results.h5")
model = ShakerMaker(crust, fault, stations)
model.run(dt=0.05, nfft=4096, tb=1000, dk=0.3, tmin=0.0, tmax=100,
          writer=writer, writer_mode='progressive')
# -> results.h5 with /Data/{xyz,internal,data_location,velocity,acceleration,displacement}
#    and /Metadata/{dt,tstart,tend,...}
```

### DRM writer, inside the OP pipeline (`run_nearest`)

```python
from shakermaker.slw_extensions import DRMHDF5StationListWriter

writer = DRMHDF5StationListWriter("motions.h5drm")
model.run_nearest(stage='all', h5_database_name='gf_db.h5',
                   dt=0.005, nfft=4096, dk=0.1, tb=500, tmax=60,
                   writer=writer, writer_mode='progressive')
# -> motions.h5drm ready for OpenSees' H5DRMLoadPattern
```

### DRM geometry without running the engine (quick review)

```python
model.export_drm_geometry("drm_geometry.h5drm")   # instant, no FK
```

### Saving/reloading a single station

```python
station.save("sta01.npz")
# ... later, another script ...
from shakermaker.station import Station
s = Station()
s.load("sta01.npz")
z, e, n, t = s.get_response()
```

## Known gotchas

- **`writer_mode` default differs between the two classes**:
  `HDF5StationListWriter` defaults to `'legacy'`; `DRMHDF5StationListWriter`
  defaults to `'progressive'`. If the kwarg is omitted, behavior is NOT the
  same between the two.
- **Progressive mode requires explicit `tmin`/`tmax`/`dt`** (passed through
  by the engine into `initialize()`) — if the engine doesn't have them
  (e.g. mis-assembled combinations built by hand outside the normal flow),
  an immediate `ValueError` with an explicit message.
- **The QA station depends on the receiver naming it exactly `"QA"`** — if
  you build a custom `StationList`/receiver by hand for use with
  `DRMHDF5StationListWriter`, you must replicate that convention or the
  last station won't be routed to `/DRM_QA_Data`.
- **`DRMHDF5StationListWriter` with a receiver that isn't a real DRM box**
  (see `examples/14_SFSI/Surface/surface_SSFI.py`, reviewed in this
  session): using `SurfaceGrid(mode='plane')` (a single plane, not a closed
  box) together with `DRMHDF5StationListWriter` is a valid pattern that
  appears in the repo, but it **does not represent a real physical DRM
  boundary** — it's sometimes used only to get the `.h5drm` container
  format because an external tool (`ShakerMakerResults`) knows how to read
  it. In principle this case should be a plain `.h5` file
  (`HDF5StationListWriter`), not a `.h5drm`. If you see this pattern in a
  script, it isn't an error — it's a documented, temporary compatibility
  decision with an external tool.
- **`acceleration`/`displacement` do not come from the FK engine** — they
  are derived inside the writer from the velocity already interpolated
  onto the final grid. Any `NaN` in the raw velocity (e.g. the zero-epicentral-distance NaN, fixed in `subfk.f` by commit `2a83ca6`)
  propagates automatically to both.
- **`Station.save`/`load` is NOT interchangeable with the HDF5 writers** —
  it's a different format (`.npz`, not HDF5), meant for a single station,
  not a full `StationList`.

## Combines with

- `04_receivers.md` — which receiver types are valid for each writer
  (`HDF5StationListWriter` accepts any `StationList`;
  `DRMHDF5StationListWriter` requires `DRMBox`/`SurfaceGrid`/
  `PointCloudDRMReceiver`).
- `06_engine_run_modes.md` — how `writer=` is passed to
  `run()`/`run_fast()`/`run_nearest()`, and which writer mode fits which
  problem size.
- `08_drm_workflow.md` — the full DRM workflow, including the QA-vs-direct
  comparison.
- `RECIPES.md` — full combined recipes (station + nearest_method,
  DRM + nearest_method, etc.) that use these writers end to end.

## See also

- `examples/07_writers/` — runnable examples of both writers and both
  modes (`drm_writer.py`, `hdf5_writer.py`, `explore_h5_output.py`,
  `save_load_station.py`), already documented in detail in
  `examples/EXAMPLES_REFERENCE.md`.
- `examples/08_drm/notebooks/drm.ipynb` — `export_drm_geometry` example
  with a 3D visualization of the result.
