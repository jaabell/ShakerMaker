# SW4 exporter

This subpackage bridges a ShakerMaker model to an SW4 run. It writes the
SW4 input file and the per-source slip-rate files, optionally drops a
local cartesian topography, and packs everything into a single HDF5
transport bundle next to a standalone unpacking script.

The two entry points are `ShakerMaker.export_sw4(...)` and
`ShakerMaker.export_sw4_topo(...)`. Both build a `SW4ExportConfig`,
hand it to `SW4Exporter`, and call `.write()`.

## Coordinate convention

ShakerMaker is the master frame. Every source, station, and topography
node is stored in ShakerMaker coordinates (georeferenced, kilometres).
SW4 only understands a local cartesian box whose origin sits at one of
its corners, so the exporter:

1. Resolves the SW4 box (extent + origin) from the union of sources,
   stations and topography.
2. Builds a `CoordinateTransform` that is a pure translation between the
   two frames. No rotation is applied.
3. Writes the SW4 `.in` file in SW4 local metres (what SW4 needs to run).
4. Stores the offset in both directions in the HDF5 package, so any
   downstream tool can come back to ShakerMaker without guessing.

In code:

```
P_sw4_m   = P_shakermaker_m - domain_origin_m
P_shakermaker_m = P_sw4_m + domain_origin_m
```

The package stores `domain_origin_m` under two names for convenience:

| Dataset                                | Meaning                                          |
|----------------------------------------|--------------------------------------------------|
| `/coordinates/sw4_origin_in_shakermaker_m` | `+domain_origin_m`. Add this to an SW4 point. |
| `/coordinates/shakermaker_to_sw4_offset_m` | `-domain_origin_m`. Subtract from an SW4 point or add to a ShakerMaker point to get SW4 local. |

The kilometre versions of both arrays are also written.

Topography files in the SW4 convention use `x = East, y = North`.
ShakerMaker uses `x = North, y = East`. `rotate_topography_to_shakermaker`
swaps those two columns on read; the cartesian grid is then resorted by
`rebuild_cartesian_topography` so the rest of the pipeline can assume
row-major order.

## What `.write()` produces

Under `config.path`:

```
shakermakerexports/
    sw4_package.h5            <- single HDF5 transport bundle
    unpack_sw4_package.py     <- standalone unpacker, no ShakerMaker dependency
sw4/
    shakermaker2sw4.in        <- only written via unpack_sw4_package_h5
    sources/source_NNNNNN.txt
    sources/sources_summary.csv
    topo/<name>_local.txt     <- only when a topography is configured
```

`SW4Exporter.write()` itself does *not* touch `sw4/` -- it embeds every
text payload inside `sw4_package.h5` and writes the standalone unpacker.
Calling `unpack_sw4_package_h5(package, output_dir)` (or running the
standalone script) recreates the `sw4/` tree on disk.

## HDF5 package layout

Top-level groups in `sw4_package.h5`:

| Group           | Contents                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `manifest`      | SW4 input path, fileio path, list of every relpath embedded in `/files`. |
| `config`        | Every knob from `SW4ExportConfig` after `__post_init__`.                 |
| `coordinates`   | ShakerMaker <-> SW4 offset, in metres and kilometres, both directions.   |
| `crust`         | Layer thickness, Vp, Vs, rho, Qp, Qs and cumulative top-depth (km).      |
| `stations`      | ShakerMaker receiver list, georef km.                                    |
| `sw4_input`     | Verbatim `.in` text + its relpath.                                       |
| `sources`       | Per-source scalar fields + a flat slip-rate buffer indexed by offsets.   |
| `topography`    | `present` flag; when present, nx/ny, ShakerMaker nodes and original bounds. |
| `receivers`     | One row per SW4 `rec` line, with georef km coordinates and kind tag.     |
| `drm_template`  | The non-QA receiver subset shaped for an h5drm build, plus the QA point. |
| `files`         | Every text file the SW4 case needs, gzipped, indexed by `relpath`.       |

Top-level attributes:

| Attribute          | Value                                  |
|--------------------|----------------------------------------|
| `package_version`  | `"1.0"`                                |
| `generator`        | `"ShakerMaker.sw4_exporter"`           |
| `purpose`          | `"transport_unpack_to_sw4_files"` (used by `_find_package`). |

## Receiver families

Every receiver row is tagged with a `kind` so downstream tools (h5drm
builder, geometry viewer) can treat each family on its own terms:

| `kind`                | Source                                            |
|-----------------------|---------------------------------------------------|
| `shakermaker`         | ShakerMaker stations at their true z.             |
| `shakermaker_surface` | Same stations forced to `depth=0`.                |
| `topography_surface`  | One `depth=0` receiver per topography node.       |
| `topography_to_z0`    | Vertical fill between topography and z=0.         |
| `sw4_domain`          | Regular grid spanning the SW4 box (DRM-style).    |

DRM-aware receiver lists (`DRMBox`, `SurfaceGrid`, `PointCloudDRMReceiver`)
carry a trailing QA station that is marked with `is_qa=True` in the
package. When the receiver list is a plain `StationList`, no QA is
defined and the centre of the receiver bounding box is used as a
fallback in `/drm_template/qa_xyz_km`.

## Coming back: building an h5drm

After running SW4, take the per-station `.txt` files and the same
`sw4_package.h5` and feed them to
`examples/sw4_2_h5drm.py::build_h5drm_from_sw4_case`. That script reads
the package, walks the receivers in writing order, attaches the
component signals, and writes a `motions.h5drm` ready for STKO.
`move_2_shakermaker_coor=True` shifts coordinates back into the
ShakerMaker georef using `/coordinates/sw4_origin_in_shakermaker_km`.

## Module map

| File              | Role                                                            |
|-------------------|-----------------------------------------------------------------|
| `exporter.py`     | `SW4Exporter`, the orchestrator. Decides the SW4 box, drives every other module. |
| `config.py`       | `SW4ExportConfig` dataclass.                                    |
| `coordinates.py`  | `CoordinateTransform` (ShakerMaker <-> SW4, translation only).  |
| `grid.py`         | SW4 `grid` line builder.                                        |
| `materials.py`    | SW4 `block` lines from a `CrustModel`.                          |
| `sources.py`      | Source row flattening, slip-rate file text, `source` lines.     |
| `receivers.py`    | One builder per receiver family. Tolerance constant `_Z_TOL_M`. |
| `topography.py`   | Cartesian topography I/O and SW4-grid extension.                |
| `input_writer.py` | Stitches the `.in` text from already-formatted pieces.          |
| `package_h5.py`   | HDF5 transport package: writer, unpacker, embedded script.      |
| `geometry_plot.py`| PyVista viewer (text or HDF5 input).                            |

## Smoke test

`examples/sw4_export_smoke_test.py` exercises the full path without the
FK core: build a small model, export, open the HDF5 package, assert the
expected groups and shapes, unpack into a tmp dir, and check the round
trip of the coordinate offset. Run it from the repo root with the
project venv whenever this subpackage is touched.
