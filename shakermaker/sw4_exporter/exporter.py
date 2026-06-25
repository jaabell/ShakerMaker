"""SW4 exporter orchestrator.

:class:`SW4Exporter` drives the full export pipeline for a ShakerMaker
model. The flow inside :meth:`SW4Exporter.write` is:

1. Read and condition the topography (optional). The file is rotated into
   the ShakerMaker (x = North, y = East) convention and resorted into a
   regular cartesian grid.
2. Collect every active point (sources, stations, topography) and decide
   the SW4 box extents and origin. The origin lives at one corner of the
   box, in ShakerMaker georef.
3. Build a :class:`CoordinateTransform` from the box origin. Every later
   step uses it to map ShakerMaker coordinates into SW4 local metres, and
   the inverse to bring SW4-side data back into the georef on demand.
4. Emit the SW4 ``.in`` file (grid, materials, sources, receivers, optional
   topography) along with the per-source slip-rate ``.txt`` files and the
   local topography file.
5. Pack everything into a single HDF5 transport bundle next to the SW4
   case and copy the standalone unpack script next to it.

The package only depends on the model's attributes, not on the FK core, so
it can run on a machine without the Fortran extension installed.
"""

from pathlib import Path
import csv
import io
import warnings
import numpy as np

from shakermaker.sl_extensions import DRMBox, SurfaceGrid, PointCloudDRMReceiver

from .config import SW4ExportConfig
from .coordinates import CoordinateTransform
from .geometry_plot import plot_sw4_geometry
from .grid import grid_line
from .input_writer import sw4_input_text
from .materials import deepest_interface, material_lines
from .package_h5 import write_sw4_package_h5, write_unpack_script
from .receivers import (
    domain_receiver_lines,
    model_receiver_lines,
    model_receiver_surface_lines,
    topography_receiver_lines,
    topography_z0_receiver_lines,
    _values_between_topography_and_z0,
)
from .sources import source_file_text, source_rows, sw4_source_lines
from .topography import (
    SEPARATOR,
    print_active_geometry_bounds,
    print_domain_diagnostics,
    print_topography_diagnostics,
    warn_active_geometry_supergrid,
    read_cartesian_topography,
    rebuild_cartesian_topography,
    rotate_topography_to_shakermaker,
    extend_topography_to_domain,
    cartesian_topography_text,
)


# Tolerance, in metres, used to compare floating-point coordinates inside
# the exporter. Same role as :data:`shakermaker.sw4_exporter.receivers._Z_TOL_M`
# but kept local so the two modules can drift independently if needed.
_COORD_TOL_M = 1.0e-9


def _snap_up_to_h(value, h):
    """Round ``value`` up to the next multiple of ``h``.

    A small tolerance keeps values that are already a multiple of ``h`` from
    being bumped to the next one by float64 noise.
    """
    h = float(h)
    return float(np.ceil(float(value) / h - _COORD_TOL_M) * h)


class SW4Exporter:
    """Top-level driver for the SW4 export.

    Inputs
    ------
    model : ShakerMaker
        Source/receiver/crust container. Only ``_source``, ``_receivers``
        and ``_crust`` are read.
    config : SW4ExportConfig
        Knob bag (see :class:`SW4ExportConfig`).

    Attributes
    ----------
    base_path : Path
        Absolute path of the export root (``config.path`` resolved).
    exports_path : Path
        ``<base_path>/shakermakerexports``. Holds the HDF5 package and the
        unpack script.
    sw4_path : Path
        ``<base_path>/sw4``. Holds the ``.in`` file and the per-source/topo
        subdirectories when the package is unpacked.
    sources_path, topo_path : Path
        Convenience aliases under ``sw4_path``.
    input_file : Path
        Where the SW4 ``.in`` lands once unpacked.
    package_h5 : Path
        Final transport HDF5 file.
    unpack_script : Path
        Standalone unpacker placed next to ``package_h5``.
    """

    def __init__(self, model, config: SW4ExportConfig):
        self.model = model
        self.config = config
        self.base_path = Path(config.path).resolve()
        self.exports_path = self.base_path / "shakermakerexports"
        self.sw4_path = self.base_path / "sw4"
        self.sources_path = self.sw4_path / "sources"
        self.topo_path = self.sw4_path / "topo"
        self.input_file = self.sw4_path / "shakermaker2sw4.in"
        self.package_h5 = self.exports_path / self.config.h5_export_name
        self.unpack_script = self.exports_path / "unpack_sw4_package.py"

    def write(self):
        """Run the full export and write the HDF5 transport package.

        The package contains every file the case needs to reproduce the SW4
        run (input file, per-source slip-rate files, local topography), plus
        a structured copy of the model metadata (crust, stations, sources,
        receivers, DRM template, coordinate offsets). The standalone unpack
        script is written next to it.

        After this call the model is decorated with two attributes so the
        rest of ShakerMaker can locate the produced files:

        - ``model.sw4_export_paths`` -- dict returned by :meth:`paths`.
        - ``model.sw4_export_config`` -- the same ``config`` passed in.

        Returns
        -------
        None
        """
        self.exports_path.mkdir(parents=True, exist_ok=True)
        stale_structure = self.exports_path / "PACKAGE_STRUCTURE.txt"
        if stale_structure.is_file():
            stale_structure.unlink()

        topo_line = None
        topo_points = None
        topo_points_local = None
        topo_points_sw4 = None
        topo_nx = topo_ny = None
        local_topo = None
        topo_nx_sw4 = topo_ny_sw4 = None
        topo_original_bounds = None

        # 1) Topography conditioning (optional).
        if self.config.topo_file is not None:
            topo_nx, topo_ny, topo_points = read_cartesian_topography(self.config.topo_file)
            topo_points = rotate_topography_to_shakermaker(topo_points)
            topo_nx, topo_ny, topo_points = rebuild_cartesian_topography(topo_points)

        # 2) Decide the SW4 box extents and origin from the model bounds.
        original_points = self._active_geometry_points_m(topo_points)
        x_domain, y_domain, z_domain, domain_origin = self._resolve_domain(original_points)
        transform = CoordinateTransform(domain_origin)
        self._store_domain_in_config(x_domain, y_domain, z_domain, transform)

        if topo_points is not None:
            topo_points_local = np.array([transform.from_original_m_to_sw4_m(p) for p in topo_points])
            topo_nx_sw4, topo_ny_sw4, topo_points_sw4 = extend_topography_to_domain(
                topo_nx, topo_ny, topo_points_local, x_domain, y_domain)
            topo_points_shaker = np.asarray(
                [transform.to_original_m(point) for point in topo_points_sw4],
                dtype=float,
            )
            local_topo = self.topo_path / f"{Path(self.config.topo_file).stem}_local{Path(self.config.topo_file).suffix}"
            print_topography_diagnostics(
                topo_points, topo_points_local, x_domain, y_domain, z_domain,
                h=self.config.h, topo_nx=topo_nx, topo_ny=topo_ny, topo_zmax=self.config.topo_zmax)
            topo_original_bounds = self._topography_bounds_array(topo_points)
            topo_line = self._topography_line(local_topo, topo_points_local)
            if self.config.topo_zmax is not None:
                topo_line += f" zmax={float(self.config.topo_zmax):.16g}"
        else:
            topo_points_shaker = None
            print_domain_diagnostics(x_domain, y_domain, z_domain, h=self.config.h)

        # 3) Sources: flatten model._source into tabular rows with both
        #    ShakerMaker and SW4 coordinates.
        rows = source_rows(self.model, transform)
        source_points = np.array([[row["x_sw4_m"], row["y_sw4_m"], row["z_sw4_m"]] for row in rows], dtype=float)

        # DRMBox/SurfaceGrid/PointCloud receivers carry a trailing QA station
        # that mirrors the rest of the array. Identify its index so the HDF5
        # package can mark it explicitly.
        has_qa = isinstance(self.model._receivers, (DRMBox, SurfaceGrid, PointCloudDRMReceiver))
        station_count = self.model._receivers.nstations
        n_drm_stations = station_count - 1 if has_qa else station_count
        qa_index = n_drm_stations if has_qa else -1

        # 4) Receivers: build the SW4 ``rec`` lines per family, in the order
        #    they should appear in the ``.in`` file.
        receiver_lines = []
        receiver_records = []
        rec_index = 1
        topo_xy_z0 = self._topography_xy_at_z0(topo_points_local)
        active_points = [source_points]

        if self.config.shakermaker_stations:
            lines = model_receiver_lines(
                self.model._receivers, transform, self.config.station_prefix,
                start_index=rec_index, topo_xy_z0=topo_xy_z0)
            if lines:
                receiver_lines.append("# ShakerMaker stations")
                receiver_lines += lines
                receiver_records += self._model_receiver_records(
                    transform, start_index=rec_index, topo_xy_z0=topo_xy_z0,
                    qa_index=qa_index)
                rec_index += len(lines)

        if self.config.shakermaker_stations_to_surface:
            surf_lines = model_receiver_surface_lines(
                self.model._receivers, transform, self.config.station_prefix,
                start_index=rec_index)
            if surf_lines:
                receiver_lines.append("# ShakerMaker stations Surface")
                receiver_lines += surf_lines
                receiver_records += self._model_receiver_surface_records(
                    transform, topo_points_local, start_index=rec_index,
                    qa_index=qa_index)
                rec_index += len(surf_lines)

        receiver_points = np.array(
            [transform.from_shakermaker_km_to_sw4_m(station.x) for station in self.model._receivers],
            dtype=float)
        active_points.append(receiver_points)

        active_all = np.vstack(active_points)
        print_active_geometry_bounds(active_all)
        warn_active_geometry_supergrid(
            active_all, x_domain, y_domain, z_domain,
            self.config.h, self.config.supergrid_gp)

        if topo_points_local is not None:
            lines = topography_receiver_lines(
                topo_points_local, start_index=rec_index, prefix=self.config.station_prefix)
            if lines:
                receiver_lines.append("# Topography surface stations (depth=0)")
                receiver_lines += lines
                receiver_records += self._topography_surface_records(
                    transform, topo_points_local, start_index=rec_index)
                rec_index += len(lines)

            if self.config.write_topography_z0_stations:
                lines = topography_z0_receiver_lines(
                    topo_points_local, h=self.config.h,
                    start_index=rec_index, prefix=self.config.station_prefix)
                if lines:
                    receiver_lines.append("# Between topography and z=0")
                    receiver_lines += lines
                    receiver_records += self._topography_z0_records(
                        transform, topo_points_local, start_index=rec_index)
                    rec_index += len(lines)

        if self.config.domain_sw4:
            domain_size = [
                self.config.domain_sw4_x if self.config.domain_sw4_x is not None else x_domain,
                self.config.domain_sw4_y if self.config.domain_sw4_y is not None else y_domain,
                self.config.domain_sw4_z if self.config.domain_sw4_z is not None else z_domain,
            ]
            domain_origin_xy = self._domain_sw4_origin_xy(domain_size, x_domain, y_domain)
            lines = domain_receiver_lines(
                self.config.h,
                domain_size,
                start_index=rec_index,
                prefix=self.config.station_prefix,
                topo_xy_z0=topo_xy_z0,
                origin_xy=domain_origin_xy)
            if lines:
                receiver_lines.append("# SW4 domain grid stations")
                receiver_lines += lines
                receiver_records += self._records_from_rec_lines(
                    transform, lines, kind="sw4_domain", start_model_index=-1)
                rec_index += len(lines)

        # 5) Assemble the ``.in`` text and the file payloads that travel
        #    with the HDF5 package.
        grid = grid_line(self.config.h, x_domain, y_domain, z_domain)
        materials = material_lines(
            self.model._crust,
            h=self.config.h,
            interface_blocks=self.config.interface_blocks,
            interface_block_delta=self.config.interface_block_delta,
        )
        source_lines = sw4_source_lines(rows, self.config.m0)
        input_text = sw4_input_text(
            grid,
            self.config.tmax,
            self.config.fileio_path,
            self.config.supergrid_gp,
            materials,
            source_lines,
            receiver_lines,
            topo_line,
        )

        paths = self.paths()
        file_payloads = {"sw4/shakermaker2sw4.in": input_text}
        for row in rows:
            file_payloads[f"sw4/sources/{Path(row['dfile']).name}"] = source_file_text(row)
        file_payloads["sw4/sources/sources_summary.csv"] = self._sources_summary_text(rows)
        topo_relpath = None
        if local_topo is not None:
            topo_relpath = f"sw4/topo/{local_topo.name}"
            file_payloads[topo_relpath] = cartesian_topography_text(topo_nx_sw4, topo_ny_sw4, topo_points_sw4)
        write_sw4_package_h5(
            paths["package_h5"],
            self.model,
            self.config,
            paths,
            rows,
            receiver_records,
            file_payloads,
            input_text,
            topography_relpath=topo_relpath,
            topography_shape=(topo_nx_sw4, topo_ny_sw4) if local_topo is not None else None,
            topography_points=topo_points_shaker,
            topography_original_bounds=topo_original_bounds,
        )
        write_unpack_script(paths["unpack_script"])

        self.model.sw4_export_paths = paths
        self.model.sw4_export_config = self.config

        print(SEPARATOR)
        print("SW4 export files")
        print(SEPARATOR)
        print(f"SW4 package   : {paths['package_h5']}")
        print(f"Unpack script : {paths['unpack_script']}")
        print(SEPARATOR)

        if self.config.plot_geometry:
            plot_sw4_geometry(paths["package_h5"], origin_m=transform.domain_origin_m)
        if self.config.plot_geometry_sw4:
            plot_sw4_geometry(paths["package_h5"])

    # -------------------------------------------------------------------
    # Domain decision and coordinate book-keeping
    # -------------------------------------------------------------------

    def _active_geometry_points_m(self, topo_points=None):
        """Gather every point that constrains the SW4 box.

        Inputs
        ------
        topo_points : ndarray, shape (N, 3), optional
            Topography nodes already in ShakerMaker metres.

        Returns
        -------
        ndarray, shape (M, 3)
            Sources and stations promoted to metres, plus topography when
            supplied. Used by :meth:`_resolve_domain`.

        Raises
        ------
        ValueError
            If no points end up in the list (empty model and no topography).
        """
        points = []
        points.extend(np.asarray(psource.x, dtype=float) * 1000.0 for psource in self.model._source)
        points.extend(np.asarray(station.x, dtype=float) * 1000.0 for station in self.model._receivers)
        if topo_points is not None:
            points.extend(np.asarray(topo_points, dtype=float))
        if not points:
            raise ValueError("Cannot build SW4 domain from empty geometry.")
        return np.asarray(points, dtype=float)

    def _resolve_domain(self, points):
        """Decide the SW4 box extents and origin from the active geometry.

        Each axis is sized so the active geometry clears the supergrid
        absorbing layer. An axis left ``None`` in the config is computed
        automatically: the horizontal extents grow by
        ``(supergrid_gp + supergrid_pad_gp) * h`` on each side, and the
        vertical extent reaches below the deepest source *and* the deepest
        material interface plus the bottom supergrid. An axis pinned in the
        config is validated against the same clearance and rejected with a
        descriptive error when it does not fit. Every returned extent is a
        multiple of ``h``.

        The origin centres the geometry in x and y and starts at z=0 (SW4
        free-surface convention; only the lateral walls and the bottom carry
        a supergrid, never the top).

        Inputs
        ------
        points : ndarray, shape (M, 3)
            Output of :meth:`_active_geometry_points_m`.

        Returns
        -------
        x_domain, y_domain, z_domain : float
            Box extents in metres, each a multiple of ``h``.
        origin : ndarray, shape (3,)
            SW4 origin expressed in ShakerMaker metres.
        """
        h = float(self.config.h)
        gp = int(self.config.supergrid_gp)
        pad_gp = int(self.config.supergrid_pad_gp)
        sponge = gp * h
        clearance = (gp + pad_gp) * h

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)

        x_domain = self._domain_length(0, mins, maxs, clearance, sponge)
        y_domain = self._domain_length(1, mins, maxs, clearance, sponge)
        z_domain = self._resolve_z_domain(float(maxs[2]), clearance, sponge)

        origin = np.array([
            center[0] - 0.5 * x_domain,
            center[1] - 0.5 * y_domain,
            0.0,
        ], dtype=float)
        return x_domain, y_domain, z_domain, origin

    def _domain_length(self, axis, mins, maxs, clearance, sponge):
        """Length of the box along one horizontal axis, supergrid-aware.

        When the config does not pin the axis, grow the geometry extent by
        ``clearance`` on each side and snap up to a multiple of ``h``. When it
        does, snap the value up to a multiple of ``h`` (warning if it changed)
        and require the geometry to clear the supergrid (``sponge`` per side),
        raising otherwise.

        Inputs
        ------
        axis : int
            0 for x, 1 for y.
        mins, maxs : ndarray, shape (3,)
            Component-wise min/max of the active geometry.
        clearance : float
            Auto padding per side (``(gp + pad_gp) * h``).
        sponge : float
            Hard supergrid width per side (``gp * h``).

        Returns
        -------
        float
            Length in metres, a multiple of ``h``.
        """
        h = float(self.config.h)
        name = "x" if axis == 0 else "y"
        configured = self.config.x_domain if axis == 0 else self.config.y_domain
        extent = float(maxs[axis] - mins[axis])
        if configured is None:
            return _snap_up_to_h(max(extent, h) + 2.0 * clearance, h)

        length = float(configured)
        snapped = _snap_up_to_h(length, h)
        if snapped > length + _COORD_TOL_M:
            warnings.warn(
                f"{name}_domain={length:g} is not a multiple of h={h:g}; "
                f"snapped up to {snapped:g}.",
                stacklevel=3,
            )
            length = snapped
        if extent + 2.0 * sponge > length + _COORD_TOL_M:
            raise ValueError(
                f"{name}_domain={length:g} m does not leave room for the "
                f"supergrid: the active geometry spans {extent:g} m and needs "
                f"at least {extent + 2.0 * sponge:g} m "
                f"({sponge:g} m clearance per side, supergrid_gp="
                f"{int(self.config.supergrid_gp)}, h={h:g}). "
                f"Pass {name}=None to size it automatically."
            )
        return length

    def _resolve_z_domain(self, deepest_source_m, clearance, sponge):
        """Vertical box extent, supergrid- and half-space-aware.

        The box must reach below both the deepest source and the deepest
        material interface, leaving the bottom supergrid clear. ``z=None``
        sizes it automatically (snapped up to a multiple of ``h``); a pinned
        value is validated and rejected when too shallow. The free surface at
        z=0 carries no supergrid, so only the bottom is padded.

        Inputs
        ------
        deepest_source_m : float
            Deepest active point in z (m).
        clearance : float
            Auto padding below the deepest feature (``(gp + pad_gp) * h``).
        sponge : float
            Hard bottom supergrid width (``gp * h``).

        Returns
        -------
        float
            z extent in metres, a multiple of ``h``.
        """
        h = float(self.config.h)
        deepest_feature = max(float(deepest_source_m), deepest_interface(self.model._crust))
        if self.config.z_domain is None:
            return _snap_up_to_h(deepest_feature + clearance, h)

        z_domain = float(self.config.z_domain)
        snapped = _snap_up_to_h(z_domain, h)
        if snapped > z_domain + _COORD_TOL_M:
            warnings.warn(
                f"z_domain={z_domain:g} is not a multiple of h={h:g}; "
                f"snapped up to {snapped:g}.",
                stacklevel=3,
            )
            z_domain = snapped
        z_min_required = deepest_feature + sponge
        if z_domain + _COORD_TOL_M < z_min_required:
            raise ValueError(
                f"z_domain={z_domain:g} m is too shallow: the deepest feature "
                f"(source/interface) is at {deepest_feature:g} m and the bottom "
                f"supergrid needs {sponge:g} m below it (>= {z_min_required:g} m "
                f"total, supergrid_gp={int(self.config.supergrid_gp)}, h={h:g}). "
                f"Pass z=None to size it automatically."
            )
        return z_domain

    def _store_domain_in_config(self, x_domain, y_domain, z_domain, transform):
        """Copy the resolved domain/origin back onto the config object.

        The config object is the single point of truth that the HDF5
        packager and downstream readers consult, so the resolved values
        have to land there.

        Inputs
        ------
        x_domain, y_domain, z_domain : float
        transform : CoordinateTransform

        Returns
        -------
        None
        """
        self.config.x_domain = float(x_domain)
        self.config.y_domain = float(y_domain)
        self.config.z_domain = float(z_domain)
        self.config.x_origin = float(transform.origin_m[0])
        self.config.y_origin = float(transform.origin_m[1])
        self.config.z_origin = float(transform.origin_m[2])

    # -------------------------------------------------------------------
    # Topography helpers
    # -------------------------------------------------------------------

    def _topography_line(self, local_topo, topo_points_local):
        """SW4 ``topography`` line plus a comment with the original bounds.

        Inputs
        ------
        local_topo : Path
            Where the local topography file will live, inside ``sw4/topo``.
        topo_points_local : ndarray, shape (N, 3)
            Topography nodes already in SW4 local metres.

        Returns
        -------
        str
            Two-line text: a ``# ShakerMaker topography_original_bounds`` row
            followed by the ``topography input=cartesian file=topo/...`` line.
        """
        xmin = float(topo_points_local[:, 0].min())
        xmax = float(topo_points_local[:, 0].max())
        ymin = float(topo_points_local[:, 1].min())
        ymax = float(topo_points_local[:, 1].max())
        bounds = (
            f"# ShakerMaker topography_original_bounds "
            f"xmin={xmin:.16g} xmax={xmax:.16g} ymin={ymin:.16g} ymax={ymax:.16g}"
        )
        line = f"topography input=cartesian file=topo/{local_topo.name}"
        return bounds + "\n" + line

    def _topography_bounds_array(self, topo_points_local):
        """``[xmin, xmax, ymin, ymax]`` of the topography, as ndarray.

        Stored verbatim in the HDF5 package for later round-trip checks.
        """
        return np.asarray([
            float(topo_points_local[:, 0].min()),
            float(topo_points_local[:, 0].max()),
            float(topo_points_local[:, 1].min()),
            float(topo_points_local[:, 1].max()),
        ], dtype=float)

    def _topography_xy_at_z0(self, topo_points_local):
        """Set of (x, y) topography nodes that sit on z=0 within tolerance.

        Used to skip duplicate receivers when a ShakerMaker station happens
        to coincide with a topography surface node. The tolerance is half a
        grid spacing -- generous enough to absorb the float64 noise that
        the topography pipeline accumulates while still avoiding accidental
        merges of two genuinely distinct grid nodes.

        Inputs
        ------
        topo_points_local : ndarray, shape (N, 3) or None

        Returns
        -------
        set of (int, int)
            Rounded (x, y) coordinates in metres. Empty when no topography
            is set.
        """
        out = set()
        if topo_points_local is None:
            return out
        h_tol = 0.5 * float(self.config.h)
        for x, y, z in topo_points_local:
            if abs(float(z)) < h_tol:
                out.add((round(float(x)), round(float(y))))
        return out

    # -------------------------------------------------------------------
    # Receiver record builders -- mirror the SW4 ``rec`` lines into the
    # HDF5 package, with ShakerMaker (georef) coordinates for downstream
    # tooling (h5drm).
    # -------------------------------------------------------------------

    def _model_receiver_records(self, transform, start_index, topo_xy_z0, qa_index):
        """Records for ShakerMaker stations, skipping the ones the topography
        block already covers (same logic as :func:`receivers.model_receiver_lines`).

        Returns
        -------
        list of dict
            One entry per kept station, with the same keys the HDF5 packager
            expects (``file``, ``kind``, ``xyz_km``, ``internal``, ``is_qa``,
            ``model_index``, ``metadata``).
        """
        records = []
        offset = int(start_index)
        for i_station, station in enumerate(self.model._receivers):
            xyz_m = transform.from_shakermaker_km_to_sw4_m(station.x)
            if abs(float(xyz_m[2])) < _COORD_TOL_M and (
                    round(float(xyz_m[0])), round(float(xyz_m[1]))) in topo_xy_z0:
                continue
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "shakermaker",
                "xyz_km": np.asarray(station.x, dtype=float),
                "internal": bool(station.is_internal),
                "is_qa": bool(i_station == qa_index),
                "model_index": int(i_station),
                "metadata": repr(station.metadata),
            })
            offset += 1
        return records

    def _model_receiver_surface_records(self, transform, topo_points_local, start_index, qa_index):
        """Records for stations forced to ``depth=0``.

        The station x/y are kept; z is set to the topography elevation at
        that (x, y) so the stored ShakerMaker coordinate matches the SW4
        receiver location.
        """
        records = []
        offset = int(start_index)
        for i_station, station in enumerate(self.model._receivers):
            xyz_m = transform.from_shakermaker_km_to_sw4_m(station.x)
            topo_z = self._topography_z_at(topo_points_local, xyz_m[0], xyz_m[1])
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "shakermaker_surface",
                "xyz_km": transform.to_original_m(np.asarray([xyz_m[0], xyz_m[1], -topo_z], dtype=float)) / 1000.0,
                "internal": bool(station.is_internal),
                "is_qa": bool(i_station == qa_index),
                "model_index": int(i_station),
                "metadata": "depth=0; " + repr(station.metadata),
            })
            offset += 1
        return records

    def _topography_z_at(self, topo_points_local, x, y):
        """Topography elevation at the closest (x, y) node, in metres.

        Nearest-neighbour lookup -- linear interpolation is unnecessary
        because the SW4 grid is already aligned with the topography grid
        after :func:`extend_topography_to_domain`.

        Returns ``0.0`` when no topography is set.
        """
        if topo_points_local is None or len(topo_points_local) == 0:
            return 0.0
        topo_points_local = np.asarray(topo_points_local, dtype=float)
        d2 = (topo_points_local[:, 0] - float(x)) ** 2 + (topo_points_local[:, 1] - float(y)) ** 2
        return float(topo_points_local[int(np.argmin(d2)), 2])

    def _topography_surface_records(self, transform, topo_points_local, start_index):
        """Records for the ``depth=0`` receiver placed on each topography node."""
        records = []
        for offset, (x, y, topo_z) in enumerate(
                np.asarray(topo_points_local, dtype=float), start=int(start_index)):
            records.append({
                "file": f"{self.config.station_prefix}{offset:05d}",
                "kind": "topography_surface",
                "xyz_km": transform.to_original_m(np.asarray([x, y, -topo_z], dtype=float)) / 1000.0,
                "internal": True,
                "is_qa": False,
                "model_index": -1,
                "metadata": "depth=0",
            })
        return records

    def _topography_z0_records(self, transform, topo_points_local, start_index):
        """Records for the column of receivers between topography and z=0."""
        records = []
        offset = int(start_index)
        for x, y, topo_z in np.asarray(topo_points_local, dtype=float):
            for z in _values_between_topography_and_z0(topo_z, self.config.h):
                records.append({
                    "file": f"{self.config.station_prefix}{offset:05d}",
                    "kind": "topography_to_z0",
                    "xyz_km": transform.to_original_m(np.asarray([x, y, z], dtype=float)) / 1000.0,
                    "internal": True,
                    "is_qa": False,
                    "model_index": -1,
                    "metadata": "between topography and z=0",
                })
                offset += 1
        return records

    def _records_from_rec_lines(self, transform, lines, kind, start_model_index):
        """Records for receivers built from a list of already-formatted ``rec`` lines.

        Used for the SW4 domain grid where the receiver positions live in
        the formatted strings, not in a model object.
        """
        records = []
        for line in lines:
            values = self._parse_rec_line(line)
            xyz_local_m = np.asarray(
                [values["x"], values["y"], values.get("z", 0.0)],
                dtype=float,
            )
            records.append({
                "file": values["file"],
                "kind": kind,
                "xyz_km": transform.to_original_m(xyz_local_m) / 1000.0,
                "internal": True,
                "is_qa": False,
                "model_index": int(start_model_index),
                "metadata": "",
            })
        return records

    def _parse_rec_line(self, line):
        """Pull ``file=``, ``x=``, ``y=`` and ``z=``/``depth=`` from a ``rec`` line.

        Returns
        -------
        dict
            ``{"file": str, "x": float, "y": float, "z"/"depth": float}``
            with the keys that were present in the line.
        """
        values = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "file":
                values[key] = value
            elif key in ("x", "y", "z", "depth"):
                values[key] = float(value)
        return values

    def _domain_sw4_origin_xy(self, domain_size, x_domain, y_domain):
        """Offset that centres the SW4 receiver sub-grid inside the box.

        Inputs
        ------
        domain_size : sequence of 3 floats
            Requested sub-grid extents.
        x_domain, y_domain : float
            Outer SW4 box extents.

        Returns
        -------
        tuple of 2 floats
            ``(ox, oy)`` in metres.
        """
        lx, ly, _lz = [float(value) for value in domain_size]
        if lx > float(x_domain) + _COORD_TOL_M or ly > float(y_domain) + _COORD_TOL_M:
            raise ValueError("domain_sw4_size x/y must fit inside size_domain.")
        return (
            0.5 * (float(x_domain) - lx),
            0.5 * (float(y_domain) - ly),
        )

    # -------------------------------------------------------------------
    # Output paths and sidecar summaries
    # -------------------------------------------------------------------

    def paths(self):
        """Dictionary of every path the exporter writes to.

        Keys: ``base``, ``exports``, ``sw4``, ``sources``, ``topo``,
        ``package_h5``, ``unpack_script``, ``input``.
        """
        return {
            "base": self.base_path,
            "exports": self.exports_path,
            "sw4": self.sw4_path,
            "sources": self.sources_path,
            "topo": self.topo_path,
            "package_h5": self.package_h5,
            "unpack_script": self.unpack_script,
            "input": self.input_file,
        }

    def _sources_summary_text(self, rows):
        """Render the source rows as a CSV string for human inspection."""
        headers = [
            "id", "x_km", "y_km", "z_km", "x_m", "y_m", "z_m",
            "x_sw4_m", "y_sw4_m", "z_sw4_m",
            "strike_deg", "dip_deg", "rake_deg", "trigger_time_s",
            "stf_local_t0_s", "dt", "stf_type", "dfile",
        ]
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return f.getvalue()
