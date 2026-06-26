"""Geometry viewer for an SW4 export.

Reads back either the SW4 ``.in`` file or the HDF5 transport package and
plots sources, receivers and topography in PyVista. The viewer can show
the geometry in SW4 local metres or shifted into ShakerMaker georef
metres when an origin offset is supplied.

A receiver in the SW4 ``.in`` can be addressed by ``z=...`` (absolute
elevation) or by ``depth=...`` (distance below the topography). The parser
keeps both alive: depth-addressed receivers are stored with their depth
encoded in the z column, then resolved against the actual topography
later. The encoding uses a very large negative offset so encoded values
are easy to detect and impossible to confuse with real coordinates.
"""

from pathlib import Path

import numpy as np


# Receivers given as ``depth=D`` are stored with z = -(D + _DEPTH_ENCODE_OFFSET)
# so they sit far outside any plausible coordinate range. The mask in
# :func:`_resolve_receivers` reads back the offset and substitutes the real
# topography elevation once it is available.
_DEPTH_ENCODE_OFFSET = 1e9


def _token_value(tokens, key):
    """Return the float value of ``key=...`` in a token list, or ``None`` if absent."""
    prefix = f"{key}="
    for token in tokens:
        if token.startswith(prefix):
            return float(token[len(prefix):])
    return None


def _token_str(tokens, key):
    """Return the string value of ``key=...`` in a token list, or ``None`` if absent."""
    prefix = f"{key}="
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _read_grid(tokens):
    """Extract ``(x, y, z)`` from an SW4 ``grid ...`` line, in metres."""
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    z = _token_value(tokens, "z")
    if None in (x, y, z):
        raise ValueError("SW4 grid line must include x, y, and z.")
    return x, y, z


def _read_point(tokens):
    """Extract a 3-D point from a ``source``/``rec`` line.

    Returns
    -------
    tuple of 3 floats or None
        ``(x, y, z)``. When ``depth=`` was used instead of ``z=``, the z
        value is encoded with :data:`_DEPTH_ENCODE_OFFSET` so the depth
        is recovered later in :func:`_resolve_receivers`.
    """
    x = _token_value(tokens, "x")
    y = _token_value(tokens, "y")
    if x is None or y is None:
        return None
    depth = _token_value(tokens, "depth")
    if depth is not None:
        return x, y, -(depth + _DEPTH_ENCODE_OFFSET)
    z = _token_value(tokens, "z")
    if z is not None:
        return x, y, z
    return x, y, 0.0


def read_sw4_geometry(path):
    """Read SW4 geometry from either an ``.in`` file or the HDF5 package.

    Inputs
    ------
    path : str or Path
        Either a ``shakermaker2sw4.in`` (text) or a ``sw4_package.h5``
        (HDF5). Dispatch is by suffix; ``.h5``/``.hdf5`` routes to
        :func:`read_sw4_geometry_h5`.

    Returns
    -------
    grid : tuple of 3 floats
        SW4 box extents in metres.
    sources : ndarray, shape (Ns, 3)
        Source positions.
    receivers : ndarray, shape (Nr, 3)
        Receiver positions. May still contain depth-encoded z values when
        the input came from a text file.
    receiver_kinds : ndarray of object, shape (Nr,)
        Family tag for each receiver (``shakermaker``, ``topography_surface``,
        etc.).
    topo_points : ndarray, shape (N, 3) or None
        Topography nodes when present.
    topo_original_bounds : dict or None
        ``{"xmin", "xmax", "ymin", "ymax"}`` from the ``.in`` header
        comment or the HDF5 ``original_bounds`` field.
    """
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return read_sw4_geometry_h5(path)

    grid = None
    sources = []
    receivers = []
    receiver_kinds = []
    topo_rel_file = None
    topo_original_bounds = None
    current_receiver_kind = "receiver"

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            label = line.lstrip("#").strip().lower()
            if label.startswith("shakermaker topography_original_bounds"):
                topo_original_bounds = _read_topography_bounds(label.split())
            elif label == "shakermaker stations surface":
                current_receiver_kind = "shakermaker_surface"
            elif label == "shakermaker stations":
                current_receiver_kind = "shakermaker"
            elif label.startswith("topography surface"):
                current_receiver_kind = "topography_surface"
            elif label.startswith("between topography"):
                current_receiver_kind = "topography_to_z0"
            elif label.startswith("sw4 domain"):
                current_receiver_kind = "sw4_domain"
            continue
        tokens = line.split()
        if tokens[0] == "grid":
            grid = _read_grid(tokens)
        elif tokens[0] == "source":
            pt = _read_point(tokens)
            if pt is not None:
                sources.append(pt)
        elif tokens[0] == "rec":
            pt = _read_point(tokens)
            if pt is not None:
                receivers.append(pt)
                receiver_kinds.append(current_receiver_kind)
        elif tokens[0] == "topography":
            topo_rel_file = _token_str(tokens, "file")

    if grid is None:
        raise ValueError(f"No grid line found in {path}.")

    topo_points = None
    if topo_rel_file is not None:
        topo_abs = path.parent / topo_rel_file
        try:
            from .topography import read_cartesian_topography
            _nx, _ny, topo_points = read_cartesian_topography(topo_abs)
        except Exception:
            pass

    return (
        grid,
        np.asarray(sources, dtype=float) if sources else np.empty((0, 3)),
        np.asarray(receivers, dtype=float) if receivers else np.empty((0, 3)),
        np.asarray(receiver_kinds, dtype=object),
        topo_points,
        topo_original_bounds,
    )


def read_sw4_geometry_h5(path):
    """Same as :func:`read_sw4_geometry`, but for the HDF5 transport package.

    Coordinates stored in the package live in ShakerMaker (georef) metres;
    this function subtracts the SW4 origin so the returned arrays are in
    SW4 local metres, matching the text-file path.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as hf:
        config = hf["config"]
        sw4_origin_m = _h5_sw4_origin_m(hf)
        grid = (
            float(config["x_domain"][()]),
            float(config["y_domain"][()]),
            float(config["z_domain"][()]),
        )
        sources = np.column_stack([
            hf["sources/x_km"][:],
            hf["sources/y_km"][:],
            hf["sources/z_km"][:],
        ]).astype(float) * 1000.0 - sw4_origin_m if "sources" in hf and len(hf["sources/id"]) else np.empty((0, 3), dtype=float)

        if "receivers" in hf and "xyz_km" in hf["receivers"]:
            receivers = np.asarray(hf["receivers/xyz_km"][:], dtype=float) * 1000.0 - sw4_origin_m
            receiver_kinds = _read_h5_strings(hf["receivers/kind"])
        else:
            receivers = np.empty((0, 3), dtype=float)
            receiver_kinds = np.empty((0,), dtype=object)

        topo_points = None
        topo_original_bounds = None
        if "topography" in hf and bool(hf["topography/present"][()]):
            topo_points = np.asarray(hf["topography/points_xyz_m"][:], dtype=float) - sw4_origin_m
            if "original_bounds" in hf["topography"]:
                bounds = np.asarray(hf["topography/original_bounds"][:], dtype=float) - np.asarray(
                    [sw4_origin_m[0], sw4_origin_m[0], sw4_origin_m[1], sw4_origin_m[1]],
                    dtype=float,
                )
                topo_original_bounds = {
                    "xmin": float(bounds[0]),
                    "xmax": float(bounds[1]),
                    "ymin": float(bounds[2]),
                    "ymax": float(bounds[3]),
                }

    return grid, sources, receivers, receiver_kinds, topo_points, topo_original_bounds


def _h5_sw4_origin_m(hf):
    """Read the SW4 origin (in ShakerMaker metres) from the HDF5 package."""
    if "coordinates" in hf and "sw4_origin_in_shakermaker_m" in hf["coordinates"]:
        return np.asarray(hf["coordinates/sw4_origin_in_shakermaker_m"][:], dtype=float)
    config = hf["config"]
    return -np.asarray([
        float(config["x_origin"][()]),
        float(config["y_origin"][()]),
        float(config["z_origin"][()]),
    ], dtype=float)


def _read_h5_strings(dataset):
    """Decode a 1-D HDF5 byte-string dataset into a Python ``object`` ndarray."""
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ], dtype=object)


def _read_topography_bounds(tokens):
    """Pull ``xmin/xmax/ymin/ymax`` from the topography-bounds comment line.

    Returns
    -------
    dict or None
        ``{"xmin", "xmax", "ymin", "ymax"}`` if all four keys are present,
        otherwise ``None``.
    """
    out = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in ("xmin", "xmax", "ymin", "ymax"):
            out[key] = float(value)
    if {"xmin", "xmax", "ymin", "ymax"} <= set(out):
        return out
    return None


def _resolve_receivers(receivers, topo_points):
    """Replace depth-encoded receiver z values with the real topography elevation.

    Receivers whose z column was set to ``-(depth + _DEPTH_ENCODE_OFFSET)``
    by :func:`_read_point` get their real elevation back here: the
    topography elevation is looked up (exact node first, nearest neighbour
    otherwise) and ``depth`` is subtracted to land the receiver below the
    free surface.

    Inputs
    ------
    receivers : ndarray, shape (N, 3)
        Receiver positions, possibly with encoded depths.
    topo_points : ndarray, shape (M, 3) or None
        Topography nodes used for the lookup.

    Returns
    -------
    ndarray, shape (N, 3)
        Resolved receiver positions.
    """
    if len(receivers) == 0:
        return receivers.copy()

    resolved = receivers.copy()
    depth_mask = resolved[:, 2] < -(_DEPTH_ENCODE_OFFSET / 2.0)
    if not depth_mask.any():
        return resolved

    topo_lookup = {}
    if topo_points is not None and len(topo_points):
        for tx, ty, tz in topo_points:
            topo_lookup[(round(float(tx)), round(float(ty)))] = float(tz)

    for i in np.where(depth_mask)[0]:
        x, y = resolved[i, 0], resolved[i, 1]
        depth = -(resolved[i, 2]) - _DEPTH_ENCODE_OFFSET
        topo_z = topo_lookup.get((round(x), round(y)))
        if topo_z is None:
            topo_z = _nearest_topography_z(topo_points, x, y)
        resolved[i, 2] = -(topo_z - depth)

    return resolved


def _nearest_topography_z(topo_points, x, y):
    """Topography elevation at the closest (x, y) node, or 0 when no topography."""
    if topo_points is None or len(topo_points) == 0:
        return 0.0
    d2 = (topo_points[:, 0] - float(x)) ** 2 + (topo_points[:, 1] - float(y)) ** 2
    return float(topo_points[int(np.argmin(d2)), 2])


def plot_sw4_geometry(path, origin_m=None):
    """Open a PyVista window with the SW4 box, sources, receivers and topography.

    The viewer expects ``pyvista``, ``pyvistaqt`` and ``PyQt5`` to be
    installed. On Linux it forces the ``xcb`` Qt platform when running
    under Wayland because PyVistaQt + PyQt5 crashes there with a fatal
    X BadWindow error during startup.

    Inputs
    ------
    path : str or Path
        Either a ``shakermaker2sw4.in`` file or the HDF5 transport package.
    origin_m : array-like, shape (3,), optional
        SW4 origin expressed in ShakerMaker metres. When given, the scene
        is shifted so receivers, sources and topography are shown in the
        georef frame. When omitted, the scene stays in SW4 local metres.

    Returns
    -------
    None
        Blocks on the Qt event loop until the window is closed.
    """
    try:
        import pyvista as pv
        from pyvistaqt import QtInteractor
        from PyQt5 import QtWidgets
    except ImportError as exc:
        raise ImportError(
            "plot_geometry=True requires pyvista, pyvistaqt, and PyQt5."
        ) from exc

    import os
    import sys

    # See the docstring: force xcb on Linux when running under Wayland.
    if sys.platform.startswith("linux"):
        qpa = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
        if qpa == "" or qpa.startswith("wayland"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"

    path = Path(path)
    (grid_x, grid_y, grid_z), sources, receivers, receiver_kinds, topo_points, topo_original_bounds = read_sw4_geometry(path)
    receivers = _resolve_receivers(receivers, topo_points)
    origin_m = None if origin_m is None else np.asarray(origin_m, dtype=float)
    georef = origin_m is not None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    title = "SW4 geometry georeferenced" if georef else "SW4 geometry local"
    window.setWindowTitle(f"{title} - {path.name}")
    window.resize(1200, 800)

    frame = QtWidgets.QFrame()
    layout = QtWidgets.QVBoxLayout(frame)
    plotter = QtInteractor(frame)
    layout.addWidget(plotter.interactor)
    window.setCentralWidget(frame)

    if georef:
        x0, y0, z0 = origin_m
        box_bounds = (x0, x0 + grid_x, y0, y0 + grid_y, z0, z0 + grid_z)
        sources = _shift_points(sources, origin_m)
        receivers = _shift_points(receivers, origin_m)
    else:
        box_bounds = (0.0, grid_x, 0.0, grid_y, 0.0, grid_z)

    box = pv.Box(bounds=box_bounds)
    plotter.add_mesh(box, color="lightgray", opacity=0.12, show_edges=False)
    plotter.add_mesh(box, color="gray", style="wireframe", line_width=2)

    topo_extended_points = np.empty((0, 3), dtype=float)
    if topo_points is not None and len(topo_points):
        topo_base_points, topo_extended_points = _split_topography_points(topo_points, topo_original_bounds)
        topo_plot = np.column_stack([
            topo_base_points[:, 0],
            topo_base_points[:, 1],
            -topo_base_points[:, 2],
        ]).astype(float)
        if georef:
            topo_plot = _shift_points(topo_plot, origin_m)
        if len(topo_plot):
            plotter.add_points(
                pv.PolyData(topo_plot),
                color="tan",
                point_size=2,
                render_points_as_spheres=False,
                opacity=0.4,
            )

    if topo_extended_points is not None and len(topo_extended_points):
        extended_plot = np.column_stack([
            topo_extended_points[:, 0],
            topo_extended_points[:, 1],
            -topo_extended_points[:, 2],
        ]).astype(float)
        if georef:
            extended_plot = _shift_points(extended_plot, origin_m)
        plotter.add_points(
            pv.PolyData(extended_plot),
            color="gray",
            point_size=3,
            render_points_as_spheres=False,
            opacity=0.18,
        )

    if len(receivers):
        surface_mask = receiver_kinds == "shakermaker_surface"
        other_receivers = receivers[~surface_mask]
        if len(other_receivers):
            rec_cloud = pv.PolyData(other_receivers.astype(float))
            rec_cloud["Z [m]"] = other_receivers[:, 2].astype(float)
            plotter.add_points(
                rec_cloud,
                scalars="Z [m]",
                cmap="RdYlBu_r",
                point_size=4,
                render_points_as_spheres=False,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Z [m]   (- above datum / + below)",
                    "vertical": True,
                },
            )
        surface_receivers = receivers[surface_mask]
        if len(surface_receivers):
            plotter.add_points(
                pv.PolyData(surface_receivers.astype(float)),
                color="black",
                point_size=10,
                render_points_as_spheres=True,
            )

    if len(sources):
        plotter.add_points(
            pv.PolyData(sources.astype(float)),
            color="red",
            point_size=18,
            render_points_as_spheres=True,
        )

    label = "SW4 georeferenced coordinates" if georef else "SW4 local Cartesian"
    plotter.add_text(label, position="upper_left", font_size=11)
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="X [m]",
        ytitle="Y [m]",
        ztitle="Z [m]  (+ down)",
        font_size=9,
        fmt="%.0f",
    )
    plotter.show_axes()
    plotter.reset_camera()

    window.show()
    app.exec_()


def _shift_points(points, origin_m):
    """Add a per-coordinate origin offset to a point cloud (no-op when empty)."""
    if len(points) == 0:
        return points
    return np.asarray(points, dtype=float) + origin_m.reshape(1, 3)


def _split_topography_points(topo_points, bounds):
    """Split topography into the original-coverage subset and the extended ring.

    Lets the viewer draw the user-supplied topography in one colour and
    the extended (auto-padded) ring in another, so it is obvious which
    region the underlying data actually constrains.

    Returns
    -------
    inside, outside : ndarray, shape (Ni, 3) and (No, 3)
    """
    if bounds is None:
        return topo_points, np.empty((0, 3), dtype=float)
    points = np.asarray(topo_points, dtype=float)
    inside = (
        (points[:, 0] >= bounds["xmin"] - 1.0e-9)
        & (points[:, 0] <= bounds["xmax"] + 1.0e-9)
        & (points[:, 1] >= bounds["ymin"] - 1.0e-9)
        & (points[:, 1] <= bounds["ymax"] + 1.0e-9)
    )
    return points[inside], points[~inside]
