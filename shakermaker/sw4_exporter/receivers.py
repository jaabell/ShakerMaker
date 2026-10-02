"""SW4 ``rec`` line builders.

The exporter calls one builder per receiver family (ShakerMaker stations,
ShakerMaker stations forced to the surface, topography surface nodes, the
gap between topography and z=0, and a regular grid spanning the SW4 box).
Every builder returns a list of plain SW4 ``rec`` lines so the input writer
can drop them as-is into the ``.in`` file.
"""

import numpy as np


# Tolerance, in metres, used to decide whether a receiver sits "on" z=0 or
# coincides with a topography-surface station at the same (x, y). Receivers
# returned by SW4 are positioned within float64 noise of the requested
# coordinates, so 1 nm is more than enough to detect a duplicate without
# folding intentionally close-but-distinct points together.
_Z_TOL_M = 1.0e-9


def model_receiver_lines(receivers, transform, prefix="sf", start_index=1, topo_xy_z0=None):
    """SW4 ``rec`` lines for ShakerMaker stations, keyed by an explicit z.

    Stations that fall on a topography surface node at z=0 are skipped to
    avoid two SW4 receivers on the exact same point. The set of (x, y) at
    z=0 already covered by topography stations is passed in via
    ``topo_xy_z0``.

    Inputs
    ------
    receivers : iterable of Station
        ShakerMaker receiver list. Each station must expose ``.x`` (km).
    transform : CoordinateTransform
        Maps ShakerMaker km to SW4 local metres.
    prefix : str
        Filename prefix for the SW4 ``file=`` field. Default ``"sf"``.
    start_index : int
        First index used in the ``file=`` suffix. Default 1.
    topo_xy_z0 : set of (int, int), optional
        Rounded (x, y) in metres of topography stations at z=0. Stations
        landing on those nodes are skipped.

    Returns
    -------
    list of str
        SW4 ``rec`` lines in writing order.
    """
    topo_xy_z0 = topo_xy_z0 or set()
    lines = []
    offset = int(start_index)
    for station in receivers:
        x, y, z = transform.from_shakermaker_km_to_sw4_m(station.x)
        if abs(float(z)) < _Z_TOL_M and (round(float(x)), round(float(y))) in topo_xy_z0:
            continue
        lines.append(
            f"rec x={float(x):.16g} y={float(y):.16g} z={float(z):.16g} "
            f"file={prefix}{offset:05d} usgsformat=1"
        )
        offset += 1
    return lines


def model_receiver_surface_lines(receivers, transform, prefix="sf", start_index=1):
    """SW4 ``rec`` lines for ShakerMaker stations forced to ``depth=0``.

    Used when the caller wants stations on the topography surface regardless
    of their original elevation in the ShakerMaker model.

    Inputs
    ------
    receivers : iterable of Station
    transform : CoordinateTransform
    prefix : str
    start_index : int

    Returns
    -------
    list of str
        SW4 ``rec`` lines, all using ``depth=0``.
    """
    lines = []
    offset = int(start_index)
    for station in receivers:
        x, y, _z = transform.from_shakermaker_km_to_sw4_m(station.x)
        lines.append(
            f"rec x={float(x):.16g} y={float(y):.16g} depth=0 "
            f"file={prefix}{offset:05d} usgsformat=1"
        )
        offset += 1
    return lines


def topography_receiver_lines(local_points, start_index=1, prefix="sf"):
    """SW4 ``rec`` lines, one per topography node, sampled at ``depth=0``.

    Inputs
    ------
    local_points : ndarray, shape (N, 3)
        Topography nodes in SW4 local metres. Only the (x, y) columns are used.
    start_index : int
    prefix : str

    Returns
    -------
    list of str
        SW4 ``rec`` lines with ``depth=0``.
    """
    lines = []
    for offset, (x, y, _z) in enumerate(np.asarray(local_points, dtype=float), start=start_index):
        lines.append(f"rec x={x:.1f} y={y:.1f} depth=0 file={prefix}{offset:05d} usgsformat=1")
    return lines


def topography_z0_receiver_lines(local_points, h, start_index=1, prefix="sf"):
    """SW4 ``rec`` lines for the vertical column between topography and z=0.

    For each topography (x, y), receivers are stacked every ``h`` metres from
    the topography elevation up (or down) to z=0. Sign of the topography
    elevation is preserved so the receivers stay inside the SW4 box.

    Inputs
    ------
    local_points : ndarray, shape (N, 3)
        Topography nodes in SW4 local metres.
    h : float
        Vertical spacing in metres.
    start_index : int
    prefix : str

    Returns
    -------
    list of str
    """
    lines = []
    offset = int(start_index)
    h = float(h)
    for x, y, topo_z in np.asarray(local_points, dtype=float):
        for z in _values_between_topography_and_z0(topo_z, h):
            lines.append(
                f"rec x={x:.1f} y={y:.1f} z={z:.16g} "
                f"file={prefix}{offset:05d} usgsformat=1"
            )
            offset += 1
    return lines


def domain_receiver_lines(h, domain_size, start_index=1, prefix="sf", topo_xy_z0=None,
                          origin_xy=(0.0, 0.0)):
    """SW4 ``rec`` lines on a regular 3-D grid spanning the SW4 box.

    Useful when a uniform DRM-style grid of receivers is wanted in addition
    to the model stations. Receivers landing on topography z=0 nodes are
    skipped, just like in :func:`model_receiver_lines`.

    Inputs
    ------
    h : float
        Grid spacing in metres along all three axes.
    domain_size : sequence of 3 floats
        ``(lx, ly, lz)`` extents in metres.
    start_index : int
    prefix : str
    topo_xy_z0 : set of (int, int), optional
        Rounded (x, y) of topography stations at z=0. Skipped here to avoid
        duplicates.
    origin_xy : tuple of 2 floats
        Offset of the grid origin inside the SW4 box, in metres. Defaults to
        the box corner ``(0, 0)``.

    Returns
    -------
    list of str
        SW4 ``rec`` lines in writing order.
    """
    topo_xy_z0 = topo_xy_z0 or set()
    lx, ly, lz = [float(value) for value in domain_size]
    ox, oy = [float(value) for value in origin_xy]
    h = float(h)
    xs = ox + _grid_values(0.0, lx, h)
    ys = oy + _grid_values(0.0, ly, h)
    zs = _grid_values(h, lz, h)
    lines = []
    offset = int(start_index)
    for z in zs:
        for y in ys:
            for x in xs:
                if abs(float(z)) < _Z_TOL_M and (round(float(x)), round(float(y))) in topo_xy_z0:
                    continue
                lines.append(
                    f"rec x={x:.1f} y={y:.1f} z={z:.1f} "
                    f"file={prefix}{offset:05d} usgsformat=1"
                )
                offset += 1
    return lines


def _grid_values(start, stop, step):
    """Inclusive axis samples ``[start, start+step, ..., stop]``.

    Trims trailing samples that overshoot ``stop`` past a small tolerance.
    """
    values = np.arange(float(start), float(stop) + 0.5 * float(step), float(step))
    return values[values <= float(stop) + _Z_TOL_M]


def _values_between_topography_and_z0(topo_z, h):
    """Z-levels between a topography elevation and z=0, with sign preserved.

    Inputs
    ------
    topo_z : float
        Topography elevation at the (x, y) being filled (metres, SW4 sign).
    h : float
        Vertical spacing in metres.

    Returns
    -------
    ndarray
        Z values ordered from near-topography toward z=0. Empty when
        ``topo_z`` is at z=0 within ``_Z_TOL_M``.
    """
    topo_z = float(topo_z)
    h = float(h)
    if abs(topo_z) < _Z_TOL_M:
        return []
    values = np.arange(h, abs(topo_z), h)
    if len(values) == 0:
        return values
    if topo_z > 0.0:
        return -values
    return values
