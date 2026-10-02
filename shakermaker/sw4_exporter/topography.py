"""Cartesian topography I/O and SW4 grid extension helpers.

The exporter consumes cartesian topography in the SW4 file format
(``Nx Ny`` header line followed by ``x y z`` rows). This module reads it,
rotates it into the ShakerMaker (x = North, y = East) convention, snaps the
grid back to a regular layout, and extends it to the SW4 local box.

The diagnostic ``print_*`` helpers print bound boxes and grid estimates while
the exporter is running. They are intentionally print-based: this is one of
the few places where seeing values on the terminal during the export is more
useful than a log line.
"""

from pathlib import Path
import warnings
import numpy as np

SEPARATOR = "-" * 50


def read_cartesian_topography(path):
    """Read an SW4 cartesian topography file.

    Inputs
    ------
    path : str or Path
        Path to a text file whose first line is ``Nx Ny`` and whose remaining
        lines list ``x y z`` triplets, one per grid node.

    Returns
    -------
    nx, ny : int
        Grid sizes along the file's x and y axes.
    points : ndarray, shape (Nx*Ny, 3)
        Topography nodes in the file's coordinate convention. Units are
        whatever the file uses (typically metres).
    """
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty topography file: {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid topography header in {path}: {lines[0]}")

    nx, ny = int(header[0]), int(header[1])
    points = np.array([[float(v) for v in line.split()[:3]] for line in lines[1:]], dtype=float)
    expected = nx * ny
    if points.shape[0] != expected:
        raise ValueError(f"Topography point count mismatch in {path}: expected {expected}, got {points.shape[0]}")

    return nx, ny, points


def rotate_topography_to_shakermaker(points):
    """Swap x and y so the grid is in ShakerMaker convention.

    SW4 stores topography with x = East, y = North. ShakerMaker uses
    x = North, y = East. The vertical column is left untouched.

    Inputs
    ------
    points : ndarray, shape (N, 3)
        Topography nodes in SW4 convention.

    Returns
    -------
    ndarray, shape (N, 3)
        Same nodes with the first two columns swapped.
    """
    points = np.asarray(points, dtype=float)
    rotated = points.copy()
    rotated[:, 0] = points[:, 1]
    rotated[:, 1] = points[:, 0]
    return rotated


def rebuild_cartesian_topography(points):
    """Resort points so they form a regular cartesian grid in row-major order.

    Required after a coordinate swap, which destroys the original ordering.
    Fails if the point set is not a complete cartesian product.

    Inputs
    ------
    points : ndarray, shape (N, 3)
        Topography nodes after rotation.

    Returns
    -------
    nx, ny : int
        Recomputed grid sizes.
    points_sorted : ndarray, shape (Nx*Ny, 3)
        Nodes laid out as ``for y in ys: for x in xs:`` so the writer can
        emit them in the order SW4 expects.
    """
    points = np.asarray(points, dtype=float)
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    z_by_xy = {(float(x), float(y)): float(z) for x, y, z in points}

    rebuilt = []
    for y in ys:
        for x in xs:
            key = (float(x), float(y))
            if key not in z_by_xy:
                raise ValueError("Topography points must form a complete cartesian grid.")
            rebuilt.append((x, y, z_by_xy[key]))

    return len(xs), len(ys), np.asarray(rebuilt, dtype=float)


def bounds(points):
    """Return the axis-aligned bounding box of a point set.

    Inputs
    ------
    points : ndarray, shape (N, 3)

    Returns
    -------
    tuple of 6 floats
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    """
    points = np.asarray(points, dtype=float)
    return (
        float(points[:, 0].min()), float(points[:, 0].max()),
        float(points[:, 1].min()), float(points[:, 1].max()),
        float(points[:, 2].min()), float(points[:, 2].max()),
    )


def cartesian_topography_text(nx, ny, points):
    """Format a topography grid back into the SW4 text format.

    Inputs
    ------
    nx, ny : int
        Grid sizes.
    points : ndarray, shape (Nx*Ny, 3)
        Topography nodes in row-major order.

    Returns
    -------
    str
        Multi-line text starting with ``Nx Ny`` and one ``x y z`` row per node.
    """
    lines = [f"{nx} {ny}"]
    for x, y, z in np.asarray(points, dtype=float):
        lines.append(f"{x:.1f} {y:.1f} {z:.6f}")
    return "\n".join(lines) + "\n"


def extend_topography_to_domain(nx, ny, points, x_domain, y_domain):
    """Resample topography onto a regular grid that covers the SW4 box.

    SW4 requires the topography to span the full ``[0, x_domain] x [0, y_domain]``
    box. If the input grid is smaller, the nearest sample is used on every
    new node that falls outside the original coverage.

    Inputs
    ------
    nx, ny : int
        Grid sizes of the input topography.
    points : ndarray, shape (Nx*Ny, 3)
        Input topography nodes in local SW4 coordinates (metres).
    x_domain, y_domain : float
        SW4 domain extents in metres.

    Returns
    -------
    new_nx, new_ny : int
        Grid sizes of the extended topography.
    extended : ndarray, shape (new_nx*new_ny, 3)
        Resampled topography covering the full domain, row-major.
    """
    points = np.asarray(points, dtype=float)
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    if len(xs) != nx or len(ys) != ny:
        raise ValueError("Topography points must form a regular cartesian grid.")

    # Use the median sample spacing so a slightly irregular input does not
    # confuse the extension. SW4 itself only sees the final regular grid.
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else float(x_domain)
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else float(y_domain)
    new_xs = _axis_to_domain(0.0, x_domain, dx)
    new_ys = _axis_to_domain(0.0, y_domain, dy)

    z_grid = points[:, 2].reshape(ny, nx)
    extended = []
    for y in new_ys:
        y_clamped = min(max(y, ys[0]), ys[-1])
        old_j = int(np.abs(ys - y_clamped).argmin())
        for x in new_xs:
            x_clamped = min(max(x, xs[0]), xs[-1])
            old_i = int(np.abs(xs - x_clamped).argmin())
            extended.append((x, y, z_grid[old_j, old_i]))

    print(f"Extended topography spacing dx={dx:.6g} dy={dy:.6g}")
    return len(new_xs), len(new_ys), np.asarray(extended, dtype=float)


def _axis_to_domain(start, end, step):
    """Return axis samples spanning ``[start, end]`` at ``step`` spacing.

    Ensures the last sample lies exactly on ``end`` even when ``step`` does
    not divide ``end - start`` evenly.

    Inputs
    ------
    start, end : float
        Axis range.
    step : float
        Nominal sample spacing.

    Returns
    -------
    ndarray
        1-D array of samples.
    """
    values = list(np.arange(float(start), float(end) + 0.5 * step, float(step)))
    if values[-1] < float(end):
        values.append(float(end))
    elif values[-1] > float(end):
        values[-1] = float(end)
    return np.asarray(values, dtype=float)


def _grid_size(length, h):
    """Number of grid nodes that fit in ``length`` with spacing ``h``.

    Equivalent to ``length / h + 1`` rounded to the nearest integer, which
    matches the way SW4 counts grid nodes (inclusive of both ends).
    """
    return int(round(float(length) / float(h))) + 1


def print_topography_diagnostics(original_points, local_points, x_domain, y_domain, z_domain,
                                 h=None, topo_nx=None, topo_ny=None, topo_zmax=None):
    """Print original/local topography bounds, centroid and SW4 grid estimate.

    Inputs
    ------
    original_points : ndarray, shape (N, 3)
        Topography in ShakerMaker (georef) coordinates, metres.
    local_points : ndarray, shape (N, 3)
        Same topography in SW4 local coordinates, metres.
    x_domain, y_domain, z_domain : float
        SW4 box extents.
    h : float, optional
        Grid spacing. Triggers the grid-size estimate.
    topo_nx, topo_ny : int, optional
        Input topography grid sizes (only used in the printout).
    topo_zmax : float, optional
        Cap on topography elevation (only used in the printout).

    Returns
    -------
    None
        Diagnostic output goes to stdout.
    """
    original_centroid = np.asarray(original_points, dtype=float).mean(axis=0)
    local_centroid = np.asarray(local_points, dtype=float).mean(axis=0)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(original_points)
    lxmin, lxmax, lymin, lymax, lzmin, lzmax = bounds(local_points)
    print(SEPARATOR)
    print("Topography bounds")
    print(SEPARATOR)
    print(f"Original  x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]  z=[{zmin:.2f}, {zmax:.2f}]")
    print(f"Centroid  [{original_centroid[0]:.1f}, {original_centroid[1]:.1f}, {original_centroid[2]:.2f}]")
    print("")
    print(f"Local     x=[{lxmin:.1f}, {lxmax:.1f}]  y=[{lymin:.1f}, {lymax:.1f}]  z=[{lzmin:.2f}, {lzmax:.2f}]")
    print(f"Centroid  [{local_centroid[0]:.1f}, {local_centroid[1]:.1f}, {local_centroid[2]:.2f}]")
    print("")
    print(f"SW4 grid  x=[0.0, {x_domain:.1f}]  y=[0.0, {y_domain:.1f}]  z=[0.0, {z_domain:.1f}]")
    if h is not None:
        nx = _grid_size(x_domain, h)
        ny = _grid_size(y_domain, h)
        nz = _grid_size(z_domain, h)
        print("")
        print("Grid estimate")
        if topo_nx is not None and topo_ny is not None:
            print(f"Topography samples  Nx={int(topo_nx)}  Ny={int(topo_ny)}")
        if topo_zmax is not None:
            print(f"Topography zmax     {float(topo_zmax):.1f}")
        print(f"SW4 grid            h={float(h):.1f}  Nx={nx}  Ny={ny}  Nz={nz}  Points={nx * ny * nz}")
    if lxmin > 0.0 or lymin > 0.0:
        print("WARNING: local topography starts above x=0 or y=0; SW4 may require coverage at the grid minimum.")
    if lxmin < 0.0 or lymin < 0.0:
        print("WARNING: local topography has negative x/y coordinates.")
    if lxmax < x_domain or lymax < y_domain:
        print("WARNING: local topography does not cover the full SW4 grid domain.")


def print_domain_diagnostics(x_domain, y_domain, z_domain, h=None):
    """Print the SW4 box extents and grid-size estimate when no topography is set.

    Inputs
    ------
    x_domain, y_domain, z_domain : float
    h : float, optional
        Grid spacing. When given, the node count per axis is also printed.

    Returns
    -------
    None
    """
    print(SEPARATOR)
    print("SW4 grid domain")
    print(SEPARATOR)
    print(f"SW4 grid domain     x=[0.0, {float(x_domain):.1f}], y=[0.0, {float(y_domain):.1f}], z=[0.0, {float(z_domain):.1f}]")
    if h is not None:
        nx = _grid_size(x_domain, h)
        ny = _grid_size(y_domain, h)
        nz = _grid_size(z_domain, h)
        print(f"Grid estimate       h={float(h):.1f}, Nx={nx}, Ny={ny}, Nz={nz}, Points={nx * ny * nz}")


def print_active_geometry_bounds(points):
    """Print the bounding box and centroid of the union of sources/stations/topo.

    Inputs
    ------
    points : ndarray, shape (N, 3)
        All points contributing to the SW4 domain decision, in metres.

    Returns
    -------
    None
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(points)
    centroid = points.mean(axis=0)
    print(SEPARATOR)
    print("Active geometry bounds")
    print(SEPARATOR)
    print(f"Active    x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]  z=[{zmin:.2f}, {zmax:.2f}]")
    print(f"Centroid  [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.2f}]")


def warn_active_geometry_supergrid(points, x_domain, y_domain, z_domain, h, supergrid_gp):
    """Warn when any active point sits inside or too close to the supergrid.

    The SW4 supergrid is an absorbing layer of ``supergrid_gp`` grid points
    along the four lateral walls and the bottom (the z=0 top is a free
    surface, not a supergrid). A source or receiver inside that layer radiates
    or records spurious, damped motion, so it must stay at least ``gp*h``
    metres from each of those walls.

    Inputs
    ------
    points : ndarray, shape (N, 3)
        Active sources/receivers in SW4 local metres.
    x_domain, y_domain, z_domain : float
        SW4 box extents in metres.
    h : float
        Grid spacing in metres.
    supergrid_gp : int
        Supergrid width in grid points.

    Returns
    -------
    None
        Emits a :class:`UserWarning` listing every wall that is too close.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return
    sponge = float(supergrid_gp) * float(h)
    xmin, xmax, ymin, ymax, _zmin, zmax = bounds(points)
    checks = [
        ("x-min wall", xmin - 0.0),
        ("x-max wall", float(x_domain) - xmax),
        ("y-min wall", ymin - 0.0),
        ("y-max wall", float(y_domain) - ymax),
        ("bottom (z-max) wall", float(z_domain) - zmax),
    ]
    problems = [
        f"{label}: {clearance:.1f} m clearance"
        for label, clearance in checks
        if clearance < sponge
    ]
    if problems:
        warnings.warn(
            "Active geometry sits inside or too close to the SW4 supergrid "
            "absorbing layer:\n  " + "\n  ".join(problems)
            + f"\n(supergrid gp={int(supergrid_gp)}, h={float(h):.1f} m -> "
            f"{sponge:.1f} m sponge per wall). Sources/receivers in the "
            "damping layer produce spurious results; enlarge the domain or "
            "leave the affected axis as None to size it automatically.",
            stacklevel=2,
        )
