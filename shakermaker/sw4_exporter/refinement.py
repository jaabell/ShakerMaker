"""Vertical mesh refinement (SW4 ``refinement zmax=...``) from a crust model.

SW4's ``refinement zmax=Z`` command adds a mesh patch, covering
``0 <= z <= Z``, whose grid spacing is half of the next coarser grid
immediately below it (SW4 User's Guide, "Mesh refinement"). Successive
``refinement`` lines stack, each halving resolution again for a shallower
sub-range; grids are aligned so every second node of a refined patch
coincides with a node of the coarser grid below, which requires each
``zmax`` to fall on a node of that coarser grid.

A single global ``h`` fine enough for the softest (shallowest) layer over-
resolves the deeper, faster layers. This module derives, from the crust
layering already passed to :func:`shakermaker.sw4_exporter.materials.material_lines`,
how many refinement levels each layer needs and where to place the
``zmax`` boundaries, using the same element-size criterion
``ShakerMaker.check_parameters`` already applies globally (see
``shakermaker/shakermaker.py``): ``h_required = (Vs * 1000) / (fmax *
n_per_wavelength)``, generalised here layer by layer.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .topography import SEPARATOR


def compute_layer_refinement(crust, h_base, fmax, n_per_wavelength=10.0, round_zmax="outward"):
    """Derive SW4 ``refinement zmax=...`` lines from a ShakerMaker crust.

    Walks the crust from the half-space up. Each layer needs a grid fine
    enough to put ``n_per_wavelength`` points across its shortest resolved
    wavelength (``Vs / fmax``); the coarsest grid (``h_base``, the ``h``
    passed to ``export_sw4``/``export_sw4_topo``) must resolve the deepest
    layer, and every shallower layer gets the smallest number of halvings
    that meets its own requirement, snapped to the alignment SW4 needs.

    Inputs
    ------
    crust : CrustModel
        Must expose ``d`` (layer thicknesses, km), ``b`` (Vs, km/s) and
        ``nlayers`` -- the same object passed to ``material_lines``.
    h_base : float
        Grid spacing (m) of the coarsest grid, i.e. the ``h`` passed to
        ``export_sw4``/``export_sw4_topo``. Used for the deepest layer.
    fmax : float
        Maximum frequency of engineering interest (Hz). Together with
        ``n_per_wavelength`` this sets how fine each layer's grid must be.
        Exposed as a required argument on purpose: this is a modelling
        choice, not something to default silently.
    n_per_wavelength : float
        Minimum grid points per shortest resolved wavelength. Default
        ``10``, matching the default in ``ShakerMaker.check_parameters``.
    round_zmax : {"outward", "nearest"}
        How to snap each computed refinement boundary to a multiple of the
        coarser grid's spacing (SW4 requires refinement boundaries to land
        on a coarse-grid node). ``"outward"`` (default) always pushes the
        boundary deeper, so the softer layer above stays fully inside the
        finer grid; ``"nearest"`` rounds to the closest node instead, which
        may leave a thin sliver of the soft layer under-resolved.

    Returns
    -------
    dict
        ``table`` : list of one dict per crust layer (top to bottom), each
        with ``layer`` (1-based), ``vs_km_s``, ``top_m``, ``bot_m`` (``None``
        for the half-space), ``h_required_m``, ``level_required``,
        ``level_assigned``, ``h_assigned_m``.
        ``refinement_lines`` : list of str, ``"refinement zmax=..."``, one
        per refinement level, already in the decreasing-``zmax`` order SW4
        expects. Empty when ``h_base`` already resolves every layer.
        ``zmax_values`` : list of float, the raw zmax used for each line,
        same order as ``refinement_lines``.
        ``levels`` : list of int, refinement level assigned to each layer
        (``0`` = base grid), same order as ``crust``.
        ``h_base``, ``fmax``, ``n_per_wavelength`` : the inputs, echoed back.
    """
    if fmax <= 0:
        raise ValueError(f"fmax must be > 0, got {fmax!r}.")
    if n_per_wavelength <= 0:
        raise ValueError(f"n_per_wavelength must be > 0, got {n_per_wavelength!r}.")
    if round_zmax not in ("outward", "nearest"):
        raise ValueError(f"round_zmax must be 'outward' or 'nearest', got {round_zmax!r}.")

    h_base = float(h_base)
    n = crust.nlayers
    depth_top_m = np.concatenate(([0.0], np.cumsum(crust.d[:-1]))) * 1000.0
    depth_bot_m = np.append(depth_top_m[1:], np.inf) if n > 1 else np.array([np.inf])
    vs_km_s = np.asarray(crust.b, dtype=float)

    h_required = (vs_km_s * 1000.0) / (fmax * n_per_wavelength)

    level_required = np.zeros(n, dtype=int)
    needs_refine = h_required < h_base
    level_required[needs_refine] = np.ceil(
        np.log2(h_base / h_required[needs_refine])
    ).astype(int)

    # Monotonic from the half-space up: a refinement line refines everything
    # above its zmax, so a layer can never end up coarser than the one below
    # it. The half-space itself is pinned to the base grid -- if it needs
    # more, h_base is simply too coarse for this fmax/n_per_wavelength.
    level = np.zeros(n, dtype=int)
    if level_required[-1] > 0:
        warnings.warn(
            f"h_base={h_base:g} m does not resolve the half-space layer "
            f"(Vs={vs_km_s[-1]:g} km/s needs h<={h_required[-1]:.3g} m at "
            f"fmax={fmax:g} Hz, n_per_wavelength={n_per_wavelength:g}); "
            "lower h or relax fmax/n_per_wavelength.",
            stacklevel=2,
        )
    for i in range(n - 2, -1, -1):
        level[i] = max(level[i + 1], level_required[i])

    max_level = int(level.max()) if n else 0
    refinement_lines = []
    zmax_values = []
    for lvl in range(1, max_level + 1):
        deepest_layer_at_level = max(i for i in range(n) if level[i] >= lvl)
        zmax_raw = depth_bot_m[deepest_layer_at_level]
        if not np.isfinite(zmax_raw):
            continue  # only the half-space would need this; it is pinned to level 0 above.
        h_coarse = h_base / (2 ** (lvl - 1))
        zmax = _snap_to_h(zmax_raw, h_coarse, round_zmax)
        zmax_values.append(zmax)
        refinement_lines.append(f"refinement zmax={zmax:.16g}")

    table = []
    for i in range(n):
        table.append({
            "layer": i + 1,
            "vs_km_s": float(vs_km_s[i]),
            "top_m": float(depth_top_m[i]),
            "bot_m": float(depth_bot_m[i]) if np.isfinite(depth_bot_m[i]) else None,
            "h_required_m": float(h_required[i]),
            "level_required": int(level_required[i]),
            "level_assigned": int(level[i]),
            "h_assigned_m": h_base / (2 ** int(level[i])),
        })

    return {
        "table": table,
        "refinement_lines": refinement_lines,
        "zmax_values": zmax_values,
        "levels": level.tolist(),
        "h_base": h_base,
        "fmax": float(fmax),
        "n_per_wavelength": float(n_per_wavelength),
    }


def _snap_to_h(value, h, mode):
    """Snap ``value`` to a multiple of ``h``, rounding ``"outward"`` or ``"nearest"``."""
    ratio = value / h
    snapped = math.ceil(ratio - 1.0e-9) if mode == "outward" else round(ratio)
    return snapped * h


def print_refinement_diagnostics(refinement):
    """Print the per-layer table and the resulting ``refinement`` lines.

    Inputs
    ------
    refinement : dict
        Output of :func:`compute_layer_refinement`.

    Returns
    -------
    None
    """
    print(SEPARATOR)
    print("SW4 vertical mesh refinement")
    print(SEPARATOR)
    print(f"h_base={refinement['h_base']:g} m  fmax={refinement['fmax']:g} Hz  "
          f"n_per_wavelength={refinement['n_per_wavelength']:g}")
    print(f"{'layer':>5} {'vs_km_s':>8} {'top_m':>10} {'bot_m':>10} "
          f"{'h_req_m':>9} {'level':>5} {'h_used_m':>9}")
    for row in refinement["table"]:
        bot = f"{row['bot_m']:.1f}" if row["bot_m"] is not None else "inf"
        print(f"{row['layer']:>5} {row['vs_km_s']:>8.3f} {row['top_m']:>10.1f} {bot:>10} "
              f"{row['h_required_m']:>9.2f} {row['level_assigned']:>5} {row['h_assigned_m']:>9.3f}")
    if refinement["refinement_lines"]:
        print("Refinement lines:")
        for line in refinement["refinement_lines"]:
            print(f"  {line}")
    else:
        print("No refinement needed: h_base already resolves every layer.")


def flat_layer_table(crust, h_base):
    """Per-layer table matching :func:`compute_layer_refinement`'s ``table``
    schema, but with a single uniform ``h_base`` everywhere (no refinement
    levels). Used to build a stratigraphy overlay when ``refine_fmax`` is
    not set, so the same downstream code works whether refinement is on
    or off.

    Inputs
    ------
    crust : CrustModel
    h_base : float
        Grid spacing (m) assigned to every layer.

    Returns
    -------
    list of dict
        Same per-row keys as ``compute_layer_refinement(...)["table"]``.
    """
    n = crust.nlayers
    depth_top_m = np.concatenate(([0.0], np.cumsum(crust.d[:-1]))) * 1000.0
    depth_bot_m = np.append(depth_top_m[1:], np.inf) if n > 1 else np.array([np.inf])
    table = []
    for i in range(n):
        table.append({
            "layer": i + 1,
            "vs_km_s": float(crust.b[i]),
            "top_m": float(depth_top_m[i]),
            "bot_m": float(depth_bot_m[i]) if np.isfinite(depth_bot_m[i]) else None,
            "h_required_m": float(h_base),
            "level_required": 0,
            "level_assigned": 0,
            "h_assigned_m": float(h_base),
        })
    return table


def topo_interpolator_from_grid(topo_points, nx, ny):
    """Build ``topo_fn(x, y) -> elevation`` from a rebuilt cartesian
    topography grid.

    Inputs
    ------
    topo_points : ndarray, shape (nx*ny, 3)
        Row-major grid (``for y in ys: for x in xs``), matching
        :func:`shakermaker.sw4_exporter.topography.rebuild_cartesian_topography`
        / ``extend_topography_to_domain``. The z column is real elevation
        (metres, +up) -- the same convention the exporter carries
        internally (see ``exporter.py``'s topography step; SW4-plot sign
        flips happen only in :mod:`geometry_plot`, not here).
    nx, ny : int
        Grid size along x and y.

    Returns
    -------
    callable
        ``topo_fn(x, y)``, vectorized over arrays of any matching shape.
    """
    from scipy.interpolate import RegularGridInterpolator

    pts = np.asarray(topo_points, dtype=float)
    x_unique = pts[:nx, 0]
    y_unique = pts[::nx, 1]
    z_grid = pts[:, 2].reshape(ny, nx)
    interp = RegularGridInterpolator((y_unique, x_unique), z_grid, bounds_error=False, fill_value=None)

    def topo_fn(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        query = np.column_stack([y.ravel(), x.ravel()])
        return interp(query).reshape(x.shape)

    return topo_fn


def stratigraphy_volume_points(crust, table, x_domain, y_domain, max_display_depth_m=None,
                                topo_fn=None, max_points_per_layer=300000):
    """Point cloud of real SW4 mesh nodes filling the full domain footprint,
    one crust layer at a time.

    Spans the whole ``[0, x_domain] x [0, y_domain]`` box -- the full cube
    for a flat (no-topography) export, or the full box beneath the real
    terrain when ``topo_fn`` follows it -- down to ``max_display_depth_m``
    (the real ``z_domain`` by default, i.e. the actual domain floor, not an
    arbitrary shallow cutoff). Uses the real per-layer node spacing
    (``h_assigned_m`` from ``table``, straight out of
    :func:`compute_layer_refinement` or :func:`flat_layer_table``) as long
    as that stays under ``max_points_per_layer``; a dense block at real
    spacing over a full SW4 domain -- down to its full depth -- is billions
    of points for the shallow, finely-refined layers, and can be just as
    large for a deep half-space slab once depth is no longer capped
    shallow. Once a layer's full-resolution point count would exceed the
    budget, x, y **and** z are decimated isotropically (`h` itself is
    unchanged -- only how many of those real nodes get drawn) just enough
    to fit it, so the budget holds regardless of which axis is large.

    Inputs
    ------
    crust : CrustModel
    table : list of dict
        ``compute_layer_refinement(...)["table"]`` or
        ``flat_layer_table(...)``.
    x_domain, y_domain : float
        Full SW4 box extents (m) -- same frame as ``topo_fn`` (SW4 local
        metres when called from the exporter).
    max_display_depth_m : float, optional
        Crop depth -- layers starting below this are skipped entirely
        (their ``h`` is still in ``table``, just not drawn). ``None``
        (default): no cap, uses each layer's own extent (the half-space
        is still unbounded below and needs a real value from the caller,
        e.g. ``z_domain``, to have any depth at all -- the exporter always
        supplies one).
    topo_fn : callable, optional
        ``topo_fn(x, y) -> elevation`` (m, +up). ``None`` -> flat surface
        at ``z=0``.
    max_points_per_layer : int
        Point budget per layer; triggers isotropic x/y/z decimation once
        the full-resolution grid for that layer would exceed it. Default
        ``300000``.

    Returns
    -------
    xyz : ndarray, shape (N, 3)
        Same sign convention as topography elsewhere in the exporter: z is
        real elevation minus depth (+up), not yet flipped for the SW4
        "+down" plot convention -- callers plotting this alongside
        ``geometry_plot`` topography should negate z the same way.
    colors_rgba : ndarray, shape (N, 4)
        Pastel1 palette, same as ``CrustModel.plot_profile()``.
    """
    import matplotlib.pyplot as plt

    if topo_fn is None:
        topo_fn = lambda x, y: np.zeros_like(x)

    colors = plt.cm.Pastel1(np.linspace(0, 1, crust.nlayers))

    xs, ys, zs, cs = [], [], [], []
    for row in table:
        top = row["top_m"]
        if max_display_depth_m is not None and top >= max_display_depth_m:
            continue  # below the display crop -- still in `table`, just not drawn

        h = row["h_assigned_m"]
        if row["bot_m"] is not None:
            bot = row["bot_m"] if max_display_depth_m is None else min(row["bot_m"], max_display_depth_m)
        else:
            if max_display_depth_m is None:
                raise ValueError(
                    "max_display_depth_m is required to bound the half-space layer "
                    "(it has no bot_m); the exporter always passes z_domain here."
                )
            bot = max_display_depth_m

        n_x_full = max(2, int(round(x_domain / h)) + 1)
        n_y_full = max(2, int(round(y_domain / h)) + 1)
        n_z_full = max(2, int(round((bot - top) / h)) + 1)

        stride = 1
        total_full = n_x_full * n_y_full * n_z_full
        if total_full > max_points_per_layer:
            stride = max(1, math.ceil((total_full / max_points_per_layer) ** (1.0 / 3.0)))
        n_x = max(2, n_x_full // stride)
        n_y = max(2, n_y_full // stride)
        n_z = max(2, n_z_full // stride)

        x = np.linspace(0.0, x_domain, n_x)
        y = np.linspace(0.0, y_domain, n_y)
        depth = np.linspace(top, bot, n_z)

        X, Y, D = np.meshgrid(x, y, depth, indexing="ij")
        Z = topo_fn(X, Y) - D  # elevation = local surface - depth below it

        xs.append(X.ravel()); ys.append(Y.ravel()); zs.append(Z.ravel())
        cs.append(np.tile(colors[row["layer"] - 1], (X.size, 1)))

    if not xs:
        return np.empty((0, 3)), np.empty((0, 4))
    xyz = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    return xyz, np.vstack(cs)


def stratigraphy_flat_cap_points(crust, table, x_domain, y_domain, z_min_real, z_max_real,
                                  topo_fn, max_points_per_layer=300000):
    """Fill the gap between a flat reference elevation and the real terrain
    with the shallowest crust layer's material -- the "cap" for
    ``stratigraphy_flat`` mode.

    Used together with :func:`stratigraphy_volume_points` called with a
    *constant* ``topo_fn`` (``lambda x, y: z_min_real``): that call gives a
    perfectly flat, horizontal layer stack anchored at ``z_min_real`` (the
    real terrain's lowest point, used as the flat model's own datum).
    Nothing in that flat stack exists above ``z_min_real``, so this
    function fills that region -- from ``z_min_real`` up to the real
    terrain surface at each ``(x, y)`` -- with the shallowest layer
    (``table[0]``) only, so the deep layers stay flat while the very top
    still shows the real relief.

    Builds a regular grid at the shallowest layer's node spacing
    (``table[0]["h_assigned_m"]``) over
    ``[0, x_domain] x [0, y_domain] x [z_min_real, z_max_real]`` -- the
    same isotropic point-budget decimation as
    :func:`stratigraphy_volume_points` -- then keeps only the nodes at or
    below the real surface (``z <= topo_fn(x, y)``); the resulting wedge is
    thin near the terrain's lowest point and thickest at its highest.

    Inputs
    ------
    crust : CrustModel
    table : list of dict
        ``compute_layer_refinement(...)["table"]`` or
        ``flat_layer_table(...)``. Only ``table[0]`` (the shallowest layer)
        is used.
    x_domain, y_domain : float
        Full SW4 box extents (m), same frame as ``topo_fn``.
    z_min_real, z_max_real : float
        Real terrain elevation range (m, +up) -- typically
        ``topo_points_sw4[:, 2].min()``/``.max()`` from the exporter, the
        same array ``topo_fn`` was built from.
    topo_fn : callable
        ``topo_fn(x, y) -> elevation`` (m, +up), the real (non-flat)
        terrain -- same interpolator used for the non-flat stratigraphy
        overlay.
    max_points_per_layer : int
        Point budget, same semantics as :func:`stratigraphy_volume_points`.
        Default ``300000``.

    Returns
    -------
    xyz : ndarray, shape (N, 3)
        Same sign convention as :func:`stratigraphy_volume_points`: z is
        real elevation (+up), not yet flipped for the SW4 "+down" plot
        convention.
    colors_rgba : ndarray, shape (N, 4)
        ``table[0]``'s Pastel1 colour, repeated for every point.
    """
    import matplotlib.pyplot as plt

    top_row = table[0]
    h = top_row["h_assigned_m"]
    color = plt.cm.Pastel1(np.linspace(0, 1, crust.nlayers))[top_row["layer"] - 1]

    n_x_full = max(2, int(round(x_domain / h)) + 1)
    n_y_full = max(2, int(round(y_domain / h)) + 1)
    n_z_full = max(2, int(round((z_max_real - z_min_real) / h)) + 1)

    stride = 1
    total_full = n_x_full * n_y_full * n_z_full
    if total_full > max_points_per_layer:
        stride = max(1, math.ceil((total_full / max_points_per_layer) ** (1.0 / 3.0)))
    n_x = max(2, n_x_full // stride)
    n_y = max(2, n_y_full // stride)
    n_z = max(2, n_z_full // stride)

    x = np.linspace(0.0, x_domain, n_x)
    y = np.linspace(0.0, y_domain, n_y)
    z = np.linspace(z_min_real, z_max_real, n_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    surface = topo_fn(X, Y)
    mask = Z <= surface

    xyz = np.column_stack([X[mask], Y[mask], Z[mask]])
    colors_rgba = np.tile(color, (xyz.shape[0], 1))
    return xyz, colors_rgba
