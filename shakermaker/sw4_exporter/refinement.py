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
