"""Configuration object for the SW4 exporter.

:class:`SW4ExportConfig` is a thin dataclass that the user fills in and
passes to :class:`SW4Exporter`. It owns every knob the exporter needs:
output path, grid spacing, domain extents (optional, computed otherwise),
topography options, station selection toggles and the HDF5 package name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class SW4ExportConfig:
    """Knobs for :class:`SW4Exporter`.

    Inputs
    ------
    path : str or Path
        Root directory for the export. The exporter creates
        ``<path>/shakermakerexports/`` and ``<path>/sw4/`` underneath.
    h : float
        SW4 grid spacing in metres. Default ``50``.
    x_domain, y_domain, z_domain : float, optional
        SW4 box extents in metres. When ``None`` (per axis), the exporter
        sizes that axis automatically so the geometry clears the supergrid
        (laterally) and the box reaches below the deepest source and material
        interface (vertically). When given, the value is validated against the
        same clearance and rejected if it does not fit. All three may be
        ``None``.
    x_origin, y_origin, z_origin : float
        Position of the ShakerMaker origin in SW4 local metres. Filled in
        by the exporter; supplying it has no effect.
    tmax : float
        SW4 simulation duration in seconds. Default ``50``.
    m0 : float
        Seismic moment scale used in every SW4 ``source`` line. Default ``1``.
    size_domain : sequence of 3 floats, optional
        Shortcut for ``[x_domain, y_domain, z_domain]``. When given, splits
        into the three fields above in ``__post_init__``.
    fileio_path : str
        Directory (relative to the SW4 ``.in`` file) where SW4 will drop
        ``rec`` output files. Default ``"shakermaker2sw4_fileio"``.
    supergrid_gp : int
        Width of the SW4 supergrid layer in grid points. Default ``30``.
    supergrid_pad_gp : int
        Extra clearance, in grid points, kept between the active geometry and
        the inner edge of the supergrid when an axis is sized automatically
        (axis value left ``None``). The total auto clearance per side is
        ``(supergrid_gp + supergrid_pad_gp) * h``. Default ``10``.
    interface_blocks : bool
        Emit one thin effective-medium ``block`` per internal material
        interface (harmonic average of mu/lambda, arithmetic average of rho).
        Default ``True``.
    interface_block_delta : float
        Half-thickness in metres of each interface block (``z1=z_k-delta``,
        ``z2=z_k+delta``). Must be ``< h/2``. Default ``1.0``.
    station_prefix : str
        Filename prefix used in every ``file=`` field. Default ``"sf"``.
    topo_file : str or Path, optional
        Cartesian topography file in SW4 format. When ``None``, no
        topography is written.
    topo_zmax : float, optional
        Maximum topography elevation appended to the ``topography`` line.
    write_topography_z0_stations : bool
        Add receivers stacked from each topography node up to z=0.
        Default ``False``.
    shakermaker_stations : bool
        Emit one ``rec`` per ShakerMaker station at its true z. Default ``True``.
    shakermaker_stations_to_surface : bool
        Emit one ``rec`` per ShakerMaker station forced to ``depth=0``.
        Default ``False``.
    domain_sw4 : bool
        Emit a regular 3-D grid of receivers spanning the SW4 box.
        Default ``False``.
    domain_sw4_size : sequence of 3 floats, optional
        Shortcut for ``[domain_sw4_x, domain_sw4_y, domain_sw4_z]``.
    domain_sw4_x, domain_sw4_y, domain_sw4_z : float, optional
        Sub-box for the SW4 receiver grid, when smaller than the full box.
    plot_geometry : bool
        Open a PyVista viewer in ShakerMaker (georef) coordinates after the
        export. Default ``False``.
    plot_geometry_sw4 : bool
        Same viewer, but in SW4 local coordinates. Default ``False``.
    h5_export_name : str
        Filename of the transport HDF5 package inside
        ``shakermakerexports/``. Default ``"sw4_package.h5"``.
    """

    path: str | Path
    h: float = 50.0
    x_domain: Optional[float] = None
    y_domain: Optional[float] = None
    z_domain: Optional[float] = None
    x_origin: float = 0.0
    y_origin: float = 0.0
    z_origin: float = 0.0
    tmax: float = 50.0
    m0: float = 1.0
    size_domain: Optional[Sequence[float]] = None
    fileio_path: str = "shakermaker2sw4_fileio"
    supergrid_gp: int = 30
    supergrid_pad_gp: int = 10
    interface_blocks: bool = True
    interface_block_delta: float = 1.0
    station_prefix: str = "sf"
    topo_file: Optional[str | Path] = None
    topo_zmax: Optional[float] = None
    write_topography_z0_stations: bool = False
    shakermaker_stations: bool = True
    shakermaker_stations_to_surface: bool = False
    domain_sw4: bool = False
    domain_sw4_size: Optional[Sequence[float]] = None
    domain_sw4_x: Optional[float] = None
    domain_sw4_y: Optional[float] = None
    domain_sw4_z: Optional[float] = None
    plot_geometry: bool = False
    plot_geometry_sw4: bool = False
    h5_export_name: str = "sw4_package.h5"

    def __post_init__(self):
        # Expand the (x, y, z) shortcuts into the per-axis fields so the
        # rest of the exporter only deals with scalars.
        if self.size_domain is not None:
            self.x_domain, self.y_domain, self.z_domain = _as_xyz(self.size_domain, "size_domain")
        if self.domain_sw4_size is not None:
            self.domain_sw4_x, self.domain_sw4_y, self.domain_sw4_z = _as_xyz(
                self.domain_sw4_size, "domain_sw4_size")


def _as_xyz(values, name):
    """Validate a 3-element sequence and cast each entry to ``float`` or ``None``.

    Inputs
    ------
    values : sequence
        Must have exactly three entries.
    name : str
        Field name used in the error message.

    Returns
    -------
    tuple of 3 (float or None)
    """
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: [x, y, z].")
    return _optional_float(values[0]), _optional_float(values[1]), _optional_float(values[2])


def _optional_float(value):
    """Return ``None`` unchanged, otherwise cast to ``float``."""
    return None if value is None else float(value)
