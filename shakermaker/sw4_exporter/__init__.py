"""SW4 export tooling for ShakerMaker models.

The public surface is small:

- :class:`SW4Exporter`        -- the orchestrator. Build it with a model and
                                 a config, call ``.write()``.
- :class:`SW4ExportConfig`    -- knob bag (paths, grid spacing, topography,
                                 receiver toggles, plotting flags).
- :func:`unpack_sw4_package_h5` -- inverse of the exporter: takes an HDF5
                                   transport package and recreates the SW4
                                   directory tree on disk.
- :func:`compute_layer_refinement` -- derives SW4 ``refinement zmax=...``
                                       lines from a crust model; used
                                       internally when ``refine_fmax`` is set,
                                       also usable standalone to inspect the
                                       per-layer table before exporting.
- :func:`build_h5drm_from_sw4_case` -- the other direction: turns an SW4
                                        case's result ``.txt`` files back
                                        into an ``.h5drm`` for a local
                                        OpenSees/DRM model. A standalone
                                        copy of this is also written to
                                        ``shakermakerexports/`` as
                                        ``build_h5drm_from_sw4.py`` at
                                        export time.

See ``shakermaker/sw4_exporter/README.md`` for the layout of the produced
files, the HDF5 package schema and the coordinate convention.
"""

from .exporter import SW4Exporter
from .config import SW4ExportConfig
from .package_h5 import unpack_sw4_package_h5
from .refinement import compute_layer_refinement
from .h5drm_from_sw4 import build_h5drm_from_sw4_case

__all__ = [
    "SW4Exporter",
    "SW4ExportConfig",
    "unpack_sw4_package_h5",
    "compute_layer_refinement",
    "build_h5drm_from_sw4_case",
]
