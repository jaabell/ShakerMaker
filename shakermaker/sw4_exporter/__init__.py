"""SW4 export tooling for ShakerMaker models.

The public surface is small:

- :class:`SW4Exporter`        -- the orchestrator. Build it with a model and
                                 a config, call ``.write()``.
- :class:`SW4ExportConfig`    -- knob bag (paths, grid spacing, topography,
                                 receiver toggles, plotting flags).
- :func:`unpack_sw4_package_h5` -- inverse of the exporter: takes an HDF5
                                   transport package and recreates the SW4
                                   directory tree on disk.

See ``shakermaker/sw4_exporter/README.md`` for the layout of the produced
files, the HDF5 package schema and the coordinate convention.
"""

from .exporter import SW4Exporter
from .config import SW4ExportConfig
from .package_h5 import unpack_sw4_package_h5

__all__ = ["SW4Exporter", "SW4ExportConfig", "unpack_sw4_package_h5"]
