import numpy as np
from shakermaker.station import Station
from shakermaker.stationlist import StationList


class SurfaceGrid(StationList):
    """
    Surface grid of stations compatible with DRMBox interface.

    Parameters
    ----------
    x0 : array-like (3,)
        Center position [x, y, z] in km.
    nelems : array-like (3,)
        Number of elements [nx, ny, nz].
    h : float or array-like (3,)
        Spacing in km. Single value or [hx, hy, hz].
    mode : str
        'plane'  - single surface defined by plane_x, plane_y, or plane_z.
        'hollow' - boundary surfaces only (like DRM box).
        'filled' - full 3D grid.
    plane_x : float or None
        If set, creates a YZ plane at x = plane_x. Uses ny and nz.
    plane_y : float or None
        If set, creates a XZ plane at y = plane_y. Uses nx and nz.
    plane_z : float or None
        If set, creates a XY plane at z = plane_z. Uses nx and ny.
    metadata : dict
        Additional metadata.

    Examples
    --------
    >>> # XY plane at z=0
    >>> sg = SurfaceGrid([6,8,0], [20,20,10], dx, mode='plane', plane_z=0.0)
    >>> # XZ plane at y=8
    >>> sg = SurfaceGrid([6,8,0], [20,20,10], dx, mode='plane', plane_y=8.0)
    >>> # YZ plane at x=6
    >>> sg = SurfaceGrid([6,8,0], [20,20,10], dx, mode='plane', plane_x=6.0)
    >>> # Full 3D grid
    >>> sg = SurfaceGrid([6,8,0], [20,20,10], dx, mode='filled')
    """

    def __init__(self, x0, nelems, h, mode='plane',
                 plane_x=None, plane_y=None, plane_z=None,
                 metadata=None):
        if metadata is None:
            metadata = {}
        StationList.__init__(self, [], metadata)

        self._x0     = np.array(x0, dtype=float)
        self._nelems = np.array(nelems, dtype=int)
        self._h      = np.array([h, h, h] if np.isscalar(h) else h, dtype=float)
        self._mode   = mode

        self._plane_x = plane_x
        self._plane_y = plane_y
        self._plane_z = plane_z

        self._xmax = np.array([-np.inf, -np.inf, -np.inf])
        self._xmin = np.array([np.inf,   np.inf,  np.inf])

        # Validate plane arguments
        if mode == 'plane':
            n_defined = sum(p is not None for p in [plane_x, plane_y, plane_z])
            if n_defined == 0:
                raise ValueError(
                    "mode='plane' requires exactly one of plane_x, plane_y, or plane_z.")
            if n_defined > 1:
                raise ValueError(
                    "Only one of plane_x, plane_y, or plane_z can be set at a time.")
        else:
            if any(p is not None for p in [plane_x, plane_y, plane_z]):
                raise ValueError(
                    f"plane_x/y/z are only valid when mode='plane', got mode='{mode}'.")

        self._create_stations()

        # QA station at center (required for DRM compatibility)
        self._new_station(self._x0, internal=False, name="QA")

        self._save_metadata()

    # ------------------------------------------------------------------

    def _new_station(self, x, internal=False, name=""):
        """Create and add a new station."""
        x = np.array(x, dtype=float)
        station = Station(x, internal=internal,
                          metadata={'id': self.nstations, 'name': name,
                                    'internal': internal})
        self.add_station(station)
        self._xmax = np.maximum(self._xmax, x)
        self._xmin = np.minimum(self._xmin, x)
        return station

    def _create_stations(self):
        """Create stations based on mode."""
        nx, ny, nz = self._nelems
        hx, hy, hz = self._h

        lx, ly, lz = nx * hx, ny * hy, nz * hz

        # Origin centred on x0
        if self._mode == 'plane':
            if self._plane_z is not None:
                # XY plane — z is fixed, center x and y
                origin = self._x0 - np.array([lx/2, ly/2, 0])
            elif self._plane_y is not None:
                # XZ plane — center x, z starts from x0[2]
                origin = self._x0 - np.array([lx/2, 0, 0])
            elif self._plane_x is not None:
                # YZ plane — center y, z starts from x0[2]
                origin = self._x0 - np.array([0, ly/2, 0])
        else:
            origin = self._x0 - np.array([lx/2, ly/2, 0])


        if self._mode == 'plane':
            self._create_plane(origin, nx, ny, nz, hx, hy, hz)
        elif self._mode == 'filled':
            self._create_filled(origin, nx, ny, nz, hx, hy, hz)
        elif self._mode == 'hollow':
            self._create_hollow(origin, nx, ny, nz, hx, hy, hz, lx, ly, lz)
        else:
            raise ValueError(
                f"mode='{self._mode}' not recognized. Use 'plane', 'filled', or 'hollow'.")

    def _create_plane(self, origin, nx, ny, nz, hx, hy, hz):
        """Create a single plane of stations."""

        if self._plane_z is not None:
            # XY plane — fixed z
            for i in range(nx + 1):
                for j in range(ny + 1):
                    pos = np.array([origin[0] + i * hx,
                                    origin[1] + j * hy,
                                    self._plane_z])
                    self._new_station(pos, internal=False, name=f".{i}.{j}.0")

        elif self._plane_y is not None:
            # XZ plane — fixed y
            for i in range(nx + 1):
                for k in range(nz + 1):
                    pos = np.array([origin[0] + i * hx,
                                    self._plane_y,
                                    origin[2] + k * hz])
                    self._new_station(pos, internal=False, name=f".{i}.0.{k}")

        elif self._plane_x is not None:
            # YZ plane — fixed x
            for j in range(ny + 1):
                for k in range(nz + 1):
                    pos = np.array([self._plane_x,
                                    origin[1] + j * hy,
                                    origin[2] + k * hz])
                    self._new_station(pos, internal=False, name=f".0.{j}.{k}")

    def _create_filled(self, origin, nx, ny, nz, hx, hy, hz):
        """Full 3D grid."""
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    pos = origin + np.array([i * hx, j * hy, k * hz])
                    self._new_station(pos, internal=False, name=f".{i}.{j}.{k}")

    def _create_hollow(self, origin, nx, ny, nz, hx, hy, hz, lx, ly, lz):
        """Boundary surfaces only."""
        # Bottom face (z = origin[2])
        for i in range(nx + 1):
            for j in range(ny + 1):
                pos = origin + np.array([i * hx, j * hy, 0])
                self._new_station(pos, internal=False, name=f".{i}.{j}.bot")

        if nz > 0:
            # Top face (z = origin[2] + lz)
            for i in range(nx + 1):
                for j in range(ny + 1):
                    pos = origin + np.array([i * hx, j * hy, lz])
                    self._new_station(pos, internal=False, name=f".{i}.{j}.top")

            # Side faces (excluding edges already created)
            for k in range(1, nz):
                z = k * hz
                # Front (y=0) and Back (y=ly)
                for i in range(nx + 1):
                    self._new_station(origin + np.array([i * hx, 0,  z]), False, f".{i}.front.{k}")
                    self._new_station(origin + np.array([i * hx, ly, z]), False, f".{i}.back.{k}")
                # Left (x=0) and Right (x=lx) — excluding corners
                for j in range(1, ny):
                    self._new_station(origin + np.array([0,  j * hy, z]), False, f".left.{j}.{k}")
                    self._new_station(origin + np.array([lx, j * hy, z]), False, f".right.{j}.{k}")

    def _save_metadata(self):
        """Save metadata compatible with DRMBox.

        The bounding-box keys use the ``drmbox_*`` prefix (not
        ``surfacegrid_*``) so that consumers such as
        :meth:`ShakerMaker.export_drm_geometry`, which look up ``drmbox_*``
        keys, capture the SurfaceGrid geometry the same way they do for a
        DRMBox. The grid-specific ``nelems`` and ``mode`` keep the
        ``surfacegrid_*`` prefix because they have no DRMBox counterpart.
        """
        self.metadata["h"]                    = self._h
        self.metadata["drmbox_x0"]            = self._x0
        self.metadata["surfacegrid_nelems"]   = self._nelems
        self.metadata["surfacegrid_mode"]     = self._mode
        self.metadata["drmbox_xmax"]          = self._xmax[0]
        self.metadata["drmbox_ymax"]          = self._xmax[1]
        self.metadata["drmbox_zmax"]          = self._xmax[2]
        self.metadata["drmbox_xmin"]          = self._xmin[0]
        self.metadata["drmbox_ymin"]          = self._xmin[1]
        self.metadata["drmbox_zmin"]          = self._xmin[2]


StationList.register(SurfaceGrid)