import numpy as np
import pandas as pd
from shakermaker.station import Station
from shakermaker.stationlist import StationList


class PointCloudDRMReceiver(StationList):
    """DRM receiver built from an arbitrary FEM point cloud.

    Equivalent to DRMBox but instead of generating a regular grid,
    it reads station positions directly from a TSV file exported from
    the FEM model (e.g. STKO).

    Coordinates are transformed from the FEM system (mm, Z-up) to the
    ShakerMaker system (km, Z-down) using the fixed STKO convention::

        T     = [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, -1]]

        xyz_km = T^{-1} @ ((xyz_fem - x0_fem) * crd_scale) + drmbox_x0

    Parameters
    ----------
    point_cloud_file : str
        Tab-separated file with columns: Node_ID  X  Y  Z  Type
        X, Y, Z in FEM units (typically mm).
        Type must be 'internal' or 'external'.
    crd_scale : float
        Scale factor from FEM units to km. For mm -> km use 1/1e6.
    x0_fem : array-like (3,)
        Reference origin in FEM units (e.g. centre-top of the DRM box
        as defined in STKO).
    drmbox_x0 : array-like (3,)
        Centre of the DRM domain in ShakerMaker coordinates (km).
        This is the same value passed as ``pos`` to DRMBox.
    metadata : dict, optional
        Additional metadata forwarded to the StationList.

    Examples
    --------
    >>> drmreceiver = PointCloudDRMReceiver(
    ...     point_cloud_file = '_drm_nodes.txt',
    ...     crd_scale        = 1/1e6,
    ...     x0_fem           = [22000., 15500., 0.],
    ...     drmbox_x0        = [6302.625, 359.909, 0.],
    ...     metadata         = {"name": "H1_s0"})
    >>> model = ShakerMaker(crustal, fault, drmreceiver)
    >>> model.run_fast_faster_op(...)
    """

    def __init__(self, point_cloud_file, crd_scale, x0_fem, drmbox_x0,
                 metadata=None):
        if metadata is None:
            metadata = {}
        StationList.__init__(self, [], metadata)

        self._x0   = np.array(drmbox_x0, dtype=float)
        self._xmax = [-np.inf, -np.inf, -np.inf]
        self._xmin = [ np.inf,  np.inf,  np.inf]

        # Fixed STKO coordinate transform: FEM (mm, Z-up) -> ShakerMaker (km, Z-down)
        # LocalX_shaker = LocalY_FEM
        # LocalY_shaker = LocalX_FEM
        # LocalZ_shaker = -Z_FEM
        T     = np.array([[0., 1., 0.],
                          [1., 0., 0.],
                          [0., 0., -1.]])
        T_inv = np.linalg.inv(T)
        x0_fem = np.array(x0_fem, dtype=float)

        # Read point cloud
        df          = pd.read_csv(point_cloud_file, sep='\t')
        node_ids    = df['Node_ID'].values
        xyz_fem     = df[['X', 'Y', 'Z']].values.astype(float)
        is_internal = df['Type'].str.strip().str.lower() == 'internal'

        # Transform all coordinates at once
        xyz_km = (T_inv @ ((xyz_fem - x0_fem) * crd_scale).T).T + self._x0

        # Create one Station per node (identical pattern to DRMBox._new_station)
        for i in range(len(node_ids)):
            self._new_station(
                xyz_km[i],
                internal=bool(is_internal.iloc[i]),
                name=f'.cloud.{int(node_ids[i])}')

        # QA station at drmbox_x0, internal=True  (identical to DRMBox)
        self._new_station(self._x0, internal=True, name="QA")

        # Metadata identical to DRMBox._create_DRM_stations
        self.metadata["h"]           = np.array([0., 0., 0.])
        self.metadata["drmbox_x0"]   = self._x0
        self.metadata["drmbox_xmax"] = self._xmax[0]
        self.metadata["drmbox_ymax"] = self._xmax[1]
        self.metadata["drmbox_zmax"] = self._xmax[2]
        self.metadata["drmbox_xmin"] = self._xmin[0]
        self.metadata["drmbox_ymin"] = self._xmin[1]
        self.metadata["drmbox_zmin"] = self._xmin[2]

    def _new_station(self, x, internal, name=""):
        """Create and register a station, updating bounding box."""
        new_station = Station(
            x,
            internal=internal,
            metadata={'id': self.nstations, 'name': name, 'internal': internal})

        self.add_station(new_station)

        self._xmax = [max(x[0], self._xmax[0]),
                      max(x[1], self._xmax[1]),
                      max(x[2], self._xmax[2])]
        self._xmin = [min(x[0], self._xmin[0]),
                      min(x[1], self._xmin[1]),
                      min(x[2], self._xmin[2])]

        return new_station


StationList.register(PointCloudDRMReceiver)
