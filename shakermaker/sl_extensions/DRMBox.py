import numpy as np
from itertools import product
from shakermaker.station import Station
from shakermaker.stationlist import StationList

#Definition of the DRM planes
class Plane:
    """A helper class to define the planes of the sides of the DRM box.

    :param v0: Origin ofset in xyz coordinates.
    :type v0: numpy array (3,)
    :param v1: Direction vector of 1 local plane coordinate in xyz coordinates. (Magnitude is the length of the side).
    :type v1: numpy array (3,)
    :param v2: Direction vector of 2 local plane coordinate in xyz coordinates. (Magnitude is the length of the side).
    :type v2: numpy array (3,)
    :param k: Plane tag
    :type k: int
    :param internal: True if plane is represents an internal boundary of the DRM box.
    :type internal: bool
    :param xi1: Natural coordinate grid of the plane. Defines spacing of stations in the 1 direction. Should be monotonically increasing spanning the interval [0,1] (including the edges)
    :type xi1: numpy array (N1,)
    :param xi2: Natural coordinate grid of the plane. Defines spacing of stations in the 2 direction. Should be monotonically increasing spanning the interval [0,1] (including the edges)
    :type xi2: numpy array (N2,)

    Note:

    This class is intended for internal use by :class:`Receivers.DRMBox`.

    """

    def __init__(self, v0, v1, v2, k, internal, xi1, xi2):
        self._v0 = v0
        self._v1 = v1
        self._v2 = v2
        self._k = k
        self._internal = internal
        self._xi1 = xi1
        self._xi2 = xi2
        self._stations = np.empty((len(xi1), len(xi2)), dtype=np.int32)

    def set_station_id(self, i, j, station_id):
        """Set the station_id for the station at the (i,j) in the local plane of the grid.

        """
        self._stations[i, j] = station_id

    def get_info(self):
        return {'v0': self._v0,
                'v1': self._v1,
                'v2': self._v2,
                'internal': self._internal,
                'stations': self._stations,
                'xi1': self._xi1,
                'xi2': self._xi2}


class DRMBox(StationList):

    def __init__(self, pos, nelems, h, metadata={},azimuth=0.):
        StationList.__init__(self,[], metadata)

        self._x0 = np.array(pos)
        self._h = np.array(h)
        self._nelems = np.array(nelems)
        self._azimuth = azimuth

        self._planes = []
        self._tstart = np.infty
        self._tend = -np.infty
        self._dt = 0
        self._xmax = [-np.infty, -np.infty, -np.infty]
        self._xmin = [np.infty, np.infty, np.infty]

        self._create_DRM_stations()

    @property
    def nplanes(self):
        return len(self._planes)

    @property
    def planes(self):
        return self._planes

    def _new_station(self, x, internal, name=""):
        new_station = Station(x, {'id': self.nstations, 'name': name, 'internal': internal})

        self.add_station(new_station)

        self._xmax = [max(x[0], self._xmax[0]), max(x[1], self._xmax[1]), max(x[2], self._xmax[2])]
        self._xmin = [min(x[0], self._xmin[0]), min(x[1], self._xmin[1]), min(x[2], self._xmin[2])]

        return new_station

    def _new_DRM_plane(self, v0, v1, v2, xi, eta, internal):
        new_plane = Plane(v0, v1, v2, self.nplanes, internal, xi, eta)
        self._planes.append(new_plane)

        for i, j in product(np.arange(xi.size), np.arange(eta.size)):
            xi_, eta_ = xi[i], eta[j]
            p = v0 + xi_*v1 + eta_*v2
            new_station = self._new_station(p, '.{}.{}.{}'.format(i, j, self.nplanes - 1), internal)
            new_plane.set_station_id(i, j, new_station.metadata['id'])

    def _create_DRM_stations(self):
        #DRM box orientation (TODO: add azimuthal rotation)
        e1 = np.array([1.,0.,0.])
        e2 = np.array([0.,1.,0.])
        e3 = np.array([0.,0.,1.])

        #Inner boundary
        internal = True
        lx = self._nelems[0] * self._h[0]
        ly = self._nelems[1] * self._h[1]
        lz = self._nelems[2] * self._h[2]
        xi_x = np.linspace(0., 1., self._nelems[0]+1)
        xi_y = np.linspace(0., 1., self._nelems[1]+1)
        xi_z = np.linspace(0., 1., self._nelems[2]+1)

        v0_e1_args = [e1, e1, -e1, -e1]
        v0_e2_args = [-e2, e2, e2, -e2]
        v1_args = [e2, e2, -e1, -e1]
        xi2_args = [xi_y, xi_x[1:-1], xi_y, xi_x[1:-1]]
        for i in range(4):
            v0 = self._x0 + (lx/2)*v0_e1_args[i] + (ly/2)*v0_e2_args[i]
            v1, v2 = lz*e3, ly*v1_args[i]
            xi1, xi2 = xi_z, xi2_args[i]
            self._new_DRM_plane(v0, v1, v2, xi1, xi2, internal)

        v0 = self._x0 - (lx/2)*e1 - (ly/2)*e2 + lz*e3
        v1, v2 = lx*e1, ly*e2
        xi1, xi2 = xi_x[1:-1], xi_y[1:-1]
        self._new_DRM_plane(v0, v1, v2, xi1, xi2, internal)

        #Outer boundary
        internal = False
        lx = (self._nelems[0] + 2) * self._h[0]
        ly = (self._nelems[1] + 2) * self._h[1]
        lz = (self._nelems[2] + 1) * self._h[2]
        xi_x = np.linspace(0., 1., self._nelems[0]+3)
        xi_y = np.linspace(0., 1., self._nelems[1]+3)
        xi_z = np.linspace(0., 1., self._nelems[2]+2)

        xi2_args = [xi_y, xi_x[1:-1], xi_y, xi_x[1:-1]]
        for i in range(4):
            v0 = self._x0 + (lx/2)*v0_e1_args[i] + (ly/2)*v0_e2_args[i]
            v1, v2 = lz*e3, ly*v1_args[i]
            xi1, xi2 = xi_z, xi2_args[i]
            self._new_DRM_plane(v0, v1, v2, xi1, xi2, internal)

        v0 = self._x0 - (lx/2)*e1 - (ly/2)*e2 + lz*e3
        v1, v2 = lx*e1, ly*e2
        xi1, xi2 = xi_x[1:-1], xi_y[1:-1]
        self._new_DRM_plane(v0, v1, v2, xi1, xi2, internal)

StationList.register(DRMBox)
