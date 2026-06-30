import abc
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as sig


def interpolator(told, yold, tnew):
    return interp1d(told, yold, bounds_error=False, fill_value=(yold[0], yold[-1]))(tnew)


class StationObserver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def station_change(self, station, t):
        raise NotImplementedError('derived class must define method station_change')


class Station:
    """This simple receiver stores response in memory.

    Internally, numpy arrays are used for storage. Optional parameters allow
    filtering of the response before outputting, although it is always stored
    raw (unfiltered), therefore the user can experiment with different
    filtering settings.

    :param x: xyz location of the station.
    :type x: numpy array (3,)
    :param metadata: metadata to store with the station
    :type dict: python dictionary
    """

    def __init__(self, x=None, internal=False, metadata=None):
        self._x = x
        self._metadata = metadata if metadata is not None else {}
        self._observers = []
        self._internal = internal
        self._initialized = False
        self._greens_functions = {}  # green functions storage (opt-in via metadata save_gf=True)

    @property
    def x(self):
        return self._x

    @property
    def metadata(self):
        return self._metadata

    @property
    def is_internal(self):
        return self._internal

    # def add_to_response(self, z, e, n, t, tmin=0, tmax=100.):
    #     if not self._initialized:
    #         dt = t[1] - t[0]
    #         self._t = np.arange(tmin, tmax, dt)
    #         self._z = 0 * self._t
    #         self._e = 0 * self._t
    #         self._n = 0 * self._t
    #         self._dt = t[1] - t[0]
    #         self._tmin = t.min()
    #         self._tmax = t.max()
    #         self._initialized = True
    #         nskip = int(t[0] / self._dt)
    #         # Bounds-check: avoid IndexError when signal extends past array end
    #         nwrite = min(len(z), len(self._z) - nskip)
    #         if nwrite > 0:
    #             self._z[nskip:(nskip + nwrite)] = z[:nwrite]
    #             self._e[nskip:(nskip + nwrite)] = e[:nwrite]
    #             self._n[nskip:(nskip + nwrite)] = n[:nwrite]
    #     else:
    #         dt = t[1] - t[0]
    #         nskip = int(t[0] / dt)
    #         # Bounds-check: avoid IndexError when signal extends past array end
    #         nwrite = min(len(z), len(self._z) - nskip)
    #         if nwrite > 0:
    #             self._z[nskip:(nskip + nwrite)] += z[:nwrite]
    #             self._e[nskip:(nskip + nwrite)] += e[:nwrite]
    #             self._n[nskip:(nskip + nwrite)] += n[:nwrite]
    #     self._notify(t)
    def add_to_response(self, z, e, n, t, tmin=0, tmax=100.):
        if not self._initialized:
            dt = t[1] - t[0]
            self._t = np.arange(tmin, tmax, dt)
            self._z = 0 * self._t
            self._e = 0 * self._t
            self._n = 0 * self._t
            self._dt = dt
            self._tmin = t.min()
            self._tmax = t.max()
            self._initialized = True

            if t[0] >= 0:
                nskip_buf = int(t[0] / dt)
                nskip_sig = 0
            else:
                nskip_buf = 0
                nskip_sig = int(-t[0] / dt)

            nwrite = min(len(z) - nskip_sig, len(self._t) - nskip_buf)
            if nwrite > 0:
                self._z[nskip_buf:(nskip_buf + nwrite)] = z[nskip_sig:(nskip_sig + nwrite)]
                self._e[nskip_buf:(nskip_buf + nwrite)] = e[nskip_sig:(nskip_sig + nwrite)]
                self._n[nskip_buf:(nskip_buf + nwrite)] = n[nskip_sig:(nskip_sig + nwrite)]
        else:
            dt = t[1] - t[0]

            if t[0] >= 0:
                nskip_buf = int(t[0] / dt)
                nskip_sig = 0
            else:
                nskip_buf = 0
                nskip_sig = int(-t[0] / dt)

            nwrite = min(len(z) - nskip_sig, len(self._t) - nskip_buf)
            if nwrite > 0:
                self._z[nskip_buf:(nskip_buf + nwrite)] += z[nskip_sig:(nskip_sig + nwrite)]
                self._e[nskip_buf:(nskip_buf + nwrite)] += e[nskip_sig:(nskip_sig + nwrite)]
                self._n[nskip_buf:(nskip_buf + nwrite)] += n[nskip_sig:(nskip_sig + nwrite)]
        self._notify(t)

    def get_response(self):
        """Return the recorded response of the station.

        :returns: Z (down), E (east), N (north), t (time) response.
        :retval: tuple of numpy arrays (z, e, n, t) with shape (Nt,)

        Example::

            z, e, n, t = station.get_response()

        """
        return self._z, self._e, self._n, self._t

    def attach(self, observer):
        assert isinstance(observer, StationObserver), \
            "Station.attach (Input error) - 'observer' Should be subclass of StationObserver"
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def _notify(self, t):
        for obs in self._observers:
            obs.station_change(self, t)

    def __str__(self):
        return (f"Station @ {self._x}\n"
                f"  Internal: {self._internal}\n"
                f"  Initialized: {self._initialized}\n"
                f"  metadata: {self._metadata}")

    # -------------------------------------------------------------------------
    # Progressive-write support  (PXP)
    # -------------------------------------------------------------------------

    def clear_response(self):
        """Release response arrays from memory after progressive write.

        Called by run_fast (Stage 2) in progressive writer_mode to keep
        RAM usage constant regardless of the number of stations.
        After clear_response() the station can be re-initialized by the
        next add_to_response() call.
        """
        self._z = None
        self._e = None
        self._n = None
        self._t = None
        self._initialized = False

    def add_greens_function(self, z, e, n, t, tdata, t0, subfault_id):
        """Store a Green's function for this station (opt-in).

        Only stores if ``save_gf=True`` is present in the station's metadata.
        Used for debugging and validation workflows.

        :param subfault_id: index of the point source / subfault
        :type subfault_id: int
        """
        if not self._metadata.get('save_gf', False):
            return
        self._greens_functions[subfault_id] = (
            z.copy(), e.copy(), n.copy(), t.copy(), tdata.copy(), t0
        )

    def get_greens_functions(self):
        """Return stored Green's functions dict (subfault_id -> tuple)."""
        return self._greens_functions

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, npzfilename):
        """Save the state of the station to an .npz file.

        :param npzfilename: Filename to save into.
        :type npzfilename: string

        Example::

            station.save("station_results.npz")

        """
        savedict = {}
        savedict["_x"] = self._x
        savedict["_metadata"] = self._metadata
        savedict["_observers"] = self._observers
        savedict["_internal"] = self._internal
        savedict["_initialized"] = self._initialized

        if self._initialized:
            savedict["_z"] = self._z
            savedict["_e"] = self._e
            savedict["_n"] = self._n
            savedict["_t"] = self._t
            savedict["_dt"] = self._dt
            savedict["_tmin"] = self._tmin
            savedict["_tmax"] = self._tmax

        np.savez(npzfilename, **savedict)
        return

    def load(self, npzfilename):
        """Load the state of a station from an .npz file.

        :param npzfilename: Filename to load from.
        :type npzfilename: string

        Example::

            station = Station()
            station.load("station_results.npz")

        """
        print(f"Loading station data from npzfilename={npzfilename}")
        loaddict = np.load(npzfilename, allow_pickle=True)
        print(f"Data={loaddict}")

        self._x = loaddict["_x"]
        self._metadata = loaddict["_metadata"][()]  # extract dict from numpy array
        self._observers = loaddict["_observers"]
        self._internal = loaddict["_internal"]
        self._initialized = loaddict["_initialized"]

        if self._initialized:
            self._z = loaddict["_z"]
            self._e = loaddict["_e"]
            self._n = loaddict["_n"]
            self._t = loaddict["_t"]
            self._dt = loaddict["_dt"]
            self._tmin = loaddict["_tmin"]
            self._tmax = loaddict["_tmax"]

        return
