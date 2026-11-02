import abc
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as sig


def interpolator(told, yold, tnew):
    return interp1d(told, yold, bounds_error=False, fill_value=(yold[0], yold[-1]))(tnew)


# Fraction of a sample below which a placement is taken as integer, so that traces
# that fall on the output grid are added exactly as before.
_FRAC_EPS = 1e-6


def lagrange4_taps(mu):
    """Integer offset and taps that place a trace at a real sample offset mu.

    A trace x whose sample k falls at output index mu + k is placed as
    z = np.convolve(x, h) added from output index m0: output n gets
    sum_j w_j x[n-m-2+j], the 4-point (cubic) Lagrange interpolation of x at the
    real index n - mu. Returns (m0, h); h is None when mu is an integer within
    _FRAC_EPS, and then x itself is added from m0 = round(mu).
    """
    m = int(np.floor(mu))
    d = mu - m
    if d < _FRAC_EPS:
        return m, None
    if d > 1.0 - _FRAC_EPS:
        return m + 1, None
    f = 1.0 - d
    w = np.array([-f * (f - 1.0) * (f - 2.0) / 6.0,
                  (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0,
                  -(f + 1.0) * f * (f - 2.0) / 2.0,
                  (f + 1.0) * f * (f - 1.0) / 6.0])
    return m - 1, w[::-1].copy()


def add_shifted(buf, x, mu):
    """buf[n] += x at the real index n - mu (mu in samples), clipped to buf."""
    m0, h = lagrange4_taps(mu)
    if h is not None:
        x = np.convolve(x, h)
    k0 = max(0, -m0)
    j0 = m0 + k0
    nw = min(len(x) - k0, len(buf) - j0)
    if nw > 0:
        buf[j0:j0 + nw] += x[k0:k0 + nw]


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
    # Placement by whole samples (int(t[0]/dt)): it dropped the fraction of a sample
    # of every trace (0 to 1 dt, different per station) and ignored tmin. Kept
    # below for reference.
    def add_to_response(self, z, e, n, t, tmin=0, tmax=100.):
        """Add a trace whose first sample is at time t[0] to the station response.

        The trace is placed at its real time, fraction of a sample included, with the
        4-point Lagrange interpolator of add_shifted; a trace on the output grid
        (t[0] - tmin a whole number of samples) is added exactly.
        """
        dt = t[1] - t[0]
        if not self._initialized:
            self._t = np.arange(tmin, tmax, dt)
            self._z = 0 * self._t
            self._e = 0 * self._t
            self._n = 0 * self._t
            self._dt = dt
            self._tmin = t.min()
            self._tmax = t.max()
            self._initialized = True
        mu = (t[0] - self._t[0]) / dt
        add_shifted(self._z, np.asarray(z, dtype=np.float64), mu)
        add_shifted(self._e, np.asarray(e, dtype=np.float64), mu)
        add_shifted(self._n, np.asarray(n, dtype=np.float64), mu)
        self._notify(t)

    def _add_to_response_whole_samples(self, z, e, n, t, tmin=0, tmax=100.):
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
