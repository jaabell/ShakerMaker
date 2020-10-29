import abc
import scipy as sp
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import scipy.signal as sig





def interpolator(told, yold, tnew):
    return interp1d(told, yold, bounds_error=False, fill_value=(yold[0], yold[-1]))(tnew)





class StationObserver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def station_change(self, station, t):
        raise NotImplementedError('derived class must define method station_change')

class Station:

    def __init__(self, x, internal=False, metadata={}):
        self._x = x
        self._metadata = metadata
        self._observers = []
        self._internal = internal
        self._initialized = False

    @property
    def x(self):
        return self._x

    @property
    def metadata(self):
        return self._metadata

    @property
    def is_internal(self):
        return self._internal

    def add_to_response(self, z, e, n, t):
        if not self._initialized:
            self._z = z
            self._e = e
            self._n = n
            self._t = t
            self._dt = t[1] - t[0]
            self._tmin = t.min()
            self._tmax = t.max()
            self._initialized = True
        else:
            dt = t[1] - t[0]
            tmin = min(self._tmin, t.min())
            tmax = max(self._tmax, t.max())
            if dt != self._dt:
                dt = max(dt, self._dt)
            tnew = sp.arange(tmin, tmax, dt)
            zz = interpolator(self._t, self._z, tnew)
            zz += interpolator(t, z, tnew)
            ee = interpolator(self._t, self._e, tnew)
            ee += interpolator(t, e, tnew)
            nn = interpolator(self._t, self._n, tnew)
            nn += interpolator(t, n, tnew)
            self._z = zz
            self._e = ee
            self._n = nn
            self._t = tnew
        self._notify(t)


    def get_response(self):
        """Return the recorded response of the station. 
        
        :param do_filter: Will/won't filter if filter parameters have been set. (Most useful to disable filtering before return)
        :type do_filter: bool
        :param interpolate: If ``True`` then will interpolate to a new time vector before filtering
        :type interpolate: bool
        :param interpolate_t: New time vector (its best if this vector spans or encompasses the old vector... otherwise artifacts will ensue)
        :type interpolate_t: numpy array (Nt,)
    
        :returns: Z (down), E (east), N (north), t (time) response of the station. 
        :retval: tuple containing numpt arrays with z, e, n, t reponse (shape (Nt,))

        Example::

            z,e,n,t = station.get_response()

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
