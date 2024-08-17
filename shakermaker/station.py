import abc
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
# from scipy.integrate import cumtrapz
import scipy.signal as sig


def interpolator(told, yold, tnew):
    return interp1d(told, yold, bounds_error=False, fill_value=(yold[0], yold[-1]))(tnew)


class StationObserver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def station_change(self, station, t):
        raise NotImplementedError('derived class must define method station_change')


class Station:
    """This simple receiver stores response in memory. 

    Internally, numpy arrays are used for storage. Optional parameters allow filtering
    of the response before outputting, althoiugh it is always stored
    raw (unfiltered), therefore the user can experiment with
    different filtering settings.

    :param x: xyz location of the station.
    :type x: numpy array (3,)
    :param metadata: metadata to store with the station
    :type dict: python dictionary

    """
    def __init__(self, x=None, internal=False, metadata={}):
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

    def add_to_response(self, z, e, n, t, tmin=0, tmax=100.):
        if not self._initialized:
            # print(f"Station {self} initializing")
            # self._z = z
            # self._e = e
            # self._n = n
            # self._t = t
            # self._dt = t[1] - t[0]
            # self._tmin = t.min()
            # self._tmax = t.max()
            # self._initialized = True

            dt = t[1] - t[0]
            self._t = np.arange(tmin,tmax,dt)
            self._z = 0*self._t
            self._e = 0*self._t
            self._n = 0*self._t
            self._dt = t[1] - t[0]
            self._tmin = t.min()
            self._tmax = t.max()
            self._initialized = True
            nskip = int(t[0]/self._dt)
            # print(f" --> {t[0]=} {dt=} {nskip=} {tmin=} {tmax=}")
            self._z[nskip:(nskip+len(z))] = z
            self._e[nskip:(nskip+len(e))] = e
            self._n[nskip:(nskip+len(n))] = n
            # print(f"{self._tmin=} {self._tmax=} {self._dt=}")
        else:
            # print(f"Station {self} interpolating!")
            dt = t[1] - t[0]
            # tmin = min(self._tmin, t.min())
            # tmax = max(self._tmax, t.max())
            # # if dt != self._dt:
            # #     dt = max(dt, self._dt)
            # print(f"{self._tmin=} {self._tmax=} {self._dt=}")
            # tnew = sp.arange(tmin, tmax, dt)
            # zz = interpolator(self._t, self._z, tnew)
            # zz += interpolator(t, z, tnew)
            # ee = interpolator(self._t, self._e, tnew)
            # ee += interpolator(t, e, tnew)
            # nn = interpolator(self._t, self._n, tnew)
            # nn += interpolator(t, n, tnew)
            # self._z = zz
            # self._e = ee
            # self._n = nn
            # self._t = tnew
            nskip = int(t[0]/dt)
            # print(f" ++> {t[0]=} {dt=} {nskip=} ")
            self._z[nskip:(nskip+len(z))] += z
            self._e[nskip:(nskip+len(e))] += e
            self._n[nskip:(nskip+len(n))] += n
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

    def __str__(self):
        return f"""Station @ {self._x} 
        Internal: {self._internal} 
        Initialized: {self._initialized} 
        metadata: {self._metadata}"""


    def save(self, npzfilename):
        """Save the state of the station to an .npz file. 
        
        :param npzfilename: String with the name of the file to save into. 
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
        
        :param npzfilename: String with the name of the file to load from. 
        :type npzfilename: string

        Example::

            station = Station()  #creates an empty station
            station.load("station_results.npz")

        """

        print(f"Loading station data from npzfilename={npzfilename}")
        loaddict = np.load(npzfilename, allow_pickle=True)

        print(f"Data={loaddict}")



        self._x = loaddict["_x"] 
        self._metadata = loaddict["_metadata"][()]  #Extract the dict from the numpy array
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
