import abc

class StationObserver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def station_change(self, station, t):
        raise NotImplementedError('derived class must define method station_change')

class Station:

    def __init__(self, x, metadata=None):
        self._x = x
        self._metadata = metadata
        self._observers = []

    @property
    def x(self):
        return self._x

    @property
    def metadata(self):
        return self._metadata

    def add_to_resonse(self, z, e, n, t):
        self._notify(t)

    def get_response(self, do_filter, interpolate, interpolate_t):
        pass

    def get_response_integral(self, ntimes, do_filter, interpolate, interpolate_t):
        pass

    def get_response_derivative(self, ntimes, do_filter, interpolate, interpolate_t):
        pass

    def attach(self, observer):
        assert isinstance(observer, StationObserver)
            , "Station.attach (Input error) - 'observer' Should be subclass of StationObserver"

        self._observers.append(observer)

    def dettach(self, observer):
        self._observers.remove(observer)

    def _notify(self, t):
        for obs in self._observers:
            obs.station_change(self, t)
