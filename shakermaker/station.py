import abc

class StationObserver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def station_change(self, station, t):
        raise NotImplementedError('derived class must define method station_change')

class Station:

    def __init__(self, x, internal=False, metadata=None):
        self._x = x
        self._metadata = metadata
        self._observers = []
        self._internal = internal

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
        self._notify(t)

    def get_response(self, do_filter=True, interpolate=False, interpolate_t=None):
        pass

    def get_response_integral(self, ntimes, do_filter=True, interpolate=False, interpolate_t=None):
        pass

    def get_response_derivative(self, ntimes, do_filter, interpolate, interpolate_t):
        pass

    def attach(self, observer):
        assert isinstance(observer, StationObserver), \
            "Station.attach (Input error) - 'observer' Should be subclass of StationObserver"

        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def _notify(self, t):
        for obs in self._observers:
            obs.station_change(self, t)
