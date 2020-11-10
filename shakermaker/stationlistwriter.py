import abc

from shakermaker.station import Station
from shakermaker.stationlist import StationList


class StationListWriter(metaclass=abc.ABCMeta):

    def __init__(self, filename, transform_function=None):
        self._filename = filename
        self._transform_function = transform_function

    def write(self, station_list, num_samples):
        assert isinstance(station_list, StationList), \
            "StationListWriter.write - 'station_list' Should be subclass of StationList"

        self.initialize(station_list, num_samples)
        self.write_metadata(station_list.metadata)
        for station in enumerate(station_list):
            self.write_station(station)
        self.close()

    @abc.abstractmethod
    def initialize(self, station_list, num_samples):
        raise NotImplementedError('derived class must define method initialize')

    @abc.abstractmethod
    def write_station(self, station, index):
        raise NotImplementedError('derived class must define method _write_stations')

    @abc.abstractmethod
    def write_metadata(self, metadata):
        raise NotImplementedError('derived class must define method write_metadata')

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError('derived class must define method close')

    @property
    def filename(self):
        return self._filename

    @property
    def transform_function(self):
        return self._transform_function

    @transform_function.setter
    def transform_function(self, transform_function):
        self._transform_function = transform_function
