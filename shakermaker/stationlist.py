from shakermaker import Station, StationObserver

class StationList(StationObserver):

    def __init__(self, stations, metadata):
        self._stations = []
        for station in stations:
            self.add_station(station)

        self._metadata = metadata
        self._is_finalized = False

    def __iter__(self):
        return self._stations.__iter__()

    def add_station(self, station):
        assert isinstance(station, Station)
            , "StationList.add_station - 'station' Should be subclass of Station"

        self._stations.append(station)
        station.attach(self)

    @property
    def metadata(self):
        return self._metadata

    @property
    def nstations(self):
        return len(self._stations)

    @property
    def is_finalized(self):
        return self._is_finalized

    def station_change(self, station, t):
        pass

    def finalize(self):
        self._is_finalized = True

StationObserver.register(StationList)
