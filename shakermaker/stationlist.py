from typing import List, Dict
from shakermaker.station import Station, StationObserver

class StationList(StationObserver):
    """This is a list of stations.

    Parameters
    ----------
    stations : list
        A list of Stations
    metadata : dict
        Metadata to store with the station list

    Example
    -------
    sta1 = Station([20,20,0])
    sta2 = Station([20,40,0])
    stations = StationList([sta1, sta2], {})
    """

    def __init__(self, stations: List[Station], metadata: Dict):
        self._stations = []
        for station in stations:
            self.add_station(station)

        self._metadata = metadata
        self._is_finalized = False

    def __iter__(self):
        return iter(self._stations)

    def add_station(self, station: Station):
        """Add a station to the list."""
        assert isinstance(station, Station), "StationList.add_station - 'station' Should be subclass of Station"
        self._stations.append(station)
        station.attach(self)

    @property
    def metadata(self):
        """Get metadata."""
        return self._metadata

    @property
    def nstations(self):
        """Get the number of stations."""
        return len(self._stations)

    @property
    def is_finalized(self):
        """Check if the station list is finalized."""
        return self._is_finalized

    def get_station_by_id(self, id: int) -> Station:
        """Get a station by its id.

        Parameters
        ----------
        id : int
            The station id.

        Returns
        -------
        Station
            The station at the specified id.

        Raises
        ------
        IndexError
            If the id is out of range.
        """
        if id >= len(self._stations) or id < 0:
            raise IndexError("Station index out of range.")
        return self._stations[id]

    def station_change(self, station: Station, t: float):
        """Handle station changes."""
        pass

    def finalize(self):
        """Finalize the station list."""
        self._is_finalized = True


StationObserver.register(StationList)
