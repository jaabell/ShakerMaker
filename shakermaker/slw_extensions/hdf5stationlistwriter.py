from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.station import Station
import h5py
import numpy as np


class HDF5StationListWriter(StationListWriter):

    def __init__(self, filename):
        StationListWriter.__init__(filename)

        self._h5file = None

    def initialize(self, station_list, num_samples):
        assert isinstance(station_list, StationList), \
            "HDF5StationListWriter.initialize - 'station_list' Should be subclass of StationList"

        # Form filename and create HDF5 dataset
        if self._filename is None or self._filename == "":
            self._filename = "case.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # Create groups
        grp_data = self._h5file.create_group("/Data")
        self._h5file.create_group("/Metadata")

        # Create data
        grp_data.create_dataset("velocity", (3 * station_list.nstations, num_samples), dtype=np.double,
                                chunks=(3, num_samples))
        grp_data.create_dataset("xyz", (station_list.nstations, 3), dtype=np.double)
        grp_data.create_dataset("internal", [station_list.nstations], dtype=np.bool)

        data_location = np.arange(0, station_list.nstations, dtype=np.int32) * 3
        grp_data.create_dataset("data_location", data=data_location)

    def write_metadata(self, metadata):
        assert self._h5file, "HDF5StationListWriter.write_metadata uninitialized HDF5 file"

        grp_metadata = self._h5file['Metadata']
        for key, value in metadata.items():
            grp_metadata.create_dataset(key, data=value)

    def write_station(self, station, index):
        assert self._h5file, "HDF5StationListWriter.write_station uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "HDF5StationListWriter.write_station 'station Should be subclass of Station"

        velocity = self._h5file['Data/velocity']
        xyz = self._h5file['Data/xyz']
        internal = self._h5file['Data/internal']

        zz, ee, nn, t = station.get_response()

        velocity[3 * index, :] = ee
        velocity[3 * index + 1, :] = nn
        velocity[3 * index + 2, :] = zz
        xyz[index, :] = station.x
        internal[index] = station.is_internal()

    def close(self):
        assert self._h5file, "HDF5StationListWriter.close uninitialized HDF5 file"

        self._h5file.close()


StationListWriter.register(HDF5StationListWriter)
