from shakermaker.slw_extensions.hdf5stationlistwriter import HDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.station import Station
import h5py
import numpy as np


class DRMHDF5StationListWriter(HDF5StationListWriter):

    def __init__(self, filename):
        HDF5StationListWriter.__init__(self, filename)

        self._h5file = None

    def initialize(self, station_list, num_samples):
        assert isinstance(station_list, DRMBox), \
            "DRMHDF5StationListWriter.initialize - 'station_list' Should be a DRMBox"

        # Form filename and create HDF5 dataset
        if self._filename is None or self._filename == "":
            self._filename = "DRMcase.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # Create groups
        grp_drm_data = self._h5file.create_group("/DRM_Data")
        self._h5file.create_group("/DRM_Metadata")
        grp_drm_planes = self._h5file.create_group("/DRM_Planes")

        # Write planes info
        grp_drm_planes.create_dataset("nplanes", data=station_list.nplanes)

        for i, plane in enumerate(station_list.planes):
            grp_this_plane = grp_drm_planes.create_group("plane_{0:02.0f}".format(i))
            for key, value in plane.get_info().items():
                grp_this_plane.create_dataset(key, data=value)

        # Create data
        grp_drm_data.create_dataset("velocity", (3 * station_list.nstations, num_samples), dtype=np.double,
                                chunks=(3, num_samples))
        grp_drm_data.create_dataset("xyz", (station_list.nstations, 3), dtype=np.double)
        grp_drm_data.create_dataset("internal", [station_list.nstations], dtype=np.bool)
        data_location = np.arange(0, station_list.nstations, dtype=np.int32) * 3
        grp_drm_data.create_dataset("data_location", data=data_location)

    def write_metadata(self, metadata):
        assert self._h5file, "DRMHDF5StationListWriter.write_metadata uninitialized HDF5 file"

        grp_metadata = self._h5file['DRM_Metadata']
        for key, value in metadata.items():
            print(f"key = {key} {value}")
            grp_metadata.create_dataset(key, data=value)

    def write_station(self, station, index):
        assert self._h5file, "DRMHDF5StationListWriter.write_station uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "DRMHDF5StationListWriter.write_station 'station Should be subclass of Station"

        velocity = self._h5file['DRM_Data/velocity']
        xyz = self._h5file['DRM_Data/xyz']
        internal = self._h5file['DRM_Data/internal']

        zz, ee, nn, t = station.get_response()
        if self.transform_function:
            zz, ee, nn, t = self.transform_function(zz, ee, nn, t)

        velocity[3 * index, :] = ee
        velocity[3 * index + 1, :] = nn
        velocity[3 * index + 2, :] = zz
        xyz[index, :] = station.x
        internal[index] = station.is_internal


HDF5StationListWriter.register(DRMHDF5StationListWriter)
