from shakermaker.slw_extensions.hdf5stationlistwriter import HDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.station import Station
from shakermaker.version import shakermaker_version
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import datetime
import os

class DRMHDF5StationListWriter(HDF5StationListWriter):

    def __init__(self, filename):
        HDF5StationListWriter.__init__(self, filename)

        self._h5file = None

        self._velocities = {}
        self._tstart = np.infty
        self._tend = -np.infty
        self._dt = 0.

    def initialize(self, station_list, num_samples):
        assert isinstance(station_list, DRMBox), \
            "DRMHDF5StationListWriter.initialize - 'station_list' Should be a DRMBox"

        # Form filename and create HDF5 dataset
        if self._filename is None or self._filename == "":
            self._filename = "DRMcase.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # Create groups
        grp_drm_data = self._h5file.create_group("/DRM_Data")
        grp_drm_qa_data = self._h5file.create_group("/DRM_QA_Data")
        self._h5file.create_group("/DRM_Metadata")
        self.nstations = station_list.nstations-1
        self.station_list = station_list
        # grp_drm_planes = self._h5file.create_group("/DRM_Planes")

        # # Write planes info
        # grp_drm_planes.create_dataset("nplanes", data=station_list.nplanes)

        # for i, plane in enumerate(station_list.planes):
        #     grp_this_plane = grp_drm_planes.create_group("plane_{0:02.0f}".format(i))
        #     for key, value in plane.get_info().items():
        #         grp_this_plane.create_dataset(key, data=value)

        # Create data
        # grp_drm_data.create_dataset("velocity", (3 * self.nstations, num_samples), dtype=np.double,
                                # chunks=(3, num_samples))
        grp_drm_data.create_dataset("xyz", (self.nstations, 3), dtype=np.double)
        grp_drm_data.create_dataset("internal", [self.nstations], dtype=np.bool)
        data_location = np.arange(0, self.nstations, dtype=np.int32) * 3
        grp_drm_data.create_dataset("data_location", data=data_location)

        grp_drm_qa_data.create_dataset("xyz", (1, 3), dtype=np.double)
        # self.station_list = station_list

    def write_metadata(self, metadata):
        assert self._h5file, "DRMHDF5StationListWriter.write_metadata uninitialized HDF5 file"

        grp_metadata = self._h5file['DRM_Metadata']


        #More metadata:

        metadata["created_by"] = "---"   # os.getlogin() produces an error on slurm
        metadata["program_used"] = f"ShakeMaker version {shakermaker_version}"
        metadata["created_on"] = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        for key, value in metadata.items():
            print(f"key = {key} {value}")
            grp_metadata.create_dataset(key, data=value)

    def write_station(self, station, index):
        assert self._h5file, "DRMHDF5StationListWriter.write_station uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "DRMHDF5StationListWriter.write_station 'station Should be subclass of Station"

        # velocity = self._h5file['DRM_Data/velocity']
        xyz = self._h5file['DRM_Data/xyz']
        xyz_QA = self._h5file['DRM_QA_Data/xyz']
        internal = self._h5file['DRM_Data/internal']

        zz, ee, nn, t = station.get_response()
        if self.transform_function:
            zz, ee, nn, t = self.transform_function(zz, ee, nn, t)

        if index < self.nstations:
            xyz[index, :] = station.x
            internal[index] = station.is_internal
        else:
            xyz_QA[0, :] = station.x

        is_QA = False
        if station.metadata["name"] == "QA":
            is_QA = True
        self._velocities[index] = (zz, ee, nn, t, is_QA)
        self._tstart = min(t[0],self._tstart)
        self._tend = max(t[-1],self._tend)
        self._dt = t[1] - t[0]





    def close(self):
        t_final = np.arange(self._tstart, self._tend, self._dt)
        num_samples = len(t_final)

        grp_drm_data = self._h5file['DRM_Data/']
        grp_drm_qa_data = self._h5file['DRM_QA_Data/']


        grp_drm_data.create_dataset("velocity", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_data.create_dataset("displacement", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_data.create_dataset("acceleration", (3 * self.nstations, num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("velocity", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("displacement", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_drm_qa_data.create_dataset("acceleration", (3 , num_samples), dtype=np.double, chunks=(3, num_samples))
        grp_metadata = self._h5file['DRM_Metadata']
        grp_metadata.create_dataset("dt", data=self._dt)
        grp_metadata.create_dataset("tstart", data=self._tstart)
        grp_metadata.create_dataset("tend", data=self._tend)


        def interpolatorfun(told, yold, tnew):
            return interp1d(told,yold,
                fill_value=(yold[0],yold[-1]),
                bounds_error=False)(tnew)


        velocity = self._h5file['DRM_Data/velocity']
        displacement = self._h5file['DRM_Data/displacement']
        acceleration = self._h5file['DRM_Data/acceleration']
        for index in self._velocities:
            ee, nn, zz, t, is_QA = self._velocities[index]
            ve = interpolatorfun(t,ee,t_final)
            vn = interpolatorfun(t,nn,t_final)
            vz = interpolatorfun(t,zz,t_final)
            ae = np.gradient(ve, t_final)
            an = np.gradient(vn, t_final)
            az = np.gradient(vz, t_final)
            de = cumulative_trapezoid(ve, t_final, initial=0.)
            dn = cumulative_trapezoid(vn, t_final, initial=0.)
            dz = cumulative_trapezoid(vz, t_final, initial=0.)
            if not is_QA:
                displacement[3 * index, :] = de
                displacement[3 * index + 1, :] = dn
                displacement[3 * index + 2, :] = dz
                velocity[3 * index, :] = ve
                velocity[3 * index + 1, :] = vn
                velocity[3 * index + 2, :] = vz
                acceleration[3 * index, :] = ae
                acceleration[3 * index + 1, :] = an
                acceleration[3 * index + 2, :] = az
            else:
                self._h5file['DRM_QA_Data/velocity'][0,:] = ve
                self._h5file['DRM_QA_Data/velocity'][1,:] = vn
                self._h5file['DRM_QA_Data/velocity'][2,:] = vz
                self._h5file['DRM_QA_Data/displacement'][0,:] = de
                self._h5file['DRM_QA_Data/displacement'][1,:] = dn
                self._h5file['DRM_QA_Data/displacement'][2,:] = dz
                self._h5file['DRM_QA_Data/acceleration'][0,:] = ae
                self._h5file['DRM_QA_Data/acceleration'][1,:] = an
                self._h5file['DRM_QA_Data/acceleration'][2,:] = az

        self._h5file.close()




HDF5StationListWriter.register(DRMHDF5StationListWriter)
