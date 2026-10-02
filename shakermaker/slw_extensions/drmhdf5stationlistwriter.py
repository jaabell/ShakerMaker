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


def _interpolate(told, yold, tnew):
    """Interpolate yold(told) onto tnew, clamping outside the sampled range."""
    return interp1d(told, yold,
                    fill_value=(yold[0], yold[-1]),
                    bounds_error=False)(tnew)


class DRMHDF5StationListWriter(HDF5StationListWriter):
    """HDF5 writer for DRM output (.h5drm format).

    Extends HDF5StationListWriter with DRM-specific groups and datasets:
    ``/DRM_Data``, ``/DRM_QA_Data``, ``/DRM_Metadata``.

    Supports two writing modes:

    - **'progressive'** (default / recommended): writes each station to disk
      immediately after ``write_station()`` is called.  Requires ``tmin``,
      ``tmax``, ``dt``.  RAM usage is O(1) per station -- essential for
      large DRM meshes (8 000+ stations).

    - **'legacy'**: accumulates all responses in RAM and writes at
      ``close()``.  Compatible with JAA workflows.

    Accepts DRMBox, SurfaceGrid, and PointCloudDRMReceiver station lists.
    """

    def __init__(self, filename):
        HDF5StationListWriter.__init__(self, filename)

        self._h5file = None

        # Legacy mode accumulators
        self._velocities = {}
        self._tstart = np.inf
        self._tend   = -np.inf
        self._dt     = 0.0

        # Progressive mode
        self._progressive_mode = False
        self._t_final = None

    # -------------------------------------------------------------------------
    # initialize
    # -------------------------------------------------------------------------

    def initialize(self, 
                    station_list, num_samples,
                    tmin=None, tmax=None, dt=None,
                    writer_mode='progressive'):
        """Initialize DRM HDF5 writer.

        Parameters
        ----------
        station_list : DRMBox | SurfaceGrid | PointCloudDRMReceiver
            DRM station collection.
        num_samples : int
            Number of time samples (used only in legacy mode).
        tmin, tmax, dt : float
            Time window parameters.  Required for progressive mode.
        writer_mode : str
            ``'progressive'`` (default) or ``'legacy'``.
        """
        from shakermaker.sl_extensions import DRMBox
        try:
            from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid
            _sg_type = SurfaceGrid
        except ImportError:
            _sg_type = None
        try:
            from shakermaker.sl_extensions.PointCloudDRMReceiver import PointCloudDRMReceiver
            _pc_type = PointCloudDRMReceiver
        except ImportError:
            _pc_type = None

        valid_types = [DRMBox]
        if _sg_type:  valid_types.append(_sg_type)
        if _pc_type:  valid_types.append(_pc_type)
        assert isinstance(station_list, tuple(valid_types)), (
            "DRMHDF5StationListWriter.initialize - 'station_list' should be "
            "DRMBox, SurfaceGrid or PointCloudDRMReceiver")

        valid_modes = ('legacy', 'progressive')
        if writer_mode not in valid_modes:
            raise ValueError(
                f"writer_mode='{writer_mode}' is invalid. "
                f"Valid options: {valid_modes}")

        if writer_mode == 'progressive' and (tmin is None or tmax is None or dt is None):
            raise ValueError(
                "Progressive mode requires tmin, tmax, and dt.\n"
                "Example: initialize(receivers, nsamples, "
                "tmin=0, tmax=100, dt=0.05, writer_mode='progressive')")

        if self._filename is None or self._filename == "":
            self._filename = "DRMcase.hdf5"

        self._h5file = h5py.File(self._filename, mode="w")

        # DRM-specific groups
        grp_drm_data    = self._h5file.create_group("/DRM_Data")
        grp_drm_qa_data = self._h5file.create_group("/DRM_QA_Data")
        self._h5file.create_group("/DRM_Metadata")

        # nstations excludes the QA station (always the last one)
        self.nstations    = station_list.nstations - 1
        self.station_list = station_list

        # Geometry datasets -- always known at init
        grp_drm_data.create_dataset("xyz",      (self.nstations, 3), dtype=np.double)
        grp_drm_data.create_dataset("internal", (self.nstations,),   dtype=bool)
        grp_drm_data.create_dataset(
            "data_location",
            data=np.arange(0, self.nstations, dtype=np.int32) * 3)
        grp_drm_qa_data.create_dataset("xyz", (1, 3), dtype=np.double)

        if writer_mode == 'progressive':
            self._progressive_mode = True
            self._dt      = dt
            self._tstart  = tmin
            self._tend    = tmax
            self._t_final = np.arange(tmin, tmax, dt)
            ns = len(self._t_final)
            grp_metadata = self._h5file['DRM_Metadata']

            meta_to_write = {
                "dt": self._dt,
                "tstart": self._tstart,
                "tend": self._tend,
                "nt": ns,
                "writer_mode": writer_mode,
            }
            for key, value in meta_to_write.items():
                if key not in grp_metadata:
                    grp_metadata.create_dataset(key, data=value)

            # Pre-allocate velocity / displacement / acceleration datasets.
            # chunks=(3, ns): one chunk per station for efficient row writes.
            for grp, nrows in [(grp_drm_data, self.nstations),
                               (grp_drm_qa_data, 1)]:
                grp.create_dataset("velocity",
                                   (3 * nrows, ns), dtype=np.double,
                                   chunks=(3, ns))
                grp.create_dataset("displacement",
                                   (3 * nrows, ns), dtype=np.double,
                                   chunks=(3, ns))
                grp.create_dataset("acceleration",
                                   (3 * nrows, ns), dtype=np.double,
                                   chunks=(3, ns))

        else:
            # Legacy: signal datasets created at close()
            self._progressive_mode = False

    # -------------------------------------------------------------------------
    # write_metadata
    # -------------------------------------------------------------------------

    def write_metadata(self, metadata):
        """Write simulation metadata to /DRM_Metadata."""
        assert self._h5file, \
            "DRMHDF5StationListWriter.write_metadata - uninitialized HDF5 file"

        grp_metadata = self._h5file['DRM_Metadata']

        metadata = dict(metadata)   # don't mutate caller's dict
        metadata["created_by"]   = "---"   # os.getlogin() fails on Slurm
        metadata["program_used"] = f"ShakerMaker version {shakermaker_version}"
        metadata["created_on"]   = datetime.datetime.now().strftime(
            "%d-%b-%Y (%H:%M:%S.%f)")

        for key, value in metadata.items():
            if key not in grp_metadata:
                grp_metadata.create_dataset(key, data=value)

    # -------------------------------------------------------------------------
    # write_station
    # -------------------------------------------------------------------------

    def write_station(self, station, index):
        """Write a single DRM station.

        The QA station (last in the list, name=='QA') is written to
        /DRM_QA_Data; all others go to /DRM_Data.
        """
        assert self._h5file, \
            "DRMHDF5StationListWriter.write_station - uninitialized HDF5 file"
        assert isinstance(station, Station), \
            "DRMHDF5StationListWriter.write_station - 'station' should be Station"

        zz, ee, nn, t = station.get_response()
        is_QA = (station.metadata.get("name") == "QA")

        if not is_QA and index < self.nstations:
            self._h5file['DRM_Data/xyz'][index, :]   = station.x
            self._h5file['DRM_Data/internal'][index] = station.is_internal
        else:
            self._h5file['DRM_QA_Data/xyz'][0, :] = station.x

        if self._progressive_mode:
            self._write_station_progressive_drm(index, zz, ee, nn, t, is_QA)
        else:
            self._velocities[index] = (zz, ee, nn, t, is_QA)
            self._tstart = min(t[0],  self._tstart)
            self._tend   = max(t[-1], self._tend)
            self._dt     = t[1] - t[0]

    # -------------------------------------------------------------------------
    # _write_station_progressive_drm  (internal)
    # -------------------------------------------------------------------------

    def _write_station_progressive_drm(self, index, zz, ee, nn, t, is_QA):
        """Interpolate, derive, integrate and write one DRM station."""

        t_final = self._t_final
        dt      = self._dt

        ve = _interpolate(t, ee, t_final)
        vn = _interpolate(t, nn, t_final)
        vz = _interpolate(t, zz, t_final)

        Nt = len(ve)
        ae = np.zeros(Nt); ae[1:] = (ve[1:] - ve[:-1]) / dt
        an = np.zeros(Nt); an[1:] = (vn[1:] - vn[:-1]) / dt
        az = np.zeros(Nt); az[1:] = (vz[1:] - vz[:-1]) / dt

        de = cumulative_trapezoid(ve, t_final, initial=0.)
        dn = cumulative_trapezoid(vn, t_final, initial=0.)
        dz = cumulative_trapezoid(vz, t_final, initial=0.)

        if not is_QA:
            row = 3 * index
            grp = self._h5file['DRM_Data']
        else:
            row = 0
            grp = self._h5file['DRM_QA_Data']

        grp['velocity'][row,     :] = ve
        grp['velocity'][row + 1, :] = vn
        grp['velocity'][row + 2, :] = vz
        grp['displacement'][row,     :] = de
        grp['displacement'][row + 1, :] = dn
        grp['displacement'][row + 2, :] = dz
        grp['acceleration'][row,     :] = ae
        grp['acceleration'][row + 1, :] = an
        grp['acceleration'][row + 2, :] = az

        self._h5file.flush()

    # -------------------------------------------------------------------------
    # close
    # -------------------------------------------------------------------------

    def close(self):
        """Finalize and close the DRM HDF5 file.

        Progressive: all data already written -- just close.
        Legacy: interpolate/integrate all accumulated data in batch.
        """
        if self._progressive_mode:
            print("[WRITER] Progressive mode: closing file (all data already written)")
            self._h5file.close()
            return

        # --- Legacy batch write ---
        t_final     = np.arange(self._tstart, self._tend + self._dt * 0.5, self._dt)
        num_samples = len(t_final)

        grp_drm_data    = self._h5file['DRM_Data/']
        grp_drm_qa_data = self._h5file['DRM_QA_Data/']

        for grp, nrows in [(grp_drm_data, self.nstations),
                           (grp_drm_qa_data, 1)]:
            grp.create_dataset("velocity",
                               (3 * nrows, num_samples), dtype=np.double,
                               chunks=(3, num_samples))
            grp.create_dataset("displacement",
                               (3 * nrows, num_samples), dtype=np.double,
                               chunks=(3, num_samples))
            grp.create_dataset("acceleration",
                               (3 * nrows, num_samples), dtype=np.double,
                               chunks=(3, num_samples))

            grp_metadata = self._h5file['DRM_Metadata']
            meta_to_write = {
                "dt": self._dt,
                "tstart": self._tstart,
                "tend": self._tend,
                "nt": num_samples,
                "writer_mode": "legacy",
            }

            for key, val in meta_to_write.items():
                if key not in grp_metadata:
                    grp_metadata.create_dataset(key, data=val)

        for index, (zz, ee, nn, t, is_QA) in self._velocities.items():
            ve = _interpolate(t, ee, t_final)
            vn = _interpolate(t, nn, t_final)
            vz = _interpolate(t, zz, t_final)

            dt = t_final[1] - t_final[0]
            Nt = len(ve)
            ae = np.zeros(Nt); ae[1:] = (ve[1:] - ve[:-1]) / dt
            an = np.zeros(Nt); an[1:] = (vn[1:] - vn[:-1]) / dt
            az = np.zeros(Nt); az[1:] = (vz[1:] - vz[:-1]) / dt

            de = cumulative_trapezoid(ve, t_final, initial=0.)
            dn = cumulative_trapezoid(vn, t_final, initial=0.)
            dz = cumulative_trapezoid(vz, t_final, initial=0.)

            if not is_QA:
                row = 3 * index
                grp = self._h5file['DRM_Data']
            else:
                row = 0
                grp = self._h5file['DRM_QA_Data']

            grp['velocity'][row,     :] = ve
            grp['velocity'][row + 1, :] = vn
            grp['velocity'][row + 2, :] = vz
            grp['displacement'][row,     :] = de
            grp['displacement'][row + 1, :] = dn
            grp['displacement'][row + 2, :] = dz
            grp['acceleration'][row,     :] = ae
            grp['acceleration'][row + 1, :] = an
            grp['acceleration'][row + 2, :] = az

        self._h5file.close()


HDF5StationListWriter.register(DRMHDF5StationListWriter)
