"""FFSP finite-fault stochastic source model.

``FFSPSource`` wraps the FFSP stochastic-rupture kernel (Pengcheng Liu, (c) 2005;
modifications by Chen Ji, 2020) and exposes it as a ShakerMaker source. It generates
physically-admissible stochastic slip / slip-rate distributions on a fault plane from a
deterministic skeleton (magnitude, geometry, mechanism, hypocentre, spectrum) plus random
fields, and produces an ensemble of realizations.

What it provides
----------------
- Generation: ``run`` computes an ensemble of realizations; ``get_realization``,
  ``set_active_realization`` and ``get_subfaults`` inspect them.
- Persistence:
  - HDF5: ``write_hdf5`` / ``load_hdf5`` / classmethod ``from_hdf5``.
  - Native FFSP text format: ``write_ffsp_format`` / ``load_ffsp_format`` /
    classmethod ``from_ffsp_format`` (with the ``_load_*`` / ``_parse_ffsp_inp`` helpers).
- Plotting (matplotlib, imported lazily): ``plot_histogram``, ``plot_spacial_distribution``,
  ``plot_rupture_snapshot``, ``plot_quality_metrics``, ``plot_temporal_metrics``,
  ``plot_spectral_comparison``, ``plot_source_time_function``, ``plot_crust_layers``,
  ``create_animation``.
- MPI-aware: ensemble generation is distributed across MPI ranks.

See ``docs/web/guides/ffsp.md`` for the user guide and ``FFSP_CITATION`` for attribution.
"""
import os
import numpy as np
from typing import Optional, Dict, List
from .crustmodel import CrustModel
from .version import shakermaker_version

FFSP_CITATION = (
    "FFSP finite-fault source - (c) 2005 Pengcheng Liu; modifications by Chen Ji "
    "(2020); based on Liu et al. (2006). Code obtained from the original authors "
    "and adapted for use within ShakerMaker."
)

FFSP_HDF5_SCHEMA_VERSION = "2.0"

#: Magnitude-to-moment constant used internally by the FFSP kernel
#: (``ffsp_wrapper.f90``): ``M0 = 10 ** (1.5 * Mw + 9.05)`` in N*m.  Hanks &
#: Kanamori (1979) use 9.1 in SI; the fixed offset of 0.05/1.5 = 0.033 in
#: magnitude is a convention difference, not a numerical error.
FFSP_MAGNITUDE_CONSTANT = 9.05


def _write_hdf_mapping(group, mapping):
    """Write a nested result mapping without losing its logical structure."""
    for key, value in mapping.items():
        if isinstance(value, dict):
            _write_hdf_mapping(group.create_group(key), value)
        elif np.isscalar(value):
            group.attrs[key] = value
        else:
            array = np.asarray(value)
            options = {'compression': 'gzip', 'shuffle': True} if array.size else {}
            group.create_dataset(key, data=array, **options)


def _read_hdf_mapping(group):
    """Reconstruct a nested result mapping written by _write_hdf_mapping."""
    result = {key: value for key, value in group.attrs.items()}
    for key, value in group.items():
        result[key] = _read_hdf_mapping(value) if hasattr(value, 'items') else value[:]
    return result


def _validate_ffsp_kernel_contract(params):
    """Reject inputs that would trigger a Fortran STOP or exceed ABI buffers."""
    positive = ('fault_length', 'fault_width', 'rv_avg', 'freq_min', 'freq_max')
    for name in positive:
        if params[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if params['freq_min'] >= params['freq_max']:
        raise ValueError("freq_min must be smaller than freq_max")
    if params['nsubx'] <= 0 or params['nsuby'] <= 0:
        raise ValueError("nsubx and nsuby must be positive")
    if params['id_ran2'] < params['id_ran1']:
        raise ValueError("id_ran2 must be greater than or equal to id_ran1")
    if params['id_ran1'] < 1:
        raise ValueError("id_ran1 must be greater than or equal to 1")

    seeds = params['seeds']
    if len(seeds) != 3:
        raise ValueError("seeds must contain exactly three integers")

    # The f2py ABI and the FFSP kernel use default 32-bit Fortran INTEGERs.  Seed
    # offsets are based on the global realization id, so validate them with Python
    # integers before NumPy/Fortran can silently wrap.  seed1 is negated by the
    # kernel for nsource > 1; excluding abs(INT32_MIN) also avoids its undefined
    # two's-complement overflow in Fortran.
    int32 = np.iinfo(np.int32)
    for index, seed in enumerate(seeds, start=1):
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed{index} must be an integer")
        if not int32.min <= int(seed) <= int32.max:
            raise ValueError(f"seed{index} is outside the int32 range")

    for realization_id in {params['id_ran1'], params['id_ran2']}:
        derived = (
            int(seeds[0]) + (realization_id - 1) * 10000,
            int(seeds[1]) + (realization_id - 1) * 20000,
            int(seeds[2]) + (realization_id - 1) * 30000,
        )
        if realization_id > 1 and abs(derived[0]) > int32.max:
            raise ValueError(
                f"derived seed1 for realization {realization_id} exceeds int32")
        for index, value in enumerate(derived[1:], start=2):
            if not int32.min <= value <= int32.max:
                raise ValueError(
                    f"derived seed{index} for realization "
                    f"{realization_id} exceeds int32")

    dx = params['fault_length'] / params['nsubx']
    dy = params['fault_width'] / params['nsuby']
    ncxp = int(params['x_hypc'] / dx) + 1
    ncyp = int(params['y_hypc'] / dy) + 1
    if not (1 <= ncxp <= params['nsubx'] and 1 <= ncyp <= params['nsuby']):
        raise ValueError("The hypocenter must lie inside the discretized fault")

    ix = max(ncxp, params['nsubx'] - ncxp)
    iy = max(ncyp, params['nsuby'] - ncyp)
    maximum_distance = np.hypot(ix * dx, iy * dy)
    minimum_average_velocity = params['rv_avg'] * 1.2 / 1.45
    required_samples = int(maximum_distance / minimum_average_velocity / 0.001 + 1.0) * 10
    ntime = 1
    while ntime < required_samples:
        ntime *= 2

    if ntime > 131072:
        raise ValueError(
            f"FFSP requires ntime={ntime}, exceeding the f2py buffer limit 131072")

    df = 1.0 / (ntime * 0.001)
    lnpt = int(np.log(ntime) / np.log(2.0) + 0.1)
    nfe1 = int(1.1 * params['freq_min'] / df) + 1
    nfe2 = int(0.9 * params['freq_max'] / df) + 1
    nbb = int(np.log(nfe1) / np.log(2.0))
    nee = int(np.log(nfe2) / np.log(2.0)) + 1
    if nbb < 1 or nee > lnpt - 1:
        raise ValueError(
            "The requested frequency band is outside the FFSP octave range: "
            f"nbb={nbb}, nee={nee}, valid=[1, {lnpt - 1}]")


# Finite Fault Stochastic Process (FFSP) source model
class FFSPSource:
    """Finite Fault Stochastic Process source model (Pengcheng Liu, (c) 2005;
    modifications by Chen Ji, 2020).

    An MPI-aware wrapper over the FFSP kernel that generates an ensemble of stochastic
    finite-fault realizations and exposes HDF5 and native FFSP-format I/O plus plotting.
    See the module docstring for the full capability list.
    """
    
    def __init__(self,
                 id_sf_type: int, freq_min: float, freq_max: float,
                 fault_length: float, fault_width: float,
                 x_hypc: float, y_hypc: float, depth_hypc: float,
                 xref_hypc: float, yref_hypc: float,
                 magnitude: float, fc_main_1: float, fc_main_2: float,
                 rv_avg: float,
                 ratio_rise: float,
                 strike: float, dip: float, rake: float,
                 pdip_max: float, prake_max: float,
                 nsubx: int, nsuby: int,
                 nb_taper_trbl: List[int],
                 seeds: List[int],
                 id_ran1: int, id_ran2: int,
                 angle_north_to_x: float,
                 is_moment: int,
                 crust_model: CrustModel,
                 output_name: str = "FFSP_OUTPUT",
                 verbose: bool = True):
        """
        Initialize FFSP source model.

        Parameters
        ----------
        id_sf_type : int
            Slip function type
        freq_min, freq_max : float
            Frequency range (Hz)
        fault_length, fault_width : float
            Fault dimensions (km)
        x_hypc, y_hypc, depth_hypc : float
            Hypocenter position (km)
        xref_hypc, yref_hypc : float
            Reference position for hypocenter
        magnitude : float
            Moment magnitude
        fc_main_1, fc_main_2 : float
            Corner frequencies (Hz)
        rv_avg : float
            Average rupture velocity (km/s)
        ratio_rise : float
            Rise time ratio
        strike, dip, rake : float
            Fault geometry (degrees)
        pdip_max, prake_max : float
            Maximum perturbations (degrees)
        nsubx, nsuby : int
            Number of subfaults along strike and dip
        nb_taper_trbl : List[int]
            Taper zones [top, right, bottom, left]
        seeds : List[int]
            Master random seeds [seed1, seed2, seed3]. Realizations are derived
            deterministically from these seeds and their global realization ID;
            realization 1 retains the historical FFSP reference stream.
        id_ran1, id_ran2 : int
            Realization range [start, end]
        angle_north_to_x : float
            Rotation angle (degrees)
        is_moment : int
            Result flag
        crust_model : CrustModel
            Velocity model
        output_name : str, optional
            Output file prefix
        verbose : bool, optional
            Print progress messages
        """
        if verbose:
            print(FFSP_CITATION)

        if not isinstance(crust_model, CrustModel):
            raise TypeError("crust_model must be a CrustModel instance")

        # Store all parameters
        self.params = {
            'id_sf_type': id_sf_type, 'freq_min': freq_min, 'freq_max': freq_max,
            'fault_length': fault_length, 'fault_width': fault_width,
            'x_hypc': x_hypc, 'y_hypc': y_hypc, 'depth_hypc': depth_hypc,
            'xref_hypc': xref_hypc, 'yref_hypc': yref_hypc,
            'magnitude': magnitude, 'fc_main_1': fc_main_1, 'fc_main_2': fc_main_2,
            'rv_avg': rv_avg,
            'ratio_rise': ratio_rise,
            'strike': strike, 'dip': dip, 'rake': rake,
            'pdip_max': pdip_max, 'prake_max': prake_max,
            'nsubx': nsubx, 'nsuby': nsuby,
            'nb_taper_trbl': nb_taper_trbl,
            'seeds': seeds,
            'id_ran1': id_ran1, 'id_ran2': id_ran2,
            'angle_north_to_x': angle_north_to_x,
            'is_moment': is_moment,
            'output_name': output_name,
        }

        self.crust_model = crust_model
        self.output_name = output_name
        self.verbose = verbose

        # Results storage
        self.all_realizations = None
        self.best_realization = None
        self.source_stats = None
        self.subfaults = None

        # Subfault dimensions
        self.dx = fault_length / nsubx
        self.dy = fault_width / nsuby
        self.area = self.dx * self.dy
    
    def run(self) -> Dict:
        """
        Run FFSP to generate fault realizations using Fortran wrapper.
        Supports MPI parallelization with automatic data gathering.
        
        Returns
        -------
        Dict
            Best realization subfault data (only valid on rank 0 if using MPI)
        """
        
        _validate_ffsp_kernel_contract(self.params)

        # Detect MPI environment
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            use_mpi = comm.Get_size() > 1
            rank = comm.Get_rank() if use_mpi else 0
            nprocs = comm.Get_size() if use_mpi else 1
        except ImportError:
            use_mpi = False
            rank = 0
            nprocs = 1
        
        # Distribute realizations across MPI ranks
        total_models = self.params['id_ran2'] - self.params['id_ran1'] + 1
        
        if use_mpi:
            models_per_rank = total_models // nprocs
            remainder = total_models % nprocs
            
            if rank < remainder:
                start = rank * (models_per_rank + 1) + self.params['id_ran1']
                end = start + models_per_rank
            else:
                start = rank * models_per_rank + remainder + self.params['id_ran1']
                end = start + models_per_rank - 1
            
            my_n_models = end - start + 1
        else:
            start = self.params['id_ran1']
            end = self.params['id_ran2']
            my_n_models = total_models
        
        if self.verbose and rank == 0:
            print(f"\nRunning FFSP: {total_models} realizations on {nprocs} MPI ranks\n")

        if use_mpi and my_n_models == 0:
            # More ranks than realizations is valid; idle ranks must not call
            # f2py with an inverted realization range.
            comm.send({'idle': True}, dest=0, tag=rank)
            self.all_realizations = None
            self.source_stats = None
            self.best_realization = None
            self.subfaults = None
            self.active_realization = None
            return None
        
        # Import Fortran wrapper module
        try:
            from shakermaker.ffsp import ffsp_core
        except ImportError:
            # Fallback: add ffsp directory to path
            import sys
            ffsp_dir = os.path.join(os.path.dirname(__file__), 'ffsp')
            sys.path.insert(0, ffsp_dir)
            import ffsp_core
        
        result = ffsp_core.ffsp_run_wrapper(
            self.params['id_sf_type'],
            self.params['freq_min'],
            self.params['freq_max'],
            self.params['fault_length'],
            self.params['fault_width'],
            self.params['x_hypc'],
            self.params['y_hypc'],
            self.params['depth_hypc'],
            self.params['xref_hypc'],
            self.params['yref_hypc'],
            self.params['magnitude'],
            self.params['fc_main_1'],
            self.params['fc_main_2'],
            self.params['rv_avg'],
            self.params['ratio_rise'],
            self.params['strike'],
            self.params['dip'],
            self.params['rake'],
            self.params['pdip_max'],
            self.params['prake_max'],
            self.params['nsubx'],
            self.params['nsuby'],
            np.array(self.params['nb_taper_trbl'], dtype=np.int32),
            np.array(self.params['seeds'], dtype=np.int32),
            start, end,
            self.params['angle_north_to_x'],
            self.params['is_moment'],
            self.crust_model.nlayers,
            self.crust_model.a.astype(np.float32),
            self.crust_model.b.astype(np.float32),
            self.crust_model.rho.astype(np.float32),
            self.crust_model.d.astype(np.float32),
            self.crust_model.qa.astype(np.float32),
            self.crust_model.qb.astype(np.float32),
        )
        
        # Unpack kinematics, metrics, shared axes, and per-realization products.
        (n_realizations, npts, x, y, z, slip, rupture_time, 
         rise_time, peak_time, strike, dip, rake,
         ave_tr, ave_tp, ave_vr, err_spectra, pdf,
         fc_main_1_effective, fc_main_2_effective,
         # Spectral data
         ntime_spec, nphf_spec, lnpt_spec,
         stf_time, stf,
         freq_spec, moment_rate, dcf,
         freq_center, logmean_synth, logmean_dcf) = result
        
        if use_mpi:
            if rank == 0:
                all_x = [x]
                all_y = [y]
                all_z = [z]
                all_slip = [slip]
                all_rupture_time = [rupture_time]
                all_rise_time = [rise_time]
                all_peak_time = [peak_time]
                all_strike = [strike]
                all_dip = [dip]
                all_rake = [rake]
                all_ave_tr = [ave_tr]
                all_ave_tp = [ave_tp]
                all_ave_vr = [ave_vr]
                all_err_spectra = [err_spectra]
                all_pdf = [pdf]
                all_stf = [stf[:ntime_spec, :]]
                all_moment_rate = [moment_rate[:nphf_spec, :]]
                all_logmean_synth = [logmean_synth[:lnpt_spec, :]]
                
                for r in range(1, nprocs):
                    recv_data = comm.recv(source=r, tag=r)
                    if recv_data.get('idle', False):
                        continue

                    remote_dims = (
                        recv_data['ntime_spec'], recv_data['nphf_spec'],
                        recv_data['lnpt_spec'])
                    if remote_dims != (ntime_spec, nphf_spec, lnpt_spec):
                        raise RuntimeError("MPI ranks returned inconsistent spectral dimensions")
                    
                    all_x.append(recv_data['x'])
                    all_y.append(recv_data['y'])
                    all_z.append(recv_data['z'])
                    all_slip.append(recv_data['slip'])
                    all_rupture_time.append(recv_data['rupture_time'])
                    all_rise_time.append(recv_data['rise_time'])
                    all_peak_time.append(recv_data['peak_time'])
                    all_strike.append(recv_data['strike'])
                    all_dip.append(recv_data['dip'])
                    all_rake.append(recv_data['rake'])
                    all_ave_tr.append(recv_data['ave_tr'])
                    all_ave_tp.append(recv_data['ave_tp'])
                    all_ave_vr.append(recv_data['ave_vr'])
                    all_err_spectra.append(recv_data['err_spectra'])
                    all_pdf.append(recv_data['pdf'])
                    all_stf.append(recv_data['stf'])
                    all_moment_rate.append(recv_data['moment_rate'])
                    all_logmean_synth.append(recv_data['logmean_synth'])

                    # Every rank must use the same discretization and DCF target.
                    for name, local, remote, length in (
                            ('stf_time', stf_time, recv_data['stf_time'], ntime_spec),
                            ('freq_spec', freq_spec, recv_data['freq_spec'], nphf_spec),
                            ('dcf', dcf, recv_data['dcf'], nphf_spec),
                            ('freq_center', freq_center, recv_data['freq_center'], lnpt_spec),
                            ('logmean_dcf', logmean_dcf, recv_data['logmean_dcf'], lnpt_spec)):
                        if not np.array_equal(local[:length], remote[:length]):
                            raise RuntimeError(f"MPI ranks returned inconsistent {name}")
                
                # Concatenate all data
                x = np.concatenate(all_x, axis=1)
                y = np.concatenate(all_y, axis=1)
                z = np.concatenate(all_z, axis=1)
                slip = np.concatenate(all_slip, axis=1)
                rupture_time = np.concatenate(all_rupture_time, axis=1)
                rise_time = np.concatenate(all_rise_time, axis=1)
                peak_time = np.concatenate(all_peak_time, axis=1)
                strike = np.concatenate(all_strike, axis=1)
                dip = np.concatenate(all_dip, axis=1)
                rake = np.concatenate(all_rake, axis=1)
                ave_tr = np.concatenate(all_ave_tr)
                ave_tp = np.concatenate(all_ave_tp)
                ave_vr = np.concatenate(all_ave_vr)
                err_spectra = np.concatenate(all_err_spectra)
                pdf = np.concatenate(all_pdf)
                stf = np.concatenate(all_stf, axis=1)
                moment_rate = np.concatenate(all_moment_rate, axis=1)
                logmean_synth = np.concatenate(all_logmean_synth, axis=1)
                
                n_realizations = total_models
                
            else:
                send_data = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'slip': slip,
                    'rupture_time': rupture_time,
                    'rise_time': rise_time,
                    'peak_time': peak_time,
                    'strike': strike,
                    'dip': dip,
                    'rake': rake,
                    'ave_tr': ave_tr,
                    'ave_tp': ave_tp,
                    'ave_vr': ave_vr,
                    'err_spectra': err_spectra,
                    'pdf': pdf,
                    'stf_time': stf_time,
                    'stf': stf[:ntime_spec, :],
                    'freq_spec': freq_spec,
                    'moment_rate': moment_rate[:nphf_spec, :],
                    'dcf': dcf,
                    'freq_center': freq_center,
                    'logmean_synth': logmean_synth[:lnpt_spec, :],
                    'logmean_dcf': logmean_dcf,
                    'ntime_spec': ntime_spec,
                    'nphf_spec': nphf_spec,
                    'lnpt_spec': lnpt_spec,
                }
                
                comm.send(send_data, dest=0, tag=rank)
                
                self.all_realizations = None
                self.source_stats = None
                self.best_realization = None
                self.subfaults = None
                self.active_realization = None
                
                return None
        
        # =====================================================================
        # Store results (only rank 0 in MPI mode, or single process)
        # =====================================================================
        
        self.params.setdefault('fc_main_1_requested', self.params['fc_main_1'])
        self.params.setdefault('fc_main_2_requested', self.params['fc_main_2'])
        self.params['fc_main_1'] = float(fc_main_1_effective)
        self.params['fc_main_2'] = float(fc_main_2_effective)

        realization_ids = np.arange(
            self.params['id_ran1'], self.params['id_ran2'] + 1,
            dtype=np.int32)

        # Every realization is complete: kinematics, metrics, STF, and spectra.
        self.all_realizations = {
            'n_realizations': n_realizations,
            'nseg': 1,
            'npts': npts,
            'realization_id': realization_ids,
            'x': x,
            'y': y,
            'z': z,
            'slip': slip,
            'rupture_time': rupture_time,
            'rise_time': rise_time,
            'peak_time': peak_time,
            'strike': strike,
            'dip': dip,
            'rake': rake,
            'metrics': {
                'ave_tr': ave_tr,
                'ave_tp': ave_tp,
                'ave_vr': ave_vr,
                'err_spectra': err_spectra,
                'pdf': pdf,
            },
            'stf_time': {
                'time': stf_time[:ntime_spec],
                'stf': stf[:ntime_spec, :],
            },
            'spectrum': {
                'freq': freq_spec[:nphf_spec],
                'moment_rate_synth': moment_rate[:nphf_spec, :],
                'moment_rate_dcf': dcf[:nphf_spec],
            },
            'spectrum_octave': {
                'freq_center': freq_center[:lnpt_spec],
                'logmean_synth': logmean_synth[:lnpt_spec, :],
                'logmean_dcf': logmean_dcf[:lnpt_spec],
            },
        }
        
        # Store statistics (for plots)
        self.source_stats = {
            'source_score': {
                'n_realizations': n_realizations,
                'ave_tr': ave_tr,
                'ave_tp': ave_tp,
                'ave_vr': ave_vr,
                'err_spectra': err_spectra,
                'pdf': pdf,
            },
        }
        
        # Identify best realization (minimum PDF)
        best_idx = int(np.argmin(pdf))
        self.best_realization = self.get_realization(best_idx)

        # Preserve the historical statistics paths as aliases to the best model.
        self.source_stats['stf_time'] = self.best_realization['stf_time']
        self.source_stats['spectrum'] = self.best_realization['spectrum']
        self.source_stats['spectrum_octave'] = self.best_realization['spectrum_octave']

        # Set best realization as active subfaults
        self.subfaults = self.best_realization
        self.active_realization = 'best'

        if self.verbose and (rank == 0 or not use_mpi):
            print(f"\nFFSP Complete: {n_realizations} realizations | Best: PDF={pdf[best_idx]:.4f}\n")
        
        return self.subfaults

    def get_realization(self, index: int) -> Dict:
        """
        Get specific realization by index (0-based).
        
        Parameters
        ----------
        index : int
            Realization index
            
        Returns
        -------
        Dict
            Realization data
        """
        
        if not hasattr(self, 'all_realizations') or not self.all_realizations:
            raise RuntimeError("No realizations available. Run FFSP first.")
        
        n = self.all_realizations['n_realizations']
        if not (0 <= index < n):
            raise IndexError(f"Index {index} out of range [0, {n-1}]")
        
        result = {
            'nseg': self.all_realizations['nseg'],
            'npts': self.all_realizations['npts'],
            'realization_id': int(self.all_realizations.get(
                'realization_id',
                np.arange(self.params['id_ran1'], self.params['id_ran2'] + 1),
            )[index]),
            'x': self.all_realizations['x'][:, index],
            'y': self.all_realizations['y'][:, index],
            'z': self.all_realizations['z'][:, index],
            'slip': self.all_realizations['slip'][:, index],
            'rupture_time': self.all_realizations['rupture_time'][:, index],
            'rise_time': self.all_realizations['rise_time'][:, index],
            'peak_time': self.all_realizations['peak_time'][:, index],
            'strike': self.all_realizations['strike'][:, index],
            'dip': self.all_realizations['dip'][:, index],
            'rake': self.all_realizations['rake'][:, index],
        }

        if 'metrics' in self.all_realizations:
            result['metrics'] = {
                key: np.asarray(value)[index]
                for key, value in self.all_realizations['metrics'].items()
            }
        if 'stf_time' in self.all_realizations:
            result['stf_time'] = {
                'time': self.all_realizations['stf_time']['time'],
                'stf': self.all_realizations['stf_time']['stf'][:, index],
            }
        if 'spectrum' in self.all_realizations:
            result['spectrum'] = {
                'freq': self.all_realizations['spectrum']['freq'],
                'moment_rate_synth': self.all_realizations['spectrum']['moment_rate_synth'][:, index],
                'moment_rate_dcf': self.all_realizations['spectrum']['moment_rate_dcf'],
            }
        if 'spectrum_octave' in self.all_realizations:
            result['spectrum_octave'] = {
                'freq_center': self.all_realizations['spectrum_octave']['freq_center'],
                'logmean_synth': self.all_realizations['spectrum_octave']['logmean_synth'][:, index],
                'logmean_dcf': self.all_realizations['spectrum_octave']['logmean_dcf'],
            }
        return result

    def set_active_realization(self, index: int):
        """Set active realization for plotting."""
        self.subfaults = self.get_realization(index)
        self.active_realization = index
        # Detect MPI rank
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except (ImportError, AttributeError):
            rank = 0

        if self.verbose and rank == 0:
            print(f"Realization #{index+1} activated")

    def get_subfaults(self) -> Dict:
        """Get currently active subfault data."""
        if self.subfaults is None:
            raise RuntimeError("No active realization. Call set_active_realization() first.")
        return self.subfaults 
    
    # ============ FILE WRITING METHODS ============
    
    
    def write_hdf5(self, filename: str):
        """
        Write results to HDF5 file (modern, efficient format).
        Stores all realizations and metadata in a single compressed file.
        
        Parameters
        ----------
        filename : str
            Output HDF5 filename (will add .h5 if not present)
        """
        
        if self.all_realizations is None:
            raise RuntimeError("No realizations available. Run FFSP first.")
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        print(f"Writing HDF5: {filename}")
        
        with h5py.File(filename, 'w') as f:
            f.attrs['schema_version'] = FFSP_HDF5_SCHEMA_VERSION
            f.attrs['shakermaker_version'] = shakermaker_version
            f.attrs['best_realization_id'] = self.best_realization['realization_id']
            grp_realizations = f.create_group('realizations')
            grp_best = f.create_group('best_realization')
            grp_stats = f.create_group('statistics')
            grp_params = f.create_group('parameters')
            
            _write_hdf_mapping(grp_realizations, self.all_realizations)
            
            if self.best_realization is not None:
                _write_hdf_mapping(grp_best, self.best_realization)
            
            if self.source_stats is not None:
                _write_hdf_mapping(grp_stats, self.source_stats)
                grp_stats.attrs['realization_id'] = self.best_realization['realization_id']
            
            for key, val in self.params.items():
                if isinstance(val, (int, float, str)):
                    grp_params.attrs[key] = val
                elif isinstance(val, list):
                    grp_params.create_dataset(key, data=np.array(val))
            
            grp_params.attrs['dx'] = self.dx
            grp_params.attrs['dy'] = self.dy
            grp_params.attrs['area'] = self.area
            
            grp_crust = grp_params.create_group('crust_model')
            grp_crust.attrs['nlayers'] = self.crust_model.nlayers
            grp_crust.create_dataset('d', data=self.crust_model.d)
            grp_crust.create_dataset('a', data=self.crust_model.a)
            grp_crust.create_dataset('b', data=self.crust_model.b)
            grp_crust.create_dataset('rho', data=self.crust_model.rho)
            grp_crust.create_dataset('qa', data=self.crust_model.qa)
            grp_crust.create_dataset('qb', data=self.crust_model.qb)
        
        print(f"[OK] HDF5 saved\n")
    
    def load_hdf5(self, filename: str):
        """
        Load results from HDF5 file into existing FFSPSource object.
        
        Parameters
        ----------
        filename : str
            Input HDF5 filename
        """
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 input. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        print(f"Loading HDF5: {filename}")
        
        with h5py.File(filename, 'r') as f:
            # Load all realizations
            self.all_realizations = _read_hdf_mapping(f['realizations'])
            
            # Load best realization
            self.best_realization = _read_hdf_mapping(f['best_realization'])
            
            # Load statistics
            self.source_stats = _read_hdf_mapping(f['statistics'])

            # Older files stored spectral products only under /statistics.
            # Keep those products attached to the loaded best realization
            # without presenting them as data for every realization.
            for group_name in ('stf_time', 'spectrum', 'spectrum_octave'):
                if group_name not in self.best_realization and group_name in self.source_stats:
                    self.best_realization[group_name] = self.source_stats[group_name]
            
            # Load parameters
            grp_params = f['parameters']
            self.params = {}
            for key in grp_params.attrs.keys():
                self.params[key] = grp_params.attrs[key]
            for key in grp_params.keys():
                if key != 'crust_model':  # Skip crust_model group
                    self.params[key] = grp_params[key][:].tolist()
            
            # Reconstruct subfault geometry attributes
            self.dx = self.params['dx']
            self.dy = self.params['dy']
            self.area = self.params['area']
            self.output_name = self.params.get('output_name', 'FFSP_OUTPUT')
            self.verbose = True  # Default
            
            # Load crust model - reconstruct layer by layer
            grp_crust = grp_params['crust_model']
            from .crustmodel import CrustModel
            
            nlayers = grp_crust.attrs['nlayers']
            self.crust_model = CrustModel(nlayers)
            
            # Add each layer
            d = grp_crust['d'][:]
            a = grp_crust['a'][:]    # vp
            b = grp_crust['b'][:]    # vs
            rho = grp_crust['rho'][:]
            qa = grp_crust['qa'][:]
            qb = grp_crust['qb'][:]
            
            for i in range(nlayers):
                self.crust_model.add_layer(d[i], a[i], b[i], rho[i], qa[i], qb[i])
            
            # Set active realization
            self.subfaults = self.best_realization
            self.active_realization = 'best'
            # Detect MPI rank
            try:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
            except (ImportError, AttributeError):
                rank = 0

            if rank == 0:
                print(f"HDF5 loaded")
                print(f"Best realization activated\n")

        
        print(f"[OK] HDF5 loaded\n")
    
    @classmethod
    def from_hdf5(cls, filename: str):
        """
        Create FFSPSource object from HDF5 file (class method).
        This is the recommended way to load saved results.
        
        Parameters
        ----------
        filename : str
            Input HDF5 filename
            
        Returns
        -------
        FFSPSource
            Fully initialized FFSPSource object with all data loaded
            
        Examples
        --------
        >>> # Save results
        >>> source.run()
        >>> source.write_hdf5('results.h5')
        >>> 
        >>> # Load in new session
        >>> source = FFSPSource.from_hdf5('results.h5')
        >>> source.plot_spectral_comparison()
        """
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 input. Install with: pip install h5py")
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        with h5py.File(filename, 'r') as f:
            # Load parameters first
            grp_params = f['parameters']
            params = {}
            for key in grp_params.attrs.keys():
                params[key] = grp_params.attrs[key]
            for key in grp_params.keys():
                if key != 'crust_model':
                    params[key] = grp_params[key][:].tolist()
            
            # Load crust model - reconstruct layer by layer
            grp_crust = grp_params['crust_model']
            from .crustmodel import CrustModel
            
            nlayers = grp_crust.attrs['nlayers']
            crust_model = CrustModel(nlayers)
            
            # Add each layer
            d = grp_crust['d'][:]
            a = grp_crust['a'][:]    # vp
            b = grp_crust['b'][:]    # vs
            rho = grp_crust['rho'][:]
            qa = grp_crust['qa'][:]
            qb = grp_crust['qb'][:]
            
            for i in range(nlayers):
                crust_model.add_layer(d[i], a[i], b[i], rho[i], qa[i], qb[i])
        
        # Create object using __init__ with loaded parameters
        obj = cls(
            id_sf_type=params['id_sf_type'],
            freq_min=params['freq_min'],
            freq_max=params['freq_max'],
            fault_length=params['fault_length'],
            fault_width=params['fault_width'],
            x_hypc=params['x_hypc'],
            y_hypc=params['y_hypc'],
            depth_hypc=params['depth_hypc'],
            xref_hypc=params['xref_hypc'],
            yref_hypc=params['yref_hypc'],
            magnitude=params['magnitude'],
            fc_main_1=params['fc_main_1'],
            fc_main_2=params['fc_main_2'],
            rv_avg=params['rv_avg'],
            ratio_rise=params['ratio_rise'],
            strike=params['strike'],
            dip=params['dip'],
            rake=params['rake'],
            pdip_max=params['pdip_max'],
            prake_max=params['prake_max'],
            nsubx=params['nsubx'],
            nsuby=params['nsuby'],
            nb_taper_trbl=params['nb_taper_trbl'],
            seeds=params['seeds'],
            id_ran1=params['id_ran1'],
            id_ran2=params['id_ran2'],
            angle_north_to_x=params['angle_north_to_x'],
            is_moment=params['is_moment'],
            crust_model=crust_model,
            output_name=params.get('output_name', 'FFSP_OUTPUT'),
            verbose=False  # Don't print during load
        )
        
        # Now load the data into the initialized object
        obj.load_hdf5(filename)
        
        return obj
    
    def write_ffsp_format(self, output_dir: str, output_name: str = None):
            """
            Write FFSP results to legacy format files.
            
            Creates the following files:
            - FFSP_OUTPUT.001, .002, ... (all realizations)
            - FFSP_OUTPUT.bst (best realization)
            - source_model.score (quality metrics)
            - source_model.list (geometry summary)
            - source_model.params (all parameters)
            - ffsp.inp (input parameters for Fortran)
            - velocity.vel (crustal velocity model)
            - calsvf.dat, calsvf_tim.dat, logsvf.dat (spectral data)
            
            Parameters
            ----------
            output_dir : str
                Output directory path
            output_name : str, optional
                Base name for output files (default: self.output_name)
            """
            if self.all_realizations is None:
                raise RuntimeError("No realizations available. Run FFSP first.")
            
            if output_name is None:
                output_name = self.output_name
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"Writing FFSP: {output_dir}")
            
            n = self.all_realizations['n_realizations']
            npts = self.all_realizations['npts']
            
            # =========================================================================
            # Write realization files (.001, .002, ...)
            # =========================================================================
            for i in range(n):
                filename = os.path.join(output_dir, f"{output_name}.{i+1:03d}")
                with open(filename, 'w') as f:
                    # Header: is_moment npts id_sf_type ratio_rise (Fortran line 210)
                    f.write(f"{self.params['is_moment']} {npts} {self.params['id_sf_type']} {self.params['ratio_rise']}\n")
                    
                    # Subfault data (Fortran lines 238-239)
                    for j in range(npts):
                        f.write(f"{self.all_realizations['x'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['y'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['z'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['slip'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['rupture_time'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['rise_time'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['peak_time'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['strike'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['dip'][j, i]:15.6e} ")
                        f.write(f"{self.all_realizations['rake'][j, i]:15.6e}\n")
            
            # =========================================================================
            # Write best realization (.bst)
            # =========================================================================
            if self.best_realization is not None:
                filename = os.path.join(output_dir, f"{output_name}.bst")
                with open(filename, 'w') as f:
                    # Header: is_moment npts id_sf_type ratio_rise
                    f.write(f"{self.params['is_moment']} {self.best_realization['npts']} {self.params['id_sf_type']} {self.params['ratio_rise']}\n")
                    
                    # Subfault data
                    for j in range(self.best_realization['npts']):
                        f.write(f"{self.best_realization['x'][j]:15.6e} ")
                        f.write(f"{self.best_realization['y'][j]:15.6e} ")
                        f.write(f"{self.best_realization['z'][j]:15.6e} ")
                        f.write(f"{self.best_realization['slip'][j]:15.6e} ")
                        f.write(f"{self.best_realization['rupture_time'][j]:15.6e} ")
                        f.write(f"{self.best_realization['rise_time'][j]:15.6e} ")
                        f.write(f"{self.best_realization['peak_time'][j]:15.6e} ")
                        f.write(f"{self.best_realization['strike'][j]:15.6e} ")
                        f.write(f"{self.best_realization['dip'][j]:15.6e} ")
                        f.write(f"{self.best_realization['rake'][j]:15.6e}\n")
            
            # =========================================================================
            # Write source_model.score (quality metrics)
            # =========================================================================
            if self.source_stats is not None:
                filename = os.path.join(output_dir, "source_model.score")
                stats = self.source_stats['source_score']
                with open(filename, 'w') as f:
                    f.write(f"{n}\n")
                    f.write("Target: average Risetime= 0.0 average peaktime= 0.0\n")
                    for i in range(n):
                        f.write(f"{output_name}.{i+1:03d}\n")
                        f.write(f"{stats['ave_tr'][i]:15.5e} ")
                        f.write(f"{stats['ave_tp'][i]:15.5e} ")
                        f.write(f"{stats['ave_vr'][i]:15.5e} ")
                        f.write(f"{stats['err_spectra'][i]:15.5e} ")
                        f.write(f"{stats['pdf'][i]:15.5e}\n")
            
            # =========================================================================
            # Write source_model.list (geometry summary)
            # =========================================================================
            filename = os.path.join(output_dir, "source_model.list")
            with open(filename, 'w') as f:
                f.write(f"{self.params['id_sf_type']} ")
                f.write(f"{self.params['nsubx']} ")
                f.write(f"{self.params['nsuby']} ")
                f.write(f"{self.dx} ")
                f.write(f"{self.dy} ")
                f.write(f"{self.params['x_hypc']} ")
                f.write(f"{self.params['y_hypc']}\n")
                f.write(f"{self.params['xref_hypc']} ")
                f.write(f"{self.params['yref_hypc']} ")
                f.write(f"{self.params['angle_north_to_x']}\n")
                f.write(f"{output_name}.bst\n")
            
            # =========================================================================
            # Write source_model.params (all parameters - Python format)
            # =========================================================================
            filename = os.path.join(output_dir, "source_model.params")
            with open(filename, 'w') as f:
                for key, val in self.params.items():
                    if isinstance(val, list):
                        f.write(f"{key} {' '.join(map(str, val))}\n")
                    else:
                        f.write(f"{key} {val}\n")
            
            # =========================================================================
            # Write ffsp.inp (Fortran input format - 16 lines)
            # =========================================================================
            filename = os.path.join(output_dir, "ffsp.inp")
            with open(filename, 'w') as f:
                # Line 1: id_sf_type freq_min freq_max
                f.write(f"{self.params['id_sf_type']} {self.params['freq_min']} {self.params['freq_max']}\n")
                
                # Line 2: fault_length fault_width
                f.write(f"{self.params['fault_length']} {self.params['fault_width']}\n")
                
                # Line 3: x_hypc y_hypc depth_hypc
                f.write(f"{self.params['x_hypc']} {self.params['y_hypc']} {self.params['depth_hypc']}\n")
                
                # Line 4: xref_hypc yref_hypc
                f.write(f"{self.params['xref_hypc']} {self.params['yref_hypc']}\n")
                
                # Line 5: magnitude fc_main_1 fc_main_2 rv_avg
                f.write(f"{self.params['magnitude']} {self.params['fc_main_1']} {self.params['fc_main_2']} {self.params['rv_avg']}\n")
                
                # Line 6: ratio_rise
                f.write(f"{self.params['ratio_rise']}\n")
                
                # Line 7: strike dip rake
                f.write(f"{self.params['strike']} {self.params['dip']} {self.params['rake']}\n")
                
                # Line 8: pdip_max prake_max
                f.write(f"{self.params['pdip_max']} {self.params['prake_max']}\n")
                
                # Line 9: nsubx nsuby
                f.write(f"{self.params['nsubx']} {self.params['nsuby']}\n")
                
                # Line 10: nb_taper_trbl[0-3]
                f.write(f"{self.params['nb_taper_trbl'][0]} {self.params['nb_taper_trbl'][1]} ")
                f.write(f"{self.params['nb_taper_trbl'][2]} {self.params['nb_taper_trbl'][3]}\n")
                
                # Line 11: seeds[0-2]
                f.write(f"{self.params['seeds'][0]} {self.params['seeds'][1]} {self.params['seeds'][2]}\n")
                
                # Line 12: id_ran1 id_ran2
                f.write(f"{self.params['id_ran1']} {self.params['id_ran2']}\n")
                
                # Line 13: velocity_filename
                f.write(f"velocity.vel\n")
                
                # Line 14: angle_north_to_x
                f.write(f"{self.params['angle_north_to_x']}\n")
                
                # Line 15: is_moment
                f.write(f"{self.params['is_moment']}\n")
                
                # Line 16: output_name
                f.write(f"{output_name}\n")
            
            # =========================================================================
            # Write velocity.vel (crustal model - FORMAT: vp vs rho thickness qa qb)
            # =========================================================================
            filename = os.path.join(output_dir, "velocity.vel")
            with open(filename, 'w') as f:
                # Line 1: nlayers dmin_source
                f.write(f"{self.crust_model.nlayers} 2.0\n")
                
                # Lines 2+: vp vs rho thickness qa qb (NOT thickness vp vs...)
                for i in range(self.crust_model.nlayers):
                    f.write(f"{self.crust_model.a[i]:.6e} ")   # vp
                    f.write(f"{self.crust_model.b[i]:.6e} ")   # vs
                    f.write(f"{self.crust_model.rho[i]:.6e} ") # rho
                    f.write(f"{self.crust_model.d[i]:.6e} ")   # thickness
                    f.write(f"{self.crust_model.qa[i]:.6e} ")  # qa
                    f.write(f"{self.crust_model.qb[i]:.6e}\n") # qb
            
            # =========================================================================
            # Write spectral data files (if available)
            # =========================================================================
            if self.source_stats and 'spectrum' in self.source_stats:
                # calsvf.dat (frequency spectrum)
                filename = os.path.join(output_dir, "calsvf.dat")
                spectrum = self.source_stats['spectrum']
                with open(filename, 'w') as f:
                    f.write(f"{len(spectrum['freq'])}\n")
                    for i in range(len(spectrum['freq'])):
                        f.write(f"{spectrum['freq'][i]:15.6e} ")
                        f.write(f"{spectrum['moment_rate_synth'][i]:15.6e} ")
                        f.write(f"{spectrum['moment_rate_dcf'][i]:15.6e}\n")
                
                # calsvf_tim.dat (time domain STF)
                filename = os.path.join(output_dir, "calsvf_tim.dat")
                stf = self.source_stats['stf_time']
                with open(filename, 'w') as f:
                    f.write(f"{len(stf['time'])}\n")
                    for i in range(len(stf['time'])):
                        f.write(f"{stf['time'][i]:15.6e} ")
                        f.write(f"{stf['stf'][i]:15.6e}\n")
                
                # logsvf.dat (octave-averaged spectrum)
                filename = os.path.join(output_dir, "logsvf.dat")
                octave = self.source_stats['spectrum_octave']
                with open(filename, 'w') as f:
                    f.write(f"{len(octave['freq_center'])}\n")
                    for i in range(len(octave['freq_center'])):
                        f.write(f"{octave['freq_center'][i]:15.6e} ")
                        f.write(f"{octave['logmean_synth'][i]:15.6e} ")
                        f.write(f"{octave['logmean_dcf'][i]:15.6e}\n")
            
            print(f"[OK] FFSP saved\n")



    def load_ffsp_format(self, input_dir: str, output_name: str = "FFSP_OUTPUT"):
        """
        Load FFSP results from modern format files (with source_model.params).
        
        Reads the following files:
        - source_model.params (all parameters)
        - velocity.vel (crustal velocity model)
        - source_model.score (quality metrics)
        - FFSP_OUTPUT.001, .002, ... (all realizations)
        - FFSP_OUTPUT.bst (best realization)
        - calsvf.dat, calsvf_tim.dat, logsvf.dat (spectral data, if available)
        
        Parameters
        ----------
        input_dir : str
            Directory containing FFSP files
        output_name : str, optional
            Base name of FFSP files (default: "FFSP_OUTPUT")
        """
        print(f"Loading FFSP: {input_dir}")
        
        # =========================================================================
        # Read source_model.params
        # =========================================================================
        params_file = os.path.join(input_dir, "source_model.params")
        params = {}
        with open(params_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                key = parts[0]
                if key in ['nb_taper_trbl', 'seeds']:
                    params[key] = [int(x) for x in parts[1:]]
                elif key in ['id_sf_type', 'nsubx', 'nsuby', 'id_ran1', 'id_ran2', 'is_moment']:
                    params[key] = int(parts[1])
                elif key == 'output_name':
                    params[key] = parts[1]
                else:
                    params[key] = float(parts[1])
        
        self.params = params
        self.dx = params['fault_length'] / params['nsubx']
        self.dy = params['fault_width'] / params['nsuby']
        self.area = self.dx * self.dy
        self.output_name = params.get('output_name', 'FFSP_OUTPUT')
        self.verbose = True
        
        # =========================================================================
        # Read velocity.vel (FORMAT: vp vs rho thickness qa qb)
        # =========================================================================
        vel_file = None
        for filename in os.listdir(input_dir):
            if filename.endswith('.vel'):
                vel_file = os.path.join(input_dir, filename)
                if self.verbose:
                    print(f"  Found velocity file: {filename}")
                break

        if vel_file is None:
            raise FileNotFoundError(f"No .vel velocity file found in {input_dir}")
        from .crustmodel import CrustModel
        with open(vel_file, 'r') as f:
            line = f.readline().strip().split()
            nlayers = int(line[0])  # First number only
            self.crust_model = CrustModel(nlayers)
            for i in range(nlayers):
                values = f.readline().split()
                # Format: vp vs rho thickness qa qb
                self.crust_model.add_layer(
                    float(values[3]),  # thickness
                    float(values[0]),  # vp
                    float(values[1]),  # vs
                    float(values[2]),  # rho
                    float(values[4]),  # qa
                    float(values[5])   # qb
                )
        
        # =========================================================================
        # Read source_model.score
        # =========================================================================
        score_file = os.path.join(input_dir, "source_model.score")
        with open(score_file, 'r') as f:
            n_realizations = int(f.readline().strip())
            f.readline()  # Skip header line
            ave_tr, ave_tp, ave_vr, err_spectra, pdf = [], [], [], [], []
            for i in range(n_realizations):
                f.readline()  # Skip filename line
                values = f.readline().split()
                ave_tr.append(float(values[0]))
                ave_tp.append(float(values[1]))
                ave_vr.append(float(values[2]))
                err_spectra.append(float(values[3]))
                pdf.append(float(values[4]))
        
        # =========================================================================
        # Read realization files (.001, .002, ...)
        # =========================================================================
        npts = int(params['nsubx']) * int(params['nsuby'])
        x = np.zeros((npts, n_realizations))
        y = np.zeros((npts, n_realizations))
        z = np.zeros((npts, n_realizations))
        slip = np.zeros((npts, n_realizations))
        rupture_time = np.zeros((npts, n_realizations))
        rise_time = np.zeros((npts, n_realizations))
        peak_time = np.zeros((npts, n_realizations))
        strike = np.zeros((npts, n_realizations))
        dip = np.zeros((npts, n_realizations))
        rake = np.zeros((npts, n_realizations))
        
        for i in range(n_realizations):
            filename = os.path.join(input_dir, f"{output_name}.{i+1:03d}")
            with open(filename, 'r') as f:
                header = f.readline().split()
                # Header is: is_moment npts id_sf_type ratio_rise
                # We extract is_moment (for legacy compatibility check)
                is_moment = int(header[0])
                
                # Read subfault data
                for j in range(npts):
                    values = f.readline().split()
                    x[j, i] = float(values[0])
                    y[j, i] = float(values[1])
                    z[j, i] = float(values[2])
                    slip[j, i] = float(values[3])
                    rupture_time[j, i] = float(values[4])
                    rise_time[j, i] = float(values[5])
                    peak_time[j, i] = float(values[6])
                    strike[j, i] = float(values[7])
                    dip[j, i] = float(values[8])
                    rake[j, i] = float(values[9])
        
        self.all_realizations = {
            'n_realizations': n_realizations, 
            'nseg': 1,  # Always 1 for single-segment faults
            'npts': npts,
            'x': x, 'y': y, 'z': z, 
            'slip': slip, 
            'rupture_time': rupture_time,
            'rise_time': rise_time, 
            'peak_time': peak_time, 
            'strike': strike,
            'dip': dip, 
            'rake': rake,
        }
        
        # =========================================================================
        # Read best realization (.bst)
        # =========================================================================
        best_file = os.path.join(input_dir, f"{output_name}.bst")
        best_x, best_y, best_z = np.zeros(npts), np.zeros(npts), np.zeros(npts)
        best_slip, best_rupture_time, best_rise_time = np.zeros(npts), np.zeros(npts), np.zeros(npts)
        best_peak_time, best_strike, best_dip, best_rake = np.zeros(npts), np.zeros(npts), np.zeros(npts), np.zeros(npts)
        
        with open(best_file, 'r') as f:
            f.readline()  # Skip header
            for j in range(npts):
                values = f.readline().split()
                best_x[j], best_y[j], best_z[j] = float(values[0]), float(values[1]), float(values[2])
                best_slip[j], best_rupture_time[j], best_rise_time[j] = float(values[3]), float(values[4]), float(values[5])
                best_peak_time[j], best_strike[j], best_dip[j], best_rake[j] = float(values[6]), float(values[7]), float(values[8]), float(values[9])
        
        self.best_realization = {
            'nseg': 1, 
            'npts': npts, 
            'x': best_x, 'y': best_y, 'z': best_z,
            'slip': best_slip, 
            'rupture_time': best_rupture_time, 
            'rise_time': best_rise_time,
            'peak_time': best_peak_time, 
            'strike': best_strike, 
            'dip': best_dip, 
            'rake': best_rake,
        }
        
        # =========================================================================
        # Store statistics
        # =========================================================================
        self.source_stats = {
            'source_score': {
                'n_realizations': n_realizations,
                'ave_tr': np.array(ave_tr), 
                'ave_tp': np.array(ave_tp), 
                'ave_vr': np.array(ave_vr),
                'err_spectra': np.array(err_spectra), 
                'pdf': np.array(pdf),
            }
        }
        
        # =========================================================================
        # Read spectral data (if available)
        # =========================================================================
        calsvf_file = os.path.join(input_dir, "calsvf.dat")
        if os.path.exists(calsvf_file):
            with open(calsvf_file, 'r') as f:
                nphf_spec = int(f.readline().strip())
                freq_spec, moment_rate, dcf = np.zeros(nphf_spec), np.zeros(nphf_spec), np.zeros(nphf_spec)
                for i in range(nphf_spec):
                    values = f.readline().split()
                    freq_spec[i], moment_rate[i], dcf[i] = float(values[0]), float(values[1]), float(values[2])
            self.source_stats['spectrum'] = {
                'freq': freq_spec, 
                'moment_rate_synth': moment_rate, 
                'moment_rate_dcf': dcf
            }
            
            # calsvf_tim.dat
            calsvf_tim = os.path.join(input_dir, "calsvf_tim.dat")
            if os.path.exists(calsvf_tim):
                with open(calsvf_tim, 'r') as f:
                    ntime_spec = int(f.readline().strip())
                    time, stf = np.zeros(ntime_spec), np.zeros(ntime_spec)
                    for i in range(ntime_spec):
                        values = f.readline().split()
                        time[i], stf[i] = float(values[0]), float(values[1])
                self.source_stats['stf_time'] = {'time': time, 'stf': stf}
            
            # logsvf.dat
            logsvf = os.path.join(input_dir, "logsvf.dat")
            if os.path.exists(logsvf):
                with open(logsvf, 'r') as f:
                    lnpt_spec = int(f.readline().strip())
                    freq_center, logmean_synth, logmean_dcf = np.zeros(lnpt_spec), np.zeros(lnpt_spec), np.zeros(lnpt_spec)
                    for i in range(lnpt_spec):
                        values = f.readline().split()
                        freq_center[i], logmean_synth[i], logmean_dcf[i] = float(values[0]), float(values[1]), float(values[2])
                self.source_stats['spectrum_octave'] = {
                    'freq_center': freq_center, 
                    'logmean_synth': logmean_synth, 
                    'logmean_dcf': logmean_dcf
                }
        
        # Set active realization
        self.subfaults = self.best_realization
        self.active_realization = 'best'
        # Detect MPI rank
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except (ImportError, AttributeError):
            rank = 0

        if self.verbose and rank == 0:
            print(f"[OK] FFSP loaded")
            print(f"Best realization activated\n")
        print(f"[OK] FFSP loaded\n")


    @classmethod
    def from_ffsp_format(cls, input_dir: str, output_name: str = "FFSP_OUTPUT", verbose: bool = True):
         """
         Load FFSP results from directory.
         Works with both Python-generated and original Fortran FFSP output.
         
         For Fortran output, requires ffsp.inp to be present in the directory.
         """
         
         params_file = os.path.join(input_dir, "source_model.params")
         
         if os.path.exists(params_file):
             # Python-generated format (has .params file)
             if verbose:
                 print(f"Loading Python-generated FFSP output")
             return cls._load_from_params_file(input_dir, output_name, verbose)
         else:
             # Original Fortran format (needs ffsp.inp)
             inp_file = os.path.join(input_dir, "ffsp.inp")
             if not os.path.exists(inp_file):
                 raise FileNotFoundError(
                     f"Cannot load FFSP data from {input_dir}\n\n"
                     f"Missing required files:\n"
                     f"  - source_model.params (Python format), OR\n"
                     f"  - ffsp.inp (Fortran format)\n\n"
                     f"At least one of these files must be present to reconstruct the source model."
                 )
             
             if verbose:
                 print(f"Loading original Fortran FFSP output")
             return cls._load_from_fortran_files(input_dir, output_name, verbose)

    @classmethod
    def _load_from_params_file(cls, input_dir: str, output_name: str, verbose: bool):
         """Load from Python-generated format with source_model.params"""
         
         # Read params
         params_file = os.path.join(input_dir, "source_model.params")
         params = {}
         
         with open(params_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key = parts[0]
                    if key in ['id_sf_type', 'nsubx', 'nsuby', 'id_ran1', 'id_ran2', 'is_moment']:
                        params[key] = int(parts[1])
                    elif key == 'nb_taper_trbl':
                        params[key] = [int(x) for x in parts[1:5]]
                    elif key == 'seeds':
                        params[key] = [int(x) for x in parts[1:4]]
                    elif key == 'output_name':
                        params[key] = parts[1]
                    else:
                        params[key] = float(parts[1])
         
         # Load velocity model
         vel_file = None
         for filename in os.listdir(input_dir):
            if filename.endswith('.vel'):
                vel_file = os.path.join(input_dir, filename)
                if verbose:
                    print(f"  Found velocity file: {filename}")
                break

         if vel_file is None:
            raise FileNotFoundError(f"No .vel velocity file found in {input_dir}")

         with open(vel_file, 'r') as f:
            line = f.readline().strip().split()
            nlayers = int(line[0])
            crust_model = CrustModel(nlayers)
            for i in range(nlayers):
                values = f.readline().split()
                # Format: vp vs rho thickness qa qb
                crust_model.add_layer(
                    float(values[3]),  # thickness
                    float(values[0]),  # vp
                    float(values[1]),  # vs
                    float(values[2]),  # rho
                    float(values[4]),  # qa
                    float(values[5])   # qb
                )
         
         # Create FFSPSource object
         source = cls(
             id_sf_type=params['id_sf_type'],
             freq_min=params['freq_min'],
             freq_max=params['freq_max'],
             fault_length=params['fault_length'],
             fault_width=params['fault_width'],
             x_hypc=params['x_hypc'],
             y_hypc=params['y_hypc'],
             depth_hypc=params['depth_hypc'],
             xref_hypc=params['xref_hypc'],
             yref_hypc=params['yref_hypc'],
             magnitude=params['magnitude'],
             fc_main_1=params['fc_main_1'],
             fc_main_2=params['fc_main_2'],
             rv_avg=params['rv_avg'],
             ratio_rise=params['ratio_rise'],
             strike=params['strike'],
             dip=params['dip'],
             rake=params['rake'],
             pdip_max=params['pdip_max'],
             prake_max=params['prake_max'],
             nsubx=params['nsubx'],
             nsuby=params['nsuby'],
             nb_taper_trbl=params['nb_taper_trbl'],
             seeds=params['seeds'],
             id_ran1=params['id_ran1'],
             id_ran2=params['id_ran2'],
             angle_north_to_x=params['angle_north_to_x'],
             is_moment=params['is_moment'],
             crust_model=crust_model,
             output_name=params.get('output_name', output_name),
             verbose=verbose
         )
         
         # Load data files using existing method
         source._load_ffsp_data_legacy(input_dir, output_name)
         
         return source

    @classmethod
    def _load_from_fortran_files(cls, input_dir: str, output_name: str, verbose: bool):
         """Reconstruct FFSPSource from original Fortran FFSP output files"""
         
         # 1. Parse ffsp.inp
         inp_file = os.path.join(input_dir, "ffsp.inp")
         params = cls._parse_ffsp_inp(inp_file)
         
         # 2. Read source_model.list for geometric info
         list_file = os.path.join(input_dir, "source_model.list")
         with open(list_file, 'r') as f:
             lines = f.readlines()
             parts = lines[0].strip().split()
             nsubx = int(parts[1])
             nsuby = int(parts[2])
         
         # 3. Count realizations
         id_ran1 = 1
         id_ran2 = 1
         while os.path.exists(os.path.join(input_dir, f"{output_name}.{id_ran2:03d}")):
             id_ran2 += 1
         id_ran2 -= 1
         
         if verbose:
             print(f"  Found {id_ran2} realizations")
             print(f"  Grid: {nsubx} × {nsuby} subfaults")
         
         # 4. Load velocity model
         vel_file = None
         for filename in os.listdir(input_dir):
            if filename.endswith('.vel'):
                vel_file = os.path.join(input_dir, filename)
                if verbose:
                    print(f"  Found velocity file: {filename}")
                break

         if vel_file is None:
            raise FileNotFoundError(f"No .vel velocity file found in {input_dir}")
         with open(vel_file, 'r') as f:
            line = f.readline().strip().split()
            nlayers = int(line[0])
            crust_model = CrustModel(nlayers)
            for i in range(nlayers):
                values = f.readline().split()
                # Format: vp vs rho thickness qa qb
                crust_model.add_layer(
                    float(values[3]),  # thickness
                    float(values[0]),  # vp
                    float(values[1]),  # vs
                    float(values[2]),  # rho
                    float(values[4]),  # qa
                    float(values[5])   # qb
                )
         
         # 5. Create FFSPSource
         source = cls(
             id_sf_type=params['id_sf_type'],
             freq_min=0.0,
             freq_max=params['freq_max'],
             fault_length=params['fault_length'],
             fault_width=params['fault_width'],
             x_hypc=params['x_hypc'],
             y_hypc=params['y_hypc'],
             depth_hypc=params['depth_hypc'],
             xref_hypc=params['xref_hypc'],
             yref_hypc=params['yref_hypc'],
             magnitude=params['magnitude'],
             fc_main_1=params['fc_main_1'],
             fc_main_2=params['fc_main_2'],
             rv_avg=params['rv_avg'],
             ratio_rise=params['ratio_rise'],
             strike=params['strike'],
             dip=params['dip'],
             rake=params['rake'],
             pdip_max=params['pdip_max'],
             prake_max=params['prake_max'],
             nsubx=nsubx,
             nsuby=nsuby,
             nb_taper_trbl=params['nb_taper_trbl'],
             seeds=params['seeds'],
             id_ran1=id_ran1,
             id_ran2=id_ran2,
             angle_north_to_x=params['angle_north_to_x'],
             is_moment=params['is_moment'],
             crust_model=crust_model,
             output_name=output_name,
             verbose=verbose
         )
         
         # 6. Load data files using existing method
         source._load_ffsp_data_legacy(input_dir, output_name)
         
         return source

    @staticmethod
    def _parse_ffsp_inp(inp_file: str) -> dict:
         """Parse ffsp.inp - extracts ALL parameters from original Fortran input"""
         
         with open(inp_file, 'r') as f:
             lines = f.readlines()
         
         params = {}
         
         # Line 1: id_sf_type freq_max
         parts = lines[0].strip().split()
         params['id_sf_type'] = int(parts[0])
         params['freq_max'] = float(parts[1])
         
         # Line 2: fault_length fault_width
         parts = lines[1].strip().split()
         params['fault_length'] = float(parts[0])
         params['fault_width'] = float(parts[1])
         
         # Line 3: x_hypc y_hypc depth_hypc
         parts = lines[2].strip().split()
         params['x_hypc'] = float(parts[0])
         params['y_hypc'] = float(parts[1])
         params['depth_hypc'] = float(parts[2])
         
         # Line 4: xref_hypc yref_hypc
         parts = lines[3].strip().split()
         params['xref_hypc'] = float(parts[0])
         params['yref_hypc'] = float(parts[1])
         
         # Line 5: moment fc_main_1 fc_main_2 rv_avg
         parts = lines[4].strip().split()
         params['magnitude'] = float(parts[0])
         params['fc_main_1'] = float(parts[1])
         params['fc_main_2'] = float(parts[2])
         params['rv_avg'] = float(parts[3])
         
         # Line 6: ratio_rise
         params['ratio_rise'] = float(lines[5].strip())
         
         # Line 7: strike dip rake
         parts = lines[6].strip().split()
         params['strike'] = float(parts[0])
         params['dip'] = float(parts[1])
         params['rake'] = float(parts[2])
         
         # Line 8: pdip_max prake_max
         parts = lines[7].strip().split()
         params['pdip_max'] = float(parts[0])
         params['prake_max'] = float(parts[1])
         
         # Line 9: nsubx nsuby (also in .list, but read from both)
         parts = lines[8].strip().split()
         params['nsubx'] = int(parts[0])
         params['nsuby'] = int(parts[1])
         
         # Line 10: nb_taper_TRBL
         parts = lines[9].strip().split()
         params['nb_taper_trbl'] = [int(x) for x in parts]
         
         # Line 11: seeds
         parts = lines[10].strip().split()
         params['seeds'] = [int(x) for x in parts]
         
         # Line 12: id_ran1 id_ran2 (counted from output files)
         parts = lines[11].strip().split()
         params['id_ran1'] = int(parts[0])
         params['id_ran2'] = int(parts[1])
         
         # Line 14: angle_north_to_x
         params['angle_north_to_x'] = float(lines[13].strip())
         
         # Line 15: is_moment
         params['is_moment'] = int(lines[14].strip())
         
         return params

    def _load_ffsp_data_legacy(self, input_dir: str, output_name: str):
        """Helper to load FFSP data files for legacy format"""
        
        # =========================================================================
        # Read source_model.score (OPTIONAL - may not exist for single realization)
        # =========================================================================
        score_file = os.path.join(input_dir, "source_model.score")
        
        if os.path.exists(score_file):
            # Multiple realizations case
            with open(score_file, 'r') as f:
                n_realizations = int(f.readline().strip())
                f.readline()  # Skip header
                ave_tr, ave_tp, ave_vr, err_spectra, pdf = [], [], [], [], []
                for i in range(n_realizations):
                    f.readline()  # Skip filename
                    values = f.readline().split()
                    ave_tr.append(float(values[0]))
                    ave_tp.append(float(values[1]))
                    ave_vr.append(float(values[2]))
                    err_spectra.append(float(values[3]))
                    pdf.append(float(values[4]))
        else:
            # Single realization case - count files manually
            n_realizations = 0
            i = 1
            while os.path.exists(os.path.join(input_dir, f"{output_name}.{i:03d}")):
                n_realizations += 1
                i += 1
            
            if n_realizations == 0:
                raise FileNotFoundError(
                    f"No realization files found in {input_dir}\n"
                    f"Expected files: {output_name}.001, {output_name}.002, etc."
                )
            
            # Create dummy statistics (not meaningful for single realization)
            ave_tr = [0.0] * n_realizations
            ave_tp = [0.0] * n_realizations
            ave_vr = [0.0] * n_realizations
            err_spectra = [0.0] * n_realizations
            pdf = [0.0] * n_realizations
        
        # =========================================================================
        # Read realization files (.001, .002, ...)
        # =========================================================================
        npts = self.params['nsubx'] * self.params['nsuby']
        x = np.zeros((npts, n_realizations))
        y = np.zeros((npts, n_realizations))
        z = np.zeros((npts, n_realizations))
        slip = np.zeros((npts, n_realizations))
        rupture_time = np.zeros((npts, n_realizations))
        rise_time = np.zeros((npts, n_realizations))
        peak_time = np.zeros((npts, n_realizations))
        strike = np.zeros((npts, n_realizations))
        dip = np.zeros((npts, n_realizations))
        rake = np.zeros((npts, n_realizations))
        
        for i in range(n_realizations):
            filename = os.path.join(input_dir, f"{output_name}.{i+1:03d}")
            with open(filename, 'r') as f:
                header = f.readline().split()
                nseg = int(header[0])
                for j in range(npts):
                    values = f.readline().split()
                    x[j, i] = float(values[0])
                    y[j, i] = float(values[1])
                    z[j, i] = float(values[2])
                    slip[j, i] = float(values[3])
                    rupture_time[j, i] = float(values[4])
                    rise_time[j, i] = float(values[5])
                    peak_time[j, i] = float(values[6])
                    strike[j, i] = float(values[7])
                    dip[j, i] = float(values[8])
                    rake[j, i] = float(values[9])
        
        self.all_realizations = {
            'n_realizations': n_realizations,
            'nseg': nseg,
            'npts': npts,
            'x': x,
            'y': y,
            'z': z,
            'slip': slip,
            'rupture_time': rupture_time,
            'rise_time': rise_time,
            'peak_time': peak_time,
            'strike': strike,
            'dip': dip,
            'rake': rake,
        }
        
        # =========================================================================
        # Read best realization (.bst) - OPTIONAL for single realization
        # =========================================================================
        best_file = os.path.join(input_dir, f"{output_name}.bst")
        
        if os.path.exists(best_file):
            # .bst file exists (multiple realizations)
            best_x = np.zeros(npts)
            best_y = np.zeros(npts)
            best_z = np.zeros(npts)
            best_slip = np.zeros(npts)
            best_rupture_time = np.zeros(npts)
            best_rise_time = np.zeros(npts)
            best_peak_time = np.zeros(npts)
            best_strike = np.zeros(npts)
            best_dip = np.zeros(npts)
            best_rake = np.zeros(npts)
            
            with open(best_file, 'r') as f:
                f.readline()  # Skip header
                for j in range(npts):
                    values = f.readline().split()
                    best_x[j] = float(values[0])
                    best_y[j] = float(values[1])
                    best_z[j] = float(values[2])
                    best_slip[j] = float(values[3])
                    best_rupture_time[j] = float(values[4])
                    best_rise_time[j] = float(values[5])
                    best_peak_time[j] = float(values[6])
                    best_strike[j] = float(values[7])
                    best_dip[j] = float(values[8])
                    best_rake[j] = float(values[9])
            
            self.best_realization = {
                'nseg': nseg,
                'npts': npts,
                'x': best_x,
                'y': best_y,
                'z': best_z,
                'slip': best_slip,
                'rupture_time': best_rupture_time,
                'rise_time': best_rise_time,
                'peak_time': best_peak_time,
                'strike': best_strike,
                'dip': best_dip,
                'rake': best_rake,
            }
        else:
            # No .bst file - use first (and only) realization as "best"
            self.best_realization = {
                'nseg': nseg,
                'npts': npts,
                'x': x[:, 0],
                'y': y[:, 0],
                'z': z[:, 0],
                'slip': slip[:, 0],
                'rupture_time': rupture_time[:, 0],
                'rise_time': rise_time[:, 0],
                'peak_time': peak_time[:, 0],
                'strike': strike[:, 0],
                'dip': dip[:, 0],
                'rake': rake[:, 0],
            }
        
        # =========================================================================
        # Store statistics
        # =========================================================================
        self.source_stats = {
            'source_score': {
                'n_realizations': n_realizations,
                'ave_tr': np.array(ave_tr),
                'ave_tp': np.array(ave_tp),
                'ave_vr': np.array(ave_vr),
                'err_spectra': np.array(err_spectra),
                'pdf': np.array(pdf),
            }
        }
        
        # =========================================================================
        # Load spectral data if available
        # =========================================================================
        calsvf_file = os.path.join(input_dir, "calsvf.dat")
        if os.path.exists(calsvf_file):
            with open(calsvf_file, 'r') as f:
                nphf_spec = int(f.readline().strip())
                freq_spec = np.zeros(nphf_spec)
                moment_rate = np.zeros(nphf_spec)
                dcf = np.zeros(nphf_spec)
                for i in range(nphf_spec):
                    values = f.readline().split()
                    freq_spec[i] = float(values[0])
                    moment_rate[i] = float(values[1])
                    dcf[i] = float(values[2])
            
            self.source_stats['spectrum'] = {
                'freq': freq_spec,
                'moment_rate_synth': moment_rate,
                'moment_rate_dcf': dcf
            }
            
            # calsvf_tim.dat
            calsvf_tim = os.path.join(input_dir, "calsvf_tim.dat")
            if os.path.exists(calsvf_tim):
                with open(calsvf_tim, 'r') as f:
                    ntime_spec = int(f.readline().strip())
                    time = np.zeros(ntime_spec)
                    stf = np.zeros(ntime_spec)
                    for i in range(ntime_spec):
                        values = f.readline().split()
                        time[i] = float(values[0])
                        stf[i] = float(values[1])
                
                self.source_stats['stf_time'] = {
                    'time': time,
                    'stf': stf
                }
            
            # logsvf.dat
            logsvf = os.path.join(input_dir, "logsvf.dat")
            if os.path.exists(logsvf):
                with open(logsvf, 'r') as f:
                    lnpt_spec = int(f.readline().strip())
                    freq_center = np.zeros(lnpt_spec)
                    logmean_synth = np.zeros(lnpt_spec)
                    logmean_dcf = np.zeros(lnpt_spec)
                    for i in range(lnpt_spec):
                        values = f.readline().split()
                        freq_center[i] = float(values[0])
                        logmean_synth[i] = float(values[1])
                        logmean_dcf[i] = float(values[2])
                
                self.source_stats['spectrum_octave'] = {
                    'freq_center': freq_center,
                    'logmean_synth': logmean_synth,
                    'logmean_dcf': logmean_dcf
                }
        
        # =========================================================================
        # Set active realization
        # =========================================================================
        self.subfaults = self.best_realization
        self.active_realization = 'best'
        
        print(f"[OK] Legacy FFSP loaded\n")
    

    # ============ PLOTTING METHODS ============
    
    def plot_histogram(self, field='slip', bins=50, figsize=(7, 5)):
        """Plot histogram of field across all realizations."""
        import matplotlib.pyplot as plt
        valid_fields = ['x', 'y', 'z', 'slip', 'rupture_time', 'rise_time',
                       'peak_time', 'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}, got '{field}'")
            
        plt.figure(figsize=figsize)
        
        # Plot all realizations
        for i in range(self.all_realizations['n_realizations']):
            var = self.all_realizations[field][:, i]
            plt.hist(var, bins=bins, alpha=0.4, label=f'Rlz{i+1}')
        
        # Plot best realization
        var_best = self.best_realization[field]
        plt.hist(var_best, bins=bins, histtype='step', color='red', 
                linewidth=2, label='Best')
        
        # Labels
        field_labels = {
            'x': 'North (m)', 'y': 'East (m)', 'z': 'Depth (m)',
            'slip': 'Slip (m)', 'rupture_time': 'Rupture Time (s)',
            'rise_time': 'Rise Time (s)', 'peak_time': 'Peak Time (s)',
            'strike': 'Strike (°)', 'dip': 'Dip (°)', 'rake': 'Rake (°)'
        }
        
        plt.xlabel(field_labels[field])
        plt.ylabel('Frequency')
        plt.title(f'{field_labels[field]} - Multiple Realizations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



    def plot_spacial_distribution(self, figsize=(10, 8), field='rise_time', rotate=False,
                                  cmap='coolwarm', show_contours=True, 
                                  contour_field='rupture_time', show_hypocenter=True,
                                  contour_interval=None, contour_color='blue',
                                  internal_ref=None, external_coord=None,
                                  save_fig=False, model_name='model' , image_type='png'):
        """
        Plot spatial distribution of subfault parameters with strike rotation.
        """
        
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        # Get colormap (new way)
        cmap_base = plt.get_cmap('YlOrRd', 256)

        # Create new colormap starting with white
        colors = cmap_base(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]  # White

        cmap_white = mcolors.ListedColormap(colors)
        if cmap == 'cmap_white':
            cmap = cmap_white


        nx = int(self.params['nsubx'])
        ny = int(self.params['nsuby'])
        lx = self.params['fault_length']
        ly = self.params['fault_width']
        dx = self.dx
        dy = self.dy
        cxp = self.params['x_hypc']
        cyp = self.params['y_hypc']
        strike = self.params['strike']
        
        # Validate fields
        valid_fields = ['slip', 'rupture_time', 'rise_time', 'peak_time', 
                       'strike', 'dip', 'rake']
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}")
        if contour_field not in valid_fields:
            raise ValueError(f"contour_field must be one of {valid_fields}")
        
        # Prepare data
        field_data = np.transpose(self.subfaults[field].reshape(nx, ny))
        contour_data = np.transpose(self.subfaults[contour_field].reshape(nx, ny))
        
        # Create local coordinates CENTERED ON HYPOCENTER
        x_local = np.linspace(-lx/2, lx/2, nx)
        y_local = np.linspace(0, ly, ny)

        # Convert strike to radians
        if rotate:
            strike_rad = np.radians(strike) + np.radians(90)
        else:
            strike_rad = np.radians(strike)

        # Create meshgrid centered on hypocenter
        X_local, Y_local = np.meshgrid(x_local, y_local)

        # Shift to center on hypocenter before rotating
        X_centered = X_local - (cxp - lx/2)
        Y_centered = Y_local - cyp

        # Rotate around hypocenter
        X_rot = X_centered * np.sin(strike_rad) + Y_centered * np.cos(strike_rad)
        Y_rot = X_centered * np.cos(strike_rad) - Y_centered * np.sin(strike_rad)
        
        # Apply coordinate transformation if provided
        if internal_ref is not None and external_coord is not None:
            ref_x, ref_y = internal_ref
            if rotate:
                ext_y, ext_x = external_coord
            else:
                ext_x, ext_y = external_coord
            
            # Rotate the reference point
            ref_x_rot = ref_x * np.sin(strike_rad) + ref_y * np.cos(strike_rad)
            ref_y_rot = ref_x * np.cos(strike_rad) - ref_y * np.sin(strike_rad)
            
            # Calculate offset to place ref at external coord
            offset_x = ext_x - ref_x_rot
            offset_y = ext_y - ref_y_rot
            
            # Apply offset
            X = X_rot + offset_x
            Y = Y_rot + offset_y
            
            # Transform hypocenter
            hypo_x = self.params['xref_hypc'] * np.sin(strike_rad) + self.params['yref_hypc'] * np.cos(strike_rad) + offset_x
            hypo_y = self.params['xref_hypc'] * np.cos(strike_rad) - self.params['yref_hypc'] * np.sin(strike_rad) + offset_y
            
            if rotate:
                ylabel = 'UTM Easting, X [km]'
                xlabel = 'UTM Northing, Y [km]'
            else:                
                xlabel = 'UTM Easting, X [km]'
                ylabel = 'UTM Northing, Y [km]'
        else:
            # No transformation, use xref/yref as offset
            X = X_rot + self.params['xref_hypc']
            Y = Y_rot + self.params['yref_hypc']
            
            hypo_x = self.params['xref_hypc']
            hypo_y = self.params['yref_hypc']
            
            if rotate:
                ylabel = 'Along Dip [km]'
                xlabel = 'Along Strike [km]'
            else:                
                xlabel = 'Along Dip [km]'
                ylabel = 'Along Strike [km]'
        
        # Labels
        field_labels = {
            'slip': 'Slip [m]', 'rupture_time': 'Rupture Time [s]',
            'rise_time': 'Rise Time [s]', 'peak_time': 'Peak Time [s]',
            'strike': 'Strike [°]', 'dip': 'Dip [°]', 'rake': 'Rake [°]',
        }
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use pcolormesh for rotated coordinates
        vmin = field_data.min()
        vmax = field_data.max()  

        im = ax.pcolormesh(X, Y, field_data, cmap=cmap, shading='auto',
                    vmin=vmin, vmax=vmax)

        if rotate:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label(field_labels[field], fontsize=10)


        else:
            plt.colorbar(im, label=field_labels[field], shrink=1.0, ax=ax)

        
        # Contours
        if show_contours:
            if contour_interval is not None:
                # Custom interval
                levels = np.arange(0, contour_data.max() + contour_interval, contour_interval)
                contours = ax.contour(X, Y, contour_data, levels=levels, colors=contour_color, linewidths=1.5)
            else:
                # Automatic levels
                contours = ax.contour(X, Y, contour_data, 8, colors='blue', linewidths=1.5)
            
            ax.clabel(contours, fontsize=10, fmt='%2.1f', inline=1)
            ax.plot([], [], color='blue', linewidth=1.5, label=f'Isochrones ({field_labels[contour_field]})')

        # Hypocenter
        if show_hypocenter:
            ax.scatter(hypo_x, hypo_y, c='red', s=300, marker='*', 
                       edgecolors='white', linewidth=2, label='Hypocenter', zorder=10)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{field_labels[field]} Distribution', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        # Add margins to the plot (10% padding)
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.01

        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()

        
        # Save figure if requested
        if save_fig:
            plt.savefig(f'{model_name}_spatial_distribution.{image_type}', 
                        format=image_type, 
                        dpi=600, 
                        bbox_inches='tight', 
                        transparent=True,
                        facecolor='none')
        
        plt.show()



    def plot_rupture_snapshot(self, time_snapshot, figsize=(10, 8), 
                             field='slip', cmap='YlOrRd',
                             show_rupture_front=True,
                             internal_ref=None, external_coord=None,
                             save_fig=False, model_name='model',  image_type='png'):
        """
        Plot rupture propagation snapshot at a specific time.
        """
        
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        
        # Custom colormap with white start
        cmap_base = plt.get_cmap('YlOrRd', 256)
        colors = cmap_base(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]  # White
        cmap_white = mcolors.ListedColormap(colors)
        if cmap == 'cmap_white':
            cmap = cmap_white
        
        # If cmap is still a string, create custom colormap
        if isinstance(cmap, str):
            cmap_base = plt.get_cmap(cmap, 256)
            colors = cmap_base(np.linspace(0, 1, 256))
            colors[0] = [1, 1, 1, 1]
            cmap_custom = mcolors.ListedColormap(colors)
        else:
            cmap_custom = cmap
        
        nx = int(self.params['nsubx'])
        ny = int(self.params['nsuby'])
        lx = self.params['fault_length']
        ly = self.params['fault_width']
        cxp = self.params['x_hypc']
        cyp = self.params['y_hypc']
        strike = self.params['strike']
        
        # Get data
        rupture_time = np.transpose(self.subfaults['rupture_time'].reshape(nx, ny))
        field_data = np.transpose(self.subfaults[field].reshape(nx, ny))
        
        # Get vmin/vmax from FULL field (before masking)
        vmin = field_data.min()
        vmax = field_data.max()
        
        # NOW apply mask
        mask = rupture_time <= time_snapshot
        field_masked = np.ma.masked_where(~mask, field_data)
        
        # Coordinates
        x_local = np.linspace(-lx/2, lx/2, nx)
        y_local = np.linspace(0, ly, ny)
        strike_rad = np.radians(strike)
        X_local, Y_local = np.meshgrid(x_local, y_local)
        
        X_centered = X_local - (cxp - lx/2)
        Y_centered = Y_local - cyp
        X_rot = X_centered * np.sin(strike_rad) + Y_centered * np.cos(strike_rad)
        Y_rot = X_centered * np.cos(strike_rad) - Y_centered * np.sin(strike_rad)
        
        if internal_ref is not None and external_coord is not None:
            ref_x, ref_y = internal_ref
            ext_x, ext_y = external_coord
            ref_x_rot = ref_x * np.sin(strike_rad) + ref_y * np.cos(strike_rad)
            ref_y_rot = ref_x * np.cos(strike_rad) - ref_y * np.sin(strike_rad)
            offset_x = ext_x - ref_x_rot
            offset_y = ext_y - ref_y_rot
            X = X_rot + offset_x
            Y = Y_rot + offset_y
            hypo_x = self.params['xref_hypc'] * np.sin(strike_rad) + self.params['yref_hypc'] * np.cos(strike_rad) + offset_x
            hypo_y = self.params['xref_hypc'] * np.cos(strike_rad) - self.params['yref_hypc'] * np.sin(strike_rad) + offset_y
            xlabel = 'UTM Easting, X [km]'
            ylabel = 'UTM Northing, Y [km]'
        else:
            X = X_rot + self.params['xref_hypc']
            Y = Y_rot + self.params['yref_hypc']
            hypo_x = self.params['xref_hypc']
            hypo_y = self.params['yref_hypc']
            xlabel = 'Along Dip [km]'
            ylabel = 'Along Strike [km]'
        
        # Labels
        field_labels = {
            'slip': 'Slip [m]',
            'rise_time': 'Rise Time [s]',
            'peak_time': 'Peak Time [s]',
        }
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Gray background for unruptured areas
        ax.set_facecolor('#E8E8E8')
        
        # Plot masked field with fixed vmin/vmax
        im = ax.pcolormesh(X, Y, field_masked, cmap=cmap_custom, shading='auto',
                           vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, label=field_labels[field], shrink=1.0, ax=ax)
        
        # Blue contour at rupture front
        if show_rupture_front:
            contours = ax.contour(X, Y, rupture_time, levels=[time_snapshot], 
                                 colors='tab:blue', linewidths=2, zorder=5)
        
        # Hypocenter
        ax.scatter(hypo_x, hypo_y, c='white', s=400, marker='*', 
                   edgecolors='black', linewidth=3, label='Hypocenter', zorder=10)
        
        ax.legend(loc='upper right', frameon=True)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'T={time_snapshot:.2f}s', fontsize=14, fontweight='bold', loc='left')
        ax.set_aspect('equal')
        
        # Margins
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.05
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        plt.tight_layout()

        if save_fig:
            plt.savefig(f'{model_name}_rupture_snapshot.{image_type}',
                        format=image_type,
                        dpi=600,
                        bbox_inches='tight',
                        transparent=True,
                        facecolor='none')

        plt.show()



    def plot_quality_metrics(self, figsize=(14, 5)):
        """Plot PDF and Spectral Error side by side."""
        import matplotlib.pyplot as plt
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return
            
        stats = self.source_stats['source_score']
        n = stats['n_realizations']
        idx = np.arange(1, n + 1)
        best_idx = np.argmin(stats['pdf'])
        
        plt.figure(figsize=figsize)
        
        # Subplot 1: PDF
        plt.subplot(1, 2, 1)
        colors = ['tab:green' if i == best_idx else 'tab:blue' for i in range(n)]
        plt.bar(idx, stats['pdf'], color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(stats['pdf'][best_idx], color='tab:red', ls='--', lw=2, 
                   label=f'Best = {stats["pdf"][best_idx]:.3f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('PDF', fontsize=12)
        plt.title(f'PDF Quality Metric (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Spectral Error
        plt.subplot(1, 2, 2)
        colors = ['tab:green' if i == best_idx else 'tab:orange' for i in range(n)]
        plt.bar(idx, stats['err_spectra'], color=colors, edgecolor='black', alpha=0.7)
        plt.axhline(stats['err_spectra'][best_idx], color='tab:red', ls='--', lw=2, 
                   label=f'Best = {stats["err_spectra"][best_idx]:.4f}')
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Spectral Error (RMS)', fontsize=12)
        plt.title(f'Spectral Error (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_temporal_metrics(self, figsize=(15, 5)):
        """Plot Rise Time, Peak Time, and Rupture Velocity."""
        import matplotlib.pyplot as plt
        if self.source_stats is None:
            print("No source statistics available. Run simulation first.")
            return

        stats = self.source_stats['source_score']
        n = stats['n_realizations']
        idx = np.arange(1, n + 1)
        best_idx = np.argmin(stats['pdf'])

        plt.figure(figsize=figsize)
        
        # Subplot 1: Rise Time
        plt.subplot(1, 3, 1)
        plt.plot(idx, stats['ave_tr'], 'o-', color='tab:purple', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_tr'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_tr"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rise Time (s)', fontsize=12)
        plt.title(f'Rise Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Subplot 2: Peak Time
        plt.subplot(1, 3, 2)
        plt.plot(idx, stats['ave_tp'], 'o-', color='tab:orange', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_tp'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_tp"][best_idx]:.3f} s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Peak Time (s)', fontsize=12)
        plt.title(f'Peak Time (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Subplot 3: Rupture Velocity
        plt.subplot(1, 3, 3)
        plt.plot(idx, stats['ave_vr'], 'o-', color='tab:cyan', lw=2, markersize=6)
        plt.plot(best_idx+1, stats['ave_vr'][best_idx], 's', color='tab:green', 
                markersize=15, label=f'Best = {stats["ave_vr"][best_idx]:.3f} km/s', zorder=10)
        plt.xlabel('Realization', fontsize=12)
        plt.ylabel('Average Rupture Velocity (km/s)', fontsize=12)
        plt.title(f'Rupture Velocity (Best: #{best_idx+1})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_spectral_comparison(self, figsize=(14, 6)):
        """Plot Moment Rate Spectrum and Octave-Averaged Spectrum side by side."""
        import matplotlib.pyplot as plt
        if self.subfaults is None or 'spectrum' not in self.subfaults:
            print("No source statistics available. Run simulation first.")
            return
        
        spectrum = self.subfaults['spectrum']
        octave = self.subfaults['spectrum_octave']
        plt.figure(figsize=figsize)
        
        # Normalize to compare shapes
        synth_norm = spectrum['moment_rate_synth'] / spectrum['moment_rate_synth'].max()
        dcf_norm = spectrum['moment_rate_dcf'] / spectrum['moment_rate_dcf'].max()
        
        # Subplot 1: Full Spectrum (normalized)
        plt.subplot(1, 2, 1)
        plt.loglog(spectrum['freq'], synth_norm, color='tab:blue', lw=2.5, label='Synthetic Model')
        plt.loglog(spectrum['freq'], dcf_norm, color='tab:red', lw=2.5, ls='--', label='DCF Target')
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Normalized Moment Rate', fontsize=12)
        plt.title('Moment Rate Spectrum', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, which='both', alpha=0.3)
        
        # Subplot 2: Octave-Averaged (normalize in log space)
        plt.subplot(1, 2, 2)

        # Normalize: subtract minimum, then divide by max to get 0-1 range
        synth_log_norm = octave['logmean_synth'] - octave['logmean_synth'].min()
        synth_log_norm = synth_log_norm / synth_log_norm.max()

        dcf_log_norm = octave['logmean_dcf'] - octave['logmean_dcf'].min()
        dcf_log_norm = dcf_log_norm / dcf_log_norm.max()

        plt.semilogx(octave['freq_center'], synth_log_norm, 'o-', color='tab:blue', lw=2.5, markersize=8, label='Synthetic (log-mean)')
        plt.semilogx(octave['freq_center'], dcf_log_norm, 's--', color='tab:red', lw=2.5, markersize=8, label='DCF Target (log-mean)')
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Normalized Log Mean Amplitude', fontsize=12)
        plt.title('Octave-Averaged Spectrum', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)

    def plot_source_time_function(self, figsize=(10, 6),xlim=None,
                                save_fig=False, model_name='source', image_type='png'):
        """Plot Source Time Function (STF).

        The stored STF is normalised to unit area by the kernel, so its unit is
        1/s, not N*m/s.  Multiply by the scalar moment to obtain a physical
        moment rate: ``M0 = 10 ** (1.5 * magnitude + 9.05)``, using FFSP's own
        magnitude constant 9.05 rather than the SI 9.1 of Hanks & Kanamori.
        """
        import matplotlib.pyplot as plt
        if self.subfaults is None or 'stf_time' not in self.subfaults:
            print("No source statistics available. Run simulation first.")
            return

        stf = self.subfaults['stf_time']
        # The kernel stores the STF with unit area, so it is a rate in 1/s.
        # Scaling by the scalar moment turns it into a physical moment rate.
        M0 = 10.0 ** (1.5 * float(self.params['magnitude']) + FFSP_MAGNITUDE_CONSTANT)
        moment_rate = np.asarray(stf['stf'], dtype=float) * M0

        plt.figure(figsize=figsize)
        plt.plot(stf['time'], moment_rate, color='black', lw=1.5, label='STF')
        plt.fill_between(stf['time'], 0, moment_rate, alpha=0.3, color='tab:cyan')
        max_idx = np.argmax(moment_rate)
        plt.plot(stf['time'][max_idx], moment_rate[max_idx], 'o', color='tab:red', markersize=12,
                 label=f'Peak {moment_rate[max_idx]:.3e} N m/s at t={stf["time"][max_idx]:.2f} s')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(r'$\dot{M}_0$ (N m/s)', fontsize=12)
        plt.title('Source Time Function (STF)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        if xlim is not None:
            plt.xlim(xlim)
        plt.tight_layout()

        # Save figure if requested
        if save_fig:
            plt.savefig(f'{model_name}_source_time_function.{image_type}',
                        format=image_type, 
                        dpi=600, 
                        bbox_inches='tight', 
                        transparent=True,
                        facecolor='none')

        plt.show()

    def plot_crust_layers(self, figsize=(6, 4)):
        """Plot crust model layers"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        nlayers = self.crust_model.nlayers
        thicknesses = []
        labels = []
        
        for i in range(nlayers):
            if self.crust_model.d[i] == 0:  # Half-space
                thicknesses.append(10)  # Arbitrary for plotting
                labels.append(f'Layer {i+1}: ∞ (Half-space)')
            else:
                thicknesses.append(self.crust_model.d[i])
                labels.append(f'Layer {i+1}: {self.crust_model.d[i]:.1f} km')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, nlayers))
        
        bottom = 0
        for i in range(nlayers):
            ax.bar(0, thicknesses[i], bottom=bottom, width=1,
                   color=colors[i], edgecolor='black', linewidth=1.5,
                   label=labels[i])
            bottom += thicknesses[i]
        
        ax.set_ylabel('Depth (km)', fontsize=12)
        ax.set_title('Crust Model Layers', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(bottom, 0)  # Inverted
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()

        
    def create_animation(self,
                             field='slip',
                             figsize=(10, 8),
                             cmap='cmap_white',
                             show_contours=True,
                             contour_field='rupture_time',
                             show_hypocenter=True,
                             contour_interval=None,
                             contour_color='black',
                             internal_ref=None,
                             external_coord=None,
                             rotate=False,
                             n_frames=50,
                             fps=10,
                             dpi=100,
                             output_dir='animation_frames',
                             output_video='rupture_animation.mp4',
                             ffmpeg_path=None):
            """Create rupture propagation animation.

            Generates one frame per time step, masking subfaults that have not
            yet ruptured, then assembles frames into a video with ffmpeg.

            Parameters
            ----------
            field : str, default 'slip'
                Field to display. One of 'slip', 'rise_time', 'peak_time'.
            figsize : tuple, default (10, 8)
            cmap : str, default 'cmap_white'
                Colormap. Use 'cmap_white' for white-start YlOrRd.
            show_contours : bool, default True
                Overlay rupture time isochrones.
            contour_field : str, default 'rupture_time'
            show_hypocenter : bool, default True
            contour_interval : float, optional
                Isochrone interval in seconds. Auto if None.
            contour_color : str, default 'black'
            internal_ref : list [x, y], optional
                Reference point in FFSP local coords (km).
            external_coord : list [x, y], optional
                Target position in ShakerMaker coords (km).
            rotate : bool, default False
            n_frames : int, default 50
            fps : int, default 10
            dpi : int, default 100
            output_dir : str, default 'animation_frames'
            output_video : str, default 'rupture_animation.mp4'
            ffmpeg_path : str, optional
                Full path to ffmpeg binary. Uses system ffmpeg if None.
            """
            import os
            import shutil
            import subprocess
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt

            os.makedirs(output_dir, exist_ok=True)

            # --- Colormap ---
            cmap_base = plt.get_cmap('YlOrRd', 256)
            colors_arr = cmap_base(np.linspace(0, 1, 256))
            colors_arr[0] = [1, 1, 1, 1]
            cmap_white = mcolors.ListedColormap(colors_arr)
            cmap_plot  = cmap_white if cmap == 'cmap_white' else plt.get_cmap(cmap)

            # --- Geometry ---
            nx       = int(self.params['nsubx'])
            ny       = int(self.params['nsuby'])
            lx       = self.params['fault_length']
            ly       = self.params['fault_width']
            cxp      = self.params['x_hypc']
            cyp      = self.params['y_hypc']
            strike   = self.params['strike']

            field_labels = {
                'slip': 'Slip [m]',
                'rise_time': 'Rise Time [s]',
                'peak_time': 'Peak Time [s]',
            }

            # --- Grid data ---
            rupture_time = np.transpose(self.subfaults['rupture_time'].reshape(nx, ny))
            field_data   = np.transpose(self.subfaults[field].reshape(nx, ny))
            contour_data = np.transpose(self.subfaults[contour_field].reshape(nx, ny))

            vmin = field_data.min()
            vmax = field_data.max()

            # --- Coordinates (same as plot_spacial_distribution) ---
            x_local = np.linspace(-lx/2, lx/2, nx)
            y_local = np.linspace(0, ly, ny)

            if rotate:
                strike_rad = np.radians(strike) + np.radians(90)
            else:
                strike_rad = np.radians(strike)

            X_local, Y_local = np.meshgrid(x_local, y_local)
            X_centered = X_local - (cxp - lx/2)
            Y_centered = Y_local - cyp
            X_rot = X_centered * np.sin(strike_rad) + Y_centered * np.cos(strike_rad)
            Y_rot = X_centered * np.cos(strike_rad) - Y_centered * np.sin(strike_rad)

            if internal_ref is not None and external_coord is not None:
                ref_x, ref_y = internal_ref
                if rotate:
                    ext_y, ext_x = external_coord
                else:
                    ext_x, ext_y = external_coord
                ref_x_rot = ref_x * np.sin(strike_rad) + ref_y * np.cos(strike_rad)
                ref_y_rot = ref_x * np.cos(strike_rad) - ref_y * np.sin(strike_rad)
                offset_x  = ext_x - ref_x_rot
                offset_y  = ext_y - ref_y_rot
                X = X_rot + offset_x
                Y = Y_rot + offset_y
                hypo_x = self.params['xref_hypc'] * np.sin(strike_rad) + self.params['yref_hypc'] * np.cos(strike_rad) + offset_x
                hypo_y = self.params['xref_hypc'] * np.cos(strike_rad) - self.params['yref_hypc'] * np.sin(strike_rad) + offset_y
                xlabel = 'UTM Easting, X [km]' if not rotate else 'UTM Northing, Y [km]'
                ylabel = 'UTM Northing, Y [km]' if not rotate else 'UTM Easting, X [km]'
            else:
                X = X_rot + self.params['xref_hypc']
                Y = Y_rot + self.params['yref_hypc']
                hypo_x = self.params['xref_hypc']
                hypo_y = self.params['yref_hypc']
                xlabel = 'Along Strike [km]' if rotate else 'Along Dip [km]'
                ylabel = 'Along Dip [km]'    if rotate else 'Along Strike [km]'

            # --- Time steps ---
            t_max      = rupture_time.max()
            time_steps = np.linspace(0, t_max, n_frames)

            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            pad = 0.05

            print(f"Rendering {n_frames} frames → {output_dir}/")

            for i, t in enumerate(time_steps):
                mask         = rupture_time <= t
                field_masked = np.ma.masked_where(~mask, field_data)

                fig, ax = plt.subplots(figsize=figsize)
                ax.set_facecolor('#E8E8E8')
                im = ax.pcolormesh(X, Y, field_masked, cmap=cmap_plot,
                                   shading='auto', vmin=vmin, vmax=vmax)

                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.set_label(field_labels.get(field, field), fontsize=10)

                if show_contours and mask.any():
                    try:
                        if contour_interval is not None:
                            levels = np.arange(0, contour_data[mask].max() + contour_interval,
                                               contour_interval)
                            contour_data_masked = np.where(mask, contour_data, np.nan)
                            cs = ax.contour(X, Y, contour_data_masked, levels=levels,
                                            colors=contour_color, linewidths=1.5)
                        else:
                            contour_data_masked = np.where(mask, contour_data, np.nan)
                            cs = ax.contour(X, Y, contour_data_masked, 8,
                                            colors=contour_color, linewidths=1.5)
                        ax.clabel(cs, fontsize=8, fmt='%.1f', inline=True)
                    except Exception:
                        pass

                if show_hypocenter:
                    ax.scatter(hypo_x, hypo_y, c='white', s=300, marker='*',
                               edgecolors='black', linewidth=2, zorder=10)

                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(f't = {t:.2f} s', fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
                ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
                ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)

                for spine in ax.spines.values():
                    spine.set_visible(True)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'frame_{i:03d}.png'), dpi=dpi)
                plt.close(fig)
                print(f'Frame {i+1}/{n_frames}', end='\r')

            print(f'\nFrames saved to: {output_dir}')

            # --- Assemble video ---
            try:
                ffmpeg_exe = ffmpeg_path or shutil.which('ffmpeg') or 'ffmpeg'
                subprocess.run([
                    ffmpeg_exe, '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(output_dir, 'frame_%03d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',
                    output_video
                ], check=True, capture_output=True)
                print(f'Video saved: {output_video}')
            except subprocess.CalledProcessError as e:
                print(f'ffmpeg error: {e.stderr.decode()[-300:]}')
            except FileNotFoundError:
                print(f'ffmpeg not found. Frames are in: {output_dir}')
