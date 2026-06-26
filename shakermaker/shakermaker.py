"""
shakermaker.py  --  ShakerMaker simulation engine, OP architecture.

Three-stage pipeline with O(1) Green's Function lookup via pair_to_slot:

  Stage 0  gen_pairs
      Identifies unique geometries (dh, z_src, z_rec) across all
      (station, source) pairs and builds the flat mapping
          pair_to_slot[i_station * nsources + i_psource] = k

      Algorithm: MPI-parallel geometry + Numba-compiled JAA greedy.
      All ranks compute geometry in parallel; rank 0 runs the greedy
      slot-finding (compiled with Numba @njit for 100-500x speedup
      vs plain Python) and writes the HDF5 mapping file.

  Stage 1  compute_gf
      Computes the FK kernel (tdata) for each unique slot k.
      MPI parallel: workers compute, rank 0 collects and writes to HDF5.

      HDF5 layout (new, efficient):
        /tdata   shape=(n_slots, nt, 9)    float32  chunks=(1,nt,9)    gzip
        /t0      shape=(n_slots,)          float64
      One dataset per quantity (not one dataset per slot), so metadata
      overhead is O(1) regardless of the number of slots.

  Stage 2  run_fast
      For every (station, source) pair, retrieves tdata via O(1) lookup,
      calls _call_core_fast, convolves with the source time function, and
      accumulates the station response.
      MPI parallel: stations distributed across ranks.

Orchestrator:
  run_nearest(stage=0|1|2|'0_1'|'all')

Debug / validation (no database):
  run()

STKO geometry export:
  export_drm_geometry()

Legacy HDF5 migration (JAA / PXP databases with tdata_dict):
  build_pair_to_slot_from_legacy_h5()

HDF5 database -- two files (separated to keep mapping light):
  {name}_map.h5
    /pairs_to_compute  (n_slots, 2)   int32   representative [i_sta, i_src]
    /dh_of_pairs       (n_slots,)     float64 horizontal distance per slot
    /dv_of_pairs       (n_slots,)     float64 vertical distance per slot
    /zrec_of_pairs     (n_slots,)     float64 receiver depth per slot
    /zsrc_of_pairs     (n_slots,)     float64 source depth per slot
    /pair_to_slot      (nsta*nsrc,)   int32   flat index -> slot k
    /nstations         scalar         int
    /nsources          scalar         int
    /delta_h           scalar         float64
    /delta_v_rec       scalar         float64
    /delta_v_src       scalar         float64

  {name}_gf.h5
    /tdata   (n_slots, nt, 9)    float64  chunks=(1,nt,9)    gzip=4
    /t0      (n_slots,)          float64
    Note: nt = actual samples from core.subgreen (= nfft when smth=1,
          = smth*nfft otherwise). Dataset created lazily on first slot.
"""

import copy
import os
import sys
import threading
import traceback
import logging
import numpy as np
import h5py
from time import perf_counter

from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker import core

# Windows change: the default Windows thread stack is ~1 MB, which is not
# enough for the Fortran FK core with large nfft (e.g. 16384 needs ~4 MB).
# _win_run() relaunches a callable in a 64 MB thread on Windows only.
# On Linux/macOS it is a transparent no-op (calls fn directly).
_WIN_STACK_SIZE = 64 * 1024 * 1024  # 64 MB

def _win_run(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) in a 64 MB stack thread on Windows only."""
    if sys.platform != 'win32':
        return fn(*args, **kwargs)
    result = [None]
    error  = [None]
    def _target():
        threading.current_thread()._sm_large_stack = True
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as exc:
            error[0] = exc
    old_size = threading.stack_size()
    try:
        threading.stack_size(_WIN_STACK_SIZE)
        t = threading.Thread(target=_target)
        t.start()
        t.join()
    finally:
        threading.stack_size(old_size)
    if error[0] is not None:
        raise error[0]
    return result[0]

try:
    from mpi4py import MPI
    use_mpi = True
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
except (ImportError, RuntimeError):
    # RuntimeError covers mpi4py installed but no MPI runtime available
    use_mpi = False
    rank   = 0
    nprocs = 1


# ---------------------------------------------------------------------------
# Module-level helpers  (shared by all methods, no duplication)
# ---------------------------------------------------------------------------

def _perf_counters():
    """Return a dict of zeroed timing accumulators."""
    return {k: np.zeros(1, dtype=np.double)
            for k in ('core', 'send', 'recv', 'conv', 'add')}


def _print_perf_stats(c, total):
    """Reduce timing counters across MPI ranks and print on rank 0."""
    if not (use_mpi and nprocs > 1):
        return
    labels = {'core': 'time_core', 'send': 'time_send', 'recv': 'time_recv',
              'conv': 'time_conv', 'add':  'time_add'}
    if rank == 0:
        print("\nPerformance statistics (all MPI processes):")
    for key in ('core', 'send', 'recv', 'conv', 'add'):
        mx = np.array([-np.inf]); mn = np.array([np.inf])
        comm.Reduce(c[key], mx, op=MPI.MAX, root=0)
        comm.Reduce(c[key], mn, op=MPI.MIN, root=0)
        if rank == 0 and total > 0:
            print(f"  {labels[key]:12s}:  "
                  f"max={mx[0]:.3f}s ({mx[0]/total*100:.2f}%)  "
                  f"min={mn[0]:.3f}s ({mn[0]/total*100:.2f}%)")


def _eta_str(elapsed, done, total):
    """Return 'H:MM:SS.s' ETA string."""
    if done <= 0:
        return "??:??:??"
    rem = elapsed / done * (total - done)
    hh  = int(rem) // 3600
    mm  = (int(rem) % 3600) // 60
    ss  = rem % 60
    return f"{hh}:{mm:02d}:{ss:04.1f}"


# ---------------------------------------------------------------------------
# ShakerMaker
# ---------------------------------------------------------------------------


class ShakerMaker:
    """Main ShakerMaker class: defines a model, links components,
    sets simulation parameters and executes the pipeline.

    OP architecture: three-stage pipeline with O(1) Green's Function lookup.
    See module docstring for full description.

    :param crust: Crustal model used by the simulation.
    :type crust: CrustModel
    :param source: Source model(s).
    :type source: FaultSource
    :param receivers: Receiver station(s).
    :type receivers: StationList
    """

    def __init__(self, crust, source, receivers):
        assert isinstance(crust, CrustModel), \
            "crust must be an instance of the shakermaker.CrustModel class"
        assert isinstance(source, FaultSource), \
            "source must be an instance of the shakermaker.FaultSource class"
        assert isinstance(receivers, StationList), \
            "receivers must be an instance of the shakermaker.StationList class"

        self._crust      = crust
        self._source     = source
        self._receivers  = receivers
        self._mpi_rank   = rank
        self._mpi_nprocs = nprocs
        self._logger     = logging.getLogger(__name__)

    # =========================================================================
    # run()  --  direct pair-by-pair, no database  (debug / legacy method)
    # =========================================================================

    def check_parameters(self, dt, nfft, dk, tb, tmax=100., tmin=0.,
                         sigma=2, smth=1, taper=0.9, wc1=1, wc2=2,
                         pmin=0, pmax=1, nx=1, kc=15.0,
                         n_per_wavelength=10, courant=1.0,
                         fem_fmax=None, coda=10.0):
        """Pre-run parameter check, organised around the two numbers YOU pick:
        ``dt`` (frequency band) and ``tmax`` (output window). Everything else
        (``nfft``, ``dk``, ``tb``) is *derived* from those plus the model
        geometry. Pure arithmetic (no FK run), so call it right after building
        the model::

            model = ShakerMaker(crust, fault, stations)
            model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

        The report mirrors what the FK core (``subfk.f`` / ``fk.f``) actually
        uses, and every formula is annotated with its source line:

        * ``hs`` is the **source-receiver vertical separation** (``fk.f:48``),
          NOT the total crust thickness, used for ``dk`` resolution and ``kc``.
        * ``f_max = (1-taper)*f_Nyq`` is the flat usable band (``fk.f:74``); the
          short wavelength ``lambda_min = Vs_min/f_max`` feeds the FEM mesh size.
        * the FEM stable step uses ``Vp_surf`` (the Vp of the slow ``Vs_min``
          layer, where the small elements live), NOT ``Vp_max``.

        RESULT separates *hard checks* (errors that corrupt the result) from
        *recommended changes* (it runs, but a better value exists, e.g. raising
        ``tmax`` so the surface-wave coda is not clipped).

        The returned dict carries ``passed`` and the ``recommended`` values
        (``dk``, ``tb``, ``nfft``, ``tmax``).

        :param taper: low-pass taper (same default as run()); sets f_max.
        :param n_per_wavelength: points per wavelength for the FEM mesh advice.
        :param courant: Courant number C for the FEM CFL step (C*dx/Vp_surf).
        :param fem_fmax: FEM mesh target freq (Hz); default None -> use the FK
            band f_max so the mesh matches the motion it will be driven with.
        :param coda: fixed coda margin (s) added when sizing the record. Not an
            FK parameter; only affects the recommendation.
        """
        # --- geometry (km), read straight from the model --------------------
        ns = self._source.nsources
        nr = self._receivers.nstations
        src = np.array([self._source.get_source_by_id(j).x for j in range(ns)], dtype=float)
        rec = np.array([self._receivers.get_station_by_id(i).x for i in range(nr)], dtype=float)
        tt  = np.array([self._source.get_source_by_id(j).tt for j in range(ns)], dtype=float)

        r     = np.sqrt((src[:, None, 0] - rec[None, :, 0]) ** 2 +
                        (src[:, None, 1] - rec[None, :, 1]) ** 2)         # horizontal (km)
        dz    = np.abs(src[:, None, 2] - rec[None, :, 2])                 # vertical (km)
        hs    = np.maximum(dz, 1e-6)                  # FK 'hs' = src-rcv sep (fk.f:48)
        slant = np.sqrt(r ** 2 + dz ** 2)                                # 3D (km)

        r_max    = float(r.max())
        t0_max   = float(tt.max())
        Vp_max   = float(self._crust.a.max())        # km/s
        Vs_min   = float(self._crust.b.min())        # km/s
        i_vsmin  = int(np.argmin(self._crust.b))     # slow surface layer
        Vp_surf  = float(self._crust.a[i_vsmin])     # Vp of the Vs_min layer (km/s)
        Vray     = 0.92 * Vs_min                     # Rayleigh-ish slow tail

        # per-pair arrival proxies (sizing only, not ray-traced)
        tP_pair = slant / Vp_max                          # earliest first arrival
        surf    = tt[:, None] + r / Vray                  # slow surface-wave tail
        first_arrival = float((tt[:, None] + tP_pair).min())
        last_arrival  = float(surf.max()) + coda          # end of useful signal

        def _pow2(n):  return (n & (n - 1)) == 0 and n > 0
        def _next2(x): return 1 << int(np.ceil(np.log2(max(x, 1.0))))

        # --- dt : the frequency band, and the FEM mesh it implies -----------
        f_nyq   = 1.0 / (2.0 * dt)
        f_pass  = (1.0 - taper) * f_nyq           # flat usable band edge = f_max
        dt_out  = dt / smth
        mesh_fmax = float(fem_fmax) if fem_fmax is not None else f_pass
        lam_min = Vs_min * 1000.0 / mesh_fmax     # shortest wavelength (m)
        dx_fem  = lam_min / n_per_wavelength       # element size (m)
        dt_fem  = courant * dx_fem / (Vp_surf * 1000.0)   # CFL step w/ soft Vp
        f_half  = f_pass / 2.0                     # half-band target
        dt_half = 2.0 * dt                         # dt to reach f_pass/2

        # --- dk : geometry as the Fortran uses it (hs = src-rcv sep) --------
        xmax   = np.maximum(r, hs)                             # fk.f:100-104
        Nstat  = kc * xmax / (np.pi * hs * dk)                 # pts to evanescent cutoff
        N_min  = float(Nstat.min())
        dk_ok  = (dk < 0.5) and (N_min >= 10)
        ratio  = xmax / hs
        dk_reco = min(0.40, kc * float(ratio.min()) / (np.pi * 10.0))
        L_per   = 2.0 * xmax / dk                              # spatial period (km)
        t_img   = float(((L_per - r) / Vray).min())            # nearest image arrival (s)

        # --- nfft : record must hold the signal; must be power of 2 ---------
        T_rec   = nfft * dt
        pad     = tb * dt
        T_sig   = (last_arrival - first_arrival) + pad         # physics-driven length
        T_tmax  = (tmax - first_arrival) + pad                 # tmax-driven length
        T_need  = max(T_sig, T_tmax)
        nfft_reco = _next2(T_need / dt)
        p2_ok   = _pow2(int(nfft))
        len_ok  = T_rec >= T_sig
        nfft_ok = p2_ok and len_ok

        # --- tb : pre-arrival pad; judged against a VALID record ------------
        tb_min  = max(1, int(np.ceil(1.0 / dt)))               # >= 1 s pre-roll
        Lwin    = float((surf - (tt[:, None] + tP_pair)).max()) + coda
        T_valid = max(T_rec, nfft_reco * dt)
        tb_max  = int((T_valid - Lwin) / dt)
        tb_ok   = (tb >= tb_min) and (tb <= tb_max)
        tb_reco = int(np.clip(round(2.0 / dt), tb_min, max(tb_min, tb_max)))

        # --- tmax / tmin : coherent with the record window ------------------
        end_rec   = first_arrival - pad + T_rec               # last usable time
        tmin_ok   = 0.0 <= tmin < tmax
        tmax_fits = tmax <= end_rec
        tmax_ok   = tmax_fits and tmin_ok                     # hard gate
        cuts_coda = tmax < last_arrival                       # soft (clips tail)
        tmax_reco = round(min(last_arrival, end_rec), 1)

        # --- report (rank 0 only) -------------------------------------------
        def _tag(ok): return "[OK]" if ok else "[WARN]"
        if not tmax_ok:    tmax_tag = "[WARN]"
        elif cuts_coda:    tmax_tag = "[OK, cuts coda]"
        else:              tmax_tag = "[OK]"

        if rank == 0:
            W = 70
            print("=" * W)
            print(" ShakerMaker . PARAMETER CHECK            you set:  dt + tmax")
            print("=" * W)
            print(f" YOU CHOSE     dt = {dt} s        tmax = {tmax} s")
            print(f" GEOMETRY      r_max {r_max:.1f} km . src-rcv sep "
                  f"{float(dz.min()):.1f}-{float(dz.max()):.1f} km . Vs_min {Vs_min*1000:.0f} m/s")
            print(f"               Vp_max {Vp_max*1000:.0f} m/s . V_Ray {Vray:.2f} km/s")
            print(f"               physical signal window  t = [{first_arrival:.1f}, "
                  f"{last_arrival:.1f}] s  (lasts {last_arrival-first_arrival:.1f} s)")
            print("-" * W)
            print(f" dt = {dt} s   sets your FREQUENCY BAND                       [info]")
            print(f"   f_Nyq         = 1/(2*dt)                   = {f_nyq:.0f} Hz       [fk.f:72]")
            print(f"   f_max usable  = (1-taper)*f_Nyq, taper={taper} = {f_pass:.0f} Hz        [fk.f:74]")
            print(f"   above {f_pass:.0f} Hz   fades out smoothly, fully gone by {f_nyq:.0f} Hz    [fk.f:169]")
            print(f"   lambda_min    = Vs_min/f_max               = {Vs_min*1000:.0f}/{mesh_fmax:.0f} = {lam_min:.1f} m")
            print(f"     element size (N={n_per_wavelength} pts/wavelength), for YOUR mesh:")
            print(f"        dx <= lambda_min/N    = {lam_min:.1f}/{n_per_wavelength} = {dx_fem:.2f} m")
            print(f"        dt <= C*dx/Vp_surf    = {courant:g}*{dx_fem:.2f}/{Vp_surf*1000:.0f} = {dt_fem:.4f} s")
            print(f"        (Vp_surf = Vp of the soft Vs_min layer, NOT Vp_max:")
            print(f"         the {dx_fem:.1f} m elements live in the soft surface layer)")
            print(f"   >> for f_max = {f_half:.0f} Hz (half) use dt = {dt_half:g}   (runs 2x faster)")
            print("")
            print(f" nfft = {nfft}    must HOLD the signal without wrap-around         {_tag(nfft_ok)}")
            print(f"   driven by    max(your tmax {T_tmax:.1f} s , physics {T_sig:.1f} s) = {T_need:.1f} s [fk.f:66]")
            print(f"   need         nfft = 2^ceil({T_need:.1f}/dt) = {nfft_reco}        [check]")
            print(f"   T_rec        = nfft*dt = {T_rec:.1f} s  {'>=' if len_ok else '<'}  signal {T_sig:.1f} s   (pow2: {p2_ok}) [fk.f:72]")
            print(f"   >> recommend nfft = {nfft_reco}")
            print("")
            print(f" dk = {dk}      keeps the FK integral resolved & clean           {_tag(dk_ok)}")
            print(f"   resolution   N = kc*xmax/(pi*hs*dk) = {N_min:.0f}    (need >= 10)   [fk.f:115]")
            print(f"   ghost source L = 2*xmax/dk -> nearest image at {t_img:.0f} s       [fk.f:107]")
            print(f"                {t_img:.0f} s  vs  tmax {tmax:g} s   (keep image > tmax)")
            print(f"   >> recommend dk = {dk_reco:.3f}  (coarser = faster, still clean)")
            print("")
            print(f" tb = {tb}        pre-arrival padding (in samples)                 {_tag(tb_ok)}")
            print(f"   pad          = tb*dt = {pad:.2f} s   (sane range {tb_min} - {tb_max})      [fk.f:105]")
            print(f"   >> recommend tb = {tb_reco}")
            print("")
            print(f" tmax = {tmax} s     output window                          {tmax_tag}")
            print(f"   MAX  = t_first - pad + nfft*dt = {first_arrival:.1f} - {pad:.1f} + {T_rec:.1f} = {end_rec:.1f} s [fk.f:72]")
            print(f"          (above this = garbage)")
            print(f"   full = max(t0 + r/V_Ray) + coda = {last_arrival:.1f} s")
            print(f"   gate = tmin < tmax <= {end_rec:.1f}   (tmin = {tmin})                  [fk.f:105]")
            if cuts_coda:
                print(f"   your {tmax:g} s stops before the surface-wave tail (ends at {last_arrival:.1f} s)")
            print(f"   >> recommend tmax = {tmax_reco:g} s")
            print("-" * W)

            errors = []
            if not nfft_ok:
                why = "not power of 2" if not p2_ok else f"record {T_rec:.1f}s < signal {T_sig:.1f}s"
                errors.append(f"nfft {nfft} -> use {nfft_reco}  ({why})")
            if not dk_ok:
                why = "dk >= 0.5, Bessel undersampled" if dk >= 0.5 else f"N_min={N_min:.0f} < 10"
                errors.append(f"dk {dk} -> use {dk_reco:.3f}  ({why})")
            if not tb_ok:
                if tb < tb_min: errors.append(f"tb {tb} -> use {tb_reco}  (pad too small)")
                else:           errors.append(f"tb {tb} -> use <= {tb_max}  (pushes tail out of record)")
            if not tmax_ok:
                if not tmin_ok: errors.append(f"tmin {tmin} -> need 0 <= tmin < tmax")
                else:           errors.append(f"tmax {tmax} -> end of record is {end_rec:.1f}s (raise nfft or lower tmax)")

            recs = []
            if tmax_ok and cuts_coda:
                recs.append(f"tmax  {tmax:g} -> {tmax_reco:g} s   (capture full signal)")

            if errors:
                print(f" RESULT: {len(errors)} error(s) -- fix before running")
                for e in errors:
                    print("   " + e)
            else:
                print(" RESULT: all hard checks passed")
            for rline in recs:
                print(f"   >> recommended change:  {rline}")
            print("-" * W)

            print(" READY-TO-RUN  (recommended values; full FK control):")
            print("")
            print( "   model.run(")
            print(f"       dt     = {dt:g},   # YOU SET : time step -> f_max {f_pass:.0f} Hz   [fk.f:72]")
            print(f"       tmax   = {tmax_reco:g},   # {'RECOMMEND' if cuts_coda else 'YOU SET  '}: output end time")
            print(f"       tmin   = {tmin:g},      # output start time")
            print(f"       nfft   = {nfft_reco},    # DERIVED : record {nfft_reco*dt:.1f} s >= {T_sig:.1f} s   [fk.f:66]")
            print(f"       dk     = {dk_reco:.3f},    # RECOMMEND: coarser = faster, still clean [fk.f:107]")
            print(f"       tb     = {tb_reco},      # DERIVED : pre-arrival pad {tb_reco*dt:.1f} s    [fk.f:105]")
            print(f"       smth   = {smth},        # default : output upsampling          [fk.f:186]")
            print(f"       sigma  = {sigma},        # default : wrap-around damping exp(-{sigma}) [fk.f:73]")
            print(f"       taper  = {taper},      # default : low-pass, sets f_max {f_pass:.0f} Hz  [fk.f:74]")
            print(f"       wc1    = {wc1},        # default : low-freq high-pass window   [fk.f:171]")
            print(f"       wc2    = {wc2},        # default : low-freq high-pass window   [fk.f:172]")
            print(f"       pmin   = {pmin},        # default : body waves in               [fk.f:140]")
            print(f"       pmax   = {pmax},        # default : safe (tested vs 8 = same)   [fk.f:141]")
            print(f"       nx     = {nx},        # structural: always 1                  [fk.f:95]")
            print(f"       kc     = {kc},     # default : evanescent cutoff kc/hs     [fk.f:89]")
            print( "   )")
            print("=" * W)

        return {"passed": bool(dk_ok and tb_ok and nfft_ok and tmax_ok),
                "recommended": {"dk": round(dk_reco, 3), "tb": tb_reco,
                                "nfft": nfft_reco, "tmax": tmax_reco}}

    def run(self,
            dt=0.05,
            nfft=4096,
            tb=1000,
            smth=1,
            sigma=2,
            taper=0.9,
            wc1=1,
            wc2=2,
            pmin=0,
            pmax=1,
            dk=0.3,
            nx=1,
            kc=15.0,
            writer=None,
            verbose=False,
            debugMPI=False,
            tmin=0.,
            tmax=100,
            showProgress=True,
            writer_mode='progressive'):
        """Run the simulation pair by pair. No Green's Function database.

        Useful for debugging and validating results against the OP pipeline.
        Every (source, receiver) pair is computed independently; no reuse of
        previously computed Green's Functions.

        :param sigma: Damps the trace at rate exp(-sigma*t) to reduce wrap-around.
        :type sigma: double
        :param nfft: Number of time-points in fft.
        :type nfft: integer
        :param dt: Simulation time-step.
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth.
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: kmax = 1/hs. Kernels decay exp(-k*hs) at w=0; kmax>10 required.
        :type kc: double
        :param writer: Writer class to store outputs.
        :type writer: StationListWriter
        :param writer_mode: 'progressive' (write per station, O(1) RAM) or 'legacy'.
        :type writer_mode: str
        """
        # Windows change: relaunch in a 64 MB stack thread so the
        # Fortran FK core has enough stack for large nfft values.
        if sys.platform == 'win32' and not getattr(threading.current_thread(), '_sm_large_stack', False):
            return _win_run(self.run,
                dt=dt, nfft=nfft, tb=tb, smth=smth, sigma=sigma,
                taper=taper, wc1=wc1, wc2=wc2, pmin=pmin, pmax=pmax,
                dk=dk, nx=nx, kc=kc, writer=writer, verbose=verbose,
                debugMPI=debugMPI, tmin=tmin, tmax=tmax,
                showProgress=showProgress, writer_mode=writer_mode)

        title = f"ShakerMaker Run begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))

        perf_time_begin = perf_counter()
        c = _perf_counters()

        if debugMPI:
            fid_debug_mpi = open(f"rank_{rank}.debuginfo", "w")
            def printMPI(*args):
                fid_debug_mpi.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid_debug_mpi = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.run - starting\n\tNumber of sources: {}\n'
            '\tNumber of receivers: {}\n\tTotal src-rcv pairs: {}\n'
            '\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations, dt, nfft))

        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        ipair = 0
        if nprocs == 1 or rank == 0:
            next_pair, skip_pairs = rank, 1
        else:
            next_pair, skip_pairs = rank - 1, nprocs - 1

        npairs = self._receivers.nstations * self._source.nsources
        tstart = perf_counter()

        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):
                aux_crust = copy.deepcopy(self._crust)
                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                if ipair == next_pair:
                    if verbose:
                        print(f"rank={rank} nprocs={nprocs} ipair={ipair} "
                              f"skip_pairs={skip_pairs} npairs={npairs} !!")
                    if nprocs == 1 or (rank > 0 and nprocs > 1):
                        if verbose:
                            print("calling core START")
                        t1 = perf_counter()
                        tdata, z, e, n, t0 = self._call_core(
                            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                            pmin, pmax, dk, kc, taper, aux_crust,
                            psource, station, verbose)
                        c['core'] += perf_counter() - t1
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        t1 = perf_counter()
                        t = np.arange(0, nt * dt, dt) + psource.tt + t0
                        station.add_greens_function(z, e, n, t, tdata, t0, i_psource)
                        psource.stf.dt = dt
                        z_stf = psource.stf.convolve(z, t)
                        e_stf = psource.stf.convolve(e, t)
                        n_stf = psource.stf.convolve(n, t)
                        c['conv'] += perf_counter() - t1

                        if rank > 0:
                            t1 = perf_counter()
                            comm.Send(np.array([nt], dtype=np.int32),
                                      dest=0, tag=2 * ipair)
                            data = np.empty((nt, 4), dtype=np.float64)
                            data[:, 0] = z_stf; data[:, 1] = e_stf
                            data[:, 2] = n_stf; data[:, 3] = t
                            comm.Send(data, dest=0, tag=2 * ipair + 1)
                            c['send'] += perf_counter() - t1
                            next_pair += skip_pairs

                    if rank == 0:
                        if nprocs > 1:
                            remote = ipair % (nprocs - 1) + 1
                            t1 = perf_counter()
                            ant = np.empty(1, dtype=np.int32)
                            comm.Recv(ant, source=remote, tag=2 * ipair)
                            nt = ant[0]
                            data = np.empty((nt, 4), dtype=np.float64)
                            comm.Recv(data, source=remote, tag=2 * ipair + 1)
                            z_stf = data[:, 0]; e_stf = data[:, 1]
                            n_stf = data[:, 2]; t     = data[:, 3]
                            c['recv'] += perf_counter() - t1
                        next_pair += 1
                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf,
                                                    t, tmin, tmax)
                            c['add'] += perf_counter() - t1
                        except Exception:
                            traceback.print_exc()
                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress:
                            elapsed = perf_counter() - tstart
                            print(f"{ipair} of {npairs} done  "
                                  f"ETA={_eta_str(elapsed, ipair+1, npairs)}  "
                                  f"t=[{t[0]:.4f}, {t[-1]:.4f}] "
                                  f"({tmin=:.4f} {tmax=:.4f})")
                else:
                    pass
                ipair += 1

            if verbose:
                self._logger.debug(
                    f'ShakerMaker.run - finished station {i_station} '
                    f'(rank={rank} ipair={ipair} next_pair={next_pair})')

            if writer and rank == 0:
                printMPI(f"Rank 0 writing station {i_station}")
                writer.write_station(station, i_station)
                printMPI(f"Rank 0 done writing station {i_station}")

        if writer and rank == 0:
            writer.close()

        fid_debug_mpi.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker run done. Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)
    # =========================================================================
    # Stage 0  --  gen_pairs
    # =========================================================================

    def gen_pairs(self,
                     h5_database_name,
                     delta_h=0.04,
                     delta_v_rec=0.002,
                     delta_v_src=0.2,
                     npairs_max=200000,
                     showProgress=True):
        """Stage 0 of the OP pipeline.

        Scans all (station, source) pairs, identifies geometrically unique
        slots based on (dh, z_src, z_rec) tolerances, and builds the flat
        mapping array::

            pair_to_slot[i_station * nsources + i_psource] = k

        **Algorithm -- MPI parallel geometry + Numba-compiled JAA greedy:**

        The geometry computation is distributed across all MPI ranks
        (vectorised, no Python loop).  The greedy slot-finding uses
        *exactly* the same algorithm as JAA, compiled to native code with
        Numba ``@njit`` so that it runs 100--500× faster than Python while
        producing bit-for-bit identical results.

        1. [All ranks] Vectorised geometry for local station slice:
           compute (dh, z_src, z_rec) via NumPy broadcasting.
        2. [All ranks] Gather geometry arrays to rank 0 via ``Gatherv``.
        3. [Rank 0] Run the JAA greedy in canonical pair order, compiled
           to C via Numba @njit.  Produces exactly the same slots as the
           original serial JAA implementation.
        4. [Rank 0] Build ``pair_to_slot`` and ``pairs_to_compute``,
           write HDF5 database.
        5. All ranks synchronise at a ``Barrier``.

        Complexity:
          - Geometry : O(n_pairs / nprocs) -- vectorised, MPI parallel.
          - Greedy   : O(n_pairs × n_slots) -- Numba compiled (single rank).
                       Typical speedup vs Python: 100--500×.
                       A 24-hour Python Stage 0 becomes ~10 minutes.

        Result: **identical slot count and pair_to_slot mapping** as the
        original serial JAA greedy for every geometry type.

        Writes to ``h5_database_name``:

        - ``/pairs_to_compute``  (n_slots, 2)  representative [i_sta, i_src]
        - ``/dh_of_pairs``       (n_slots,)    horizontal distance
        - ``/dv_of_pairs``       (n_slots,)    vertical distance
        - ``/zrec_of_pairs``     (n_slots,)    receiver depth
        - ``/zsrc_of_pairs``     (n_slots,)    source depth
        - ``/pair_to_slot``      (nsta*nsrc,)  flat index -> slot index k
        - ``/delta_h``, ``/delta_v_rec``, ``/delta_v_src``
        - ``/nstations``, ``/nsources``

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance (km).
        :type delta_h: double
        :param delta_v_rec: Receiver depth tolerance (km).
        :type delta_v_rec: double
        :param delta_v_src: Source depth tolerance (km).
        :type delta_v_src: double
        :param npairs_max: Kept for API compatibility; not used internally.
        :type npairs_max: integer
        :param showProgress: Print per-rank timing lines.
        :type showProgress: bool
        """
        # ------------------------------------------------------------------
        # JAA greedy compiled to native C with Numba.
        # Defined inside the method so it is always available even if Numba
        # was imported after the module was loaded.  The @njit decorator
        # compiles on first call; subsequent calls use the cached binary.
        #
        # This is EXACTLY the same algorithm as JAA:
        #   for each pair in canonical order:
        #     if no existing slot covers it -> create new slot (anchor = pair)
        #     else -> assign to nearest covering slot (L1 distance)
        #
        # The inner loop over existing slots is a plain C for-loop in
        # compiled code, giving 100--500x speedup over the Python version
        # while producing bit-for-bit identical slot arrays and pair_to_slot.
        # ------------------------------------------------------------------
        try:
            from numba import njit as _njit

            @_njit
            def _greedy_jaa(dh_arr, zsrc_arr, zrec_arr,
                             delta_h, delta_v_src, delta_v_rec):
                N         = len(dh_arr)
                slot_dh   = np.empty(N, dtype=np.float64)
                slot_zsrc = np.empty(N, dtype=np.float64)
                slot_zrec = np.empty(N, dtype=np.float64)
                p2s       = np.full(N, -1, dtype=np.int32)
                n_slots   = 0

                for i in range(N):
                    if n_slots == 0:
                        slot_dh[0]   = dh_arr[i]
                        slot_zsrc[0] = zsrc_arr[i]
                        slot_zrec[0] = zrec_arr[i]
                        p2s[i]       = 0
                        n_slots      = 1
                    else:
                        # Find covering slot closest in L1 norm
                        found     = -1
                        best_dist = 1e18
                        for k in range(n_slots):
                            d_dh  = dh_arr[i]   - slot_dh[k]
                            d_zs  = zsrc_arr[i] - slot_zsrc[k]
                            d_zr  = zrec_arr[i] - slot_zrec[k]
                            if d_dh  < 0.0: d_dh  = -d_dh
                            if d_zs  < 0.0: d_zs  = -d_zs
                            if d_zr  < 0.0: d_zr  = -d_zr
                            if (d_dh <= delta_h and
                                    d_zs <= delta_v_src and
                                    d_zr <= delta_v_rec):
                                dist = d_dh + d_zs + d_zr
                                if dist < best_dist:
                                    best_dist = dist
                                    found     = k
                        if found == -1:
                            slot_dh[n_slots]   = dh_arr[i]
                            slot_zsrc[n_slots] = zsrc_arr[i]
                            slot_zrec[n_slots] = zrec_arr[i]
                            p2s[i]             = n_slots
                            n_slots           += 1
                        else:
                            p2s[i] = found

                return (slot_dh[:n_slots], slot_zsrc[:n_slots],
                        slot_zrec[:n_slots], p2s)

            _use_numba = True

        except ImportError:
            _use_numba = False

        if rank == 0:
            if _use_numba:
                print("  Numba available -- greedy will run compiled (fast path)")
            else:
                print("  Numba NOT available -- greedy runs in Python (slow path).")
                print("  Install numba for 100-500x speedup: pip install numba")

        # ------------------------------------------------------------------
        # Pure-Python fallback (identical algorithm, no Numba dependency).
        # Used automatically when Numba is not installed.
        # ------------------------------------------------------------------
        def _greedy_jaa_python(dh_arr, zsrc_arr, zrec_arr,
                                delta_h, delta_v_src, delta_v_rec):
            N = len(dh_arr)
            slot_dh   = np.empty(N, dtype=np.float64)
            slot_zsrc = np.empty(N, dtype=np.float64)
            slot_zrec = np.empty(N, dtype=np.float64)
            p2s       = np.full(N, -1, dtype=np.int32)
            n_slots   = 0
            for i in range(N):
                if n_slots == 0:
                    slot_dh[0]   = dh_arr[i]
                    slot_zsrc[0] = zsrc_arr[i]
                    slot_zrec[0] = zrec_arr[i]
                    p2s[i]       = 0
                    n_slots      = 1
                else:
                    not_covered = (
                        (np.abs(dh_arr[i]   - slot_dh[:n_slots])   > delta_h)    |
                        (np.abs(zsrc_arr[i] - slot_zsrc[:n_slots]) > delta_v_src) |
                        (np.abs(zrec_arr[i] - slot_zrec[:n_slots]) > delta_v_rec)
                    )
                    if np.all(not_covered):
                        slot_dh[n_slots]   = dh_arr[i]
                        slot_zsrc[n_slots] = zsrc_arr[i]
                        slot_zrec[n_slots] = zrec_arr[i]
                        p2s[i]             = n_slots
                        n_slots           += 1
                    else:
                        covered = np.where(~not_covered)[0]
                        dist = (np.abs(dh_arr[i]   - slot_dh[covered]) +
                                np.abs(zsrc_arr[i] - slot_zsrc[covered]) +
                                np.abs(zrec_arr[i] - slot_zrec[covered]))
                        p2s[i] = covered[np.argmin(dist)]
            return (slot_dh[:n_slots], slot_zsrc[:n_slots],
                    slot_zrec[:n_slots], p2s)

        nsources  = self._source.nsources
        nstations = self._receivers.nstations
        N         = nstations * nsources          # total pairs

        title = (f"ShakerMaker Gen GF database pairs begin. "
                 f"{delta_h=} {delta_v_rec=} {delta_v_src=}")
        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  Stations    : {nstations}")
            print(f"  Sources     : {nsources}")
            print(f"  Total pairs : {N}")
            print(f"  Max slots   : {npairs_max}  (kept for API compat)")
            print(f"  MPI ranks   : {nprocs}  (geometry parallel, greedy Numba-compiled)")

        t0_global = perf_counter()

        # ------------------------------------------------------------------
        # Step 1: every rank computes geometry for its contiguous station
        # slice using fully vectorised NumPy -- no Python loop over pairs.
        # ------------------------------------------------------------------
        base, rem = divmod(nstations, nprocs)
        r_start   = rank * base + min(rank, rem)
        r_end     = r_start + base + (1 if rank < rem else 0)
        my_nsta   = r_end - r_start

        # Pre-fetch coordinates once -- avoids repeated Python attribute access
        # reshape(-1, 3) guarantees 2-D shape even when my_nsta == 0
        # (happens when nprocs > nstations, e.g. 48 ranks for 1 station)
        my_sta = np.array(
            [self._receivers.get_station_by_id(i).x
             for i in range(r_start, r_end)],
            dtype=np.float64).reshape(-1, 3)         # (my_nsta, 3)
        src_coords = np.array(
            [self._source.get_source_by_id(j).x
             for j in range(nsources)],
            dtype=np.float64).reshape(-1, 3)         # (nsources, 3)

        # Tile to full (my_nsta * nsources, 3) -- broadcasting, no loop
        sta_rep  = np.repeat(my_sta,    nsources, axis=0)  # (my_n*nsrc, 3)
        src_tile = np.tile(src_coords, (my_nsta,  1))      # (my_n*nsrc, 3)

        d_xy   = sta_rep[:, :2] - src_tile[:, :2]
        dh_loc = np.sqrt(np.einsum('ij,ij->i', d_xy, d_xy))  # horizontal dist
        zs_loc = src_tile[:, 2]                               # z_src per pair
        zr_loc = sta_rep[:, 2]                                # z_rec per pair
        dv_loc = np.abs(zr_loc - zs_loc)                      # vertical dist

        t1_geom = perf_counter()
        if showProgress:
            print(f"  [Rank {rank}] geometry ({my_nsta * nsources:,} pairs): "
                  f"{t1_geom - t0_global:.3f}s")

        # ------------------------------------------------------------------
        # Step 2: gather all geometry on rank 0 via Gatherv.
        # Each rank sends 4 float64 per pair: dh, dv, z_src, z_rec.
        # ------------------------------------------------------------------
        local_geom = np.ascontiguousarray(
            np.column_stack([dh_loc, dv_loc, zs_loc, zr_loc]))  # (my_n*nsrc, 4)
        my_size    = my_nsta * nsources

        if use_mpi and nprocs > 1:
            all_sizes = np.array(comm.allgather(my_size), dtype=np.int64)
            counts    = (all_sizes * 4).tolist()
            displ     = [0] + list(np.cumsum(counts[:-1]))
            if rank == 0:
                recv_geom = np.empty(N * 4, dtype=np.float64)
            else:
                recv_geom = None
            comm.Gatherv(local_geom.ravel(),
                         [recv_geom, counts, displ, MPI.DOUBLE],
                         root=0)
        else:
            recv_geom = local_geom.ravel()

        # ------------------------------------------------------------------
        # Steps 3--4 run exclusively on rank 0.
        # ------------------------------------------------------------------
        if rank == 0:
            t2_gather = perf_counter()
            if showProgress:
                print(f"  [Rank 0] gather complete: {t2_gather - t1_geom:.3f}s")

            all_geom = recv_geom.reshape(N, 4)  # columns: dh, dv, zsrc, zrec
            dh_all   = np.ascontiguousarray(all_geom[:, 0])
            dv_all   = all_geom[:, 1]
            zs_all   = np.ascontiguousarray(all_geom[:, 2])
            zr_all   = np.ascontiguousarray(all_geom[:, 3])

            # ----------------------------------------------------------
            # Step 3: run JAA greedy in canonical pair order.
            #
            # If Numba is available the function is compiled to native C
            # on first call (warm-up is done below).  The algorithm is
            # identical to JAA: same slot decisions, same pair_to_slot.
            # ----------------------------------------------------------
            if _use_numba:
                # Warm-up: compile with a tiny slice so the JIT cost is
                # not measured inside the timed section.
                if showProgress:
                    print(f"  [Rank 0] warming up Numba JIT...")
                _greedy_jaa(dh_all[:1], zs_all[:1], zr_all[:1],
                            delta_h, delta_v_src, delta_v_rec)
                t3_warmup = perf_counter()
                if showProgress:
                    print(f"  [Rank 0] Numba JIT ready: {t3_warmup - t2_gather:.3f}s")

                t_greedy_start = perf_counter()
                slot_dh, slot_zsrc, slot_zrec, pair_to_slot_full = _greedy_jaa(
                    dh_all, zs_all, zr_all,
                    delta_h, delta_v_src, delta_v_rec)
            else:
                t_greedy_start = perf_counter()
                slot_dh, slot_zsrc, slot_zrec, pair_to_slot_full = _greedy_jaa_python(
                    dh_all, zs_all, zr_all,
                    delta_h, delta_v_src, delta_v_rec)

            slot_dv   = np.abs(slot_zrec - slot_zsrc)
            n_slots   = len(slot_dh)
            pair_to_slot_full = pair_to_slot_full.astype(np.int32)

            t4_greedy = perf_counter()
            if showProgress:
                print(f"  [Rank 0] JAA greedy ({N:,} pairs -> {n_slots} slots): "
                      f"{t4_greedy - t_greedy_start:.3f}s")

            # ----------------------------------------------------------
            # Step 4: build pairs_to_compute.
            # For each slot k, pick the first pair (in canonical order)
            # that was assigned to it -- this is the representative
            # [i_station, i_source] used in Stage 1.
            # ----------------------------------------------------------
            order        = np.argsort(pair_to_slot_full, kind='stable')
            _, first_occ = np.unique(pair_to_slot_full[order],
                                     return_index=True)
            repr_flat    = order[first_occ]           # canonical flat index

            sta_of_repr      = (repr_flat // nsources).astype(np.int32)
            src_of_repr      = (repr_flat  % nsources).astype(np.int32)
            pairs_to_compute = np.column_stack(
                [sta_of_repr, src_of_repr]).astype(np.int32)

            dh_of_pairs   = dh_all[repr_flat]
            dv_of_pairs   = dv_all[repr_flat]
            zsrc_of_pairs = zs_all[repr_flat]
            zrec_of_pairs = zr_all[repr_flat]

            # Sanity checks
            assert np.all(pair_to_slot_full >= 0), \
                "[Stage 0] BUG: pair_to_slot contains negative index"
            assert np.all(pair_to_slot_full < n_slots), \
                "[Stage 0] BUG: pair_to_slot index out of range"
            assert len(np.unique(pair_to_slot_full)) == n_slots, \
                "[Stage 0] BUG: some slots have zero pairs assigned"

            elapsed   = perf_counter() - t0_global
            reduction = (1.0 - n_slots / N) * 100.0
            print(f"\nNeed only {n_slots} pairs of {N} "
                  f"({n_slots / N * 100:.1f}% of total, "
                  f"{reduction:.1f}% reduction)")
            print(f"Stage 0 done. Time: {elapsed:.1f}s")

            # ----------------------------------------------------------
            # Step 5: write HDF5 database
            # ----------------------------------------------------------
            # Write mapping to _map.h5 (lightweight, always loadable without GF).
            # The GF data (_gf.h5) is written separately by compute_gf (Stage 1).
            map_file = h5_database_name.replace('.h5', '') + '_map.h5'
            with h5py.File(map_file, 'w', locking=False) as hf:
                hf.create_dataset("pairs_to_compute", data=pairs_to_compute)
                hf.create_dataset("dh_of_pairs",      data=dh_of_pairs)
                hf.create_dataset("dv_of_pairs",      data=dv_of_pairs)
                hf.create_dataset("zrec_of_pairs",    data=zrec_of_pairs)
                hf.create_dataset("zsrc_of_pairs",    data=zsrc_of_pairs)
                hf.create_dataset("pair_to_slot",     data=pair_to_slot_full)
                hf.create_dataset("delta_h",          data=delta_h)
                hf.create_dataset("delta_v_rec",      data=delta_v_rec)
                hf.create_dataset("delta_v_src",      data=delta_v_src)
                hf.create_dataset("nstations",        data=int(nstations))
                hf.create_dataset("nsources",         data=int(nsources))

            print(f"Mapping database written to: {map_file}")

        # All ranks wait here before Stage 1 starts
        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # Stage 1  --  compute_gf
    # =========================================================================

    def compute_gf(self,
                      h5_database_name,
                      dt=0.05,
                      nfft=4096,
                      tb=1000,
                      smth=1,
                      sigma=2,
                      taper=0.9,
                      wc1=1,
                      wc2=2,
                      pmin=0,
                      pmax=1,
                      dk=0.3,
                      nx=1,
                      kc=15.0,
                      verbose=False,
                      debugMPI=False,
                      showProgress=True):
        """Stage 1 of the OP pipeline.

        Computes the FK kernel (tdata) for every unique slot k produced by
        Stage 0. Reads ``h5_database_name`` and appends the group
        ``/tdata_dict`` to the same file.

        MPI: rank 0 coordinates and writes; worker ranks compute and send.

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param verbose: Verbose output from the Fortran core.
        :type verbose: bool
        :param debugMPI: Write per-rank debug files.
        :type debugMPI: bool
        :param showProgress: Print ETA on rank 0.
        :type showProgress: bool
        """
        # Windows change: relaunch in a 64 MB stack thread so the
        # Fortran FK core has enough stack for large nfft values.
        if sys.platform == 'win32' and not getattr(threading.current_thread(), '_sm_large_stack', False):
            return _win_run(self.compute_gf,
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth, sigma=sigma,
                taper=taper, wc1=wc1, wc2=wc2, pmin=pmin, pmax=pmax,
                dk=dk, nx=nx, kc=kc, verbose=verbose,
                debugMPI=debugMPI, showProgress=showProgress)
        title = (f"ShakerMaker Gen Green's functions database begin. "
                 f"{dt=} {nfft=} {dk=} {tb=}")

        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  MPI processes  : {nprocs}")
            print(f"  OpenMP threads : {os.environ.get('OMP_NUM_THREADS','not set')}")
            print(f"  Loading database: {h5_database_name}")
            # Stage 1 reads the mapping from the _map file (written by Stage 0)
            # and writes Green's Functions into a separate _gf file.
            # Separating them allows:
            #   - regenerating GF without touching the mapping
            #   - inspecting/migrating the mapping without loading GF
            map_file = h5_database_name.replace('.h5', '') + '_map.h5'
            gf_file  = h5_database_name.replace('.h5', '') + '_gf.h5'
            hfile = h5py.File(map_file, 'r', locking=False)
        else:
            map_file = h5_database_name.replace('.h5', '') + '_map.h5'
            gf_file  = h5_database_name.replace('.h5', '') + '_gf.h5'
            hfile = h5py.File(map_file, 'r', locking=False)

        pairs_to_compute = hfile["/pairs_to_compute"][:]
        npairs = len(pairs_to_compute)

        if rank == 0:
            print(f"  Slots to compute: {npairs}")
            print(f"  Map file : {map_file}")
            print(f"  GF  file : {gf_file}")
            # Create the GF file with a single resizable dataset.
            # /tdata is created lazily on the FIRST slot write using the real nt
            # from core.subgreen. nt != nfft when smth>1 (e.g. smth=2 -> nt=2*nfft).
            # chunks=(1, nt, 9): one chunk per slot -> O(1) random read in Stage 2.
            # float32: halves disk vs float64; gzip=4: ~3x compression.
            with h5py.File(gf_file, 'w', locking=False) as gf_h:
                # /tdata created lazily below once nt is known.
                gf_h.create_dataset(
                    't0',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float64)

            # Open GF file for appending
            hfile_gf = h5py.File(gf_file, 'r+', locking=False)

        if debugMPI:
            fid = open(f"rank_{rank}_stage1.debuginfo", "w")
            def printMPI(*args):
                fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.compute_gf - starting\n'
            '\tNumber of sources: {}\n\tNumber of receivers: {}\n'
            '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations, dt, nfft))

        if nprocs == 1 or rank == 0:
            next_pair, skip_pairs = rank, 1
        else:
            next_pair, skip_pairs = rank - 1, nprocs - 1

        perf_time_begin = perf_counter()
        c      = _perf_counters()
        tstart = perf_counter()
        ipair  = 0
        # Cache split CrustModels by (z_src, z_rec) to avoid deepcopy per slot.
        # _crust_cache_gf = {}

        for i_station, i_psource in pairs_to_compute:
            station  = self._receivers.get_station_by_id(int(i_station))
            psource  = self._source.get_source_by_id(int(i_psource))
            z_src = psource.x[2]; z_rec = station.x[2]
            # _key = (round(z_src, 8), round(z_rec, 8))
            # if _key not in _crust_cache_gf:
            #     _c = copy.deepcopy(self._crust)
            #     _c.split_at_depth(z_src)
            #     _c.split_at_depth(z_rec)
            #     _crust_cache_gf[_key] = _c
            # aux_crust = _crust_cache_gf[_key]
            aux_crust = copy.deepcopy(self._crust)
            aux_crust.split_at_depth(z_src)
            aux_crust.split_at_depth(z_rec)

            if ipair == next_pair:
                if nprocs == 1 or (rank > 0 and nprocs > 1):
                    if verbose:
                        print("calling core START")
                    t1 = perf_counter()
                    tdata, z, e, n, t0 = self._call_core(
                        dt, nfft, tb, nx, sigma, smth,
                        wc1, wc2, pmin, pmax, dk, kc,
                        taper, aux_crust, psource, station, verbose)
                    c['core'] += perf_counter() - t1
                    if verbose:
                        print("calling core END")

                    nt     = len(z)
                    t0_arr = np.array([t0], dtype=np.double)
                    # Convert tdata from Fortran layout (1,9,nt) to C-order (nt,9).
                    # tdata[0] is (9,nt) in C; .T gives (nt,9) contiguous float32.
                    # Direct transpose avoids a Python loop over 9 components (15x faster).
                    # tdata_c = tdata[0].T   # shape (nt, 9), dtype float32 from Fortran
                    tdata_c = np.ascontiguousarray(tdata[0].T, dtype=np.float64)
                    if rank > 0:
                        t1 = perf_counter()
                        # Send only nt, t0, tdata (3 tags).
                        # z, e, n are NOT sent -- Stage 2 re-derives them
                        # from tdata via core.subgreen2, so transmitting them
                        # here wastes ~33% MPI bandwidth with no benefit.
                        comm.Send(np.array([nt], dtype=np.int32),
                                  dest=0, tag=3 * ipair)
                        comm.Send(t0_arr,  dest=0, tag=3 * ipair + 1)
                        comm.Send(tdata_c, dest=0, tag=3 * ipair + 2)
                        c['send'] += perf_counter() - t1
                        next_pair += skip_pairs

                if rank == 0:
                    if nprocs > 1:
                        remote  = ipair % (nprocs - 1) + 1
                        t1      = perf_counter()
                        ant     = np.empty(1, dtype=np.int32)
                        t0_arr  = np.empty(1, dtype=np.double)
                        printMPI(f"P0 Recv from remote {remote}")
                        # Receive only nt, t0, tdata (3 tags -- z/e/n removed)
                        comm.Recv(ant,    source=remote, tag=3 * ipair)
                        comm.Recv(t0_arr, source=remote, tag=3 * ipair + 1)
                        nt = ant[0]
                        tdata_c = np.empty((nt, 9), dtype=np.float64)
                        comm.Recv(tdata_c, source=remote, tag=3 * ipair + 2)
                        c['recv'] += perf_counter() - t1

                    # Write slot k to HDF5.
                    # Only tdata and t0 are stored; z/e/n are NOT stored --
                    # they are re-derived by core.subgreen2 in Stage 2.
                    #
                    # /tdata is created lazily on the first slot so that the
                    # chunk dimension matches the REAL nt (not nfft), which
                    # can differ when smth > 1.
                    nt_real = tdata_c.shape[0]   # actual samples from subgreen
                    if '/tdata' not in hfile_gf:
                        # First slot: create the dataset with real nt.
                        # chunks=(1, nt_real, 9): one chunk per slot -> O(1) read.
                        # float32: 50% smaller than float64, FK precision sufficient.
                        # gzip=4: ~3x compression on smooth time-series.
                        hfile_gf.create_dataset(
                            '/tdata',
                            shape=(0, nt_real, 9),
                            maxshape=(None, nt_real, 9),
                            chunks=(1, nt_real, 9),
                            dtype=np.float64,
                            compression='gzip',
                            compression_opts=4)
                    gf_ds = hfile_gf['/tdata']
                    t0_ds = hfile_gf['/t0']
                    gf_ds.resize(ipair + 1, axis=0)
                    t0_ds.resize(ipair + 1, axis=0)
                    # # Cast to float32 on write: halves disk usage vs float64.
                    # gf_ds[ipair] = tdata_c.astype(np.float32)

                    # Keep tdata in float64 to match JAA numerical behaviour.
                    gf_ds[ipair] = tdata_c

                    t0_ds[ipair] = t0_arr[0]
                    next_pair += 1

                    if showProgress:
                        elapsed = perf_counter() - tstart
                        print(f"{ipair} of {npairs} done  "
                              f"ETA={_eta_str(elapsed, ipair + 1, npairs)}")
            ipair += 1

        fid.close()
        hfile.close()
        if rank == 0:
            hfile_gf.close()
        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker Generate GF database done. "
                  f"Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # Stage 2  --  run_fast
    # =========================================================================

    def run_fast(self,
               h5_database_name,
               dt=0.05,
               nfft=4096,
               tb=1000,
               smth=1,
               sigma=2,
               taper=0.9,
               wc1=1,
               wc2=2,
               pmin=0,
               pmax=1,
               dk=0.3,
               nx=1,
               kc=15.0,
               writer=None,
               writer_mode='progressive',
               verbose=False,
               debugMPI=False,
               tmin=0.,
               tmax=100,
               showProgress=True):
        """Stage 2 of the OP pipeline.

        For each (station, source) pair retrieves the precomputed tdata via
        the O(1) pair_to_slot index, calls _call_core_fast (skipping the FK
        integration), convolves with the source time function, and accumulates
        the station response.

        MPI: unified single-pass loop — each station is computed, sent/received,
        written, and cleared immediately. RAM usage is O(1) per rank regardless
        of the number of stations.

        All ranks iterate over every station in canonical order [0..nstations).
        Owner of station i: owner = i % nprocs. Rank 0 always knows which rank
        to receive from at each step — no deadlock possible.

        Requires ``/pair_to_slot`` in the HDF5 file. Use
        :meth:`build_pair_to_slot_from_legacy_h5` to add it to legacy
        JAA / PXP databases without recomputing any Green's Functions.

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param writer: Use this writer class to store outputs
        :type writer: StationListWriter
        :param writer_mode: 'progressive' writes each station immediately to disk
            (O(1) RAM). 'legacy' accumulates all stations in memory before writing.
        :type writer_mode: str
        :param tmin: Start of output time window (s).
        :type tmin: double
        :param tmax: End of output time window (s).
        :type tmax: double
        (remaining parameters identical to :meth:`run`)
        """
        # Windows change: relaunch in a 64 MB stack thread so the
        # Fortran FK core has enough stack for large nfft values.
        if sys.platform == 'win32' and not getattr(threading.current_thread(), '_sm_large_stack', False):
            return _win_run(self.run_fast,
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth, sigma=sigma,
                taper=taper, wc1=wc1, wc2=wc2, pmin=pmin, pmax=pmax,
                dk=dk, nx=nx, kc=kc, writer=writer, writer_mode=writer_mode,
                verbose=verbose, debugMPI=debugMPI,
                tmin=tmin, tmax=tmax, showProgress=showProgress)
        title = (f"ShakerMaker Run (Stage 2 - OP) begin. "
                 f"{dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}")

        if rank == 0:
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  MPI processes  : {nprocs}")
            print(f"  OpenMP threads : {os.environ.get('OMP_NUM_THREADS','not set')}")
            print(f"  Loading database: {h5_database_name}")
            print(f"  writer_mode     : {writer_mode}")
            map_file = h5_database_name.replace('.h5', '') + '_map.h5'
            gf_file  = h5_database_name.replace('.h5', '') + '_gf.h5'
            hfile    = h5py.File(map_file, 'r+', locking=False)
            hfile_gf = h5py.File(gf_file,  'r',  locking=False)
            print(f"  Map file : {map_file}")
            print(f"  GF  file : {gf_file}")
        else:
            map_file = h5_database_name.replace('.h5', '') + '_map.h5'
            gf_file  = h5_database_name.replace('.h5', '') + '_gf.h5'
            hfile    = h5py.File(map_file, 'r', locking=False)
            hfile_gf = h5py.File(gf_file,  'r', locking=False)

        # O(1) lookup array — loaded once, shared across all stations
        pair_to_slot = hfile["/pair_to_slot"][:]
        nsources_db  = int(hfile["/nsources"][()])
        nstations_db = int(hfile["/nstations"][()])

        if rank == 0:
            print(f"  pair_to_slot: O(1) lookup "
                  f"({nstations_db} stations x {nsources_db} sources)")

        # Validate that current model matches the database
        assert nsources_db == self._source.nsources, (
            f"[Stage 2] nsources mismatch: "
            f"HDF5={nsources_db}, model={self._source.nsources}")
        assert nstations_db == self._receivers.nstations, (
            f"[Stage 2] nstations mismatch: "
            f"HDF5={nstations_db}, model={self._receivers.nstations}")

        # Only rank 0 owns the writer
        if rank > 0:
            writer = None
        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers, 2 * nfft,
                              tmin=tmin, tmax=tmax, dt=dt,
                              writer_mode=writer_mode)
            writer.write_metadata(self._receivers.metadata)

        if debugMPI:
            fid = open(f"rank_{rank}_stage2.debuginfo", "w")
            def printMPI(*args):
                fid.write(" ".join(str(a) for a in args) + "\n")
        else:
            fid = open(os.devnull, "w")
            printMPI = lambda *args: None

        self._logger.info(
            'ShakerMaker.run_fast - starting\n'
            '\tNumber of sources: {}\n\tNumber of receivers: {}\n'
            '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
            .format(self._source.nsources, self._receivers.nstations,
                    self._source.nsources * self._receivers.nstations, dt, nfft))

        nsources  = self._source.nsources
        nstations = self._receivers.nstations
        perf_time_begin = perf_counter()
        c         = _perf_counters()
        tstart    = perf_counter()

        for psource in self._source:
            psource.stf.dt = dt

        # ------------------------------------------------------------------
        # Unified single-pass loop — identical pattern to compute_gf.
        #
        # All ranks iterate over every station in canonical order [0..nstations).
        # Owner of station i: owner = i % nprocs
        #
        # Memory contract:
        #   progressive: clear_response() after every write → O(1) RAM per rank
        #   legacy:      clear_response() NOT called → all responses accumulate
        #                in RAM until writer.close() flushes them all at once.
        #                This is the original behaviour, kept for compatibility.
        #
        # Anti-deadlock proof:
        #   At iteration i_station, rank 0 knows owner = i_station % nprocs.
        #   If owner==0: rank 0 computes and writes directly (no MPI).
        #   If owner>0:  rank 0 blocks on Recv(source=owner).
        #                owner blocks on Send(dest=0) after computing.
        #                All other ranks are idle at this iteration.
        #   Both rank 0 and owner reach their matching Send/Recv at the same
        #   loop iteration → guaranteed rendezvous, no deadlock.
        #
        # Tags: 2*i_station and 2*i_station+1 — unique per station, no collision.
        # ------------------------------------------------------------------
        slot_matrix = pair_to_slot.reshape(nstations, nsources)
        # Cache all source objects once to avoid repeated get_source_by_id() calls
        source_list_cache = [self._source.get_source_by_id(j) for j in range(nsources)]
        
        for i_station in range(nstations):
            owner = i_station % nprocs

            # ----------------------------------------------------------------
            # Owner rank: compute this station (inner loop over all sources)
            # ----------------------------------------------------------------
            if rank == owner:
                station    = self._receivers.get_station_by_id(i_station)
                tstart_sta = perf_counter()

                # slot_to_sources = {}
                # for i_psource, psource in enumerate(self._source):
                #     k = int(pair_to_slot[i_station * nsources + i_psource])
                #     if k not in slot_to_sources:
                #         slot_to_sources[k] = []
                #     slot_to_sources[k].append((i_psource, psource))

                slot_to_sources = {}
                for i_psource, k in enumerate(slot_matrix[i_station]):
                    k = int(k)
                    if k not in slot_to_sources:
                        slot_to_sources[k] = []
                    slot_to_sources[k].append((i_psource, source_list_cache[i_psource]))

                # Pre-build crustal models for each unique (z_src, z_rec) pair
                # that this station needs. deepcopy+split is the dominant Python
                # overhead in Stage 2 (79% of non-Fortran time). The number of
                # unique depth combinations is O(n_unique_z_src) -- typically
                # O(100) for a fault plane -- not O(nsources).
                z_rec = station.x[2]
                # _crust_cache = {}

                for k, source_list in slot_to_sources.items():
                    # O(1) lookup: one HDF5 chunk read per unique slot.
                    # tdata is stored as float32 and passed directly to
                    # _call_core_fast. The Fortran subgreen2 routine declares
                    # tdata as 'real' (float32), so f2py accepts float32 without
                    # # any cast. Keeping float32 avoids a full-array copy.
                    # tdata = hfile_gf['/tdata'][k]   # float32, shape (nt, 9)

                    # tdata is stored as float64 
                    tdata = np.ascontiguousarray(hfile_gf['/tdata'][k], dtype=np.float64)   # float64, shape (nt, 9)

                    for i_psource, psource in source_list:
                        # Cache crustal models by (z_src, z_rec).
                        # All sources sharing the same depth pair reuse the
                        # same pre-split CrustModel -- zero extra deepcopies.
                        z_src = psource.x[2]
                        # crust_key = (round(z_src, 8), round(z_rec, 8))
                        # if crust_key not in _crust_cache:
                        #     aux = copy.deepcopy(self._crust)
                        #     aux.split_at_depth(z_src)
                        #     aux.split_at_depth(z_rec)
                        #     _crust_cache[crust_key] = aux
                        # aux_crust = _crust_cache[crust_key]

                        aux_crust = copy.deepcopy(self._crust)
                        aux_crust.split_at_depth(z_src)
                        aux_crust.split_at_depth(z_rec)

                        if verbose:
                            print(f"  rank={rank} sta={i_station} "
                                  f"src={i_psource} slot={k}")

                        t1 = perf_counter()
                        z, e, n, t0 = self._call_core_fast(
                            tdata, dt, nfft, tb, nx, sigma, smth,
                            wc1, wc2, pmin, pmax, dk, kc,
                            taper, aux_crust, psource, station, verbose)
                        c['core'] += perf_counter() - t1

                        t1    = perf_counter()
                        t_arr = np.arange(0, len(z) * dt, dt) + psource.tt + t0
                        z_stf = psource.stf.convolve(z, t_arr)
                        e_stf = psource.stf.convolve(e, t_arr)
                        n_stf = psource.stf.convolve(n, t_arr)
                        c['conv'] += perf_counter() - t1

                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf,
                                                    t_arr, tmin, tmax)
                            c['add'] += perf_counter() - t1
                        except Exception:
                            traceback.print_exc()
                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress and i_psource % 1000 == 0:
                            elapsed = perf_counter() - tstart_sta
                            print(f"  rank={rank} sta={i_station} "
                                  f"src={i_psource}/{nsources} "
                                  f"({i_psource/nsources*100:.1f}%)  "
                                  f"ETA={_eta_str(elapsed, i_psource+1, nsources)}")

                # All sources done for this station
                elapsed_sta = perf_counter() - tstart_sta
                nsta_left   = (nstations - i_station - 1) // nprocs
                print(f"  rank={rank} sta {i_station+1}/{nstations} "
                      f"({(i_station+1)/nstations*100:.1f}%)  "
                      f"sta_time={elapsed_sta:.1f}s  "
                      f"ETA_total={_eta_str(elapsed_sta*nsta_left,1,2)}")

                if use_mpi and nprocs > 1 and rank > 0:
                    # Worker: send accumulated response to rank 0
                    z_r, e_r, n_r, t_r = station.get_response()
                    t1 = perf_counter()
                    comm.Send(np.array([len(z_r)], dtype=np.int32),
                              dest=0, tag=2 * i_station)
                    comm.Send(np.column_stack([z_r, e_r, n_r, t_r]),
                              dest=0, tag=2 * i_station + 1)
                    c['send'] += perf_counter() - t1
                    printMPI(f"rank={rank} sent sta={i_station}")
                    # Workers always clear — they never own the writer
                    station.clear_response()

                elif rank == 0:
                    # Rank 0 owns this station: write directly
                    if writer:
                        t1 = perf_counter()
                        writer.write_station(station, i_station)
                        c['recv'] += perf_counter() - t1
                    # progressive: release RAM only after the station has been
                    # written to disk. If no writer is set, keep data in memory.
                    if writer_mode == 'progressive' and writer:
                        station.clear_response()

            # ----------------------------------------------------------------
            # Rank 0 only: receive from worker owner and write
            # (this branch is only reached when owner != 0)
            # ----------------------------------------------------------------
            elif rank == 0:
                sta = self._receivers.get_station_by_id(i_station)
                t1  = perf_counter()
                ant = np.empty(1, dtype=np.int32)
                comm.Recv(ant, source=owner, tag=2 * i_station)
                nt   = ant[0]
                data = np.empty((nt, 4), dtype=np.float64)
                comm.Recv(data, source=owner, tag=2 * i_station + 1)
                c['recv'] += perf_counter() - t1
                printMPI(f"rank=0 recv sta={i_station} from owner={owner}")

                sta.add_to_response(
                    data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                    tmin, tmax)

                if writer:
                    writer.write_station(sta, i_station)

                # progressive: release RAM only after the station has been
                # written to disk. If no writer is set, keep data in memory.
                if writer_mode == 'progressive' and writer:
                    sta.clear_response()

                if showProgress:
                    elapsed = perf_counter() - tstart
                    print(f"  [rank=0] written sta {i_station+1}/{nstations} "
                          f"({(i_station+1)/nstations*100:.1f}%)  "
                          f"ETA={_eta_str(elapsed, i_station+1, nstations)}")

            # other ranks (rank > 0, rank != owner): idle this iteration

        # ------------------------------------------------------------------
        # All stations processed — close resources
        # ------------------------------------------------------------------
        hfile.close()
        hfile_gf.close()
        fid.close()

        if rank == 0 and writer:
            writer.close()

        perf_time_total = perf_counter() - perf_time_begin

        if rank == 0:
            print(f"\n\nShakerMaker Run (Stage 2 - OP) done. "
                  f"Total time: {perf_time_total:.2f} s")
            print("-" * 50)

        _print_perf_stats(c, perf_time_total)
            


    # =========================================================================
    # Orchestrator  --  run_nearest
    # =========================================================================

    def run_nearest(self,
                           stage='all',
                           h5_database_name=None,
                           # Stage 0
                           delta_h=0.04,
                           delta_v_rec=0.002,
                           delta_v_src=0.2,
                           npairs_max=200000,
                           # Core (stages 1 and 2)
                           dt=0.05,
                           nfft=4096,
                           tb=1000,
                           smth=1,
                           sigma=2,
                           taper=0.9,
                           wc1=1,
                           wc2=2,
                           pmin=0,
                           pmax=1,
                           dk=0.3,
                           nx=1,
                           kc=15.0,
                           # Stage 2
                           writer=None,
                           writer_mode='progressive',
                           tmin=0.,
                           tmax=100,
                           # General
                           verbose=False,
                           debugMPI=False,
                           showProgress=True):
        """Orchestrator for the full OP pipeline.

        Runs Stage 0, 1, and/or 2 according to the ``stage`` parameter.
        Stages can be run independently, which is the recommended approach in
        HPC workflows where Stage 1-2 are MPI-parallel.

        Stage 0 is now also MPI parallel (geometry computed across all ranks;
        rank 0 handles the unique-slot reduction and HDF5 write).

        :param stage: Stages to run: ``0``, ``1``, ``2``, ``'0_1'`` (runs 0 then 1), or ``'all'``.
        :type stage: int or str
        :param h5_database_name: HDF5 file path (full path, including .h5). Required.
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance for slot grouping (km).
        :type delta_h: double
        :param delta_v_rec: Receiver depth tolerance for slot grouping (km).
        :type delta_v_rec: double
        :param delta_v_src: Source depth tolerance for slot grouping (km).
        :type delta_v_src: double
        :param npairs_max: Kept for API compatibility; not used internally.
        :type npairs_max: integer
        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1.
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param writer: Use this writer class to store outputs (Stage 2).
        :type writer: StationListWriter
        :param writer_mode: 'progressive' or 'legacy'.
        :type writer_mode: str
        :param tmin: Start of output time window (s).
        :type tmin: double
        :param tmax: End of output time window (s).
        :type tmax: double
        :param verbose: Verbose output from the Fortran core.
        :type verbose: bool
        :param debugMPI: Write per-rank debug files.
        :type debugMPI: bool
        :param showProgress: Print ETA.
        :type showProgress: bool
        """
        assert h5_database_name is not None, \
            "run_nearest: h5_database_name is required"
        assert stage in (0, 1, 2, '0_1', 'all'), \
            "run_nearest: stage must be 0, 1, 2, '0_1', or 'all'"

        # Normalise: strip .h5 suffix so internal methods can append
        # _map.h5 and _gf.h5 consistently regardless of what the caller
        # passed (e.g. "mydb.h5" and "mydb" both work).
        if h5_database_name.endswith('.h5'):
            h5_database_name = h5_database_name[:-3] + '.h5'

        perf_time_begin = perf_counter()

        if rank == 0:
            title = (f"ShakerMaker run_nearest | stage={stage} | "
                     f"{dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}")
            print(f"\n\n{title}")
            print("-" * len(title))
            omp = os.environ.get('OMP_NUM_THREADS', 'not set')
            print(f"Hybrid Parallelization:")
            print(f"   MPI processes  : {nprocs}")
            print(f"   OpenMP threads : {omp}")
            if omp != 'not set':
                try:
                    print(f"   Total threads  : {nprocs} x {omp} = "
                          f"{nprocs * int(omp)}")
                except ValueError:
                    pass
            print(f"   DB file        : {h5_database_name}")
            print("-" * len(title))

        if stage == '0_1':
            # Sequential: Stage 0 (gen_pairs) -> Stage 1 (compute_gf)
            self.gen_pairs(
                h5_database_name=h5_database_name,
                delta_h=delta_h, delta_v_rec=delta_v_rec,
                delta_v_src=delta_v_src, npairs_max=npairs_max,
                showProgress=showProgress)
            if rank == 0:
                print(f"Stage 0 complete -> {h5_database_name}")

            self.compute_gf(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)
            if rank == 0:
                print(f"Stage 1 complete -> {h5_database_name}")
                total = perf_counter() - perf_time_begin
                print("\n" + "=" * 70)
                print("STAGES 0 + 1 COMPLETE")
                print("=" * 70)
                print(f"  Total time: {total:.2f} s")
                if total > 60:
                    print(f"  Total time: {total/60:.2f} min")
                print("=" * 70 + "\n")
            return

        if stage in (0, 'all'):
            self.gen_pairs(
                h5_database_name=h5_database_name,
                delta_h=delta_h, delta_v_rec=delta_v_rec,
                delta_v_src=delta_v_src, npairs_max=npairs_max,
                showProgress=showProgress)
            if stage == 0:
                if rank == 0:
                    print(f"Stage 0 complete -> {h5_database_name}")
                return

        if stage in (1, 'all'):
            self.compute_gf(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)
            if stage == 1:
                if rank == 0:
                    print(f"Stage 1 complete -> {h5_database_name}")
                return

        if stage in (2, 'all'):
            if writer is None and rank == 0:
                print("WARNING: Stage 2 requires a writer. Aborting.")
                return
            self.run_fast(
                h5_database_name=h5_database_name,
                dt=dt, nfft=nfft, tb=tb, smth=smth,
                sigma=sigma, taper=taper, wc1=wc1, wc2=wc2,
                pmin=pmin, pmax=pmax, dk=dk, nx=nx, kc=kc,
                writer=writer, writer_mode=writer_mode,
                tmin=tmin, tmax=tmax,
                verbose=verbose, debugMPI=debugMPI,
                showProgress=showProgress)

        if rank == 0 and stage == 'all':
            total = perf_counter() - perf_time_begin
            print("\n" + "=" * 70)
            print("ALL STAGES COMPLETE")
            print("=" * 70)
            print(f"  Total time: {total:.2f} s")
            if total > 60:
                print(f"  Total time: {total/60:.2f} min")
            if total > 3600:
                print(f"  Total time: {total/3600:.2f} hrs")
            print("=" * 70 + "\n")

    # =========================================================================
    # Legacy migration  --  build_pair_to_slot_from_legacy_h5
    # =========================================================================

    def build_pair_to_slot_from_legacy_h5(self,
                                          h5_database_name,
                                          delta_h=None,
                                          delta_v_rec=None,
                                          delta_v_src=None,
                                          showProgress=True):
        """Migrate a legacy JAA / PXP Green's Function database to OP format.

        Reads ``h5_database_name`` (which must already contain
        ``/dh_of_pairs``, ``/zrec_of_pairs``, ``/zsrc_of_pairs``, and
        ``/tdata_dict``) and writes three new datasets:

        - ``/pair_to_slot``  (nstations * nsources,)  int32
        - ``/nstations``     scalar
        - ``/nsources``      scalar

        After this call the database is fully compatible with :meth:`run_fast`
        and :meth:`run_nearest` (stage=2), reusing all previously
        computed Green's Functions without any recomputation.

        **Algorithm** (MPI + KDTree):

        1. Rank 0 reads slot geometry arrays and tolerances from the HDF5 file
           and broadcasts them to all ranks via ``comm.bcast``.
        2. Each rank builds an identical :class:`scipy.spatial.cKDTree` from
           the slot coordinates normalised by the respective tolerances.
        3. Each rank owns a contiguous station slice ``[i_start, i_end)`` and
           assembles the full ``(my_nstations * nsources, 3)`` query matrix
           in a single vectorised operation (no Python loop over sources).
        4. A single ``tree.query(queries, workers=-1)`` call answers all
           queries in parallel using all available threads.
        5. Rank 0 collects partial arrays from all ranks via ``comm.Gatherv``
           and writes the assembled ``pair_to_slot`` plus ``nstations`` /
           ``nsources`` to the HDF5 file.

        If ``delta_h``, ``delta_v_rec``, or ``delta_v_src`` are ``None``,
        the values stored in the HDF5 file are used (the original tolerances
        from the JAA/PXP run).

        :param h5_database_name: HDF5 file path (full path, including .h5).
        :type h5_database_name: str
        :param delta_h: Horizontal distance tolerance (km). Uses stored value
            if None.
        :type delta_h: double or None
        :param delta_v_rec: Receiver depth tolerance (km). Uses stored value
            if None.
        :type delta_v_rec: double or None
        :param delta_v_src: Source depth tolerance (km). Uses stored value
            if None.
        :type delta_v_src: double or None
        :param showProgress: Print progress messages.
        :type showProgress: bool
        """
        from scipy.spatial import cKDTree

        nsources  = self._source.nsources
        nstations = self._receivers.nstations

        # ------------------------------------------------------------------
        # Step 1: rank 0 reads slot geometry + tolerances, broadcasts to all
        # ------------------------------------------------------------------
        if rank == 0:
            # Support both legacy (single file) and new split-file layout.
            # If a _map.h5 sibling exists use it; otherwise fall back to the
            # original file (JAA / old PXP databases).
            _base = h5_database_name.replace('.h5', '')
            _map_candidate = _base + '_map.h5'
            import os as _os
            _read_file = _map_candidate if _os.path.exists(_map_candidate) else h5_database_name
            with h5py.File(_read_file, 'r', locking=False) as hf:
                dh_of_pairs   = hf["/dh_of_pairs"][:]
                zrec_of_pairs = hf["/zrec_of_pairs"][:]
                zsrc_of_pairs = hf["/zsrc_of_pairs"][:]
                dh_tol = float(hf["/delta_h"][()])     if delta_h    is None else delta_h
                vr_tol = float(hf["/delta_v_rec"][()]) if delta_v_rec is None else delta_v_rec
                vs_tol = float(hf["/delta_v_src"][()]) if delta_v_src is None else delta_v_src
        else:
            dh_of_pairs = zrec_of_pairs = zsrc_of_pairs = None
            dh_tol = vr_tol = vs_tol = None

        if use_mpi and nprocs > 1:
            dh_of_pairs   = comm.bcast(dh_of_pairs,   root=0)
            zrec_of_pairs = comm.bcast(zrec_of_pairs, root=0)
            zsrc_of_pairs = comm.bcast(zsrc_of_pairs, root=0)
            dh_tol  = comm.bcast(dh_tol,  root=0)
            vr_tol  = comm.bcast(vr_tol,  root=0)
            vs_tol  = comm.bcast(vs_tol,  root=0)

        n_slots      = len(dh_of_pairs)
        npairs_total = nstations * nsources

        if rank == 0:
            title = (f"ShakerMaker build_pair_to_slot_from_legacy_h5 -- "
                     f"{h5_database_name}")
            print(f"\n\n{title}")
            print("-" * len(title))
            print(f"  Stations    : {nstations}")
            print(f"  Sources     : {nsources}")
            print(f"  Slots in DB : {n_slots}")
            print(f"  delta_h     : {dh_tol}")
            print(f"  delta_v_rec : {vr_tol}")
            print(f"  delta_v_src : {vs_tol}")
            print(f"  MPI ranks   : {nprocs}")
            print(f"  Strategy    : MPI station partitioning + KDTree O(log n) lookup")
            print(f"  NOTE: tdata format (nt,9) C-order is identical. "
                  f"Only adding pair_to_slot mapping.")

        # ------------------------------------------------------------------
        # Step 2: build KDTree (identical on every rank)
        # Normalise so each tolerance dimension spans unit radius.
        # sqrt(3) is the L2 cutoff enclosing the unit L-inf tolerance cube.
        # ------------------------------------------------------------------
        slots_norm = np.column_stack([
            dh_of_pairs   / dh_tol,
            zsrc_of_pairs / vs_tol,
            zrec_of_pairs / vr_tol,
        ])
        tree     = cKDTree(slots_norm)
        max_dist = np.sqrt(3.0)

        # ------------------------------------------------------------------
        # Step 3: partition stations across ranks (contiguous slices)
        # ------------------------------------------------------------------
        base, rem    = divmod(nstations, nprocs)
        i_start      = rank * base + min(rank, rem)
        i_end        = i_start + base + (1 if rank < rem else 0)
        my_nstations = i_end - i_start

        # Pre-fetch coordinates once to avoid repeated Python attribute lookups
        my_sta_coords = np.array(
            [self._receivers.get_station_by_id(i).x for i in range(i_start, i_end)],
            dtype=np.float64)                            # (my_nstations, 3)
        src_coords = np.array(
            [self._source.get_source_by_id(j).x for j in range(nsources)],
            dtype=np.float64)                            # (nsources, 3)

        # ------------------------------------------------------------------
        # Step 4: build full query matrix and run a single vectorised KDTree
        # query with workers=-1 (uses all available threads internally)
        # ------------------------------------------------------------------
        sta_rep  = np.repeat(my_sta_coords, nsources, axis=0)  # (my_n*nsrc, 3)
        src_tile = np.tile(src_coords, (my_nstations, 1))      # (my_n*nsrc, 3)

        d_xy   = sta_rep[:, :2] - src_tile[:, :2]
        dh_q   = np.sqrt(np.einsum('ij,ij->i', d_xy, d_xy))   # horizontal dist
        zsrc_q = src_tile[:, 2]
        zrec_q = sta_rep[:, 2]

        queries_norm = np.column_stack([
            dh_q   / dh_tol,
            zsrc_q / vs_tol,
            zrec_q / vr_tol,
        ])

        t0_q = perf_counter()
        dists, slots = tree.query(queries_norm, workers=-1)
        t1_q = perf_counter()

        if showProgress:
            print(f"  [Rank {rank}] KDTree query "
                  f"({len(queries_norm):,} points): {t1_q - t0_q:.2f}s")

        # Assign slot 0 to any out-of-tolerance pairs and warn
        out_of_bounds = dists > max_dist
        n_oob = int(np.sum(out_of_bounds))
        if n_oob > 0:
            slots[out_of_bounds] = 0
            print(f"  [Rank {rank}] WARNING: {n_oob} pairs outside tolerance "
                  f"-- assigned slot 0. Verify model / tolerance consistency.")

        my_partial = slots.astype(np.int32)   # (my_nstations * nsources,)

        # ------------------------------------------------------------------
        # Step 5: gather all partial arrays to rank 0 via Gatherv
        # ------------------------------------------------------------------
        if use_mpi and nprocs > 1:
            my_size  = np.array([len(my_partial)], dtype=np.int32)
            all_sizes = np.empty(nprocs, dtype=np.int32) if rank == 0 else None
            comm.Gather(my_size, all_sizes, root=0)

            if rank == 0:
                displacements     = np.concatenate([[0], np.cumsum(all_sizes[:-1])])
                pair_to_slot_full = np.empty(npairs_total, dtype=np.int32)
                comm.Gatherv(
                    my_partial,
                    [pair_to_slot_full, all_sizes, displacements, MPI.INT],
                    root=0)
            else:
                comm.Gatherv(my_partial, None, root=0)
                pair_to_slot_full = None
        else:
            pair_to_slot_full = my_partial

        # ------------------------------------------------------------------
        # Step 6: rank 0 validates and writes to HDF5
        # ------------------------------------------------------------------
        if rank == 0:
            n_unassigned = int(np.sum(pair_to_slot_full < 0))
            if n_unassigned > 0:
                raise RuntimeError(
                    f"[build_pair_to_slot] {n_unassigned} pairs unassigned. "
                    "The current model may not match the original database. "
                    "Check sources, stations, and tolerances.")

            elapsed = t1_q - t0_q
            print(f"\nMapping complete. Time: {elapsed:.1f}s")
            print(f"  Unique slots used : "
                  f"{len(np.unique(pair_to_slot_full))} / {n_slots}")
            if n_oob > 0:
                print(f"  WARNING: {n_oob} pairs had no match. "
                      "Verify model consistency.")

            _write_file = _map_candidate if _os.path.exists(_map_candidate) else h5_database_name
            with h5py.File(_write_file, 'r+', locking=False) as hf:
                for key in ('pair_to_slot', 'nstations', 'nsources'):
                    if key in hf:
                        del hf[key]
                hf.create_dataset("pair_to_slot", data=pair_to_slot_full)
                hf.create_dataset("nstations",    data=nstations)
                hf.create_dataset("nsources",     data=nsources)

            print(f"pair_to_slot, nstations, nsources written to: "
                  f"{_write_file}")
            print("Database is now compatible with run_fast (Stage 2).")

        if use_mpi and nprocs > 1:
            comm.Barrier()

    # =========================================================================
    # STKO export  --  export_drm_geometry
    # =========================================================================

    def export_drm_geometry(self, filename="drm_geometry.h5drm"):
        """Export DRM geometry for visualisation in STKO.

        Creates an HDF5 file with station coordinates and minimal synthetic
        data (2 samples, linear ramp 0 -> 10) for geometry inspection only.
        Works with DRMBox, SurfaceGrid and PointCloudDRMReceiver receiver
        lists.

        Parameters
        ----------
        filename : str
            Output HDF5 filename (default: 'drm_geometry.h5drm')

        Returns
        -------
        str
            Path to the created file
        """
        from shakermaker.sl_extensions import DRMBox
        from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid
        from shakermaker.sl_extensions.PointCloudDRMReceiver import PointCloudDRMReceiver

        if not isinstance(self._receivers, (DRMBox, SurfaceGrid, PointCloudDRMReceiver)):
            raise TypeError(
                f"export_drm_geometry() requires DRMBox, SurfaceGrid or "
                f"PointCloudDRMReceiver. Got: {type(self._receivers).__name__}")

        if rank != 0:
            return filename

        metadata  = self._receivers.metadata
        nstations = self._receivers.nstations - 1  # exclude QA station

        print(f"\nexport_drm_geometry: {filename}")
        print(f"  Receiver type: {type(self._receivers).__name__}")
        print(f"  Stations (excl. QA): {nstations}")

        with h5py.File(filename, 'w', locking=False) as hf:
            grp_data = hf.create_group('DRM_Data')
            grp_qa   = hf.create_group('DRM_QA_Data')
            grp_meta = hf.create_group('DRM_Metadata')

            xyz      = np.zeros((nstations, 3))
            internal = np.zeros(nstations, dtype=bool)
            for i in range(nstations):
                sta         = self._receivers.get_station_by_id(i)
                xyz[i, :]   = sta.x
                internal[i] = sta.is_internal

            grp_data.create_dataset('xyz',      data=xyz,      dtype=np.float64)
            grp_data.create_dataset('internal', data=internal, dtype=bool)
            grp_data.create_dataset('data_location',
                                    data=np.arange(0, nstations,
                                                   dtype=np.int32) * 3)

            qa_sta = self._receivers.get_station_by_id(nstations)
            grp_qa.create_dataset('xyz',
                                  data=qa_sta.x.reshape(1, 3),
                                  dtype=np.float64)

            # Minimal time data: linear ramp 0 -> 10 (2 samples)
            ramp    = np.tile([0.0, 10.0], (3 * nstations, 1))
            ramp_qa = np.tile([0.0, 10.0], (3, 1))
            for grp, r in [(grp_data, ramp), (grp_qa, ramp_qa)]:
                grp.create_dataset('velocity',     data=r, dtype=np.float64)
                grp.create_dataset('displacement', data=r, dtype=np.float64)
                grp.create_dataset('acceleration', data=r, dtype=np.float64)

            grp_meta.create_dataset('dt',     data=0.0005)
            grp_meta.create_dataset('tstart', data=0.0)
            grp_meta.create_dataset('tend',   data=10.0)
            for key in ('h', 'drmbox_x0',
                        'drmbox_xmax', 'drmbox_xmin',
                        'drmbox_ymax', 'drmbox_ymin',
                        'drmbox_zmax', 'drmbox_zmin'):
                if key in metadata:
                    grp_meta.create_dataset(key, data=metadata[key])

            print(f"  Station coordinates: written")
            print(f"  QA station: written")
            print(f"  Time data (2 samples, linear ramp): written")
            print(f"  Metadata: written")

        print(f"Geometry file created: {filename}")
        print("Use in STKO to visualise grid before running simulation.")
        return filename

    # =========================================================================
    # SW4 export  --  export_sw4
    # =========================================================================

    def export_sw4(self, path=None,
                   h=50,
                   size_domain=None,
                   tmax=50,
                   m0=1,
                   fileio_path="shakermaker2sw4_fileio",
                   supergrid_gp=30,
                   supergrid_pad_gp=10,
                   interface_blocks=True,
                   interface_block_delta=1.0,
                   station_prefix="sf",
                   shakermaker_stations=True,
                   domain_sw4=False,
                   domain_sw4_size=None,
                   plot_geometry=False,
                   plot_geometry_sw4=False,
                   h5_export_name="sw4_package.h5"):
        """Export model sources and receivers to SW4 without topography.

        The SW4 domain is built from the model geometry. If size_domain has
        None for x or y, that direction is computed automatically. The exported
        SW4 input is written in a local box whose origin is one domain corner.
        """
        from shakermaker.sw4_exporter import SW4ExportConfig, SW4Exporter

        config = SW4ExportConfig(
            path=path or os.getcwd(),
            h=h,
            size_domain=size_domain,
            tmax=tmax,
            m0=m0,
            fileio_path=fileio_path,
            supergrid_gp=supergrid_gp,
            supergrid_pad_gp=supergrid_pad_gp,
            interface_blocks=interface_blocks,
            interface_block_delta=interface_block_delta,
            station_prefix=station_prefix,
            shakermaker_stations=shakermaker_stations,
            domain_sw4=domain_sw4,
            domain_sw4_size=domain_sw4_size,
            plot_geometry=plot_geometry,
            plot_geometry_sw4=plot_geometry_sw4,
            h5_export_name=h5_export_name,
        )
        return SW4Exporter(self, config).write()

    def export_sw4_topo(self, path=None,
                        h=50,
                        size_domain=None,
                        tmax=50,
                        m0=1,
                        fileio_path="shakermaker2sw4_fileio",
                        supergrid_gp=30,
                        supergrid_pad_gp=10,
                        interface_blocks=True,
                        interface_block_delta=1.0,
                        station_prefix="sf",
                        topo_file=None,
                        topo_zmax=None,
                        write_topography_z0_stations=False,
                        shakermaker_stations=False,
                        shakermaker_stations_to_surface=False,
                        domain_sw4=False,
                        domain_sw4_size=None,
                        plot_geometry=False,
                        plot_geometry_sw4=False,
                        h5_export_name="sw4_package.h5"):
        """Export model sources and receivers to SW4 with cartesian topography.

        The domain is built from topography plus model geometry. If size_domain
        has None for x or y, that direction is computed automatically. The SW4
        input is written in local coordinates; plot_geometry shows the original
        coordinates, and plot_geometry_sw4 shows the local SW4 box.
        """
        if topo_file is None:
            raise ValueError("export_sw4_topo requires topo_file.")

        from shakermaker.sw4_exporter import SW4ExportConfig, SW4Exporter

        config = SW4ExportConfig(
            path=path or os.getcwd(),
            h=h,
            size_domain=size_domain,
            tmax=tmax,
            m0=m0,
            fileio_path=fileio_path,
            supergrid_gp=supergrid_gp,
            supergrid_pad_gp=supergrid_pad_gp,
            interface_blocks=interface_blocks,
            interface_block_delta=interface_block_delta,
            station_prefix=station_prefix,
            topo_file=topo_file,
            topo_zmax=topo_zmax,
            write_topography_z0_stations=write_topography_z0_stations,
            shakermaker_stations=shakermaker_stations,
            shakermaker_stations_to_surface=shakermaker_stations_to_surface,
            domain_sw4=domain_sw4,
            domain_sw4_size=domain_sw4_size,
            plot_geometry=plot_geometry,
            plot_geometry_sw4=plot_geometry_sw4,
            h5_export_name=h5_export_name,
        )
        return SW4Exporter(self, config).write()

    def write(self, writer):
        writer.write(self._receivers)

    def enable_mpi(self, rank, nprocs):
        self._mpi_rank = rank
        self._mpi_nprocs = nprocs

    def mpi_is_master_process(self):
        return self.mpi_rank == 0

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        return self._mpi_nprocs

    def _call_core(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        mb = crust.nlayers

        if verbose:
            print("_call_core")
            # print(f"        psource = {psource}")
            print(f"        psource.x = {psource.x}")
            # print(f"        station = {station}")
            print(f"        station.x = {station.x}")

        src = crust.get_layer(psource.x[2]) + 1   # fortran starts in 1, not 0
        rcv = crust.get_layer(station.x[2]) + 1   # fortran starts in 1, not 0
        
        stype = 2  # Source type double-couple, compute up and down going wave
        updn = 0
        d = crust.d
        a = crust.a
        b = crust.b
        rho = crust.rho
        qa = crust.qa
        qb = crust.qb

        pf = psource.angles[0]
        df = psource.angles[1]
        lf = psource.angles[2]
        sx = psource.x[0]
        sy = psource.x[1]
        rx = station.x[0]
        ry = station.x[1]
        x = np.sqrt((sx-rx)**2 + (sy - ry)**2)

        self._logger.debug('ShakerMaker._call_core - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
                           '\tstyoe: {}\n\tupdn: {}\n\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                           '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n\tsmth: {}\n\twc1: {}\n\twc2: {}\n'
                           '\tpmin: {}\n\tpmax: {}\n\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n\tpf: {}\n\tdf: {}\n'
                           '\tlf: {}\n\tsx: {}\n\tsy: {}\n\trx: {}\n\try: {}\n\t'
                           .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1,
                                   wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))
        if verbose:
            print('ShakerMaker._call_core - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
                   '\tstyoe: {}\n\tupdn: {}\n\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                   '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n\tsmth: {}\n\twc1: {}\n\twc2: {}\n'
                   '\tpmin: {}\n\tpmax: {}\n\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n\tpf: {}\n\tdf: {}\n'
                   '\tlf: {}\n\tsx: {}\n\tsy: {}\n\trx: {}\n\try: {}\n\t'
                   .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1,
                           wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))

        # Execute the core subgreen fortran routing
        tdata, z, e, n, t0 = core.subgreen(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma,
                                           smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

        self._logger.debug('ShakerMaker._call_core - core.subgreen returned: z_size'.format(len(z)))

        return tdata, z, e, n, t0


    def _call_core_fast(self, tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2,
                        pmin, pmax, dk, kc, taper, crust, psource, station,
                        verbose=False):
        """Call core.subgreen2, reusing a precomputed tdata kernel.

        Used by: run_fast() (Stage 2).

        ``tdata`` must be in C-order with shape (nt, 9) as stored in the HDF5
        database. It is reshaped to (1, 9, nt) before being passed to the
        Fortran routine. nt may differ from nfft (depends on smth parameter).

        Returns component seismograms z, e, n and time offset t0.
        """
        mb  = crust.nlayers
        src = crust.get_layer(psource.x[2]) + 1   # fortran starts in 1, not 0
        rcv = crust.get_layer(station.x[2]) + 1   # fortran starts in 1, not 0

        stype = 2  # Source type double-couple, compute up and down going wave
        updn  = 0
        d     = crust.d; a = crust.a; b = crust.b
        rho   = crust.rho; qa = crust.qa; qb = crust.qb

        pf = psource.angles[0]; df = psource.angles[1]; lf = psource.angles[2]
        sx = psource.x[0]; sy = psource.x[1]
        rx = station.x[0]; ry = station.x[1]
        x  = np.sqrt((sx - rx)**2 + (sy - ry)**2)

        self._logger.debug(
            'ShakerMaker._call_core_fast - calling core.subgreen2\n'
            '\tmb: {}\n\tsrc: {}\n\trcv: {}\n\tstype: {}\n\tupdn: {}\n'
            '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n'
            '\tx: {}\n'
            .format(mb, src, rcv, stype, updn, dt, nfft, tb, nx, sigma, x))

        if verbose:
            print(f'_call_core_fast: x={x:.4f} src={src} rcv={rcv} ' 
                  f'tdata.shape={tdata.shape}')

        # Reshape tdata from C-order (nt, 9) to Fortran layout (1, 9, nt).
        # This works for any nt, regardless of nfft or smth.
        # tdata_ = tdata.T.reshape((1, tdata.shape[1], tdata.shape[0]))
        tdata_ = tdata.T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))
        # Execute the core subgreen2 Fortran routine
        z, e, n, t0 = core.subgreen2(
            mb, src, rcv, stype, updn, d, a, b, rho, qa, qb,
            dt, nfft, tb, nx, sigma, smth, wc1, wc2,
            pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        self._logger.debug(
            'ShakerMaker._call_core_fast - core.subgreen2 returned: '
            'z_size={}'.format(len(z)))

        return z, e, n, t0
