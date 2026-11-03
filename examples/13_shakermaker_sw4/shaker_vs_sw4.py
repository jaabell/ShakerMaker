# 13 - ShakerMaker vs SW4 cross-validation.
#
# Reconstruct a ShakerMaker model from the compact SW4 export package
# (model_summary.h5: crust + 100 FFSP point sources + 2 stations), run the FK
# engine, read the matching SW4 finite-difference output, band-pass it with
# ObsPy, and overlay the two solutions at each station.
#
# Resources live in ./data/ (copied from a real SW4 run):
#   data/model_summary.h5      compact package (crust, sources, stations)
#   data/shakermaker2sw4.in    SW4 input (used only for station coordinates)
#   data/sf00001.txt           SW4 velocity time series, station 1
#   data/sf00002.txt           SW4 velocity time series, station 2
#
# NOTE: the FK run is heavy (nfft=32768, 100 sources x 2 stations = 200 core calls,
# 40 min to 6 h on one process depending on the machine). Launch under MPI:
#   mpiexec -n 6 python shaker_vs_sw4.py
# Both solutions are band-passed to FMIN-FMAX; the script fails if the correlation or the
# peak ratio with SW4 leaves the limits below. Figures go to _output/.

import ast
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt

from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Discrete

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PACKAGE_H5 = os.path.join(DATA, "model_summary.h5")
IN_FILE = os.path.join(DATA, "shakermaker2sw4.in")

OUT = os.path.join(HERE, "_output")

# Band used to compare the two solutions (Hz).
FMIN, FMAX = 0.25, 15.0
# Agreement required with SW4, per component, on the 0-15 s window (reference run: correlation 0.83-0.97,
# peak ratio 0.84-1.11; the band reaches above what the SW4 grid resolves, so the limits are loose).
CC_MIN, PEAK_RANGE = 0.80, (0.80, 1.20)


# =============================================================================
# 1. Rebuild the ShakerMaker model from the SW4 package
# =============================================================================

def load_crust():
    """Crust layers stored in the package -> CrustModel."""
    with h5py.File(PACKAGE_H5, "r") as f:
        g = f["crust"]
        thickness = g["thickness_km"][:]
        vp, vs = g["vp_km_s"][:], g["vs_km_s"][:]
        rho = g["rho_g_cm3"][:]
        qp, qs = g["qp"][:], g["qs"][:]
    crust = CrustModel(len(thickness))
    for i in range(len(thickness)):
        crust.add_layer(thickness[i], vp[i], vs[i], rho[i], qp[i], qs[i])
    return crust


def load_fault():
    """100 point sources, each with a Discrete slip-rate STF."""
    with h5py.File(PACKAGE_H5, "r") as f:
        s = f["sources"]
        x, y, z = s["x_km"][:], s["y_km"][:], s["z_km"][:]
        strike, dip, rake = s["strike_deg"][:], s["dip_deg"][:], s["rake_deg"][:]
        tt = s["trigger_time_s"][:]
        dt_all, npts_all = s["dt"][:], s["npts"][:]
        offsets, values = s["data_offsets"][:], s["data_values"][:]

    sources = []
    for i in range(len(x)):
        dt, npts, off = float(dt_all[i]), int(npts_all[i]), int(offsets[i])
        slip_rate = values[off:off + npts]
        stf = Discrete(slip_rate, np.arange(npts) * dt)
        stf.dt = dt
        sources.append(PointSource([x[i], y[i], z[i]],
                                   [strike[i], dip[i], rake[i]],
                                   tt=tt[i], stf=stf))
    return FaultSource(sources, metadata={"name": "from_sw4_package"})


def load_stations():
    """Receiver stations stored in the package."""
    with h5py.File(PACKAGE_H5, "r") as f:
        g = f["stations"]
        xyz = g["xyz_km"][:]
        internal = g["internal"][:]
        meta_raw = g["metadata"][:]
    stations = []
    for i in range(len(xyz)):
        txt = meta_raw[i].decode() if isinstance(meta_raw[i], bytes) else meta_raw[i]
        try:
            meta = ast.literal_eval(txt)
        except Exception:
            meta = {"name": f"sta_{i}"}
        stations.append(Station(xyz[i], internal=bool(internal[i]), metadata=meta))
    return StationList(stations, metadata={"name": "from_sw4_package"})


# =============================================================================
# 2. Read the SW4 output
# =============================================================================

def read_sw4_station(name):
    """Read an SW4 receiver .txt (time, x, y, z) into ShakerMaker Z/E/N.

    SW4 columns are East/North/Up-ish; map to the ShakerMaker convention:
        u_z = column z,  u_e = column y,  u_n = column x.
    """
    path = os.path.join(DATA, f"{name}.txt")
    arr = np.loadtxt(path, skiprows=13)
    t = arr[:, 0]
    uz, ue, un = arr[:, 3], arr[:, 2], arr[:, 1]
    return t, uz, ue, un


def bandpass(sig, dt):
    """Zero-phase 4-pole band-pass with ObsPy, matching the SW4/FK band."""
    from obspy import Trace
    tr = Trace(data=sig.astype(np.float32))
    tr.stats.delta = dt
    tr.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
    return tr.data


# =============================================================================
# 3. Run and compare
# =============================================================================

def main():
    crust = load_crust()
    fault = load_fault()
    stations = load_stations()
    print(crust)
    print(f"Sources: {fault.nsources}  Stations: {stations.nstations}")

    model = shakermaker.ShakerMaker(crust, fault, stations)

    # Pre-flight (prints the dt/tmax-centred report; safe to read before run).
    model.check_parameters(dt=0.0025, nfft=8192 * 4, dk=0.2, tb=800, tmax=60)

    model.run(dt=0.0025, nfft=8192 * 4, dk=0.2, tb=800, tmax=60,
              tmin=0.0, sigma=2, pmax=1, nx=1, kc=15.0, verbose=False)

    # Under MPI only rank 0 holds the station responses.
    if not model.mpi_is_master_process():
        return

    os.makedirs(OUT, exist_ok=True)
    sw4_names = ["sf00001", "sf00002"]
    failed = []
    for sid, sw4_name in enumerate(sw4_names):
        sta = stations.get_station_by_id(sid)
        z_sm, e_sm, n_sm, t_sm = sta.get_response()
        np.savez(os.path.join(OUT, f"shakermaker_{sw4_name}.npz"), t=t_sm, z=z_sm, e=e_sm, n=n_sm)

        t_s, z_s, e_s, n_s = read_sw4_station(sw4_name)
        dt_s = t_s[1] - t_s[0]
        dt_sm = t_sm[1] - t_sm[0]
        sw = [bandpass(c, dt_s) for c in (z_s, e_s, n_s)]
        sm = [bandpass(c, dt_sm) for c in (z_sm, e_sm, n_sm)]

        # correlation (zero lag) and peak ratio ShakerMaker / SW4 on the common 0-15 s window
        tc = np.arange(0.0, 15.0, dt_s)
        cc, peak = [], []
        for a_sm, a_sw in zip(sm, sw):
            a = np.interp(tc, t_sm, a_sm)
            b = np.interp(tc, t_s, a_sw)
            cc.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)))
            peak.append(float(np.abs(a).max() / max(np.abs(b).max(), 1e-300)))
        print(f"{sw4_name}: cc with SW4 (z, e, n) = " + ", ".join(f"{c:.3f}" for c in cc)
              + "   peak ratio = " + ", ".join(f"{p:.2f}" for p in peak))
        for k, comp in enumerate("zen"):
            if cc[k] < CC_MIN or not PEAK_RANGE[0] <= peak[k] <= PEAK_RANGE[1]:
                failed.append(f"{sw4_name} {comp}: cc {cc[k]:.3f}, peak ratio {peak[k]:.2f}")

        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for k, lab in enumerate([r"$u_z$", r"$u_e$", r"$u_n$"]):
            ax[k].plot(t_s, sw[k], "--", color="#7f9fc6", lw=1.2, label="SW4")
            ax[k].plot(t_sm, sm[k], color="#c0392b", lw=0.9, label="ShakerMaker")
            ax[k].set_xlim(0, 15)
            ax[k].set_ylabel(lab)
            ax[k].grid(True)
        ax[0].legend(loc="upper left")
        ax[2].set_xlabel("Time [s]")
        fig.suptitle(f"ShakerMaker vs SW4, station {sw4_name}, {FMIN:g}-{FMAX:g} Hz")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"compare_{sw4_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    assert not failed, "ShakerMaker no longer matches SW4: " + "; ".join(failed)
    print("PASS")


if __name__ == "__main__":
    main()
