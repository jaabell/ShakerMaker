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
# NOTE: the FK run is heavy (nfft=32768, 100 sources). Launch under MPI for
# real use:  mpiexec -n 8 python shaker_vs_sw4.py

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

# Band used to compare the two solutions (Hz).
FMIN, FMAX = 0.25, 15.0


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
              tmin=0.0, sigma=2, pmax=1, nx=1, kc=15.0, verbose=True)

    sw4_names = ["sf00001", "sf00002"]
    for sid, sw4_name in enumerate(sw4_names):
        sta = stations.get_station_by_id(sid)
        z_sm, e_sm, n_sm, t_sm = sta.get_response()

        t_s, z_s, e_s, n_s = read_sw4_station(sw4_name)
        dt_s = t_s[1] - t_s[0]
        z_f, e_f, n_f = (bandpass(z_s, dt_s),
                         bandpass(e_s, dt_s),
                         bandpass(n_s, dt_s))

        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for k, (sm, sw, lab) in enumerate([(z_sm, z_f, r"$u_z$"),
                                           (e_sm, e_f, r"$u_e$"),
                                           (n_sm, n_f, r"$u_n$")]):
            ax[k].plot(t_sm, sm, "r", lw=1, label="ShakerMaker")
            ax[k].plot(t_s, sw, "--", label="SW4")
            ax[k].set_xlim(0, 15)
            ax[k].set_ylabel(lab)
            ax[k].grid(True)
        ax[0].legend(loc="upper left")
        ax[2].set_xlabel("Time [s]")
        fig.suptitle(f"ShakerMaker vs SW4 — station {sw4_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"compare_{sw4_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved compare_{sw4_name}.png")

    print("PASS")


if __name__ == "__main__":
    main()
