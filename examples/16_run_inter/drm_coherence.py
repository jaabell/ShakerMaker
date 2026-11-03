# 16 - DRM box with Green's functions reused (run_nearest) or interpolated (run_inter).
#
# A DRM box turns the difference between its two layers of nodes into forces, so neighbouring
# nodes must stay coherent. This example drives a small DRM box (2.5 m elements) with a 2 x 2
# patch of point sources and compares, against a reference with one Green's function per pair:
#   - run_nearest with reuse tolerances (40 m in distance, 5 m and 200 m in depth),
#   - run_inter, tensor scheme (depths exact, cubic interpolation in distance).
# Metric: error of the difference between every interior node and its nearest exterior node,
# per frequency band. Figure: run_inter_drm_coherence.png.
#
#   python drm_coherence.py              (or: mpirun -n 4 python drm_coherence.py)
# 2026-11-02

import os
import tempfile

import numpy as np

from shakermaker import shakermaker as sm
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.sl_extensions import DRMBox
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.stf_extensions.gaussian import Gaussian

HERE = os.path.dirname(os.path.abspath(__file__))
FK = dict(dt=0.005, nfft=1024, tb=100, dk=0.1, kc=30.0, sigma=2, taper=0.9, pmax=1, smth=1)
TMAX, BAND = 4.0, 10.0


def crust():
    """Four layers: 200 m at Vs 0.75 km/s, 800 m at 1.57 km/s, 14.5 km at 3.14 km/s, half-space."""
    c = CrustModel(4)
    c.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
    c.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    c.add_layer(14.50, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    c.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    return c


def fault():
    """2 x 2 point sources, 250 m apart on a plane dipping 40 degrees, centred at 2 km depth."""
    st, dp = np.deg2rad(358.0), np.deg2rad(40.0)
    srcs = []
    for i in range(2):
        for j in range(2):
            a, b = (i - 0.5) * 0.25, (j - 0.5) * 0.25
            xyz = [a * np.cos(st) - b * np.cos(dp) * np.sin(st), a * np.sin(st) + b * np.cos(dp) * np.cos(st),
                   2.0 + b * np.sin(dp)]
            stf = Gaussian(t0=0.12, freq=50.0, M0=1.0 / 4, derivative=False)
            srcs.append(PointSource(xyz, [358.0, 40.0, 113.0], tt=np.hypot(a, b) / 2.5, stf=stf))
    return FaultSource(srcs, {"name": "patch"})


def box():
    return DRMBox([1.5, 1.5, 0.0], [8, 8, 4], [0.0025, 0.0025, 0.0025], {"name": "box"})


class Keep(StationListWriter):
    def __init__(self):
        super().__init__("unused")
        self.v = {}

    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        pass

    def write_metadata(self, metadata):
        pass

    def write_station(self, station, index):
        z, e, n, t = station.get_response()
        self.v[index] = np.array([z, e, n])

    def close(self):
        pass


def run(kind, db):
    if sm.rank == 0:
        os.makedirs(os.path.dirname(db), exist_ok=True)
    if sm.use_mpi:
        sm.comm.Barrier()
    b = box()
    model = sm.ShakerMaker(crust(), fault(), b)
    w = Keep()
    kw = dict(writer=w, writer_mode="progressive", showProgress=False, tmin=0.0, tmax=TMAX, **FK)
    if kind == "inter":
        model.run_inter(stage="all", h5_database_name=db, scheme="tensor", f_err=BAND, freq_cut=1.25 * BAND,
                        dh_min=0.020 * 20.0 / BAND, **kw)
    else:
        tol = {"exact": (1e-5, 1e-5, 1e-5), "reuse": (0.040, 0.005, 0.200)}[kind]
        model._crust.depth_tol = 1e-6
        if kind == "exact":
            os.environ["SM_GF_XMAX"] = "common"        # one wavenumber step per pair of depths
        model.run_nearest(stage="all", h5_database_name=db, delta_h=tol[0], delta_v_rec=tol[1],
                          delta_v_src=tol[2], **kw)
        os.environ.pop("SM_GF_XMAX", None)
    if sm.rank != 0:
        return None, None
    n = min(v.shape[1] for v in w.v.values())
    return np.stack([w.v[i][:, :n] for i in range(b.nstations)]), b


def layer_difference_error(vel, ref, b, f_lo, f_hi):
    """Median error of (interior - nearest exterior) against the reference, band [f_lo, f_hi]."""
    xyz = np.array([b.get_station_by_id(i).x for i in range(b.nstations)])
    inner = np.array([b.get_station_by_id(i).is_internal for i in range(b.nstations)])
    ii, oo = np.flatnonzero(inner), np.flatnonzero(~inner)
    pair = oo[np.argmin(np.linalg.norm(xyz[ii][:, None, :] - xyz[oo][None, :, :], axis=2), axis=1)]
    nt = min(vel.shape[-1], ref.shape[-1])
    f = np.fft.rfftfreq(2 * nt, FK["dt"])
    m = (f >= f_lo) & (f <= f_hi)

    def band(x):
        return np.fft.irfft(np.fft.rfft(x[..., :nt], 2 * nt, axis=-1) * m, 2 * nt, axis=-1)[..., :nt]

    d, dr = band(vel[ii] - vel[pair]), band(ref[ii] - ref[pair])
    return float(np.median(np.sqrt(np.sum((d - dr) ** 2, axis=(1, 2)) / np.sum(dr ** 2, axis=(1, 2)))))


if __name__ == "__main__":
    work = tempfile.mkdtemp(prefix="run_inter_example_") if sm.rank == 0 else None
    if sm.use_mpi:
        work = sm.comm.bcast(work, root=0)
    res = {k: run(k, os.path.join(work, k, "gf")) for k in ("exact", "reuse", "inter")}
    if sm.rank == 0:
        ref, b = res["exact"]
        bands = [(0.1, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]
        err = {k: [layer_difference_error(res[k][0], ref, b, *bb) for bb in bands] for k in ("reuse", "inter")}
        for k in err:
            print(f"{k:5s}: " + "  ".join(f"{bb[0]:.1f}-{bb[1]:.1f} Hz {100 * e:7.2f} %" for bb, e in zip(bands, err[k])))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(bands)); w_ = 0.38
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        ax.bar(x - w_ / 2, [100 * e for e in err["reuse"]], w_, color="#7f9fc6", label="run_nearest, reuse 40 m / 5 m / 200 m")
        ax.bar(x + w_ / 2, [100 * e for e in err["inter"]], w_, color="#c0392b", label="run_inter, tensor scheme")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{a:g}-{c:g} Hz" for a, c in bands])
        ax.set_ylabel("error of the DRM layer\ndifference (median, %)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "run_inter_drm_coherence.png"), dpi=150)
        assert max(err["inter"]) < 0.05 < max(err["reuse"]), err
        print("PASS")
