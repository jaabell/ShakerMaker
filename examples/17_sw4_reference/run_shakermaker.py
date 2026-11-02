# 17 - SW4 reference cases: regenerates the ShakerMaker reference (data/sm/), one engine at a time.
#
#   python run_shakermaker.py <case> <engine> [out_dir] [gf_dir]
#   mpiexec -n 32 python run_shakermaker.py layered_f1 inter_tensor
#
# case:   loh1_p1 | loh1_p2 | loh1_f1 | layered_p1 | layered_p2 | layered_f1   (see make_sw4_inputs.py)
# engine: run           direct engine, one Green's function per pair, depth_tol 1 mm (the exact reference)
#         exact         run_nearest without reuse (all tolerances 1 cm), one wavenumber step per pair of depths
#                       (SM_GF_XMAX=common), depth_tol 1 mm: the exact reference of tests/engine, batched, used for
#                       the 256-subfault cases where run() takes hours
#         nearest       run_nearest with the production tolerances (40 m, 5 m, 200 m) and the default depth_tol
#         inter_grid    run_inter, scheme='grid', 40 m distance grid, cubic interpolation in reduced time
#         inter_tensor  run_inter, scheme='tensor', the production configuration (freq_cut, zfp20, decimated)
#
# Same receivers (data/receivers_<medium>.csv, already on SW4 grid nodes), same sources (data/fault_<medium>.csv
# for f1) and same Gaussian as SW4. Writes <out_dir>/<case>_<engine>.npz (default data/sm/) with t, names and
# zen (nrec, 3, nt) in ShakerMaker units, and the wall time.
#
# This is not a smoke test: on one 32-core node a point source takes 10 s to 5 min per engine and the 256-subfault
# cases 5 to 30 min (run() on f1 takes hours, so f1 uses exact). Without arguments it prints SKIP.

import csv
import os
import sys
import time

import numpy as np

from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.stf_extensions.gaussian import Gaussian

here = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 3:
    print("SKIP: regenerates the reference only with <case> <engine> (see the header)")
    sys.exit(0)
case, engine = sys.argv[1], sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "data", "sm")
gf_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(here, "_output", "gf")
medium, src = case.split("_")

# media, sources and time functions of make_sw4_inputs.py (moment in N m; 1 ShakerMaker unit = 1e15 N m)
layers = {"loh1": [(1.0, 4.0, 2.0, 2.6), (0.0, 6.0, 3.464, 2.7)],
          "layered": [(0.2, 1.32, 0.75, 2.40), (0.8, 2.75, 1.57, 2.50), (14.5, 5.50, 3.14, 2.50), (0.0, 7.00, 4.00, 2.67)]}[medium]
sigma = {"loh1": 0.06, "layered": 0.1}[medium]
zsrc = {"loh1": {"p1": 2.0, "p2": 0.5}, "layered": {"p1": 6.0, "p2": 0.5}}[medium]
tmax = {"loh1": 12.0, "layered": 15.0}[medium]
angles = [30.0, 60.0, 45.0]
m0_point = 1.0e16

# FK parameters: dt 0.02 s keeps the flat band of the core (0.1 f_nyquist = 2.5 Hz) above the 2 Hz of the
# comparison; nfft 2048 = 41 s; dk 0.025 avoids the wavenumber ghost sources of the 0.5 km source (INF_010 V0)
fk = dict(dt=0.02, nfft=2048, tb=100, dk=0.025, smth=1, sigma=2, kc=15.0)

crust = CrustModel(len(layers))
for d, vp, vs, rho in layers:
    crust.add_layer(d, vp, vs, rho, 1000.0, 1000.0)

if src in ("p1", "p2"):
    sources = [PointSource([0.0, 0.0, zsrc[src]], angles,
                           stf=Gaussian(t0=6 * sigma, freq=1 / sigma, M0=m0_point / 1e15, derivative=False))]
else:
    rows = list(csv.DictReader(open(os.path.join(here, "data", f"fault_{medium}.csv"))))
    sources = [PointSource([float(r["north_km"]), float(r["east_km"]), float(r["depth_km"])], angles, tt=float(r["tr_s"]),
                           stf=Gaussian(t0=6 * sigma, freq=1 / sigma, M0=float(r["m0_nm"]) / 1e15, derivative=False))
               for r in rows]
recs = list(csv.DictReader(open(os.path.join(here, "data", f"receivers_{medium}.csv"))))
stations = [Station([float(r["north_km"]), float(r["east_km"]), float(r["depth_km"])], metadata={"name": r["name"]})
            for r in recs]
model = shakermaker.ShakerMaker(crust, FaultSource(sources, {"name": case}), StationList(stations, {"name": medium}))


class Capture(StationListWriter):
    """Keeps every station response in memory (rank 0 only)."""

    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        self.data = {}

    def write_metadata(self, metadata):
        pass

    def write_station(self, station, index):
        z, e, n, t = station.get_response()
        self.data[index] = (np.array(z), np.array(e), np.array(n), np.array(t))

    def close(self):
        pass


os.makedirs(gf_dir, exist_ok=True)
db = os.path.join(gf_dir, f"{case}_{engine}")
w = Capture("unused")
common = dict(writer=w, writer_mode="progressive", tmin=0.0, tmax=tmax, showProgress=False, **fk)
t0 = time.perf_counter()
if engine == "run":
    crust.depth_tol = 1e-6
    model.run(**common)
elif engine == "exact":
    os.environ["SM_GF_XMAX"] = "common"
    crust.depth_tol = 1e-6
    model.run_nearest(stage="all", h5_database_name=db, delta_h=1e-5, delta_v_rec=1e-5, delta_v_src=1e-5, **common)
elif engine == "nearest":
    model.run_nearest(stage="all", h5_database_name=db, delta_h=0.04, delta_v_rec=0.005, delta_v_src=0.2, **common)
elif engine == "inter_grid":
    model.run_inter(stage="all", h5_database_name=db, scheme="grid", dh_grid=0.04, order=3, **common)
elif engine == "inter_tensor":
    model.run_inter(stage="all", h5_database_name=db, scheme="tensor", freq_cut=2.5, dh_min=0.02,
                    eps_v=0.01, eps_s=0.05, store="zfp20", decimate=True, **common)
else:
    raise SystemExit(f"unknown engine {engine}")
wall = time.perf_counter() - t0

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0
if rank == 0:
    idx = sorted(w.data)
    assert len(idx) == len(stations), (len(idx), len(stations))
    # run() returns every station on its own time window (it starts at its own t0): put all on [0, tmax)
    t = np.arange(0.0, tmax, fk["dt"])
    zen = np.stack([np.stack([np.interp(t, w.data[i][3], c, left=0.0, right=c[-1]) for c in w.data[i][:3]])
                    for i in idx]).astype(np.float32)
    assert np.all(np.isfinite(zen)), "NaN or inf in the response"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{case}_{engine}.npz"), t=t, names=np.array([s.metadata["name"] for s in stations]),
                        zen=zen, components=np.array(["z", "e", "n"]), wall_s=wall, fk=str(fk))
    print(f"{case} {engine}: {len(idx)} receivers, {zen.shape[2]} samples, wall {wall:.1f} s, peak {np.abs(zen).max():.3e}")
    print("PASS")
