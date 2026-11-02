# 17 - Smoke test: the Green's function engines against the exact reference on a small DRM box (~25 s).
#
# A 3 x 3 x 2 DRM box with 5 m elements in a layered crust (200 m at Vs 0.75 km/s, 800 m at 1.57 km/s, half-space
# at 3.14 km/s) under six point sources at three depths, with different mechanisms and rupture times.
#
#   exact    run_nearest with 1 cm tolerances (one Green's function per pair) and one wavenumber step per pair of
#            depths (SM_GF_XMAX=common), depth_tol 1 mm: the reference
#   nearest  run_nearest with the production tolerances (40 m, 5 m, 200 m): one Green's function per slot, copied
#   grid     run_inter(scheme='grid'), 10 m distance grid, cubic interpolation in reduced time
#   tensor   run_inter(scheme='tensor'), adaptive distance grid, tolerances 1 % / 5 %
#
# Two errors per engine, relative to the peak of the reference: the traces, and the difference between every node
# and its nearest neighbour (what a DRM layer turns into effective forces). The interpolating engines must stay
# close to the reference; nearest is printed (its neighbour difference is wrong by tens of percent, the defect the
# interpolation fixes) but not checked.
#
#   python smoke_engines.py     # PASS: grid < 0.5 % / 3 %, tensor < 1 % / 5 % (traces / neighbour differences)

import os
import tempfile

import numpy as np

from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.sl_extensions import DRMBox
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.stf_extensions import Discrete

fk = dict(dt=0.01, nfft=256, dk=0.05, tb=40, smth=1, sigma=2)
window = dict(tmin=0.0, tmax=2.0, showProgress=False)


class Capture(StationListWriter):
    """Keeps every station response in memory."""

    def initialize(self, station_list, num_samples, tmin=None, tmax=None, dt=None, writer_mode=None):
        self.data = {}

    def write_metadata(self, metadata):
        pass

    def write_station(self, station, index):
        z, e, n, t = station.get_response()
        self.data[index] = np.array([z, e, n])

    def close(self):
        pass


def make_model():
    crust = CrustModel(3)
    crust.add_layer(0.2, 1.32, 0.75, 2.4, 1000.0, 1000.0)
    crust.add_layer(0.8, 2.75, 1.57, 2.5, 1000.0, 1000.0)
    crust.add_layer(0.0, 5.50, 3.14, 2.5, 1000.0, 1000.0)
    rng = np.random.default_rng(3)
    sources = []
    for i in range(2):
        for j in range(3):
            tr = 0.2 + 0.1 * rng.random()
            t = np.linspace(0, tr, int(tr / 0.005))
            svf = np.sin(np.pi * t / tr)
            svf /= np.trapz(svf, dx=0.005)
            sources.append(PointSource([1.0 + 0.4 * i, 2.0 + 0.3 * j, 1.5 + 0.25 * j],
                                       [350 + 20 * rng.random(), 35 + 10 * rng.random(), 80 + 40 * rng.random()],
                                       tt=0.05 * i + 0.03 * j, stf=Discrete(svf * (0.5 + rng.random()), t)))
    box = DRMBox([0.3, 0.2, 0.0], [3, 3, 2], [0.005] * 3, {"name": "box"})
    return shakermaker.ShakerMaker(crust, FaultSource(sources, {"name": "sources"}), box)


def run(engine, folder):
    model = make_model()
    db = os.path.join(folder, engine)
    w = Capture("unused")
    os.environ.pop("SM_GF_XMAX", None)
    if engine == "exact":
        os.environ["SM_GF_XMAX"] = "common"
        model._crust.depth_tol = 1e-6
        model.run_nearest(stage="all", h5_database_name=db, delta_h=1e-5, delta_v_rec=1e-5, delta_v_src=1e-5,
                          writer=w, **window, **fk)
        os.environ.pop("SM_GF_XMAX")
    elif engine == "nearest":
        model.run_nearest(stage="all", h5_database_name=db, delta_h=0.04, delta_v_rec=0.005, delta_v_src=0.2,
                          writer=w, **window, **fk)
    elif engine == "grid":
        model.run_inter(stage="all", h5_database_name=db, scheme="grid", dh_grid=0.010, order=3, writer=w, **window, **fk)
    else:
        model.run_inter(stage="all", h5_database_name=db, scheme="tensor", eps_v=0.01, eps_s=0.05, store="f32",
                        writer=w, **window, **fk)
    xyz = np.array([s.x for s in model._receivers])
    return np.array([w.data[i] for i in sorted(w.data)]), xyz


with tempfile.TemporaryDirectory() as folder:
    out = {e: run(e, folder) for e in ("exact", "nearest", "grid", "tensor")}

ref, xyz = out["exact"]
nodes = len(ref)
dist = np.linalg.norm(xyz[:nodes, None, :] - xyz[None, :nodes, :], axis=-1) + np.eye(nodes) * 1e9
neighbour = dist.argmin(axis=1)
peak = np.abs(ref).max()
dref = ref - ref[neighbour]
errors = {}
for engine in ("nearest", "grid", "tensor"):
    u = out[engine][0]
    du = u - u[neighbour]
    errors[engine] = (np.abs(u - ref).max() / peak, np.median(np.abs(du - dref).max(axis=(1, 2)) / np.abs(dref).max(axis=(1, 2))))
    print(f"{engine:8s} traces {100 * errors[engine][0]:7.3f} %   neighbour difference (median) {100 * errors[engine][1]:7.2f} %")

assert errors["grid"][0] < 5e-3 and errors["grid"][1] < 3e-2, "run_inter grid drifted from the exact reference"
assert errors["tensor"][0] < 1e-2 and errors["tensor"][1] < 5e-2, "run_inter tensor drifted from the exact reference"
print("PASS")
