# 07 - Walk an .h5/.h5drm file and print every group/dataset name and shape.
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox

try:
    import h5py
except Exception:
    print("SKIP: h5py not available"); raise SystemExit(0)

f_drm = "explore_geometry.h5drm"

# Produce a tiny file without an FK run via export_drm_geometry.
crust = SCEC_LOH_1()
sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2
stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})

dx = 0.5
drm = DRMBox([6, 8, 0], [3, 3, 2], [dx, dx, dx], metadata={"name": "drm_small"})
model = ShakerMaker(crust, fault, drm)
model.export_drm_geometry(f_drm)

assert os.path.exists(f_drm)


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET {name}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"GROUP   {name}")


with h5py.File(f_drm, "r") as hf:
    print(f"Exploring: {f_drm}")
    hf.visititems(walk)

print("PASS")
