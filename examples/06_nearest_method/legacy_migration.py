# 06 - Migrate a legacy JAA/PXP GF database to OP format (adds pair_to_slot).
# 2026-06-06

import os
from shakermaker.shakermaker import ShakerMaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

# Path to a pre-existing legacy database (must hold dh_of_pairs, zrec_of_pairs,
# zsrc_of_pairs, tdata_dict). No legacy DB ships with the repo, so we skip.
legacy_db = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "legacy_gf_database.h5")

if not os.path.exists(legacy_db):
    print("SKIP: no legacy database available")
    raise SystemExit(0)

_m = 1e-3

crust = SCEC_LOH_1()
stf = Gaussian(t0=6 * 0.06, freq=1 / 0.06, M0=1e18 / 5e14 / 2, derivative=False)
source = PointSource([0, 0, 2], [0., 90., 0.], stf=stf)
fault = FaultSource([source], metadata={"name": "src"})
stations = SurfaceGrid([6.0, 8.0, 0.0], [5, 5, 0], [2.0, 2.0, 2.0],
                       mode='plane', plane_z=0.0, metadata={"name": "grid"})

model = ShakerMaker(crust, fault, stations)

# Uses tolerances stored in the file when delta_* are None.
model.build_pair_to_slot_from_legacy_h5(legacy_db,
                                        delta_h=2.5 * _m,
                                        delta_v_rec=2.5 * _m,
                                        delta_v_src=2.5 * _m)

assert os.path.exists(legacy_db)
print("PASS")
