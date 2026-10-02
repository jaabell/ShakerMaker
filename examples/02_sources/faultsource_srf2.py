# 02 - Build a 2x5 fault grid of SRF2 subfaults with random rake/slip.
# 2026-06-06

import numpy as np
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.srf2 import SRF2

crust = SCEC_LOH_1()                        # layered crust (context)

rng = np.random.default_rng(0)              # fixed seed -> reproducible

# hypocenter (km) and grid spacing (km)
x0, y0, z0 = 0.0, 0.0, 3.0
dx, dz = 0.5, 0.5
strike, dip = 0.0, 90.0
base_rake = 0.0

sources = []
for i in range(2):                          # 2 along-strike
    for j in range(5):                      # 5 down-dip -> 10 subfaults
        x = x0 + i * dx
        z = z0 + j * dz                     # deeper down-dip
        rake = base_rake + rng.uniform(-10, 10)        # deg
        slip = rng.uniform(0.5, 1.5)                    # m
        stf = SRF2(Tr=2.0, Tp=0.1, Te=1.5, dt=0.01, slip=slip, a=1.0, b=1.0)
        sources.append(PointSource([x, y0, z], [strike, dip, rake], stf=stf))

fault = FaultSource(sources, metadata={"name": "fault_srf2"})

assert fault.nsources == 10
print("PASS")
