# 02 - Build a single PointSource with a Gaussian STF.
# 2026-06-06

from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian

crust = SCEC_LOH_1()                       # layered crust (context)

sigma = 0.06                               # Gaussian width (s)
stf = Gaussian(t0=6 * sigma, freq=1 / sigma, M0=1e18 / 5e14 / 2)

source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)   # [x,y,z] km ; [strike,dip,rake] deg
fault = FaultSource([source], metadata={"name": "pointsource"})

assert fault.nsources == 1
print("PASS")
