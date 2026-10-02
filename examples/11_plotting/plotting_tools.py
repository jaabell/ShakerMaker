# 11 - Build a small FaultSource and save a SourcePlot (no FK run).
# 2026-06-06

import os
import matplotlib
matplotlib.use("Agg")

from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.tools.plotting import SourcePlot

# Two point sources at slightly different depths, same Gaussian STF.
stf = Gaussian(t0=0.36, freq=16.6667, M0=1.0)
ps1 = PointSource([0.0, 0.0, 2.0], [0., 90., 0.], stf=stf)
ps2 = PointSource([0.0, 0.0, 2.5], [0., 90., 0.], stf=stf)

fault = FaultSource([ps1, ps2], metadata={"name": "two_sources"})

assert fault.nsources == 2

here = os.path.dirname(os.path.abspath(__file__))
fig = SourcePlot(fault, show=False)
fig.savefig(os.path.join(here, "sources_plot.png"), dpi=150, bbox_inches="tight")

assert os.path.exists(os.path.join(here, "sources_plot.png"))
print("Saved sources_plot.png | nsources:", fault.nsources)
print("PASS")
