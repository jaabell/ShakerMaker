import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.tools.plotting import ZENTPlot

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")

# SCEC LOH.1 crust: 1 km slow layer over a half-space
crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)

# strike-slip double couple at 2 km, Gaussian source time function
sigma = 0.06
stf = Gaussian(t0=6 * sigma, freq=1 / sigma, M0=1e18 / 5e14 / 2)
source = PointSource([0, 0, 2], [0, 90, 0], stf=stf)
fault = FaultSource([source], metadata={"name": "LOH1"})

# one receiver at (6, 8) km
s = Station([6, 8, 0], metadata={"name": "STA"})
stations = StationList([s], {})

ShakerMaker(crust, fault, stations).run(dt=0.005, nfft=4096, dk=0.05, tb=1000)

ZENTPlot(s, xlim=[0, 20], show=False,
         savefigname=os.path.join(IMG, "example_0_quick_example.png"))
plt.close("all")

print("quick example OK")
