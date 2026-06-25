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
from shakermaker.tools.plotting import ZENTPlot

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")

crust = CrustModel(2)
crust.add_layer(1.0, 4.0, 2.0, 2.6, 10000., 10000.)
crust.add_layer(0.0, 6.0, 3.464, 2.7, 10000., 10000.)

src = PointSource([0, 0, 4], [90, 90, 0])
fault = FaultSource([src], metadata={"name": "single-point-source"})

sta = Station([0, 4, 0], metadata={"name": "STA01"})
stations = StationList([sta], metadata=sta.metadata)

ShakerMaker(crust, fault, stations).run(dt=0.005, nfft=2048, dk=0.1, tb=500)

ZENTPlot(sta, xlim=[0, 30], show=False,
         savefigname=os.path.join(IMG, "seismogram_velocity.png"))
plt.close("all")
ZENTPlot(sta, xlim=[0, 30], show=False, integrate=1,
         savefigname=os.path.join(IMG, "seismogram_displacement.png"))
plt.close("all")
ZENTPlot(sta, xlim=[0, 30], show=False, differentiate=1,
         savefigname=os.path.join(IMG, "seismogram_acceleration.png"))
plt.close("all")

print("seismogram OK")
