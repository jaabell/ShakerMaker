from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot

#Initialize two-layer CrustModel
crust = CrustModel(2)

#Slow layer
Vp=4.000			# P-wave speed (km/s)
Vs=2.000			# S-wave speed (km/s)
rho=2.600			# Density (gm/cm**3)
Qp=10000.			# Q-factor for P-wave
Qs=10000.			# Q-factor for S-wave
thickness = 1.0		# Self-explanatory
crust.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

#Halfspace
Vp=6.000
Vs=3.464
rho=2.700
Qp=10000.
Qs=10000.
thickness = 0   #Zero thickness --> half space
crust.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

#Initialize Source
source = PointSource([0,0,4], [90,90,0])
fault = FaultSource([source], metadata={"name":"single-point-source"})

#Initialize Receiver
s = Station([0,4,0],metadata={"name":"a station"})
stations = StationList([s], metadata=s.metadata)


model = ShakerMaker(crust, fault, stations)

model.run()

ZENTPlot(s, xlim=[0,60], show=True)