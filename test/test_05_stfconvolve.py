###############################################################################
# shakermaker Test Suite
# Test # 00 - Source Time Functions
# file: /tests/test_05_stfconvolve.py
#
# Description
#
# This test exercices several source time functions and plots them.
#
###############################################################################

from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Brune
from shakermaker.tools.plotting import ZENTPlot

x0 = 1.
y0 = 1.
zsrc = 1.0
strike = 0.
dip = 45.
rake = 0.
dt = 0.01

#Initialize CrustModel
crust = SCEC_LOH_1()
# crust = SCEC_LOH_3()

#Initialize Source
brune = Brune(f0=10.)
source = PointSource([0,0,zsrc], [strike,dip,rake], stf=brune)
fault = FaultSource([source], metadata={"name":"source"})


#Initialize Receiver
s = Station([x0,y0,0], 
	metadata={
		"name":"Your House", 
		"filter_results":True, 
		"filter_parameters":{"fmax":10.}
	})
stations = StationList([s], metadata=s.metadata)

model = shakermaker.ShakerMaker(crust, fault, stations)

print("Running shakermaker")
model.run(dt=dt)
print("Done shakermaker")

fig = ZENTPlot(s, show=True, xlim=[0,20])

