###############################################################################
# shakermaker Test Suite
# Test # 00 - Source Time Functions
# file: /tests/test_04_pointsourcelist.py
#
# Description
#
# This test exercices a composite source made up of a list of PointSources
#
###############################################################################


from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
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

#Initialize Sources
subfaults = []
subfaults.append(PointSource([0,0,zsrc], [strike,dip,rake]))
subfaults.append(PointSource([0,0,zsrc], [strike,dip,rake],tt=3.0))

# source = PointSourceList([source1])
fault = FaultSource(subfaults, metadata="TwoSubFaults")

#Initialize Receiver
s = Station([x0,y0,0], 
	metadata={"name":"A Station", 
	"filter_results":False, 
	"filter_parameters":{"fmax":10.}})
stations = StationList([s], metadata=s.metadata)



model = shakermaker.ShakerMaker(crust, fault, stations)
print("Running shakermaker")
model.run(dt=dt)
print("Done shakermaker")

fig = ZENTPlot(s, show=True, xlim=[0,5])


