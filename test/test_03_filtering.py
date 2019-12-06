###############################################################################
# shakermaker Test Suite
# Test # 03 - Filtering
# file: /tests/test_03_filtering.py
#
# Description
#
# This test replicates tests # 02 adding filtering of results
#
###############################################################################

#
# Descriptionb
#
# This test verifies that the shakermaker interface works for one simple source, 
# crustmodel and plots. 
#
###############################################################################

import sys
import os 
import logging

print("="*80)
print(sys.path)
print("="*80)

logfname = os.path.basename(__file__).replace(".py",".log")
logging.basicConfig(filename=logfname,level=logging.DEBUG)

from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.pointsource import PointSource 
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.faultsource import FaultSource
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
source = PointSource([0,0,zsrc], [strike,dip,rake])

fault = FaultSource([source], metadata={"name":"source"})


#Initialize Receiver
s = Station([x0,y0,0], metadata={"name":"Your House", "filter_results":True, "filter_parameters":{"fmax":10.}})
stations = StationList([s], metadata=s.metadata)

model = shakermaker.ShakerMaker(crust, fault, stations)

print("Running shakermaker")
model.run(dt=dt)
print("Done shakermaker")

fig = ZENTPlot(s, show=True, xlim=[0,5])


