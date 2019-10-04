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
# Description
#
# This test verifies that the shakermaker interface works for one simple source, 
# crustmodel and plots. 
#
###############################################################################

from shakermaker import shakermaker
from shakermaker.CrustModels.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.Sources import PointSource 
from shakermaker.Receivers import SimpleStation
from shakermaker.Tools.Plotting import ZENTPlot

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

#Initialize Receiver
receiver = SimpleStation([x0,y0,0], name="Your House", filter_results=True, filter_parameters={"fmax":10.})

model = shakermaker.shakermaker(crust, source, receiver)
model.setup(dt=dt)
print "Running shakermaker"
model.run()
print "Done shakermaker"

fig = ZENTPlot(receiver, show=True, xlim=[0,5])


