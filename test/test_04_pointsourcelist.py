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
from shakermaker.CrustModels.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.Sources import PointSource, PointSourceList
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

#Initialize Sources
source1 = PointSource([0,0,zsrc], [strike,dip,rake])
source2 = PointSource([0,0,zsrc], [strike,dip,rake],tt=3.0)

# source = PointSourceList([source1])
source = PointSourceList([source1, source2])

#Initialize Receiver
receiver = SimpleStation([x0,y0,0], name="Your House")

model = shakermaker.shakermaker(crust, source, receiver)
model.setup(dt=dt)
print "Running shakermaker"
model.run(debug_subgreen=False, debug_mpi=True)
print "Done shakermaker"

fig = ZENTPlot(receiver, show=True, xlim=[0,5])


