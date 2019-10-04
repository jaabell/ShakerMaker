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
from shakermaker.CrustModels.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.Sources import PointSource 
from shakermaker.Receivers import SimpleStation
from shakermaker.SourceTimeFunctions import Brune
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
brune = Brune(f0=10.)
source = PointSource([0,0,zsrc], [strike,dip,rake], stf=brune)

#Initialize Receiver
receiver = SimpleStation([x0,y0,0], name="Your House")

model = shakermaker.shakermaker(crust, source, receiver)
model.setup(dt=dt)
print "Running shakermaker"
model.run()
print "Done shakermaker"

fig = ZENTPlot(receiver, show=True)#, xlim=[0,5])


|