###############################################################################
# shakermaker Test Suite
# Test # 07 - MSMR - Many Source Many Receiver 
# file: /tests/test_07_msmr.py
#
# Description
#
#
# References:
# 
###############################################################################

from shakermaker import shakermaker
from shakermaker.CrustModels.SOCal_LF import SOCal_LF
from shakermaker.Sources import RuptureFromFile
from shakermaker.Tools.Plotting import SourcePlot



crust = SOCal_LF()

print crust

source = RuptureFromFile("rupture.srf", crust, dx=140.157480315, dy=141.26984127)


# fig = SourcePlot(source, show=True, autoscale=True, colorbar=True, colorby="tt")  #colorby="stf"
# fig = SourcePlot(source, show=True, autoscale=True, colorbar=True, colorby="maxstf")  #colorby="stf"
fig = SourcePlot(source, show=True, autoscale=True, colorbar=True, colorby="slip")  #colorby="stf"
