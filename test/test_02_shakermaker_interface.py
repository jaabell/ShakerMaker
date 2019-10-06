###############################################################################
# shakermaker Test Suite
# Test # 02 - shakermaker interface
# file: /tests/test_02_shakermaker_interface.py
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

import cPickle

# fid = open("station.pickle")
# receiver = cPickle.load(fid)
# fid.close()


receiver = SimpleStation.load("station.pickle")


fig = ZENTPlot(receiver, show=True, xlim=[0,5])


