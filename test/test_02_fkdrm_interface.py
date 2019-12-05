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
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3
from shakermaker.pointsource import PointSource 
from shakermaker.station import Station
# from shakermaker.Tools.Plotting import ZENTPlot

import pickle

# fid = open("station.pickle")
# receiver = pickle.load(fid)
# fid.close()


receiver = Station.load("station.pickle")


fig = ZENTPlot(receiver, show=True, xlim=[0,5])


