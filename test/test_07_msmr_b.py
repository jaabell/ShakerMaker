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

import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from shakermaker.Tools.Plotting import ZENTPlot
plt.style.use("ggplot")

files = glob.glob("./test_07_results_np_*.pickle")
files.sort()

if len(files) == 0:
	print "Seems that you have not run test_07_msmr_a.py yet"
	exit(-1)


for f in files:
	print f
	data = pickle.load(open(f,"rb"))
	nprocs = data[0]
	receivers = data[1]

	print nprocs
	print receivers

	for i, rcv in enumerate(receivers):
		print rcv
		ZENTPlot(rcv,fig=i+1,xlim=[0,60],label="{}".format(nprocs))
		plt.legend()
plt.show()
