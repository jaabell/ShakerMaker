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
from shakermaker.CrustModels.AbellThesis import AbellThesis
from shakermaker.Sources import PointSource
from shakermaker.SourceTimeFunctions import Brune
from shakermaker.Receivers import DRMBox
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


crust = AbellThesis()

#Fault mechanism geometry
strike = 0.
dip =  45.
rake = 90.

#Source Time Function parameters
t0 = 0.
f0 = 10.

#Low-pass filter for results, cutoff frequency
fmax = 10.

#Simulation settings
factor = 2
dx = 0.005*factor
vs = 0.500
vp = 1.000
dt = dx/vp/factor
tmax = 6.
# nfft = 512*8
nfft = 2**(np.int32(np.log2(tmax / dt))+1)
filter_results = True


zsrcs = [0.550]#, 0.850, 1.200]


from shakermaker.Tools.Plotting import StationPlot

for zsrc in [zsrcs[0]]:
	#Setup source time function
	stf = Brune(f0=f0, t0=t0)

	#Initialize Source
	source = PointSource([0,0,zsrc], [strike,dip,rake], tt=0, stf=stf)


	receivers = DRMBox([0,2.5,0],[62/factor,62/factor,13/factor],[dx,dx,dx],
		filter_results=filter_results, 
		filter_parameters={"fmax":fmax},
		name="AbellThesis_z{0:04.0f}".format(zsrc*1000))

	model = shakermaker.shakermaker(crust, source, receivers)
	model.setup(dt=dt)
	model.setup(nfft=nfft)

	StationPlot(receivers, show=True, autoscale=True)

	# model.run(progressbar=True)#debug_mpi=True)

