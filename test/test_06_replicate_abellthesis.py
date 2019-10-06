###############################################################################
# shakermaker Test Suite
# Test # 06 - Replicate Jose Abell's thesis results
# file: /tests/test_06_replicate_abellthesis.py
#
# Description
#
# This test replicates the surface response seen in the examples in Jose Abell's
# PhD thesis and article. 
#
# References:
# 
# Abell, J. A. (2016). Earthquake-Soil-Structure Interaction Modeling of Nuclear 
#   Power Plants for Near-Field Events. University of California, Davis.
#
# Abell, J. A., Orbovic, N., McCallen, D. B., & Jeremic, B. (2018). Earthquake 
# 	soil-structure interaction of nuclear power plants, differences in response 
# 	to 3-D, 3 x 1-D, and 1-D excitations. 
# 	 Earthquake Engineering and Structural Dynamics, 47(6), 1478-1495. 
# 	https://doi.org/10.1002/eqe.3026
###############################################################################

from shakermaker import shakermaker
from shakermaker.CrustModels.AbellThesis import AbellThesis
from shakermaker.Sources import PointSource, PointSourceList
from shakermaker.Receivers import SimpleStation
from shakermaker.SourceTimeFunctions import Brune
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
plt.style.use("ggplot")

#Receiver location (km)
x0 = 0.
y0 = 2.5

#3 Sources at depths (km)
zsrcs = [0.550, 0.850, 1.200]

#Fault mechanism geometry
strike = 0.
dip =  45.
rake = 90.

#Source Time Function parameters
t0 = 0.
f0 = 10.

#Simulation settings
factor = 1
dx = 0.005*factor
vs = 0.500
vp = 1.000
dt = dx/vp/factor
tmax = 6.
# nfft = 512*8
nfft = 2**(np.int32(np.log2(tmax / dt))+1)
filter_results = True
# filter_results = False

#Low-pass filter for results, cutoff frequency
fmax = 10.


#############################################################
#  Run cases
#############################################################


for i_zsrc, zsrc in enumerate(zsrcs):

	#Crustal model
	crust = AbellThesis()

	#Setup recording station 
	station = SimpleStation([x0,y0,0], filter_results=filter_results, filter_parameters={"fmax":fmax})

	#Setup source time function
	stf = Brune(f0=f0, t0=t0, smoothed=True)

	#Initialize Source
	source = PointSource([0,0,zsrc], [strike,dip,rake], tt=0, stf=stf)
	
	#Initialize shakermaker and set custom parameters
	model = shakermaker.shakermaker(crust, source, station)
	model.setup(dt=dt)
	model.setup(nfft=nfft)

	#Run sim for current source depth
	print "Running shakermaker for source at z_src = {}".format(zsrc)
	model.run(debug_subgreen=True)
	print "Done shakermaker for source at z_src = {}".format(zsrc)


	#############################################################
	#############################################################
	# Plots
	#############################################################
	#############################################################
	
	#Get displacements and accelerations
	z,e,n,t = station.get_response_integral()
	z_ddot,e_ddot,n_ddot,t_ddot = station.get_response_derivative()

	#Normalize results to get unit E-W displacement
	emax = max(abs(e))
	e, z, n = e/emax, z/emax, n/emax
	e_ddot, z_ddot, n_ddot = e_ddot/emax, z_ddot/emax, n_ddot/emax

	#Plot displacements
	linecolor_zsrc = ['g', 'b', 'r'][i_zsrc]

	plt.figure(1)
	for i,comp in enumerate([e,n,z]):
		if i == 0:
			ax0 = plt.subplot(3,1,i+1)
		else:
			plt.subplot(3,1,i+1,sharex=ax0,sharey=ax0)

		plt.plot(t, comp, linecolor_zsrc)
		
		plt.xlim([0,5])
		plt.xticks(np.arange(0,5.5,0.5))
		plt.yticks(np.arange(-1,1.5,0.5))
		plt.ylabel(["$u_E$","$u_N$","$u_Z$"][i])
	plt.xlabel("Time, $t$ (s)")

	#Plot accelerations
	plt.figure(2)
	for i,comp in enumerate([e_ddot,n_ddot,z_ddot]):
		if i == 0:
			ax0 = plt.subplot(3,1,i+1)
		else:
			plt.subplot(3,1,i+1,sharex=ax0,sharey=ax0)

		plt.plot(t, comp, linecolor_zsrc)
		
		plt.xlim([0,5])
		plt.xticks(np.arange(0,5.5,0.5))
		plt.ylabel(["$\\ddot{u}_E$","$\\ddot{u}_N$","$\\ddot{u}_Z$"][i])
	plt.xlabel("Time, $t$ (s)")


plt.show()
