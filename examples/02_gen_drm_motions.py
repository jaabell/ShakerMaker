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
from shakermaker.cm_library.AbellThesis import AbellThesis
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import Brune
# from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot

from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")




#Fault mechanism geometry
strike = 0.
dip =  90.
rake = 0.
mech = "strikeslip"
zsrc = 2.

#Source Time Function parameters
t0 = 0.
f0 = 2.

#Low-pass filter for results, cutoff frequency
fmax = 10.

#Simulation settings
vp=6.000
vs=3.500
rho=2.700
Qa=10000.
Qb=10000.

dt = 0.005
tmax = 10.
nfft = 2048
tb = 500
dk = 0.1
filter_results = False


λ = vs / fmax
dx = λ / 15

print(f"λ = {λ}")
print(f"dx ={dx}")
print(f"dt <{dx/vp} (required)")
print(f"dt ={dt} (supplied)")


#DRM Box Spec
nx = 10#32
ny = 10#32
nz = 4#9
dx = dx
dy = dx
dz = dx
x0 = [10.,10.,0]


#Halfspace
crust = CrustModel(1)
thickness = 0   #Infinite thickness!
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)


#Setup source time function
stf = Brune(f0=f0, t0=t0)

#Initialize Source fault
source = PointSource([0,0,zsrc], [strike,dip,rake], tt=0, stf=stf)
fault = FaultSource([source], 
	metadata={"name":"just a point source"})

box = DRMBox(x0,[nx,ny,nz],[dx,dy,dz],
	metadata={
	"filter_results":filter_results, 
	"filter_parameters":{"fmax":fmax},
	"name":"datasets/DRM_simple_z{0:04.0f}_f0{1:02.0f}_{2}".format(zsrc*1000, f0, mech)
	})

exit(0)

model = shakermaker.shakermaker(crust, source, receivers)
model.run(dt=dt,nfft=nfft,tb=tb,smth=1,dk=dk)

