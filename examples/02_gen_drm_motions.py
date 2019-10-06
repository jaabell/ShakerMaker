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
from shakermaker.CrustModel import CrustModel
from shakermaker.Sources import PointSource
from shakermaker.SourceTimeFunctions import Brune
from shakermaker.Receivers import DRMBox
from shakermaker.Receivers import SimpleStation
from shakermaker.Receivers import StationList
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


lam = vs / fmax

dx = lam / 15
print "lam = ", lam
print "dx = ", dx
print "dt < ", dx/vp, " (required)"
print "dt = ", dt, " (supplied)"

# exit(0)

#DRM Box Spec
nx = 10#32
ny = 10#32
nz = 4#9
dx = dx
dy = dx
dz = dx
x0 = [10.,10.,0]



# exit(0)

#Setup Crust
# crust = AbellThesis()

crust = CrustModel(1)

#Halfspace

thickness = 0   #Infinite thickness!
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)


#Setup source time function
stf = Brune(f0=f0, t0=t0)

#Initialize Source
source = PointSource([0,0,zsrc], [strike,dip,rake], tt=0, stf=stf)

receivers = DRMBox(x0,[nx,ny,nz],[dx,dy,dz],
	filter_results=filter_results, 
	filter_parameters={"fmax":fmax},
	name="datasets/DRM_simple_z{0:04.0f}_f0{1:02.0f}_{2}".format(zsrc*1000, f0, mech))


model = shakermaker.shakermaker(crust, source, receivers)
model.setup(dt=dt)
model.setup(nfft=nfft)
model.setup(tb=tb)
model.setup(smth=1)
model.setup(dk=dk)

model.run(progressbar=True)

