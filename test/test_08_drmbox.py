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
dip = -45.
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
tmax = 12.
# nfft = 512*8
nfft = 2**(np.int32(np.log2(tmax / dt))+1)
filter_results = True

#Low-pass filter for results, cutoff frequency
fmax = 10.

zsrcs = [0.550, 0.850, 1.200]


# for zsrc in zsrcs:
zsrc = zsrcs[0]
#Setup source time function
stf = Brune(f0=f0, t0=t0)

#Initialize Source
source = PointSource([0,0,zsrc], [strike,dip,rake], tt=0, stf=stf)



receivers = DRMBox([0.0,2.5,0],[62,62,13],[dx,dx,dx],
	filter_results=filter_results, 
	filter_parameters={"fmax":fmax},
	name="AbellThesis_TEST")#.format(zsrc*1000))

model = shakermaker.shakermaker(crust, source, receivers)
model.setup(dt=dt)
model.setup(nfft=nfft)

# model.run(progressbar=True)#debug_mpi=True)

x_rcv = []
y_rcv = []
z_rcv = []
for rcv in receivers:
	x = rcv.get_pos()
	x_rcv.append(x[0])
	y_rcv.append(x[1])
	z_rcv.append(-x[2])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# p=ax.scatter(Y.reshape(nstrike,ndip), X.reshape(nstrike,ndip), zs=-Z.reshape(nstrike,ndip), zdir='z',c=DSIGMA)
ax.scatter(y_rcv, x_rcv,  z_rcv, "b")

# fig.colorbar(p)

scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']); ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
ax.set_zlabel("Depth (km)")

plt.show()
exit(0)
