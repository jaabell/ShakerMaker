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
from shakermaker.CrustModels.LOH import SCEC_LOH_1
from shakermaker.Sources import PointSource, MathyFaultPlane
from shakermaker.Receivers import SimpleStation, StationList
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


crust = SCEC_LOH_1()

strike = 20.
dip = 45.

n = 3 
lstrike = 5.
ldip = 3.
nstrike = 5*n
ndip = 3*n
x0 = [0., 0., 1.]

vrup = 2.800
maxslip = 1.3

dfun = lambda xi,eta : np.sqrt(lstrike**2*(xi-0.5)**2+ldip**2*((eta-0.5)**2))
bubblefun = lambda xi,eta : xi*(1.-xi)*eta*(1-eta)*16

rakefun = lambda xi,eta : 90. + 0*xi
slipfun = lambda xi,eta : maxslip*bubblefun(xi,eta)
dsigmafun = lambda xi,eta : 15.e6 + 0*xi
ttfun = lambda xi,eta : dfun(xi,eta) / vrup

source = MathyFaultPlane(x0, strike, dip, lstrike, ldip, nstrike, ndip, rakefun, slipfun, dsigmafun, ttfun, crust)

X,Y,Z,RAKE,DSIGMA,M0,VS,TT,SLIP = source.get_data()

th = np.linspace(0, np.pi, 2 )
R = 8.
x_rcv = R*np.cos(th)
y_rcv = R*np.sin(th)
z_rcv = 0*th

rcvlist = []
for x,y,z,th_ in zip(x_rcv, y_rcv, z_rcv,th):
	receiver = SimpleStation([x,y,z], name="th{0:2.0f}".format(180*th_/np.pi), filter_results=True, filter_parameters={"fmax":10.})
	rcvlist.append(receiver)
receivers = StationList(rcvlist)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p=ax.scatter(Y.reshape(nstrike,ndip), X.reshape(nstrike,ndip), zs=-Z.reshape(nstrike,ndip), zdir='z',c=DSIGMA)
ax.scatter(y_rcv, x_rcv,  z_rcv, "b")

fig.colorbar(p)

scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']); ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")
ax.set_zlabel("Depth (km)")

plt.show()
exit(0)


model = shakermaker.shakermaker(crust, source, receivers)


model.run()#debug_mpi=True)

import pickle
if model.is_master_process():
	nprocs = model.get_nprocs()
	data = [nprocs, receivers]
	pickle.dump(data, open("test_07_results_np_{}.pickle".format(nprocs),"wb"))
	# from shakermaker.Tools.Plotting import ZENTPlot
	# for rcv in receivers:
	# 	ZENTPlot(rcv,xlim=[0,60])
	# plt.show()
