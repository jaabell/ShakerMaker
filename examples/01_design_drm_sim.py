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
from shakermaker.Tools.Plotting import ZENTPlot
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


#Fault mechanism geometry
strike = 0.
dip =  90.
rake = 0.
zsrcs = [2.]
zsrc = 2.

#Source Time Function parameters
t0 = 0.
f0 = 2.

#Low-pass filter for results, cutoff frequency
fmax = 15.

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


#Simulation
rcvs = []
tp = []
ts = []
distance = []

lx = nx*dx
ly = ny*dy
lz = nz*dz

xsrc = np.array([0,0,zsrc])
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

xcorners = []
xcorners.append(x0 )
xcorners.append(x0 - lx/2*e1 - ly/2*e2 + 0*e3)
xcorners.append(x0 + lx/2*e1 - ly/2*e2 + 0*e3)
xcorners.append(x0 + lx/2*e1 + ly/2*e2 + 0*e3)
xcorners.append(x0 - lx/2*e1 + ly/2*e2 + 0*e3)
xcorners.append(x0 - lx/2*e1 - ly/2*e2 + lz*e3)
xcorners.append(x0 + lx/2*e1 - ly/2*e2 + lz*e3)
xcorners.append(x0 + lx/2*e1 + ly/2*e2 + lz*e3)
xcorners.append(x0 - lx/2*e1 + ly/2*e2 + lz*e3)

for xrcv in xcorners:
    print xrcv
    dist = norm(xrcv - xsrc)
    distance.append(dist)
    tp.append(dist/vp)
    ts.append(dist/vs)
    rcvs.append(SimpleStation(xrcv,
        filter_results=filter_results, 
        filter_parameters={"fmax":fmax}))
receivers = StationList(rcvs)
# xrcv = x0
# one_station = SimpleStation(xrcv,
#         filter_results=filter_results, 
#         filter_parameters={"fmax":fmax})

# model = shakermaker.shakermaker(crust, source, one_station)
model = shakermaker.shakermaker(crust, source, receivers)
model.setup(dt=dt)
model.setup(nfft=nfft)
model.setup(tb=tb)
model.setup(smth=1)
model.setup(dk=dk)


t0 = np.sqrt((x0[0])**2 + (x0[1])**2 + (x0[2] - zsrc)**2) / vp

print " t0 = ", t0

model.run(progressbar=True, debug_subgreen=True)
print " t0 = ", t0


# ZENTPlot(one_station, show=True, integrate=1)
# ZENTPlot(one_station, show=True, integrate=1)

if model.get_rank() == 0:
    ns = receivers.get_nstations()
    plt.figure()
    for i,rcv in enumerate(receivers):
        if i == 0:
            ax0 = plt.subplot(ns,1,i+1)
        else:
            plt.subplot(ns,1,i+1,sharex=ax0)#,sharey=ax0)

        z,e,n,t = rcv.get_response_integral()
        t1 = t[tb:]
        z1 = z[tb:]
        e1 = e[tb:]
        n1 = n[tb:]
        t2 = t[:tb]
        z2 = z[:tb]
        e2 = e[:tb]
        n2 = n[:tb]
        
        pz = plt.plot(t1,z1,label="z",linewidth=2)
        pe = plt.plot(t1,e1,label="e",linewidth=2)
        pn = plt.plot(t1,n1,label="n",linewidth=2)
        cz = pz[0].get_color()
        ce = pe[0].get_color()
        cn = pn[0].get_color()
        plt.plot(t2,z2,color=cz)#,linestyle="--")
        plt.plot(t2,e2,color=ce)#,linestyle="--")
        plt.plot(t2,n2,color=cn)#,linestyle="--")

        plt.axvline(tp[i], color="r", linestyle="--")
        plt.axvline(t[0]+tb*dt, color="y", linestyle="--")
        plt.axvline(t[0] + nfft*dt, color="k", linestyle="--")
        # plt.xlim([0,5])
        if i < ns - 1:
            pass
        else:
            plt.xlabel("Time")
            plt.legend()
        plt.ylabel("$d = {0:3.1f} $".format(distance[i]), rotation=0,  horizontalalignment='right')
        plt.gca().get_yaxis().set_ticks([])
        plt.suptitle("$z_{src}"+" = {}km$".format(zsrc))

    plt.show()



