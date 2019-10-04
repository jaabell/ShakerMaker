###############################################################################
# FKDRM Test Suite
# Test # 00 - Source Time Functions
# file: /tests/test_00_stf.pytdata,z,e,n,t0 = fkcore.subgreen(mb,src,rcv,stype,updn,d,a,b,rho,qa,qb,dt,nfft,tb,nx,sigma,smth,wc1,wc2,pmin,pmax,dk,kc,taper,x,pf,df,lf,sx,sy,rx,ry)
#
# Description
#
# This test exercices several source time functions and plots them.
#
###############################################################################

import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plotit(fig, stf, title, dt=0.001, label=""):
	plt.figure(fig).set_size_inches([5,2], forward=True)
	stf.dt = dt
	y = stf.data
	t = stf.t
	plt.plot(t, y,label=label)
	plt.xlabel("Time (s)")
	plt.ylabel("STF")
	plt.title(title)



###############################################################################
# Kostrov Source
###############################################################################
# from shakermaker.stf_extensions import Kostrov

# tr = 2.0
# tp = 0.1
# kostrov = Kostrov(tr=tr, tp=tp)
# plotit(1,kostrov, 'Kostrov')
# plt.savefig("../docs/images/stf_kostrov.png")


###############################################################################
# Brune Source
###############################################################################
from shakermaker.stf_extensions import Brune

f0 = 10.
t0 = 0.5
brune1 = Brune(f0=f0, t0=t0)
brune2 = Brune(f0=f0, t0=t0, smoothed=True)
plotit(2,brune1, "", label="Original")
plotit(2,brune2, "Brune sources. f0={} (Hz) t0={} (s) ".format(f0, t0), label="Smoothed")
plt.legend()
plt.savefig("../docs/images/stf_brune.png")
# plt.savefig("stf_brune.png")



###############################################################################
# Gaussian Source
###############################################################################
# from shakermaker.stf_extensions import Gaussian

# f = 10.
# gaussian = Gaussian(f)
# plotit(3,gaussian, "Guassian source. f={} (Hz) ".format(f))
# plt.savefig("../docs/images/stf_gaussian.png")





###############################################################################
plt.show()
