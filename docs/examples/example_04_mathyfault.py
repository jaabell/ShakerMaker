from fkdrm.CrustModels.LOH import SCEC_LOH_1
from fkdrm.Sources import  MathyFaultPlane
from fkdrm.Tools.Plotting import SourcePlot
import numpy as np

#Use the LOH 1 crustal model
crust = SCEC_LOH_1()

#Fault geometry
x0 = [0., 0., 1.]   #Top left corner
strike = 20.
dip = 30.
n = 4 
lstrike = 5.
ldip = 3.
nstrike = 5*n
ndip = 3*n

vrup = 2.800
maxslip = 1.3

dfun = lambda xi,eta : np.sqrt(lstrike**2*(xi-0.5)**2+ldip**2*((eta-0.5)**2))
bubblefun = lambda xi,eta : xi*(1.-xi)*eta*(1-eta)*16

rakefun = lambda xi,eta : 90. + 0*xi
slipfun = lambda xi,eta : maxslip*bubblefun(xi,eta)
dsigmafun = lambda xi,eta : 15.e6 + 0*xi
ttfun = lambda xi,eta : dfun(xi,eta) / vrup

source = MathyFaultPlane(x0, strike, dip, lstrike, ldip, nstrike, ndip, rakefun, slipfun, dsigmafun, ttfun, crust)

fig = SourcePlot(source, show=True, autoscale=True, colorbar=True, colorby="tt")  #colorby="stf"

fig.savefig("example_04.png")