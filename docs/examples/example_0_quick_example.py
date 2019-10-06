#example.py
from fkdrm import fkdrm
from fkdrm.CrustModel import CrustModel
from fkdrm.Sources import PointSource 
from fkdrm.Receivers import SimpleStation

#Initialize two-layer CrustModel
crust = CrustModel(2)

#Slow layer
Vp=4.000			# P-wave speed (km/s)
Vs=2.000			# S-wave speed (km/s)
rho=2.600			# Density (gm/cm**3)
Qp=10000.			# Q-factor for P-wave
Qs=10000.			# Q-factor for S-wave
thickness = 1.0		# Self-explanatory
crust.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

#Halfspace
Vp=6.000
Vs=3.464
rho=2.700
Qp=10000.
Qs=10000.
thickness = 0   #Zero thickness --> half space
crust.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

#Initialize Source
source = PointSource([0,0,4], [30,90,0])

#Initialize Receiver
receiver = SimpleStation([0,4,0])

model = fkdrm.fkdrm(crust, source, receiver)

model.run()

from fkdrm.Tools.Plotting import ZENTPlot
fig = ZENTPlot(receiver, show=True, xlim=[0,3])

