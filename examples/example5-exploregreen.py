from shakermaker import shakermaker
from shakermaker.cm_library.SOCal_LF import SOCal_LF
from shakermaker.cm_library.LOH import SCEC_LOH_3
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot
from shakermaker.core import subgreen
import matplotlib.pyplot as plt
import numpy as np
#Import from pre-packaged crustal models
# crust = SCEC_LOH_3()

# #Create source
# z = 5.0                 # Source depth (km)
# s,d,r = 0., 45., 90.     # Fault plane angles (deg)
# source = PointSource([0,0,z], [s,d,r])
# fault = FaultSource([source], metadata={"name":"source"})

# #Create recording station
# x0,y0 = 0.,7.           # Station location
# s = Station([x0,y0,0], 
#         metadata={
#         "name":"Your House", 
#         "filter_results":False, 
#         "filter_parameters":{"fmax":10.}
#         })
# stations = StationList([s], metadata=s.metadata)

#Create model, set parameters and run
# model = shakermaker.ShakerMaker(crust, fault, stations)
# model.run(
#  dt=0.005,   # Output time-step
#  nfft=2048,  # N timesteps
#  dk=0.05,     # wavenumber discretization
#  tb=0,      # Initial zero-padding
#  verbose=True
#  )

mb = 3
src = 3
rcv = 1
stype = 2
updn = 0
d = [1., 4., 0.]
a = [4., 6., 6.]
b = [2.,    3.464, 3.464]
rho = [2.6, 2.7, 2.7]
qa = [54.65, 69.3,  69.3 ]
qb = [137.95, 120.,   120.  ]
dt = 0.005
nfft = 2048
tb = 0
nx = 1
sigma = 2
smth = 1
wc1 = 1
wc2 = 2
pmin = 0
pmax = 1
dk = 0.05
kc = 15.0
taper = 0.9
x = 7.0
pf = 0.0
df = 0.7853981633974483
lf = 1.5707963267948966


sx = 0.0
sy = 0.0
rx = 0.0
ry = 7.0


plt.figure(1)
plt.subplot(3,3,1)

Δ = 0.1

dxdy = [
	[0, 0, 0., 0.],
	[0, Δ, 0., 0.],
	[0, 0, 0., Δ],
]

data = []

for dsx, dsy, drx, dry in dxdy:
	new_sx = sx + dsx
	new_sy = sy + dsy
	new_rx = rx + drx
	new_ry = ry + dry

	print(f"sx={new_sx} sy={new_sy} rx={new_rx} sy={new_ry}")

	x = np.sqrt((new_sx-new_rx)**2 + (new_sy - new_ry)**2)

	tdata, z, e, n, t0 = subgreen(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, new_sx, new_sy, new_rx, new_ry)

	Nt = len(z)
	data.append((tdata,t0,Nt))

	print(f" ---> {t0[0]=}")

	t = np.arange(Nt)*dt + t0 


	for i in range(9):
		plt.subplot(3,3,1+i)
		plt.plot(t, tdata[0,i,:])
		plt.title(str(i))


plt.figure(1)
plt.subplot(3,3,1)



	for i in range(9):
		plt.subplot(3,3,1+i)
		plt.plot(t, tdata[0,i,:])
		plt.title(str(i))


plt.show()