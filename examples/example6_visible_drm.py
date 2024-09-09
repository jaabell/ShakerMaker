from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions import Brune
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.tools.plotting import ZENTPlot

import numpy as np

do_DRM = False

#Fault mechanism geometry
ϕ,θ,λ = 0., 90., 0.    #Strike, dip, rake angles
zsrc = 1.0              #Source at 1-km depth

#Source Time Function 
t0, f0 = 0., 20			#Peak time and corner frequency
stf = Brune(f0=f0, t0=t0)

#Create fault (single source)
source = PointSource([0,0,zsrc], [ϕ,θ,λ], tt=0, stf=stf)
fault = FaultSource([source],metadata={"name":"fault"})

#Create crust model, single layer
vp,vs,rho,Qa,Qb=6.000,3.500,2.700,10000.,10000.
crust = CrustModel(1)
thickness = 0   
crust.add_layer(thickness, vp, vs, rho, Qa, Qb)

#Design a DRM box for maximum frequency fmax
fmax = 20.  #Hz
dx = vs / fmax / 5


#DRM Box Specification
nx, ny, nz = 30, 30, 12
x0 = [10.,10.,0]
# dt = 1/(10*fmax)
dt = 1/(2*fmax)
T_trav = dx*nx/vs
print(f"{1000*dx=} {nx=} {1000*dx*nx=} {1000*vs / fmax=} {dt=} {T_trav=}")

d0 = np.sqrt(x0[0]**2 + x0[1]**2 + zsrc**2)

ts = d0/vs
tp = d0/vp

print(f"{tp=} {ts=}")

if do_DRM:
	reciever = DRMBox(x0,[nx,ny,nz],[dx,dx,dx],
		metadata={"name":"example2"})
	#H5DRM writer
	writer = DRMHDF5StationListWriter("motions.h5drm")
else:
	station = Station(x=x0, metadata={"name":"station"})

	reciever = StationList([station], metadata={"name":"station"})
	writer = None



#Instantiate model
model = shakermaker.ShakerMaker(crust, fault, reciever)

#Execute model
model.run(
	dt=dt,  	#Time-step
	nfft=2048/2, 			#Half number of time-samples
	tb=0,    			#Number of samples before first arrival
	dk=0.05,    			#Discretization in wavenumber space
	writer=writer,
	tmin = 0.,
	tmax=55.)


if not do_DRM:
	#Visualize results
	ZENTPlot(station, show=False,  integrate=1)#, xlim=[0,10])
	ZENTPlot(station, show=False,  integrate=0)#, xlim=[0,10])
	ZENTPlot(station, show=False,  differentiate=1)#, xlim=[0,10])


	import matplotlib.pyplot as plt
	from scipy.fft import fft, fftfreq, fftshift
	z,e,n,t = station.get_response()

	z,e,n,t = z[t<10.], e[t<10.], n[t<10.], t[t<10.]

	Z, E, N = fft(z), fft(e), fft(n)

	f = fftfreq(z.size, t[1]-t[0])

	plt.figure()
	plt.plot(f[f>=0.],np.abs(Z[f>=0.]))
	plt.plot(f[f>=0.],np.abs(E[f>=0.]))
	plt.plot(f[f>=0.],np.abs(N[f>=0.]))

	plt.show()
