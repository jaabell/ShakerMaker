from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.stf_extensions import Brune
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox


#Fault mechanism geometry
ϕ,θ,λ = 0., 90., 0.    #Strike, dip, rake angles
zsrc = 1.0              #Source at 1-km depth

#Source Time Function 
t0, f0 = 0., 2.			#Peak time and corner frequency
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
fmax = 10.  #Hz
dx = vs / fmax / 15

#DRM Box Specification
nx, ny, nz = 10, 10, 4
x0 = [10.,10.,0]
drmreceiver = DRMBox(x0,[nx,ny,nz],[dx,dx,dx],
	metadata={"name":"example2"})

#H5DRM writer
writer = DRMHDF5StationListWriter("motions.h5drm")

#Instantiate model
model = shakermaker.ShakerMaker(crust, fault, drmreceiver)

#Execute model
model.run(
	dt=1/(2*fmax),  	#Time-step
	nfft=2048, 			#Half number of time-samples
	tb=500,    			#Number of samples before first arrival
	dk=0.1,    			#Discretization in wavenumber space
	writer=writer)
