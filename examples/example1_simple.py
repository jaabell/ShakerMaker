from shakermaker import shakermaker
from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.tools.plotting import ZENTPlot

#Import from pre-packaged crustal models
crust = SCEC_LOH_1()

#Create source
z = 1.0                 # Source depth (km)
s,d,r = 0., 45., 0.     # Fault plane angles (deg)
source = PointSource([0,0,z], [s,d,r])
fault = FaultSource([source], metadata={"name":"source"})

#Create recording station
x0,y0 = 1.,1.           # Station location
s = Station([x0,y0,0], 
        metadata={
        "name":"Your House", 
        "filter_results":True, 
        "filter_parameters":{"fmax":10.}
        })
stations = StationList([s], metadata=s.metadata)

#Create model, set parameters and run
model = shakermaker.ShakerMaker(crust, fault, stations)
model.run(
 dt=0.005,   # Output time-step
 nfft=2048,  # N timesteps
 dk=0.1,     # wavenumber discretization
 tb=20,      # Initial zero-padding
 )

#Visualize results
ZENTPlot(s, show=True, xlim=[0,3])