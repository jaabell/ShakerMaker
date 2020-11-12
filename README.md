![ShakerMaker](/docs/source/images/logo.png)

ShakerMaker is intended to provide a simple tool allowing earthquake engineers and seismologists to easily use the frequency-wavenumber method (FK) to produce ground-motion datasets for analysis using the Domain Reduction Method (DRM). DRM motions are stored directly into the H5DRM format.

The FK method, the core of ShakerMaker, is implemented in fortran (originally from http://www.eas.slu.edu/People/LZhu/home.html with several modifications), and interfaced with python through f2py wrappers. Classes are built on top of this wrapper to simplify common modeling tasks such as crustal model specification, generation of source faults (from simple point sources to full kinematic rupturespecifications), generating single recording stations, grids and other arrays of recording stations and stations arranged to meet the requirements of the DRM. Filtering and simple plotting tools are provided to ease model setup. Finally, computation of motion traces is done by pairing all sources and all receivers, which is parallelized using MPI. This means that ShakerMaker can run on simple personal computers all the way up to large supercomputing clusters. 

Installation
------------

For now, only though the git repo::

	git clone git@github.com:jaabell/ShakerMaker.git

Use the `setup.py` script, using setuptools, to compile and install::

	sudo python setup.py install

If you dont' have sudo, you can install locally for your user with::

	sudo python setup.py install --user


Dependencies
------------

- `h5py`
- `f2py`
- `numpy`
- `scipy`
- `mpi4py` (optional but highly recommended for parallel computing of the response)
- `matplotlib` (optional, for plotting)

You can get all these packages with `pip`::

	sudo pip install mpi4py h5py f2py numpy scipy matplotlib

or, for your user::

	sudo pip install --user mpi4py f2py h5py numpy scipy matplotlib

Quickstart usage
----------------

Using ShakerMaker is simple. You need to specify a :class:`CrustModel` (choose from the available
predefined models or create your own), a :class:`SourceModel` (from a simple 
:class:`PointSource` to a complex fully-customizable extended source with :class:`MathyFaultPlane`) 
and, finally, a :class:`Receiver` specifying a place to record motions (and store them
in memory or text format).

In this simple example, we specify a simple strike-slip (strike=90, that is due east) 
point source at the origin and a depth of 4km, on a custom two-layer crustal model, 
and a single receiver 5km away to the north::

	from shakermaker.shakermaker import ShakerMaker
	from shakermaker.crustmodel import CrustModel
	from shakermaker.pointsource import PointSource 
	from shakermaker.faultsource import FaultSource
	from shakermaker.station import Station
	from shakermaker.stationlist import StationList
	from shakermaker.tools.plotting import ZENTPlot

	#Initialize two-layer CrustModel
	model = CrustModel(2)

	#Slow layer
	Vp=4.000			# P-wave speed (km/s)
	Vs=2.000			# S-wave speed (km/s)
	rho=2.600			# Density (gm/cm**3)
	Qp=10000.			# Q-factor for P-wave
	Qs=10000.			# Q-factor for S-wave
	thickness = 1.0		# Self-explanatory
	model.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

	#Halfspace
	Vp=6.000
	Vs=3.464
	rho=2.700
	Qp=10000.
	Qs=10000.
	thickness = 0   #Zero thickness --> half space
	model.add_layer(thickness, vp, vs, rho, Qp, Qs)

	#Initialize Source
	source = PointSource([0,0,4], [90,90,0])
	fault = FaultSource([source], metadata={"name":"single-point-source"})


	#Initialize Receiver
	s = Station([0,4,0],metadata={"name":"a station"})
	stations = StationList([s], metadata=s.metadata)


These are fed into the shakermaker model class::

	model = ShakerMaker(crust, fault, stations)

Which is executed::

	model.run()

Results at the station can be readily visualized using the utility function :func:`Tools.Plotting.ZENTPlot`::

	from shakermaker.Tools.Plotting import ZENTPlot
	ZENTPlot(s, xlim=[0,60], show=True)

Yielding:

![ShakerMaker](/examples/example0_fig1.png)
