Point sources and faults
========================

.. figure::  ../../docs/source/images/fault_geom.png
   :scale: 80%
   :align:   center

`ShakerMaker` defines it's coordinate system with :math:`x` positive towards the north, 
:math:`y` positive towards the east and :math:`z` positive downwards. 

Strike is defined clockwise from the north, dip is measured from the horizontal, and rake increases in the down-dip direction.

Faults are specified using the ``FaultSource`` which are just lists of sub-faults which
are of the ``PointSource`` type. Faults can have arbitrary shape and complexity. 



PointSource 
--------------------------------

.. automodule:: shakermaker.pointsource
    :members:
    :undoc-members:
    :show-inheritance:



FaultSource 
--------------------------------

.. automodule:: shakermaker.faultsource
    :members:
    :undoc-members:
    :show-inheritance:

