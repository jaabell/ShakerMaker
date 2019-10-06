fkdrm python module
*******************

.. _coordinate-system:

Coordinate system in fkdrm
==========================

.. figure::  ../docs/images/fault_geom.png
   :scale: 80%
   :align:   center

FKDRM defines it's coordinate system with :math:`x` positive towards the north, 
:math:`y` positive towards the east and :math:`z` positive downwards. 

Strike is defined clockwise from the north, dip is measured from the horizontal, and rake increases in the down-dip direction.

The following figure illustrates this situation for a 
generic finite-fault that might be modeled using the
:obj:`MathyFaultPlane` class.




fkdrm main class
================



Methods
-------

.. autoclass:: fkdrm.fkdrm.fkdrm
    :members: __init__, setup, get_parameter, run, get_rank, get_nprocs, is_master_process


Simulation parameters
---------------------

These are the parameters that control the conditions of the simulations. 

.. note::

    In general, user will only need to play with ``nfft``, ``dt`` and possibly ``tb``.

.. autoattribute:: fkdrm.fkdrm.fkdrm.nfft
.. autoattribute:: fkdrm.fkdrm.fkdrm.dt
.. autoattribute:: fkdrm.fkdrm.fkdrm.tb
.. autoattribute:: fkdrm.fkdrm.fkdrm.dk

Less important parameters
+++++++++++++++++++++++++

.. container:: toggle

    .. container:: header

        (Click to view)

    .. container::

        They don't need tweaking in general. 

        .. autoattribute:: fkdrm.fkdrm.fkdrm.sigma
        .. autoattribute:: fkdrm.fkdrm.fkdrm.taper
        .. autoattribute:: fkdrm.fkdrm.fkdrm.smth
        .. autoattribute:: fkdrm.fkdrm.fkdrm.wc1
        .. autoattribute:: fkdrm.fkdrm.fkdrm.wc2
        .. autoattribute:: fkdrm.fkdrm.fkdrm.pmin
        .. autoattribute:: fkdrm.fkdrm.fkdrm.pmax
        .. autoattribute:: fkdrm.fkdrm.fkdrm.dk
        .. autoattribute:: fkdrm.fkdrm.fkdrm.kc
        .. autoattribute:: fkdrm.fkdrm.fkdrm.nx
        .. autoattribute:: fkdrm.fkdrm.fkdrm.sigma
        .. autoattribute:: fkdrm.fkdrm.fkdrm.smth
        .. autoattribute:: fkdrm.fkdrm.fkdrm.wc1
        .. autoattribute:: fkdrm.fkdrm.fkdrm.wc2
        .. autoattribute:: fkdrm.fkdrm.fkdrm.pmin
        .. autoattribute:: fkdrm.fkdrm.fkdrm.pmax
        .. autoattribute:: fkdrm.fkdrm.fkdrm.kc

    
Modeling Subclasses
-------------------

.. toctree::
    :maxdepth: 1

    fkdrm.CrustModels
    fkdrm.Receivers
    fkdrm.SourceTimeFunctions
    fkdrm.Sources


CrustModel
----------

.. automodule:: fkdrm.CrustModel
    :members:
    :show-inheritance:

Modeling Aids
-------------
.. toctree::
    :maxdepth: 1

    fkdrm.Tools

fkdrmBase - Abstract Base Classes
----------------------------------


.. automodule:: fkdrm.fkdrmBase
    :members:
    :show-inheritance: