SourceTimeFunction
==================

Ground motion responses (seismic traces) in `ShakerMaker` are computed
by convolving the medium's Green's function evaluated at
the receiver point with the source time function. This
convolution is done numerically using  :func:`scipy.signal.convolve`. Therefore, it is most convenient to specify source
time functions as slip rate functions, with the resultant 
traces corresponding to the ground velocity history at the 
point of interest. 

.. note:: 
    
    **T.L.D.R.** These are all slip-rate functions. Treat them as such.


Dirac 
--------------------------------------

.. automodule:: shakermaker.stf_extensions.dirac
    :members:
    :no-undoc-members:
    :show-inheritance:



Brune 
--------------------------------------

.. automodule:: shakermaker.stf_extensions.brune
    :members:
    :no-undoc-members:
    :show-inheritance:


Discrete 
-----------------------------------------

.. automodule:: shakermaker.stf_extensions.discrete
    :members:
    :no-undoc-members:
    :show-inheritance:



