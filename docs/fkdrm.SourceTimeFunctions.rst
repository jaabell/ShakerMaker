SourceTimeFunctions
=========================

Ground motion responses (seismic traces) in FKDRM are computed
by convolving the medium's Green's function evaluated at
the receiver point with the source time function. This
convolution is done numerically using  :func:`scipy.signal.convolve`. Therefore, it is most convenient to specify source
time functions as slip rate functions, with the resultant 
traces corresponding to the ground velocity history at the 
point of interest. 

.. note:: 
    
    **T.L.D.R.** These are all slip-rate functions. Treat them as such.


Brune 
--------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.Brune
    :members:
    :no-undoc-members:
    :show-inheritance:

Kostrov 
-----------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.Kostrov
    :members:
    :no-undoc-members:
    :show-inheritance:

Gaussian 
-----------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.Gaussian
    :members:
    :no-undoc-members:
    :show-inheritance:

Discrete 
-----------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.Discrete
    :members:
    :no-undoc-members:
    :show-inheritance:



MathFunction 
---------------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.MathFunction
    :members:
    :no-undoc-members:
    :show-inheritance:

Dirac 
--------------------------------------

.. automodule:: fkdrm.SourceTimeFunctions.Dirac
    :members:
    :no-undoc-members:
    :show-inheritance:



