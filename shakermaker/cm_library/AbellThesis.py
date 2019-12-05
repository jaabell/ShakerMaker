# -*- coding: utf-8 -*-
"""

"""


from shakermaker.crustmodel import CrustModel 
layers = [
[1.000, 0.500, 2.000, 0.050],
[1.894, 0.947, 2.018, 0.050],
[2.265, 1.132, 2.035, 0.100],
[2.789, 1.394, 2.070, 0.100],
[3.191, 1.595, 2.105, 0.100],
[3.530, 1.765, 2.140, 0.100],
[3.828, 1.914, 2.175, 0.100],
[4.098, 2.049, 2.210, 0.100],
[4.347, 2.173, 2.245, 0.100],
[4.578, 2.289, 2.280, 0.100],
[4.795, 2.397, 2.315, 0.100],
[5.000, 2.500, 2.350, 0]
]

def AbellThesis(split=1):
    """ Crustal model in Jose Abell's PhD thesis and paper

    .. figure::  ../docs/images/crust_model_abellthesis.png
       :scale: 60%
       :align:   center

    .. note::
        Zero anelastic attenuation has been approximated 
        using high values for the Q-factor. 

    Arguments:
    ==========
    :param split: The layering can be subdivided if needed.
    :type split: int

    Returns:
    ==========
    :returns: :class:`shakermaker.CrustModel`

    References: 
    + Abell, J. A. (2016). Earthquake-Soil-Structure Interaction Modeling of Nuclear Power Plants for Near-Field Events. University of California, Davis.
    + Abell, J. A., Orbović, N., McCallen, D. B., & Jeremic, B. (2018). Earthquake soil-structure interaction of nuclear power plants, differences in response to 3-D, 3 × 1-D, and 1-D excitations. Earthquake Engineering and Structural Dynamics, 47(6), 1478–1495. https://doi.org/10.1002/eqe.3026
    
    """

    #Initialize CrustModel
    model = CrustModel(11*split+1)
    Qa=1000.
    Qb=1000.

    for k, props in enumerate(layers):
        vp=props[0]
        vs=props[1]
        rho=props[2]

        if k == 11:
            split = 1
        thickness=props[3]/split

        for i in range(split):
            model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    return model


