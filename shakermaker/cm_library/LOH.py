# -*- coding: utf-8 -*-
"""

"""

from shakermaker.crustmodel import CrustModel

def SCEC_LOH_1():
    """This is an shakermaker Crustal Model for problem LOH.1  
    from the  SCEC test suite. 

    This is a slow layer over a half-space with no attenuation.

    .. note::
        Zero anelastic attenuation has been approximated 
        using high values for the Q-factor. 

    Reference:
    + Steven Day et al., Tests of 3D Elastodynamic Codes:
    Final report for lifelines project 1A01, Pacific Eartquake
    Engineering Center, 2001
    
    """

    #Initialize CrustModel
    model = CrustModel(2)

    #Slow layer
    vp=4.000
    vs=2.000
    rho=2.600
    Qa=10000.
    Qb=10000.
    thickness = 1.0

    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    #Halfspace
    vp=6.000
    vs=3.464
    rho=2.700
    Qa=10000.
    Qb=10000.
    thickness = 0   #Infinite thickness!
    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    return model

def SCEC_LOH_3():
    """This is an shakermaker Crustal Model for problem LOH.3  
    from the  SCEC test suite.

    This is a slow layer over a half-space with attenuation.

    Reference:
    + Steven Day et al., Tests of 3D Elastodynamic Codes:
    Final report for lifelines project 1A01, Pacific Eartquake
    Engineering Center, 2001

    """

    #Initialize CrustModel
    model = CrustModel(2)

    #Slow layer
    vp=4.000
    vs=2.000
    rho=2.600
    Qa=54.65
    Qb=137.95
    thickness = 1.

    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    #Halfspace
    vp=6.000
    vs=3.464
    rho=2.700
    Qa=69.3
    Qb=120.
    thickness = 0   #Infinite thickness!
    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    return model

