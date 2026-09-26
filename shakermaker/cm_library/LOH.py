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

    This is the same slow layer over a half-space as LOH.1, but with
    anelastic attenuation.

    The benchmark fixes the attenuation by two rules,

    .. math::
        Q_S = V_S\\,[\\mathrm{m/s}]\\,/\\,50,
        \\qquad
        Q_P = \\frac{3}{4}\\left(\\frac{V_P}{V_S}\\right)^2 Q_S ,

    the second one being the statement that the bulk is lossless
    (:math:`Q_\\kappa \\to \\infty`) and all dissipation happens in shear.
    They give :math:`Q_S = 40`, :math:`Q_P = 120` in the layer and
    :math:`Q_S = 69.3`, :math:`Q_P = 155.9` in the half-space, which are
    exactly the values of the SW4 reference input deck
    ``examples/scec/LOH.3-h50.in``.

    .. warning::
        Until 2026-09-26 this function carried ``Qa=54.65, Qb=137.95`` in the
        layer and ``Qa=69.3, Qb=120.`` in the half-space. Those came from the
        *interface* block of the SW4 deck -- the arithmetic averages
        ``(40+69.3)/2 = 54.65`` and ``(120+155.9)/2 = 137.95`` that SW4 uses to
        smear the material discontinuity over one grid cell -- and on top of
        that with :math:`Q_P` and :math:`Q_S` swapped. Results obtained with
        the old values are not LOH.3.

    .. note::
        ``add_layer`` takes ``qp`` before ``qs``.

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
    Qa=120.         #Q_P = (3/4)(vp/vs)^2 Q_S
    Qb=40.          #Q_S = vs[m/s]/50
    thickness = 1.

    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    #Halfspace
    vp=6.000
    vs=3.464
    rho=2.700
    Qa=155.9        #Q_P = (3/4)(vp/vs)^2 Q_S
    Qb=69.3         #Q_S = vs[m/s]/50
    thickness = 0   #Infinite thickness!
    model.add_layer(thickness, vp, vs, rho, Qa, Qb)

    return model

