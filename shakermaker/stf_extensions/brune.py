import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction


class Brune(SourceTimeFunction):
    """The Brune Source Time Function
    
    .. figure::  ../../docs/source/images/stf_brune.png
       :scale: 90%
       :align:   center


    Implements the classic STF as a slip rate function

    .. math::

        f_s(t) = \\Delta\\cdot\\omega_w0^2 \\cdot \\left(t - t_0\\right)\\cdot\\exp\\left\\lbrace-w_0(t - t_0)\\right\\rbrace \\quad \\mathrm{for}\\, t \\geq t_0

    Where :math:`\\Delta` is the total slip across the fault, :math:`w_0 = 2 \\pi f_0` and :math:`f_0` is the corner-frequency defined by:

    .. math::

        f_0 = 4.9 \\times 10^6 V_s \\left(\\dfrac{\\Delta\\sigma}{M_0}\\right)^{1/3}

    :math:`V_s` is the local shear-wave speed in km/s, :math:`M_0` is the seismic-moment in dyne-cm, and :math:`\\Delta\\sigma` is the
    stress-drop in bars. 

    The source is defined by the slip (``slip``) and the fault trigger time (``t0``) and either of: **(i)** the corner frequency directly ``f0`` or **(ii)** the stress drop ``dsigma``, seismic moment ``m0`` and local shear-wave speed ``Vs``. 

    .. note:: 

        The ``t0`` parameter displaces the STF in its own time vector, it is more convenient to use the point source's trigger time``tt`` to specify the rupture process. 

    :param slip: Total slip across the fault. 
    :type slip: double
    :param f0: Corner frequency. 
    :type f0: double
    :param t0: Trigger time. 
    :type t0: double
    :param dsigma: Stress-drop. 
    :type dsigma: double
    :param M0: Seismic moment. 
    :type M0: double
    :param Vs: Local shear-wave speed. 
    :type Vs: double
    :param smoothed: Use a smoothed version of the source function. 
    :type smoothed: bool

    """
    def __init__(self, slip=1.0, f0=0.0, t0=0.0, dsigma=0.0, M0=1.0, Vs=0.0, smoothed=False):
        SourceTimeFunction.__init__(self)

        specified_corner_frequency = f0 > 0
        specified_stress_drop = dsigma > 0 and M0 > 0 and Vs > 0

        if specified_stress_drop and not specified_corner_frequency:
            f0 = 4.9e6 * (Vs/1000) * np.power((dsigma*10./1e6) / (M0*1e7), 1./3)

        self._slip = slip
        self._f0 = f0
        self._M0 = M0
        self._tr = 2/f0
        self._t0 = t0
        self._smoothed = smoothed

    def _generate_data(self):
        assert self._dt > 0, "Brune.get_data() - dt not set!! dt = {}".format(self._dt)

        w0 = 2*np.pi*self._f0
        self._t = np.arange(0, 4*self._tr + self._t0, self._dt/10)
        if self._smoothed:
            self._data = self._brune_impulse_smoothed(self._t, w0, t0=self._t0)
        else:
            self._data = self._brune_impulse(self._t, w0, t0=self._t0)

    def _brune_impulse(self, t, w0, t0=0.0):
        return self._slip*w0**2*(t - t0)*np.exp(-w0*(t - t0))*(t>=t0)

    def _brune_impulse_smoothed(self, t, w0, tau=2.31, t0=0.):
        tstar = w0*(t - t0)
        y = np.zeros(t.shape)

        i = tstar < tau
        y[i] = 1. - np.exp(-tstar[i])*(
            1 + tstar[i] +
            (tstar[i])**2/2 -
            3/(2*tau)*(tstar[i])**3 +
            3/(2 * tau**2) * (tstar[i])**4 -
            1/(2 * tau**3) * (tstar[i])**5
            )
        i = tstar >= tau
        y[i] = 1 - np.exp(-tstar[i]) * (1 + tstar[i])
        y[t < t0] = 0
        ydot = np.gradient(y, t)
        return self._slip*ydot


SourceTimeFunction.register(Brune)
