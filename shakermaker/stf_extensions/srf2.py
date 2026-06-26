import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction

class SRF2(SourceTimeFunction):
    """
    SRF2 slip-rate source time function.

    Builds a slip-rate (slip-velocity) function on ``[0, Tr)`` from three
    phases: a rising sine branch for ``t < Tp``, a ``sqrt(a + b/t**2)``
    plateau for ``Tp <= t < Te``, and a decaying sine tail for ``t >= Te``.
    The function is normalised to unit area and then scaled by ``slip``.

    :param Tr: Total rise time / duration of the function (s); ``t`` spans ``[0, Tr)``.
    :type Tr: float
    :param Tp: Peak time of the initial rising branch (s).
    :type Tp: float
    :param Te: Time at which the decaying tail begins (s).
    :type Te: float
    :param dt: Time step of the generated function (s).
    :type dt: float
    :param slip: Final slip used to scale the unit-area slip-rate.
    :type slip: float
    :param a: Constant term of the ``sqrt(a + b/t**2)`` envelope.
    :type a: float
    :param b: ``1/t**2`` coefficient of the ``sqrt(a + b/t**2)`` envelope.
    :type b: float
    """
    def __init__(self, Tr, Tp, Te, dt, slip , a, b):
        SourceTimeFunction.__init__(self)
        self._Tr = Tr
        self._Tp = Tp
        self._Te = Te
        self._dt = dt
        self._slip = slip
        self._a = a
        self._b = b

    
    def _generate_data(self):
        t, svf = self._srf2_function()
        self._t = t
        self._svf = svf
        self._data = svf * self._slip

    def _srf2_function(self):
        a = self._a
        b = self._b

        t = np.arange(0, self._Tr, self._dt)
        Nt = len(t)
        svf = 0 * t

        i1 = t < self._Tp
        svf[i1] = t[i1] / self._Tp * np.sqrt(a + b / self._Tp**2) * np.sin(np.pi * t[i1] / (2 * self._Tp))
        i2 = np.logical_and(self._Tp <= t, t < self._Te)
        svf[i2] = np.sqrt(a + b / t[i2]**2)
        i3 = t >= self._Te
        svf[i3] = np.sqrt(a + b / t[i3]**2) * np.sin(5/3 * np.pi * (self._Tr - t[i3]) / self._Tr)

        A = np.trapz(svf, dx=self._dt)

        svf /= A

        return t, svf

# Register the new SRF2 class with SourceTimeFunction
SourceTimeFunction.register(SRF2)
