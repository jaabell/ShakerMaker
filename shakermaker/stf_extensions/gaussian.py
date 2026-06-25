import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction


class Gaussian(SourceTimeFunction):
    """
    Gaussian source-time function or its time derivative.

    .. math::
        g(t) = \\frac{\\omega}{\\sqrt{2\\pi}}\\,
               e^{-\\tfrac{1}{2}\\,(\\omega\\,(t - t_0))^2}

    where :math:`\\omega` is ``freq`` and :math:`t_0` is ``t0``. When
    ``derivative=True`` the time derivative :math:`\\mathrm{d}g/\\mathrm{d}t`
    is used instead. The result is scaled by the seismic moment ``M0``.

    :param t0: Time shift (s)
    :type t0: float
    :param freq: Frequency parameter (1/s)
    :type freq: float
    :param M0: Seismic moment. Default is 1 (unit source).
    :type M0: float
    :param derivative: If True, use the time derivative of the Gaussian.
    :type derivative: bool

    Example:
        >>> stf = Gaussian(t0=0.36, freq=16.6667, M0=1.0 , derivative=True)
    """

    def __init__(self, t0=0.36, freq=16.6667, M0=1.0 , derivative=False):
        SourceTimeFunction.__init__(self)
        self._t0 = t0
        self._freq = freq
        self._M0 = M0
        self._derivative = derivative

    def _generate_data(self):
        assert self._dt > 0, "Gaussian.get_data() - dt not set!"
        # t
        tmax = self._t0 + 5.0 / self._freq
        self._t = np.arange(0, tmax, self._dt)
        # Gaussian
        g = (self._freq / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (self._freq * (self._t - self._t0))**2)
        dg_dt = -(self._freq**3 * (self._t - self._t0) / np.sqrt(2.0 * np.pi)) * np.exp(
            -0.5 * (self._freq * (self._t - self._t0))**2
        )
        self._data = self._M0 * (dg_dt if self._derivative else g)

SourceTimeFunction.register(Gaussian)
