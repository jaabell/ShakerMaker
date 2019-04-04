import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction

class Brune(SourceTimeFunction):

    def __init__(self, slip=1.0, f0=0.0, t0=0.0, dsigma=0.0, M0=1.0, Vs=0.0
        , smoothed=False):
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
        assert self._dt > 0
            , "Brune.get_data() - dt not set!! dt = {}".format(self._dt)

        w0 = 2*np.pi*self._f0
        self._t = np.arange(0, 4*self._tr + self._t0, self._dt/10)
        if self._smoothed:
            self._data = self.brune_impulse_smoothed(self._t, w0, t0=self._t0)
        else:
            self._data = self.brune_impulse(self._t, w0, t0=self._t0)

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
