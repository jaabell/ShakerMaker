import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction
from scipy.interpolate import interp1d

def interpolator(told, yold, tnew):
    return interp1d(told, yold, bounds_error=False, fill_value=(yold[0], yold[-1]))(tnew)


class Discrete(SourceTimeFunction):
    """Specify the STF using discrete values at your discretion. 

    :param data: STF values
    :type data: numpy vector shape (Nt,0)
    :param t: STF time-values. Must start and end at 0, can be un-evenly spaced.
    :type t: numpy vector shape (Nt,0). 

    .. note::

        If the supplied STF specification is un-evenly spaced it 
        gets interpolated to the simulation time-step before numerical 
        convolution. 

    Example::

        t = np.array([0,0.01,0.02,0.1,0.2])
        slip = np.array([0,0.2,1,0.4,0])
        stf = Discrete(data,t)

    """
    def __init__(self, data, t):
        SourceTimeFunction.__init__(self)
        self._data_orig = data
        self._t_orig = t
        self._dt_orig = t[1] - t[0]

    def _generate_data(self):
        self._t = np.arange(self._t_orig[0], self._t_orig[-1],self._dt)
        self._data = interpolator(self._t_orig, self._data_orig, self._t)


SourceTimeFunction.register(Discrete)
