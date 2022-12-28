import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction


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
        self._data = data
        self._t = t
        self._dt = t[1] - t[0]

    def _generate_data(self):
        pass


SourceTimeFunction.register(Discrete)
