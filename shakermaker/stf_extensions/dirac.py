import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction


class Dirac(SourceTimeFunction):
    """The Dirac delta

    Generate Green's functions using a Diract delta 
    source-time-function. 

    """
    def __init__(self):
        SourceTimeFunction.__init__(self)

    def _generate_data(self):
        self._data = np.array([1.0])
        self._t = np.array([0.0])


SourceTimeFunction.register(Dirac)
