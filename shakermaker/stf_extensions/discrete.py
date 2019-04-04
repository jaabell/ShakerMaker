import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction

class Discrete(SourceTimeFunction):

    def __init__(self, data, t):
        SourceTimeFunction.__init__(self)
        self._data = data
        self._t = t

    def _generate_data(self):
        pass

SourceTimeFunction.register(Discrete)
