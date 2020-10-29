import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction
from shakermaker.stf_extensions import Dirac


class PointSource:

    def __init__(self, x, angles, stf=Dirac(), tt=0):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(angles, list):
            angles = np.array(angles)

        assert isinstance(stf, SourceTimeFunction), \
            "PointSource (Input error) - 'stf' Should be subclass of SourceTimeFunction"
        assert x.shape[0] == 3, \
            "PointSource (Input error) - 'x' Should be a numpy array with x.shape[0]=3"
        assert angles.shape[0] == 3, \
            "PointSource (Input error) - 'angles' Should be a numpy array with x.shape[0]=3"

        self._x = x
        self._angles = np.pi*angles/180
        self._stf = stf
        self._tt = tt

    @property
    def x(self):
        return self._x

    @property
    def angles(self):
        return self._angles

    @property
    def tt(self):
        return self._tt

    @property
    def stf(self):
        return self._stf
