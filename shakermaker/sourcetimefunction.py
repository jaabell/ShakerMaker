import numpy as np
import abc
import scipy.signal as sig
from scipy.interpolate import interp1d


class SourceTimeFunction(metaclass=abc.ABCMeta):

    def __init__(self, dt=-1):
        self._dt = dt
        self._data = None
        self._t = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        assert value > 0, "SourceTimeFunction - dt must be > 0. Got dt = {}".format(value)

        self._dt = value
        self._generate_data()

    @property
    def data(self):
        if self._data is None:
            self._generate_data()
        return self._data

    @property
    def t(self):
        if self._t is None:
            self._generate_data()
        return self._t

    @abc.abstractmethod
    def _generate_data(self):
        raise NotImplementedError('derived class must define method generate_data')

    def convolve(self, val):
        if len(self.data) == 1:
            val_stf = val*self.data[0]
        else:
            tstf_resampled = np.arange(0, self.t.max(), self.dt)
            dstf_resampled = interp1d(self.t, self.data)(tstf_resampled)
            val_stf = sig.convolve(val, dstf_resampled, mode="full")[0:len(val)]

        return val_stf
