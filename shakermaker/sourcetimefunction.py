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

    def convolve(self, val, t, debug=False):
        if len(self.data) == 1:
            val_stf = val*self.data[0]
        else:
            dt_old = t[1] - t[0]
            dt_new = self.dt
            # print(f"Resampling VAL from {dt_old} to {dt_new}")
            # val = 0*val
            # val[len(val)//2]=1
            t_resampled = np.arange(t[0], t[-1], dt_new)
            val_resampled = interp1d(t, val, bounds_error=False, fill_value=(val[0], val[-1]))(t_resampled)
            # val_resampled = sig.resample(val, len(t_resampled))
            # val_resampled *= 0
            # val_resampled[len(val_resampled)//2]=1
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="full")[0:len(val_resampled)]*dt_new
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="full")[0:len(val_resampled)] / sum(self.data)
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="same") / (sum(self.data)*dt_new)
            val_stf_resampled = sig.convolve(val_resampled, self.data, mode="full")[0:len(val_resampled)] * dt_new / dt_old
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="same") * dt_new / dt_old
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="same") 
            # val_stf_resampled = sig.convolve(val_resampled, self.data, mode="full")[0:len(val_resampled)]
            val_stf = interp1d(t_resampled, val_stf_resampled, bounds_error=False, fill_value=(val[0], val[-1]))(t)
            # print(f"Resampling VAL from {t_resampled[1] - t_resampled[0]} to {t[1] - t[0]}")
            # print(f"  len before = {len(val_stf_resampled)} len after = {len(val_stf)}")
            # interp1d
            # tstf_resampled = np.arange(0, self.t.max(), self.dt)
            # dstf_resampled = interp1d(self.t, self.data)(tstf_resampled)
            # val_stf = sig.convolve(val, dstf_resampled, mode="full")[0:len(val)]
            # sig.convolve
            if debug:
                import matplotlib.pylab as plt

                plt.figure(1)
                # plt.plot(tstf_resampled, dstf_resampled, label="resampled STF")
                # plt.legend()
                plt.plot(t, val, label="original val")
                plt.plot(t_resampled, val_resampled, ".", label="resampled val")

                t1 = t_resampled[val_stf_resampled.argmax()]
                t2 = self.t[self.data.argmax()]
                plt.legend()

                plt.figure(2)
                plt.plot(t, val,  label="original val")
                plt.plot(t_resampled, val_resampled, ".", label="original val")
                plt.plot(t_resampled, val_stf_resampled, ".", label="convolved resampled val")
                plt.plot(t, val_stf, label="convolved val")
                plt.plot(self.t-t2+t1, self.data, label="original STF")
                plt.legend()
                plt.show()


        return val_stf
