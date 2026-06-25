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
        raise NotImplementedError('derived class must define method _generate_data')

    def convolve(self, val, t, debug=False):
        """Convolve the signal ``val`` sampled on ``t`` with this source time function.

        The source time function is resampled onto the signal sampling step
        ``dt = t[1] - t[0]`` and convolved with ``val`` using an FFT. The
        result is returned on the same time grid as ``val``, which is the grid
        used by the rest of the pipeline (``add_to_response``, HDF5 output).

        The STF is resampled onto the signal grid (rather than the signal onto
        the STF grid) because the STF is always short while ``val`` has ``nfft``
        points, so resampling the STF is the cheaper operation.

        :param val: Signal samples.
        :type val: np.array
        :param t: Time vector for ``val``; assumed uniformly sampled.
        :type t: np.array
        :param debug: If True, plot the signal, the convolved signal and the STF.
        :type debug: bool

        :returns: ``val`` convolved with the source time function, on grid ``t``.
        :rtype: np.array
        """
        if len(self.data) == 1:
            # Dirac or unit impulse: trivial scaling, no convolution needed.
            return val * self.data[0]

        dt = t[1] - t[0]

        # Resample the STF to the signal's dt. self.t may have a finer
        # resolution than dt (e.g. Brune generates self.t at self._dt/10 for
        # internal waveform accuracy); this brings it onto the signal grid.
        t_stf_r = np.arange(self.t[0], self.t[-1] + dt, dt)
        stf_r   = interp1d(self.t, self.data,
                            bounds_error=False, fill_value=0.0)(t_stf_r)

        # FFT convolution. Result truncated to the val grid and scaled by dt.
        val_stf = sig.fftconvolve(val, stf_r, mode="full")[0:len(val)] * dt

        if debug:
            import matplotlib.pylab as plt
            plt.figure(1)
            plt.plot(t, val,     label="original val")
            plt.plot(t, val_stf, label="convolved val")
            plt.plot(self.t, self.data / (self.data.max() or 1) * np.abs(val).max(),
                     label="STF (normalised)", alpha=0.6)
            plt.legend(); plt.show()

        return val_stf
