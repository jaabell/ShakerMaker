import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction
from shakermaker.stf_extensions import Dirac


class PointSource:
    """A source that is a point. 

    Defined by its position and spatial oriention, this source can also be
    given a trigger time and a source time function (to be convolved after
    the Green's function is computed).


    :param x: Position of the source in xyz coordinates.
    :type x: numpy array (shape (3,))
    :param anlges: Orientation of the fault ``angles = [strike, dip rake]`` in degrees.
    :type angles: numpy array (shape (3,))
    :param tt: trigger time for the fault (s)
    :type tt: double
    :param stf: source time function to convolver
    :type stf: :obj:`fkdrm.fkdrmBase.SourceTimeFunction`


    """
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
        """Numpy array with the (x,y,z) coordinates of the source"""
        return self._x

    @property
    def angles(self):
        """Numpy array with the (strike,dip,rake) angles of the source fault plane in degrees"""
        return self._angles

    @property
    def tt(self):
        """Scalar trigger time"""
        return self._tt

    @property
    def stf(self):
        """The source time-function to be convolved with."""
        return self._stf
