"""ShakerMaker <-> SW4 coordinate transforms.

ShakerMaker coordinates are the leading frame in this codebase: they hold
the georeferenced positions of every source, station and topography node.
SW4 only understands a local cartesian box whose origin sits at one of its
corners. The transform is therefore a single translation between the two
frames; rotation is not applied.

The offset is stored once in :class:`CoordinateTransform` and used in both
directions. Every method takes/returns plain ``ndarray``-compatible inputs;
no in-place mutation is performed.
"""

import numpy as np


class CoordinateTransform:
    """Translation between the ShakerMaker georef and the SW4 local box.

    Inputs
    ------
    domain_origin_m : array-like, shape (3,)
        Position of the SW4 origin expressed in ShakerMaker coordinates
        (metres). All conversions are pure translations against this offset.

    Attributes
    ----------
    domain_origin_m : ndarray, shape (3,)
        SW4 origin in ShakerMaker metres (the input, kept as ndarray).
    origin_m : ndarray, shape (3,)
        ShakerMaker origin expressed in SW4 local metres. Equals
        ``-domain_origin_m``.
    origin_km : ndarray, shape (3,)
        Same as ``origin_m`` but in kilometres.
    """

    def __init__(self, domain_origin_m):
        self.domain_origin_m = np.asarray(domain_origin_m, dtype=float)
        self.origin_m = -self.domain_origin_m
        self.origin_km = self.origin_m / 1000.0

    def from_shakermaker_km_to_sw4_m(self, xyz_km):
        """Convert a ShakerMaker point in km to SW4 local metres.

        Inputs
        ------
        xyz_km : array-like, shape (3,) or (N, 3)

        Returns
        -------
        ndarray
            Same shape as the input, expressed in SW4 local metres.
        """
        return np.asarray(xyz_km, dtype=float) * 1000.0 - self.domain_origin_m

    def from_shakermaker_km_to_sw4_km(self, xyz_km):
        """Convert a ShakerMaker point in km to SW4 local kilometres.

        Inputs
        ------
        xyz_km : array-like, shape (3,) or (N, 3)

        Returns
        -------
        ndarray
            Same shape as the input, expressed in SW4 local kilometres.
        """
        return self.from_shakermaker_km_to_sw4_m(xyz_km) / 1000.0

    def from_original_m_to_sw4_m(self, xyz_m):
        """Convert a ShakerMaker point in metres to SW4 local metres.

        Inputs
        ------
        xyz_m : array-like, shape (3,) or (N, 3)

        Returns
        -------
        ndarray
            Same shape as the input, expressed in SW4 local metres.
        """
        return np.asarray(xyz_m, dtype=float) - self.domain_origin_m

    def from_original_m_to_sw4_km(self, xyz_m):
        """Convert a ShakerMaker point in metres to SW4 local kilometres.

        Inputs
        ------
        xyz_m : array-like, shape (3,) or (N, 3)

        Returns
        -------
        ndarray
            Same shape as the input, expressed in SW4 local kilometres.
        """
        return self.from_original_m_to_sw4_m(xyz_m) / 1000.0

    def to_original_m(self, xyz_sw4_m):
        """Convert an SW4 local point in metres back to ShakerMaker metres.

        Inputs
        ------
        xyz_sw4_m : array-like, shape (3,) or (N, 3)

        Returns
        -------
        ndarray
            Same shape as the input, expressed in ShakerMaker metres.
        """
        return np.asarray(xyz_sw4_m, dtype=float) + self.domain_origin_m
