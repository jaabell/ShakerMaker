class FaultSource:
    """A fault is a collection of point-sources

    If you want to have more than one point-source in space 
    generating motions, you use this class. 

    :param sources: A list of PointSources
    :type list: ``list`` containing :obj:`PointSource`

    Example::

        #Two strike-slip sources at z=1.0 and 1.5 (km)
        ps1 = PointSource([0, 0, 1],[0, 90, 0])
        ps2 = PointSource([0, 0, 1.5],[0, 90, 0])

        fault = FaultSource([ps1, ps2])

    """
    def __init__(self, sources, metadata):
        self._pslist = sources
        self._metadata = metadata

    def __iter__(self):
        return self._pslist.__iter__()

    @property
    def nsources(self):
        """Number of sub-faults"""
        return len(self._pslist)

    @property
    def metadata(self):
        """Source metadata, such as fault name. """
        return self._metadata
