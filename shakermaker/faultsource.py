from typing import List, Dict
from shakermaker.pointsource import PointSource

class FaultSource:
    """A fault is a collection of point-sources.

    If you want to have more than one point-source in space 
    generating motions, you use this class. 

    Parameters
    ----------
    sources : list
        A list of PointSources
    metadata : dict
        Metadata to store with the fault source

    Example
    -------
    ps1 = PointSource([0, 0, 1],[0, 90, 0])
    ps2 = PointSource([0, 0, 1.5],[0, 90, 0])
    fault = FaultSource([ps1, ps2], {})
    """
    def __init__(self, sources: List[PointSource], metadata: Dict):
        self._pslist = sources
        self._metadata = metadata

    def __iter__(self):
        return iter(self._pslist)

    @property
    def nsources(self):
        """Get the number of sub-faults."""
        return len(self._pslist)

    @property
    def metadata(self):
        """Get source metadata, such as fault name."""
        return self._metadata

    def get_source_by_id(self, id: int) -> PointSource:
        """Get a source by its id.

        Parameters
        ----------
        id : int
            The source id.

        Returns
        -------
        PointSource
            The source at the specified id.

        Raises
        ------
        IndexError
            If the id is out of range.
        """
        if id >= len(self._pslist) or id < 0:
            raise IndexError("Source index out of range.")
        return self._pslist[id]
