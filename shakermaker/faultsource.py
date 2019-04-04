class FaultSource:

    def __init__(self, sources, metadata):
        self._pslist = sources
        self._metadata = metadata

    def __iter__(self):
        return self._pslist.__iter__()

    @property
    def nsources(self):
        return len(self._pslist)

    @property
    def metadata(self):
        return self._metadata
