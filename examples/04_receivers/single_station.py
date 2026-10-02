# 04 - Build a single Station and wrap it in a StationList.
# 2026-06-06

from shakermaker.station import Station
from shakermaker.stationlist import StationList

# Station([x, y, z] in km, metadata={...}). internal flags DRM-interior points.
sta = Station([6.0, 8.0, 0.0], metadata={"name": "surface_rcv"})

stations = StationList([sta], metadata={"name": "single"})

s0 = stations.get_station_by_id(0)

assert stations.nstations == 1
assert list(s0.x) == [6.0, 8.0, 0.0]
assert s0.is_internal is False

print("A Station holds: x =", s0.x, "| metadata =", s0.metadata, "| internal =", s0.is_internal)
print("StationList nstations:", stations.nstations)
print("PASS")
