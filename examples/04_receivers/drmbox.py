# 04 - Build a small DRMBox receiver layout (<=50 stations).
# 2026-06-06

from shakermaker.sl_extensions import DRMBox

# DRMBox(center_km, [nx, ny, nz], [hx, hy, hz]_km). azimuth was removed.
# Keep nelems tiny: a 1x1x1 box yields a handful of boundary stations + QA.
pos = [6.0, 8.0, 0.0]
nelems = [1, 1, 1]
h = [0.010, 0.010, 0.010]   # 10 m spacing

stations = DRMBox(pos, nelems, h, metadata={"name": "small_drm"})

assert stations.nstations > 0
assert stations.nstations <= 50

print("DRMBox stations:", stations.nstations)
print("box center (drmbox_x0):", stations.metadata["drmbox_x0"])
print("PASS")
