# 04 - Build SurfaceGrid receivers: plane, hollow and filled (<=50 each).
# 2026-06-06

from shakermaker.sl_extensions import SurfaceGrid

# SurfaceGrid(center_km, [nx, ny, nz], [hx, hy, hz], mode=..., plane_*=...).
# Large spacing keeps each layout small. A plane uses (n+1)*(n+1) points.
pos = [6.0, 8.0, 0.0]
dx = 2.0   # 2 km spacing -> few points

# XY plane at z=0  -> (5+1)*(5+1) = 36 + QA
plane = SurfaceGrid(pos, [5, 5, 0], [dx, dx, dx],
                    mode='plane', plane_z=0.0,
                    metadata={"name": "plane_z0"})

# Hollow box boundary (nz=1) -> bottom+top faces only here
hollow = SurfaceGrid(pos, [3, 3, 1], [dx, dx, dx],
                     mode='hollow',
                     metadata={"name": "hollow"})

# Filled 3D grid -> (2+1)*(2+1)*(2+1) = 27 + QA
filled = SurfaceGrid(pos, [2, 2, 2], [dx, dx, dx],
                     mode='filled',
                     metadata={"name": "filled"})

assert plane.nstations > 0 and plane.nstations <= 50
assert hollow.nstations > 0 and hollow.nstations <= 50
assert filled.nstations > 0 and filled.nstations <= 50

print("plane  stations:", plane.nstations)
print("hollow stations:", hollow.nstations)
print("filled stations:", filled.nstations)
print("PASS")
