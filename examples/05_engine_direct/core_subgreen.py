# 05 - Direct FK kernel call via shakermaker.core.subgreen.
# 2026-06-06

import numpy as np
from shakermaker.core import subgreen

# Layered model (SCEC LOH.1: slow 1 km layer over half-space)
mb = 3
src = 3
rcv = 1
stype = 2
updn = 0
d = [1., 4., 0.]
a = [4., 6., 6.]
b = [2., 3.464, 3.464]
rho = [2.6, 2.7, 2.7]
qa = [54.65, 69.3, 69.3]
qb = [137.95, 120., 120.]

dt = 0.025
nfft = 1024
tb = 0
nx = 1
sigma = 2
smth = 1
wc1 = 1
wc2 = 2
pmin = 0
pmax = 1
dk = 0.1
kc = 15.0
taper = 0.9
pf = 0.0
df = 0.7853981633974483
lf = 1.5707963267948966

sx, sy = 0.0, 0.0
rx, ry = 0.0, 7.0
x = np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2)

tdata, z, e, n, t0 = subgreen(
    mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx,
    sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf,
    sx, sy, rx, ry)

assert tdata.shape == (nx, 9, 2 * nfft)
assert z.shape == (2 * nfft,)
assert e.shape == (2 * nfft,)
assert n.shape == (2 * nfft,)
assert t0.shape == (nx,)
print("PASS")
