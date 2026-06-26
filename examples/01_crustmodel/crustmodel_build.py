# 01 - Build a layered CrustModel and inspect its properties.
# 2026-06-06

import numpy as np
from shakermaker.crustmodel import CrustModel
from shakermaker.cm_library.LOH import SCEC_LOH_1, SCEC_LOH_3

# CrustModel(nlayers); add_layer(thickness_km, vp, vs, rho, Qp, Qs)
crust = CrustModel(2)
crust.add_layer(1.0, 4.000, 2.000, 2.600, 10000., 10000.)   # slow layer (1 km)
crust.add_layer(0.0, 6.000, 3.464, 2.700, 10000., 10000.)   # half-space (thickness = 0)

# modify one layer in place (raise Vs of the slow layer)
crust.modify_layer(0, vs=2.200)

# properties interpolated at given depths (km)
z = np.array([0.0, 0.5, 1.0, 2.0])
a, b, rho, qa, qb = crust.properties_at_depths(z)

# predefined library models
loh1 = SCEC_LOH_1()
loh3 = SCEC_LOH_3()

print(crust)
print("z (km):", z)
print("Vp   :", a)
print("Vs   :", b)

assert crust.nlayers == 2 and loh1.nlayers == 2 and loh3.nlayers == 2
print("PASS")
