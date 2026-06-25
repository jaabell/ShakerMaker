# 03 - Instantiate the 5 source time functions and check they generate data.
# 2026-06-06

import numpy as np
from shakermaker.stf_extensions.dirac import Dirac
from shakermaker.stf_extensions.discrete import Discrete
from shakermaker.stf_extensions.brune import Brune
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.stf_extensions.srf2 import SRF2

DT = 0.001

# synthetic pulse for Discrete (must start/end at 0)
t_user = np.linspace(0, 0.3, 301)
data_user = np.exp(-((t_user - 0.1) / 0.02) ** 2)

stfs = [
    Dirac(),
    Discrete(data_user, t_user),
    Brune(f0=10.0, t0=0.5, slip=1.0, smoothed=False),
    Gaussian(t0=0.1, freq=60.0, M0=1.0),
    SRF2(Tr=2.0, Tp=0.1, Te=1.5, dt=DT, slip=12.0, a=1.0, b=1.0),
]

for stf in stfs:
    stf.dt = DT                             # triggers _generate_data
    assert len(stf.t) > 0 and len(stf.data) > 0

print("PASS")
