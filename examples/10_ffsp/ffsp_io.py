# 10 - FFSP I/O: run a small source, write HDF5 + legacy format, load back and check.
# 2026-06-06

try:
    import h5py
except Exception:
    print("SKIP: h5py not available")
    raise SystemExit(0)

import numpy as np
from shakermaker.crustmodel import CrustModel
from shakermaker.ffspsource import FFSPSource

# 4-layer crust (same as example8_ffsp.py): add_layer(thickness, vp, vs, rho, qa, qb)
crust = CrustModel(4)
crust.add_layer(0.200,  1.32, 0.75, 2.40, 1000., 1000.)
crust.add_layer(0.800,  2.75, 1.57, 2.50, 1000., 1000.)
crust.add_layer(14.500, 5.50, 3.14, 2.50, 1000., 1000.)
crust.add_layer(0.000,  7.00, 4.00, 2.67, 1000., 1000.)

source = FFSPSource(
    id_sf_type=8, freq_min=0.01, freq_max=24.0,
    fault_length=30.0, fault_width=16.0,
    x_hypc=15.0, y_hypc=8.0, depth_hypc=8.0,
    xref_hypc=0.0, yref_hypc=0.0,
    magnitude=6.0, fc_main_1=0.09, fc_main_2=3.0,
    rv_avg=3.0,
    ratio_rise=0.3,
    strike=358.0, dip=40.0, rake=113.0,
    pdip_max=15.0, prake_max=30.0,
    nsubx=16, nsuby=8,
    nb_taper_trbl=[5, 5, 5, 5],
    seeds=[52, 448, 4446],
    id_ran1=1, id_ran2=1,
    angle_north_to_x=0.0,
    is_moment=3,
    crust_model=crust,
    output_name="FFSP_OUTPUT",
    verbose=True,
)

# Run the Fortran kernel (slow), then write both formats.
source.run()
source.write_hdf5("ffsp.h5")
source.write_ffsp_format("FFSP_OUTPUT")

# Load back via classmethod and verify the best realization matches.
reloaded = FFSPSource.from_hdf5("ffsp.h5")

a = source.best_realization
b = reloaded.best_realization
assert a["npts"] == b["npts"]
assert np.allclose(a["slip"], b["slip"])
assert np.allclose(a["rupture_time"], b["rupture_time"])

print("PASS")
