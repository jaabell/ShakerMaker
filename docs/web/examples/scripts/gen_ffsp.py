import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.crustmodel import CrustModel
from shakermaker.ffspsource import FFSPSource

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "..", "assets", "images")
SCRATCH = os.path.join(HERE, "_ffsp_scratch")
os.makedirs(SCRATCH, exist_ok=True)
os.chdir(SCRATCH)

crust = CrustModel(4)
crust.add_layer(0.200, 1.32, 0.75, 2.40, 1000., 1000.)
crust.add_layer(0.800, 2.75, 1.57, 2.50, 1000., 1000.)
crust.add_layer(14.50, 5.50, 3.14, 2.50, 1000., 1000.)
crust.add_layer(0.000, 7.00, 4.00, 2.67, 1000., 1000.)

source = FFSPSource(
    id_sf_type=8, freq_min=0.01, freq_max=24.0,
    fault_length=30.0, fault_width=16.0,
    x_hypc=15.0, y_hypc=8.0, depth_hypc=8.0,
    xref_hypc=0.0, yref_hypc=0.0,
    magnitude=6.5, fc_main_1=0.09, fc_main_2=3.0,
    rv_avg=3.0, ratio_rise=0.3,
    strike=358.0, dip=40.0, rake=113.0,
    pdip_max=15.0, prake_max=30.0,
    nsubx=256, nsuby=128,
    nb_taper_trbl=[5, 5, 5, 5],
    seeds=[52, 448, 4446],
    id_ran1=1, id_ran2=1,
    angle_north_to_x=0.0, is_moment=3,
    crust_model=crust,
    output_name="FFSP_SMALL",
    verbose=False,
)
source.run()

source.plot_spacial_distribution(field="slip")
plt.gcf().savefig(os.path.join(IMG, "ffsp_slip.png"), dpi=150, bbox_inches="tight")
plt.close("all")

source.plot_spacial_distribution(field="rise_time")
plt.gcf().savefig(os.path.join(IMG, "ffsp_rise_time.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("ffsp OK")
