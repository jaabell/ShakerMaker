import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.crustmodel import CrustModel
from shakermaker.cm_library.LOH import SCEC_LOH_1

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")

crust = SCEC_LOH_1()
crust.plot_profile()
plt.gcf().savefig(os.path.join(IMG, "crust_loh1.png"), dpi=150, bbox_inches="tight")
plt.close("all")

basin = CrustModel(3)
basin.add_layer(0.5, 1.0, 0.5, 1.8, 80., 50.)
basin.add_layer(4.0, 4.0, 2.0, 2.4, 300., 200.)
basin.add_layer(0.0, 6.0, 3.5, 2.7, 1500., 1000.)
basin.plot_profile()
plt.gcf().savefig(os.path.join(IMG, "crust_basin.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("crust profiles OK")
