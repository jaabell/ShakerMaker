import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shakermaker.sl_extensions import DRMBox
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid
from shakermaker.tools.plotting import StationPlot

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images")

grid = SurfaceGrid([4.0, 4.0, 0.0], [20, 20, 1], [0.1, 0.1, 0.1],
                   mode="plane", plane_z=0.0, metadata={"name": "surface_grid"})
fh = StationPlot(grid, show=False)
fh.savefig(os.path.join(IMG, "geom_surface_grid.png"), dpi=150, bbox_inches="tight")
plt.close("all")

drm = DRMBox([6.0, 8.0, 0.0], [10, 10, 4], [0.05, 0.05, 0.05],
             metadata={"name": "drm_box"})
fh = StationPlot(drm, show=False)
fh.savefig(os.path.join(IMG, "geom_drmbox.png"), dpi=150, bbox_inches="tight")
plt.close("all")

hollow = SurfaceGrid([6.0, 8.0, 0.0], [11, 11, 4], [0.05, 0.05, 0.05],
                     mode="hollow", metadata={"name": "hollow_box"})
fh = StationPlot(hollow, show=False)
fh.savefig(os.path.join(IMG, "geom_hollow_box.png"), dpi=150, bbox_inches="tight")
plt.close("all")

print("station geometry OK")
