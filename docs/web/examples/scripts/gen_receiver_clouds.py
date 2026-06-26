"""Receiver point-cloud gallery: every geometry, internal=red / external=blue.

Denser than the smoke-test examples (this is only for the figure, so the point
clouds read clearly). One panel per geometry, a single shared legend.
"""
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shakermaker.sl_extensions import DRMBox, PointCloudDRMReceiver
from shakermaker.sl_extensions.SurfaceGrid import SurfaceGrid

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "assets", "images")

RED, BLUE = "#dc2626", "#2563eb"   # internal, external


def scatter(ax, stations, title):
    """Scatter a StationList, internal=red / external=blue (QA point skipped)."""
    pin, pex = [], []
    for s in stations:
        if s.metadata.get("name") == "QA":
            continue
        (pin if s.is_internal else pex).append(s.x)
    pin, pex = np.array(pin), np.array(pex)
    # plot in ShakerMaker convention: y=Easting, x=Northing, depth up
    if len(pex):
        ax.scatter(pex[:, 1], pex[:, 0], -pex[:, 2], c=BLUE, s=6, depthshade=False)
    if len(pin):
        ax.scatter(pin[:, 1], pin[:, 0], -pin[:, 2], c=RED, s=6, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Y E (km)", fontsize=7)
    ax.set_ylabel("X N (km)", fontsize=7)
    ax.set_zlabel("Z (km)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.invert_zaxis()
    n_in = 0 if not len(pin) else len(pin)
    n_ex = 0 if not len(pex) else len(pex)
    ax.text2D(0.5, -0.04, f"internal {n_in} · external {n_ex}",
              transform=ax.transAxes, ha="center", fontsize=7, color="#52525b")


def dense_pointcloud_file(path, n=6, h_mm=300.0, x0=(22000., 15500., 0.)):
    """A filled FEM grid: boundary nodes = external, interior nodes = internal."""
    with open(path, "w") as f:
        f.write("Node_ID\tX\tY\tZ\tType\n")
        nid = 0
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    nid += 1
                    edge = i in (0, n) or j in (0, n) or k in (0, n)
                    typ = "external" if edge else "internal"
                    X = x0[0] + i * h_mm
                    Y = x0[1] + j * h_mm
                    Z = x0[2] - k * h_mm          # FEM Z-up, going down
                    f.write(f"{nid}\t{X}\t{Y}\t{Z}\t{typ}\n")


pos = [6.0, 8.0, 0.0]

# --- the geometries, all bumped up in density ---
plane = SurfaceGrid(pos, [40, 40, 1], [0.1, 0.1, 0.1],
                    mode="plane", plane_z=0.0, metadata={"name": "plane"})
# vertical XZ sheet (fixed Easting): a depth cross-section, not just z=0
plane_v = SurfaceGrid(pos, [40, 1, 20], [0.1, 0.1, 0.1],
                      mode="plane", plane_y=8.0, metadata={"name": "plane_v"})
hollow = SurfaceGrid(pos, [16, 16, 6], [0.05, 0.05, 0.05],
                     mode="hollow", metadata={"name": "hollow"})
filled = SurfaceGrid(pos, [10, 10, 8], [0.05, 0.05, 0.05],
                     mode="filled", metadata={"name": "filled"})
drm = DRMBox(pos, [14, 14, 6], [0.05, 0.05, 0.05], metadata={"name": "drm"})

tmp = os.path.join(tempfile.gettempdir(), "_dense_cloud.txt")
dense_pointcloud_file(tmp)
cloud = PointCloudDRMReceiver(point_cloud_file=tmp, crd_scale=1 / 1e6,
                              x0_fem=[22000., 15500., 0.], drmbox_x0=pos,
                              metadata={"name": "cloud"})

panels = [
    (plane,   "SurfaceGrid · plane (z=0)"),
    (plane_v, "SurfaceGrid · plane (vertical XZ)"),
    (hollow,  "SurfaceGrid · hollow"),
    (filled,  "SurfaceGrid · filled"),
    (drm,     "DRMBox"),
    (cloud,   "PointCloudDRMReceiver"),
]

fig = plt.figure(figsize=(15, 9))
for idx, (st, title) in enumerate(panels, start=1):
    ax = fig.add_subplot(2, 3, idx, projection="3d")
    scatter(ax, st, title)

# figure-level legend (no empty cell left now that all six are plots)
red_proxy = plt.Line2D([], [], marker="o", linestyle="", color=RED, label="internal")
blue_proxy = plt.Line2D([], [], marker="o", linestyle="", color=BLUE, label="external")
fig.legend(handles=[red_proxy, blue_proxy], loc="lower center", ncol=2,
           frameon=False, fontsize=12, markerscale=1.4,
           title="DRM node type", title_fontsize=11)

fig.suptitle("Receiver geometries — internal (red) vs external (blue)",
             fontsize=13)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(os.path.join(IMG, "receiver_clouds.png"), dpi=150,
            bbox_inches="tight")
plt.close("all")
print("receiver clouds OK")
