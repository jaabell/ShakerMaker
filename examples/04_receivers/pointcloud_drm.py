# 04 - Build a PointCloudDRMReceiver from a tiny FEM node TSV.
# 2026-06-06

import os
from shakermaker.sl_extensions import PointCloudDRMReceiver

# TSV columns required by PointCloudDRMReceiver: Node_ID  X  Y  Z  Type (tab-sep).
# X/Y/Z in FEM units (mm here); Type is 'internal' or 'external'.
here = os.path.dirname(os.path.abspath(__file__))
nodes_file = os.path.join(here, "_drm_nodes.txt")

# Transform args follow the user's main_HalfSpace block:
#   crd_scale 1/1e6 (mm -> km), x0_fem at the STKO box origin, drmbox_x0 in km.
stations = PointCloudDRMReceiver(
    point_cloud_file=nodes_file,
    crd_scale=1 / 1e6,
    x0_fem=[22000., 15500., 0.],
    drmbox_x0=[6.0, 8.0, 0.0],
    metadata={"name": "PointCloud_DRM"})

# 6 nodes from the file + 1 QA station appended by the class.
assert stations.nstations > 0
assert stations.nstations == 7

internal = sum(1 for s in stations if s.is_internal)
print("PointCloud stations:", stations.nstations, "| internal:", internal)
print("drmbox_x0:", stations.metadata["drmbox_x0"])
print("PASS")
