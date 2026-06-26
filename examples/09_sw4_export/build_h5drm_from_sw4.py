# 09 - Build an .h5drm from this SW4 case (needs SW4 result .txt files).
# 2026-06-06

import os

try:
    import h5py
except Exception:
    print("SKIP: h5py not available")
    raise SystemExit(0)

import sys

# Reuse the reference builder shipped alongside this example.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from build_h5drm_from_sw4_case import build_h5drm_from_sw4_case

# This SW4 case is produced by export_sw4.py in _sw4_out.
case_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sw4_out")
package = os.path.join(case_path, "shakermakerexports", "sw4_package.h5")

# Need the compact package first (run export_sw4.py).
if not os.path.exists(package):
    print("SKIP: SW4 package not found - run export_sw4.py first")
    raise SystemExit(0)

# Need SW4 result station .txt files inside sw4/<fileio_path>/.
# These only exist after running SW4 on the case; SKIP if absent.
sw4_dir = os.path.join(case_path, "sw4")
fileio_dir = os.path.join(sw4_dir, "shakermaker2sw4_fileio")
has_txt = os.path.isdir(fileio_dir) and any(
    f.endswith(".txt") for f in os.listdir(fileio_dir)
)
if not has_txt:
    print("SKIP: SW4 result .txt files not found - run SW4 on the case first")
    raise SystemExit(0)

output_h5drm = build_h5drm_from_sw4_case(
    case_path=case_path,
    package_h5=package,
    output_name="motions.h5drm",
)

motions = os.path.join(case_path, "shakermakerexports", "motions.h5drm")
assert os.path.exists(motions), f"missing h5drm: {motions}"
print("PASS")
