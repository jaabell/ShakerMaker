import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "gen_quick_example.py",
    "gen_crust_profiles.py",
    "gen_stf_gallery.py",
    "gen_station_geometry.py",
    "gen_receiver_clouds.py",
    "gen_seismogram.py",
    "gen_ffsp.py",
]

for name in SCRIPTS:
    print(f"--- {name} ---")
    runpy.run_path(os.path.join(HERE, name), run_name="__main__")

print("all figures generated")
