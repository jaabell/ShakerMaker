# 01 - CRUST1.0 tool: crustal profile at given lat/lon and a CrustModel snippet.
# Data: Laske et al., CRUST1.0 (attribution printed on import).
# 2026-06-06

from shakermaker.crust1 import Crust1   # CRUST1_CITATION prints on import

# sites: (lat, lon) in degrees
sites = [(-33.42, -70.61), (37.77, -122.51)]

crust1 = Crust1()

# per-site response: profile summary + ready-to-paste CrustModel snippet
for lat, lon in sites:
    p = crust1.profile_at(lat, lon)
    print(f"\nsite ({lat}, {lon})  avg_Vs={p['avg_vs']:.3f} km/s  moho={p['moho_depth_km']:.1f} km")
    crust1.print_shakermaker((lat, lon))

print("PASS")


# cell index (row, col) of the 1deg cell containing the point
j, i = crust1.cell_index(lat, lon)

# midpoint (lat, lon) of that cell
mlat, mlon = crust1.cell_midpoint(lat, lon)

# full 9-layer profile at the point (returns a dict)
p = crust1.profile_at(lat, lon)

# geological type code of the cell at the point
t = crust1.type_at(lat, lon)

# print a full layer table for the site
crust1.print_tables((lat, lon))

# print a ready-to-paste ShakerMaker CrustModel snippet
crust1.print_shakermaker((lat, lon))

# Vp, Vs, rho vs depth for the site
crust1.plot_profile((lat, lon))

# global topography with the site marked
crust1.plot_global_topo((lat, lon))

# zoomed topography with 1deg grid and cell outline
crust1.plot_regional_topo((lat, lon))

# regional map of CRUST 1.0 geological type groups
crust1.plot_regional_geological((lat, lon))

# global Vp, Vs, rho maps for one layer (default layer=5)
crust1.plot_global_velocity((lat, lon))

# layer-thickness stratigraphic columns, one per site
crust1.plot_stacked_columns((lat, lon))