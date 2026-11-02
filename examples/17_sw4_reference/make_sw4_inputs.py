# 17 - SW4 reference cases: writes the six SW4 decks and the receiver and fault tables.
#
# Two media (m1 = SCEC LOH.1, m2 = layered 1D crust) x three sources (p1 deep point, p2 shallow point,
# f1 = 16 x 16 = 256 subfaults). The same receivers for the three sources of a medium: a ring at the
# surface (r = 1, 4, 8 km every 30 deg), three verticals, the epicentre and 24 nodes of a DRM box.
# Every receiver is moved to the nearest SW4 grid node (SW4 records at the nearest node) and the
# tables keep the moved coordinates, so ShakerMaker computes exactly the points SW4 records.
#
#   python make_sw4_inputs.py --write     # writes sw4/<case>.in and data/{receivers_<medium>,fault_<medium>}.csv
#
# Without --write it only prints SKIP, so the smoke suite never overwrites the reference inputs.
# Convention of both codes: x = north, y = east, z = down (SW4 grid az=0); moment in N m here,
# 1 ShakerMaker moment unit = 1e15 N m.

import csv
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

# (thickness km, vp, vs, rho) top to bottom; last row is the half-space
media = {
    "loh1": dict(layers=[(1.0, 4.0, 2.0, 2.6), (0.0, 6.0, 3.464, 2.7)],
                 zmax=12.0, tmax=12.0, sigma=0.06, zsrc={"p1": 2.0, "p2": 0.5},
                 refine=[1.0], near_iface=[0.95, 1.1]),
    "layered": dict(layers=[(0.2, 1.32, 0.75, 2.40), (0.8, 2.75, 1.57, 2.50), (14.5, 5.50, 3.14, 2.50), (0.0, 7.00, 4.00, 2.67)],
                zmax=20.0, tmax=15.0, sigma=0.1, zsrc={"p1": 6.0, "p2": 0.5},
                refine=[1.0, 0.3], near_iface=[0.175, 0.95, 1.1]),
}
h0 = 0.1                       # coarse grid spacing (km); each refinement halves it above its zmax
lx = ly = 24.0                 # horizontal size of the SW4 volume (km); the epicentre is at its centre
strike, dip, rake = 30.0, 60.0, 45.0
m0_point = 1.0e16              # N m, point sources
m0_fault = 3.5e16              # N m, Mw ~5.0, split evenly over the 256 subfaults
nsub, flen, fwid, ftop = 16, 4.0, 4.0, 0.3


def grid_h(medium, z):
    """Grid spacing (km) of SW4 at depth z: h0 halved once per refinement whose zmax is >= z."""
    return h0 / 2 ** sum(z <= zr + 1e-9 for zr in media[medium]["refine"])


def snap(v, h):
    return round(v / h) * h


def moment_tensor(m0):
    """Aki & Richards Box 4.4, x north, y east, z down (the formulas of subfocal.f)."""
    f, d, l = np.radians([strike, dip, rake])
    mxx = -m0 * (np.sin(d) * np.cos(l) * np.sin(2 * f) + np.sin(2 * d) * np.sin(l) * np.sin(f) ** 2)
    mxy = m0 * (np.sin(d) * np.cos(l) * np.cos(2 * f) + 0.5 * np.sin(2 * d) * np.sin(l) * np.sin(2 * f))
    mxz = -m0 * (np.cos(d) * np.cos(l) * np.cos(f) + np.cos(2 * d) * np.sin(l) * np.sin(f))
    myy = m0 * (np.sin(d) * np.cos(l) * np.sin(2 * f) - np.sin(2 * d) * np.sin(l) * np.cos(f) ** 2)
    myz = -m0 * (np.cos(d) * np.cos(l) * np.sin(f) - np.cos(2 * d) * np.sin(l) * np.cos(f))
    mzz = m0 * np.sin(2 * d) * np.sin(l)
    return dict(mxx=mxx, myy=myy, mzz=mzz, mxy=mxy, mxz=mxz, myz=myz)


def receivers(medium):
    """(name, north_km, east_km, depth_km, group), every point on an SW4 grid node."""
    m = media[medium]
    rows = []
    hs = grid_h(medium, 0.0)
    for r in (1, 4, 8):
        for az in range(0, 360, 30):
            a = np.radians(az)
            rows.append((f"ring_r{r}_a{az:03d}", snap(r * np.cos(a), hs), snap(r * np.sin(a), hs), 0.0, "ring"))
    # verticals on 100 m nodes (common to every grid level)
    depths = sorted(set([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0] + m["near_iface"]))
    for r, az in ((1, 0), (4, 120), (8, 240)):
        a = np.radians(az)
        n, e = snap(r * np.cos(a), h0), snap(r * np.sin(a), h0)
        for z in depths:
            rows.append((f"prof_r{r}_z{round(z * 1000):05d}", n, e, z, "prof"))
    # epicentre: surface and 0.1 km above and below each point source
    epi = sorted(set([0.0] + [zs + dz for zs in m["zsrc"].values() for dz in (-0.1, 0.1)]))
    for z in epi:
        rows.append((f"epi_z{round(z * 1000):05d}", 0.0, 0.0, z, "epi"))
    # DRM box 1 x 1 x 0.5 km, centre 4 km at 45 deg moved to a 100 m node; 50 m elements
    cn, ce = snap(4 * np.cos(np.radians(45)), h0), snap(4 * np.sin(np.radians(45)), h0)
    for i, (dn, de) in enumerate(((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))):
        rows.append((f"drm_corner{i}", cn + dn, ce + de, 0.0, "drm"))
    for name, dn, de, on, oe in (("n", 0.5, 0, 0.05, 0), ("s", -0.5, 0, -0.05, 0), ("e", 0, 0.5, 0, 0.05), ("w", 0, -0.5, 0, -0.05)):
        rows.append((f"drm_face_{name}_in", cn + dn, ce + de, 0.25, "drm"))
        rows.append((f"drm_face_{name}_out", cn + dn + on, ce + de + oe, 0.25, "drm"))
    for k in range(12):
        z = 0.05 * k
        rows.append((f"drm_col_z{round(z * 1000):04d}", cn, ce, z, "drm"))
    # check: every point on a node of its grid level
    for name, n, e, z, _ in rows:
        h = grid_h(medium, z)
        for v in (n, e, z):
            assert abs(v / h - round(v / h)) < 1e-6 or z <= 1e-9 and v == z, (name, v, h)
    return [(nm, round(n, 6), round(e, 6), round(z, 6), g) for nm, n, e, z, g in rows]


def fault(medium):
    """16 x 16 subfaults of the f1 fault: (n, e, z, m0, tr) in km, N m, s; hypocentre at the centre of the base."""
    m = media[medium]
    us = np.array([np.cos(np.radians(strike)), np.sin(np.radians(strike)), 0.0])               # along strike (n, e, z)
    ud = np.array([-np.sin(np.radians(strike)) * np.cos(np.radians(dip)),
                   np.cos(np.radians(strike)) * np.cos(np.radians(dip)), np.sin(np.radians(dip))])  # down dip
    zbot = ftop + fwid * np.sin(np.radians(dip))
    hypo = np.array([0.0, 0.0, zbot])
    tops = np.concatenate([[0.0], np.cumsum([l[0] for l in m["layers"][:-1]])])
    vs_hypo = m["layers"][int(np.searchsorted(tops, zbot, side="right")) - 1][2]
    vr = 0.8 * vs_hypo
    out = []
    for i in range(nsub):
        for j in range(nsub):
            a = (i + 0.5) / nsub * flen - flen / 2
            b = (j + 0.5) / nsub * fwid - fwid          # up dip from the base: negative down-dip coordinate
            p = hypo + a * us + b * ud
            assert min(abs(p[2] - t) for t in tops[1:]) > 0.010, ("subfault within 10 m of an interface", p)
            out.append((p[0], p[1], p[2], m0_fault / nsub ** 2, np.hypot(a, b) / vr))
    return out, vr


def deck(medium, src):
    m = media[medium]
    zs = m["zsrc"]
    lines = [f"# 17_sw4_reference: {medium} {src}. Elastic (no attenuation), x north, y east, z down.",
             "fileio path=out verbose=1",
             f"grid h={h0 * 1000:g} x={lx * 1000:g} y={ly * 1000:g} z={m['zmax'] * 1000:g} extrapolate=1",
             f"time t={m['tmax']:g}"]
    lines += [f"refinement zmax={zr * 1000:g}" for zr in m["refine"]]
    layers = m["layers"]
    tops = np.concatenate([[0.0], np.cumsum([l[0] for l in layers[:-1]])])
    # half-space first, then each layer from the deepest up (a later block overrides an earlier one)
    hs = layers[-1]
    lines.append(f"block vp={hs[1] * 1000:g} vs={hs[2] * 1000:g} rho={hs[3] * 1000:g}")
    for k in range(len(layers) - 2, -1, -1):
        l = layers[k]
        lines.append(f"block vp={l[1] * 1000:g} vs={l[2] * 1000:g} rho={l[3] * 1000:g} z2={tops[k + 1] * 1000:g}")
    sig = m["sigma"]
    x0, y0 = lx / 2, ly / 2
    if src in ("p1", "p2"):
        sources = [(0.0, 0.0, zs[src], m0_point, 0.0)]
    else:
        sources, _ = fault(medium)
    for n, e, z, m0, tr in sources:
        mt = moment_tensor(m0)
        lines.append(f"source x={(x0 + n) * 1000:.3f} y={(y0 + e) * 1000:.3f} z={z * 1000:.3f} "
                     + " ".join(f"{k}={v:.6e}" for k, v in mt.items())
                     + f" t0={6 * sig + tr:.6f} freq={1 / sig:.6f} type=Gaussian")
    for name, n, e, z, _ in receivers(medium):
        lines.append(f"rec x={(x0 + n) * 1000:g} y={(y0 + e) * 1000:g} z={z * 1000:g} file={name} "
                     "usgsformat=1 sacformat=0 variables=velocity nsew=0")
    longest = max(len(l) for l in lines)
    assert longest < 250, f"deck line of {longest} characters (SW4 buffer is 256)"
    return "\n".join(lines) + "\n"


def main():
    if "--write" not in sys.argv:
        print("SKIP: writes the SW4 reference inputs only with --write")
        return
    os.makedirs(os.path.join(here, "sw4"), exist_ok=True)
    os.makedirs(os.path.join(here, "data"), exist_ok=True)
    for medium in media:
        with open(os.path.join(here, "data", f"receivers_{medium}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "north_km", "east_km", "depth_km", "group"])
            w.writerows(receivers(medium))
        rows, vr = fault(medium)
        with open(os.path.join(here, "data", f"fault_{medium}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["north_km", "east_km", "depth_km", "m0_nm", "tr_s", "strike", "dip", "rake"])
            w.writerows([(f"{n:.6f}", f"{e:.6f}", f"{z:.6f}", f"{m0:.6e}", f"{tr:.6f}", strike, dip, rake) for n, e, z, m0, tr in rows])
        for src in ("p1", "p2", "f1"):
            with open(os.path.join(here, "sw4", f"{medium}_{src}.in"), "w") as f:
                f.write(deck(medium, src))
        print(f"{medium}: {len(receivers(medium))} receivers, fault vr = {vr:.3f} km/s")
    print("PASS")


if __name__ == "__main__":
    main()
