"""Explanatory figures for docs/web/background/conventions.md and guides/drm.md.

These are schematics: every statement they draw is checked against SW4 and
the SCEC LOH.1 analytical solution (see the verification figures in the same
folder). Run from this folder:

    python make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# one fixed colour per physical axis, the same in every figure
C = {"N": "#1f5fa8", "E": "#d9771e", "Down": "#2e7d32", "Up": "#8bc34a", "other": "#7a7a7a"}
# oblique screen projection: East to the right, North into the page, Up upwards
SCREEN = {"E": (1.0, 0.0), "N": (0.55, 0.42), "Up": (0.0, 1.0), "Down": (0.0, -1.0)}


def arrow(ax, p0, d, color, lw=2.6):
    ax.add_patch(FancyArrowPatch(p0, (p0[0] + d[0], p0[1] + d[1]), arrowstyle="-|>",
                                 mutation_scale=18, color=color, lw=lw))


def box(ax, x, y, w, text, phys, h=0.6, fs=10):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                fc=C.get(phys, C["other"]), ec="k", lw=0.6))
    ax.text(x + w / 2, y, text, ha="center", va="center", color="white", fontsize=fs, fontweight="bold")


def triad(ax, o, axes, title, note):
    for lab, phys in axes:
        d = SCREEN[phys]
        arrow(ax, o, (0.9 * d[0], 0.9 * d[1]), C[phys])
        ax.text(o[0] + 1.05 * d[0], o[1] + 1.05 * d[1] + (0.06 if d[1] >= 0 else -0.14), lab,
                color=C[phys], fontsize=11, fontweight="bold", ha="center")
    ax.plot(*o, "ko", ms=5)
    ax.text(o[0], o[1] + 1.55, title, ha="center", fontsize=12, fontweight="bold")
    ax.text(o[0], o[1] - 1.45, note, ha="center", fontsize=9, va="top")


def fig_frames():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    triad(ax, (0.6, 0), [("x = North", "N"), ("y = East", "E"), ("z = down (+)", "Down")],
          "ShakerMaker", "positions in km\norigin chosen by the user\n(e.g. the epicentre)\nx × y = z: right-handed (NED)")
    triad(ax, (4.6, 0), [("x = North", "N"), ("y = East", "E"), ("z = down (+)", "Down")],
          "SW4 (az = 0, the default)", "positions in m\norigin at the corner (0, 0, 0) of the domain\nsame frame as ShakerMaker:\nthe exporter only scales and translates")
    triad(ax, (8.6, 0), [("Y = North", "N"), ("X = East", "E"), ("Z = up (+)", "Up")],
          "OpenSees model (ENU)", "the frame the FE model is built in\nX × Y = Z: right-handed (ENU)\nreached with the matrix T")
    ax.set_xlim(-0.8, 10.2); ax.set_ylim(-2.4, 1.9); ax.axis("off")
    fig.suptitle("The three axis systems (oblique view: East to the right, North into the page)", fontsize=12)
    fig.tight_layout(); fig.savefig("frames.png", dpi=140); plt.close(fig)


def fig_component_order():
    rows = [
        ("ShakerMaker input", [("x = North", "N"), ("y = East", "E"), ("z = depth, +down", "Down")],
         "Station, PointSource (km)", "POSITION"),
        ("ShakerMaker positions in files", [("North", "N"), ("East", "E"), ("depth, +down", "Down")],
         ".h5 Data/xyz, .h5drm DRM_Data/xyz, QA, drmbox_x0", "POSITION"),
        ("ShakerMaker .h5 / .h5drm (nodes and QA)", [("u East", "E"), ("u North", "N"), ("u down", "Down")],
         "rows 3i, 3i+1, 3i+2: velocity, displacement, acceleration", "MOTION"),
        ("ShakerMaker get_response() / .npz", [("u down", "Down"), ("u East", "E"), ("u North", "N")],
         "z, e, n, t = sta.get_response()", "MOTION"),
        ("ShakerMaker Green's functions (save_gf)", [("g down", "Down"), ("g East", "E"), ("g North", "N")],
         "(z, e, n, t, tdata, t0): as get_response, no STF", "MOTION"),
        ("SW4 input", [("x = North", "N"), ("y = East", "E"), ("z = depth, +down", "Down")],
         "grid, source, rec (m, origin at the corner)", "POSITION"),
        ("SW4 rec, nsew = 0 (exporter)", [("u North", "N"), ("u East", "E"), ("u down", "Down")],
         "columns X, Y, Z of the .txt", "MOTION"),
        ("SW4 rec, nsew = 1", [("u East", "E"), ("u North", "N"), ("u up", "Up")],
         "columns EW, NS, UD (SW4 flips z)", "MOTION"),
    ]
    fig, ax = plt.subplots(figsize=(13, 8.6))
    y = 0
    for lab, comps, sub, kind in rows:
        ax.text(-0.15, y, lab, ha="right", va="center", fontsize=10.5, fontweight="bold")
        ax.text(-0.15, y - 0.32, sub, ha="right", va="center", fontsize=8, color="#444")
        for k, (txt, phys) in enumerate(comps):
            box(ax, 0.1 + 1.55 * k, y, 1.35, txt, phys)
        ax.text(4.9, y, kind, va="center", fontsize=9, color="#555", fontweight="bold")
        y -= 1.0
    for k, t in enumerate(["1st component", "2nd component", "3rd component"]):
        ax.text(0.1 + 1.55 * k + 0.675, 0.65, t, ha="center", fontsize=9)
    ax.set_xlim(-5.2, 5.8); ax.set_ylim(y + 0.4, 1.0); ax.axis("off")
    fig.suptitle("Which component sits where, by format (colour = physical axis; all positive towards\n"
                 "North, East and down, except SW4 nsew = 1, which uses up)", fontsize=12)
    fig.tight_layout(); fig.savefig("component_order.png", dpi=140); plt.close(fig)


def fig_greens():
    fig, ax = plt.subplots(figsize=(14, 3.8))
    steps = [
        ("FK core (subfk)\n9 fundamental Green's\nfunctions: tdata",
         "internal cylindrical frame\n(vertical, radial, transverse\nfor DD, DS, SS); independent\nof mechanism and azimuth"),
        ("subfocal / subgreen2\ncombine tdata with\nstrike, dip, rake, azimuth",
         "vertical, radial, transverse\n→ down, East, North\n(azimuth from North\ntowards East)"),
        ("Station Green's\nfunction\n(z, e, n)",
         "save_gf = True stores\n(z, e, n, t, tdata, t0)\n= (down, East, North)"),
        ("Convolution with the STF\nand sum over sources\nget_response()",
         "(z, e, n) = (down, East, North)\n.npz the same"),
        ("Writers\n.h5 and .h5drm",
         "rows (e, n, z)\n= (East, North, down)"),
    ]
    for k, (t, sub) in enumerate(steps):
        x = k * 2.85
        ax.add_patch(FancyBboxPatch((x, 0.6), 2.4, 1.35, boxstyle="round,pad=0.03", fc="#e8eef7", ec="#1f5fa8"))
        ax.text(x + 1.2, 1.28, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + 1.2, 0.35, sub, ha="center", va="top", fontsize=8.3)
        if k < len(steps) - 1:
            arrow(ax, (x + 2.45, 1.28), (0.35, 0), "k", lw=1.4)
    ax.text(0, -1.25, "OP pipeline database (_gf.h5): stores /tdata (n_slots, nt, 9) per distance/depth slot; "
                      "Stage 2 recombines it with subgreen2 for every real pair and follows the same chain.", fontsize=9)
    ax.set_xlim(-0.2, 14.2); ax.set_ylim(-1.45, 2.1); ax.axis("off")
    fig.suptitle("Green's functions: from the 9 core components to what the writers store", fontsize=12)
    fig.tight_layout(); fig.savefig("greens_functions.png", dpi=140); plt.close(fig)


def fig_T():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    left = [("North", "N"), ("East", "E"), ("depth (+down)", "Down")]
    right = [("model X = East", "E"), ("model Y = North", "N"), ("model Z = up", "Up")]
    for (t, p), y in zip(left, [2, 1, 0]):
        box(ax, 0, y, 2.6, t, p)
    for (t, p), y in zip(right, [2, 1, 0]):
        box(ax, 6.4, y, 2.9, t, p)
    for (y0, y1, lab, dx, dy) in [(2, 1, "+1 (North → Y)", 3.4, 2.05), (1, 2, "+1 (East → X)", 3.4, 0.82),
                                  (0, 0, "−1 (sign flip)", 4.5, 0.15)]:
        arrow(ax, (2.65, y0), (3.7, y1 - y0), "k", lw=1.6)
        ax.text(dx, dy, lab, ha="center", fontsize=9.5)
    ax.text(1.3, 2.75, "Positions in the file\n(xyz, in order)", ha="center", fontweight="bold")
    ax.text(7.85, 2.75, "Model axes\n(what T produces)", ha="center", fontweight="bold")
    mat = ("T (rows = model axes,\ncolumns = file axes)\n\n"
           "        N    E   depth\n"
           " X  [   0    1    0  ]\n"
           " Y  [   1    0    0  ]\n"
           " Z  [   0    0   −1  ]\n\n"
           "STKO: Local X = (0, 1, 0)\n      Local Y = (1, 0, 0)\n      Local Z = X × Y = (0, 0, −1)")
    ax.text(10.0, 1.0, mat, family="monospace", fontsize=9.5, va="center",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999"))
    ax.text(4.6, -1.15, "Example: a node 50 m North, 50 m East and 50 m deep from the box centre\n"
                        "lands in the model at  X = 50 m (East),  Y = 50 m (North),  Z = −50 m (below the surface)",
            ha="center", fontsize=10)
    ax.set_xlim(-0.4, 13.6); ax.set_ylim(-1.7, 3.3); ax.axis("off")
    fig.suptitle("The matrix T acts on the POSITIONS: it swaps North and East and flips the depth", fontsize=12)
    fig.tight_layout(); fig.savefig("t_matrix_positions.png", dpi=140); plt.close(fig)


def fig_motion():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    rows = [("u East", "E"), ("u North", "N"), ("u down", "Down")]
    applied = [("u East", "E"), ("u North", "N"), ("u up", "Up")]
    model = [("DOF X = East", "E"), ("DOF Y = North", "N"), ("DOF Z = up", "Up")]
    for k in range(3):
        y = 2 - k
        for x0, (t, p) in ((0, rows[k]), (3.2, applied[k]), (6.4, model[k])):
            box(ax, x0, y, 2.5, t, p, fs=9.5)
        arrow(ax, (2.55, y), (0.6, 0), "k", lw=1.4)
        arrow(ax, (5.75, y), (0.6, 0), "k", lw=1.4)
        ax.text(9.05, y, "✓", fontsize=16, color="green", va="center")
    ax.text(1.25, 2.75, "row in the .h5drm", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(4.45, 2.75, "what is applied\n(H5DRM flips row 2 on read)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(7.65, 2.75, "model axis (STKO T)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(4.6, -0.95, "The MOTION (displacement and acceleration, which H5DRM reads) is not rotated by T:\n"
                        "row 0 → DOF X, row 1 → DOF Y, row 2 → DOF Z. The file therefore already carries (East, North),\n"
                        "and the vertical is flipped once, by H5DRM itself.", ha="center", fontsize=9.5)
    ax.set_xlim(-0.2, 9.5); ax.set_ylim(-1.6, 3.3); ax.axis("off")
    fig.suptitle("The motion enters unrotated: how each row reaches a Z-up model", fontsize=12)
    fig.tight_layout(); fig.savefig("motion_in_opensees.png", dpi=140); plt.close(fig)


def fig_mistakes():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    cases = [
        ("T = identity (Local X = (1,0,0), Local Y = (0,1,0))\nmodel X = North, Y = East, Z = down",
         [("DOF X = North", "N", "receives u East", "E"), ("DOF Y = East", "E", "receives u North", "N"),
          ("DOF Z = down", "Down", "receives u up", "Up")],
         "North/East swapped, vertical flipped"),
        ("STKO T, but the file's vertical flipped by hand\nmodel X = East, Y = North, Z = up",
         [("DOF X = East", "E", "receives u East", "E"), ("DOF Y = North", "N", "receives u North", "N"),
          ("DOF Z = up", "Up", "receives u down", "Down")],
         "vertical flipped twice"),
    ]
    for i, (title, rows, verdict) in enumerate(cases):
        x0 = i * 6.6
        ax.text(x0 + 2.9, 2.85, title, ha="center", fontsize=9.5, fontweight="bold")
        for k, (m, mp, r, rp) in enumerate(rows):
            y = 2 - k
            box(ax, x0, y, 2.7, m, mp, fs=9.5)
            box(ax, x0 + 3.0, y, 2.7, r, rp, fs=9.5)
            ok = mp == rp
            ax.text(x0 + 5.9, y, "✓" if ok else "✗", fontsize=16, color="green" if ok else "red", va="center")
        ax.text(x0 + 2.9, -0.75, verdict, ha="center", color="red", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 13.3); ax.set_ylim(-1.2, 3.3); ax.axis("off")
    fig.suptitle("Two common mistakes", fontsize=12)
    fig.tight_layout(); fig.savefig("common_mistakes.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    fig_frames(); fig_component_order(); fig_greens(); fig_T(); fig_motion(); fig_mistakes()
    print("ok")
