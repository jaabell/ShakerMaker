# 18 - SW4 reference cases: the five ShakerMaker engines against SW4 and against the exact engine.
#
#   python compare.py [sm_dir] [out_dir]      # defaults: data/sm and ../17_sw4_reference/data/sm, _output
#
# SW4 (data/sw4/<case>.npz): velocity, x north, y east, z down. With Gaussian(derivative=False) the ShakerMaker
# output is the time integral of the SW4 velocity for type=Gaussian (checked in example 15), so the SW4 velocity is
# integrated in time (cumulative trapezoid; no baseline correction). The z components share the sign.
# Amplitudes are compared in absolute terms (1 ShakerMaker moment unit = 1e15 N m, no fitted factor); the least-squares
# factor over the p1 case without the epicentral receivers is printed only as a check (it should be ~1).
# Engines are compared with run below 2 Hz (low-pass), the band of the comparison: inter_tensor cuts the core at
# freq_cut = 2.5 Hz, so above it the engines differ by construction.
#
# Writes comparison_sw4.csv (case, engine, band, receiver, component: cc, best-lag cc, lag, peak ratio, nrms),
# comparison_engines.csv (each engine against run, raw traces) and comparison_drm.csv (difference between the two
# layers of the DRM box faces), and prints a summary.

import csv
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

here = os.path.dirname(os.path.abspath(__file__))
# Inputs of 17_sw4_reference (receivers, faults, SW4 solutions and the exact, nearest and tensor results);
# this folder adds the results of run() and of the grid scheme (data/sm).
ref17 = os.path.join(here, "..", "17_sw4_reference", "data")
if not os.path.isdir(ref17):
    print("SKIP: needs ../17_sw4_reference/data")
    sys.exit(0)
sm_dirs = [sys.argv[1]] if len(sys.argv) > 1 else [os.path.join(here, "data", "sm"), os.path.join(ref17, "sm")]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "_output")
os.makedirs(out_dir, exist_ok=True)

cases = ["loh1_p1", "loh1_p2", "loh1_f1", "layered_p1", "layered_p2", "layered_f1"]
engines = ["run", "exact", "nearest", "inter_grid", "inter_tensor"]
bands = {"0.1-2.0": (0.1, 2.0), "0.1-0.5": (0.1, 0.5), "0.5-1.0": (0.5, 1.0), "1.0-2.0": (1.0, 2.0)}
tmax = {"loh1": 12.0, "layered": 15.0}
dt = 0.01
maxlag = 0.1                     # s, search window of the best-lag correlation (SW4 places an interface half a cell deeper)
drm_pairs = [(f"drm_face_{s}_out", f"drm_face_{s}_in") for s in "nsew"] + [("drm_col_z0550", "drm_col_z0500")]


def groups(medium):
    return {r["name"]: r["group"] for r in csv.DictReader(open(os.path.join(ref17, f"receivers_{medium}.csv")))}


def load_sw4(case):
    d = np.load(os.path.join(ref17, "sw4", f"{case}.npz"))
    t, v = d["t"], d["vel"].astype(float)                    # (nrec, 3, nt): north, east, down
    u = np.concatenate([np.zeros(v.shape[:2] + (1,)), np.cumsum(0.5 * (v[..., 1:] + v[..., :-1]) * np.diff(t), axis=-1)], axis=-1)
    tc = np.arange(0.0, tmax[case.split("_")[0]], dt)
    zen = np.stack([u[:, 2], u[:, 1], u[:, 0]], axis=1)       # -> z (down), e, n
    return list(d["names"]), np.array([[np.interp(tc, t, c) for c in r] for r in zen])


def load_sm(case, engine):
    found = [os.path.join(dd, f"{case}_{engine}.npz") for dd in sm_dirs]
    found = [q for q in found if os.path.exists(q)]
    if not found:
        return None, None
    p = found[0]
    d = np.load(p)
    tc = np.arange(0.0, tmax[case.split("_")[0]], dt)
    return list(d["names"]), np.array([[np.interp(tc, d["t"], c, left=0.0, right=c[-1]) for c in r] for r in d["zen"].astype(float)])


def band(x, fmin, fmax):
    return sosfiltfilt(butter(4, [fmin, fmax], btype="band", fs=1 / dt, output="sos"), x, axis=-1)


def lowpass(x, f=2.0):
    return sosfiltfilt(butter(4, f, btype="low", fs=1 / dt, output="sos"), x, axis=-1)


def best_lag_cc(a, b):
    nl = int(maxlag / dt)
    best = (-2.0, 0)
    for k in range(-nl, nl + 1):
        x, y = (a[k:], b[:len(b) - k]) if k >= 0 else (a[:k], b[-k:])
        c = np.corrcoef(x, y)[0, 1]
        if c > best[0]:
            best = (c, k)
    return best[0], best[1] * dt


sw4 = {c: load_sw4(c) for c in cases}
sm = {(c, e): load_sm(c, e) for c in cases for e in engines}

# amplitude factor per medium from p1 / run, broad band
scale = {}
for med in ("loh1", "layered"):
    names, ref = sw4[f"{med}_p1"]
    _, s = sm[(f"{med}_p1", "run")]
    if s is None:
        continue
    keep = [i for i, n in enumerate(names) if not n.startswith("epi")]
    a, b = band(ref[keep], *bands["0.1-2.0"]), band(s[keep], *bands["0.1-2.0"])
    print(f"least-squares amplitude factor ShakerMaker -> SW4, {med} (check, not applied): {np.sum(a * b) / np.sum(b * b):.4f}")
    scale[med] = 1.0

rows_sw4, rows_eng, rows_drm = [], [], []
for c in cases:
    med = c.split("_")[0]
    if med not in scale:
        continue
    grp = groups(med)
    names, ref = sw4[c]
    for e in engines:
        names_sm, s = sm[(c, e)]
        if s is None:
            continue
        assert names_sm == names
        s = scale[med] * s
        for bname, (f1, f2) in bands.items():
            a, b = band(ref, f1, f2), band(s, f1, f2)
            for i, n in enumerate(names):
                energy = np.linalg.norm(a[i], axis=-1)
                for k, comp in enumerate("ZEN"):
                    if energy[k] < 0.05 * energy.max():
                        continue                                  # nodal component, no signal in SW4
                    cc = float(np.corrcoef(a[i, k], b[i, k])[0, 1])
                    ccb, lag = best_lag_cc(a[i, k], b[i, k])
                    rows_sw4.append(dict(case=c, engine=e, band=bname, receiver=n, group=grp[n], comp=comp, cc=cc, cc_best=ccb,
                                         lag_s=lag, peak=float(np.abs(b[i, k]).max() / np.abs(a[i, k]).max()),
                                         nrms=float(np.linalg.norm(b[i, k] - a[i, k]) / np.linalg.norm(a[i, k]))))
            # DRM face layers: difference outer - inner, ShakerMaker against SW4
            for o, inn in drm_pairs:
                io, ii = names.index(o), names.index(inn)
                for k, comp in enumerate("ZEN"):
                    da, db = a[io, k] - a[ii, k], b[io, k] - b[ii, k]
                    rows_drm.append(dict(case=c, engine=e, ref="sw4", band=bname, pair=f"{o}-{inn}", comp=comp,
                                         cc=float(np.corrcoef(da, db)[0, 1]), nrms=float(np.linalg.norm(db - da) / np.linalg.norm(da))))
        # engines against the exact reference: run, or exact where run was not computed (the 256-subfault cases)
        refname = "run" if sm[(c, "run")][1] is not None else "exact"
        _, r = sm[(c, refname)]
        if e != refname and r is not None:
            r, s = lowpass(scale[med] * r), lowpass(s)
            for i, n in enumerate(names):
                pk = np.abs(r[i]).max()
                rows_eng.append(dict(case=c, engine=e, ref=refname, receiver=n, group=grp[n],
                                     max_rel=float(np.abs(s[i] - r[i]).max() / pk),
                                     nrms=float(np.linalg.norm(s[i] - r[i]) / np.linalg.norm(r[i]))))
            for o, inn in drm_pairs:
                io, ii = names.index(o), names.index(inn)
                for k, comp in enumerate("ZEN"):
                    dr, ds = r[io, k] - r[ii, k], s[io, k] - s[ii, k]
                    rows_drm.append(dict(case=c, engine=e, ref=refname, band="lp2", pair=f"{o}-{inn}", comp=comp,
                                         cc=float(np.corrcoef(dr, ds)[0, 1]), nrms=float(np.linalg.norm(ds - dr) / np.linalg.norm(dr))))

for name, rows in (("comparison_sw4.csv", rows_sw4), ("comparison_engines.csv", rows_eng), ("comparison_drm.csv", rows_drm)):
    if rows:
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

# summary
print("\nShakerMaker vs SW4: median / min correlation (best lag), median peak ratio, median nrms")
for c in cases:
    for e in engines:
        for bname in bands:
            sel = [r for r in rows_sw4 if r["case"] == c and r["engine"] == e and r["band"] == bname]
            if sel:
                cc = np.array([r["cc_best"] for r in sel])
                print(f"{c:8s} {e:12s} {bname:8s} cc {np.median(cc):.3f} / {cc.min():.3f}  peak {np.median([r['peak'] for r in sel]):.2f}"
                      f"  nrms {np.median([r['nrms'] for r in sel]):.3f}  lag {np.median([r['lag_s'] for r in sel]):+.3f} s  (n={len(sel)})")
print("\nEngines vs the exact reference (low-pass 2 Hz): median / max of max|diff|/peak")
for c in cases:
    for e in engines[1:]:
        sel = [r for r in rows_eng if r["case"] == c and r["engine"] == e]
        if sel:
            v = [r["max_rel"] for r in sel]
            print(f"{c:8s} {e:12s} vs {sel[0]['ref']:5s} {np.median(v):.2e} / {np.max(v):.2e}")
print("\nDRM layer difference vs run (low-pass 2 Hz): median nrms")
for c in cases:
    for e in engines[1:]:
        sel = [r["nrms"] for r in rows_drm if r["case"] == c and r["engine"] == e and r["ref"] != "sw4"]
        if sel:
            print(f"{c:8s} {e:12s} {np.median(sel):.3e}")
print("\nWorst receivers vs SW4 (run, band 0.1-2.0, best-lag cc < 0.95):")
for r in sorted([r for r in rows_sw4 if r["engine"] == "run" and r["band"] == "0.1-2.0" and r["cc_best"] < 0.95], key=lambda r: r["cc_best"])[:25]:
    print(f"  {r['case']:8s} {r['receiver']:16s} {r['comp']} cc {r['cc_best']:.3f} peak {r['peak']:.2f}")
print("PASS")
