"""run_inter: Green's functions on a regular distance grid, interpolated in reduced time.

run_nearest copies the Green's function of the nearest slot representative to every
pair within the grouping tolerances, so the motion is piecewise constant in
distance and depth; two DRM nodes 2.5 m apart get the same function (difference
zero) or functions a whole slot apart. run_inter keeps the receiver depth exact,
groups the source depth by source only (delta_v_src), and for every pair of depths
(z_src level, z_rec) computes the Green's functions on a regular grid of horizontal
distances (spacing dh_grid, multiples of dh_grid, one k grid per pair of depths:
SM_GF_XMAX=common). Every pair mixes the grid nodes around its distance with
4-point Lagrange (cubic, order=3) or linear (order=1) weights. Each tdata starts at
its own first arrival t0, so mixing them sample by sample is an interpolation in
reduced time; the result is placed at the pair's own t0, fraction of a sample
included. Stage 2 is linear in tdata, so a pair is a weighted sum of node
contributions.

Stage 0 (stage0_inter) writes the grid nodes as "virtual pairs" in the map format
compute_gf reads (/pairs_to_compute, /dh_of_pairs, /dv_of_pairs), so Stage 1 is
compute_gf itself (stage1_inter). Stage 2 (stage2_inter) is the grouped Stage 2
with a weight per contribution.
"""

import copy
import os
from time import perf_counter

import h5py
import numpy as np

from shakermaker.faultsource import FaultSource
from shakermaker.pointsource import PointSource
from shakermaker.station import Station, lagrange4_taps
from shakermaker.stationlist import StationList


def _levels(z, tol):
    """Depth levels: sorted unique depths grouped greedily within tol (km).

    Returns the level depths (the first depth of each group) and the level of
    every input depth (nearest level). tol below the spacing of the depths gives
    one level per distinct depth.
    """
    zr = np.round(np.asarray(z, dtype=np.float64), 9)
    reps = []
    for v in np.unique(zr):
        if not reps or v - reps[-1] > tol:
            reps.append(v)
    reps = np.array(reps)
    idx = np.searchsorted(reps, zr, side="right") - 1
    nxt = np.minimum(idx + 1, len(reps) - 1)
    idx = np.where(np.abs(reps[nxt] - zr) < np.abs(reps[idx] - zr), nxt, idx)
    return reps, idx


def interp_weights(u, n, order):
    """Grid nodes and weights for local coordinates u (in grid steps) on n nodes.

    order=3: 4-point Lagrange through the nodes k-1..k+2 around u (k = floor(u)),
    one-sided at the ends of the grid; order=1: linear between k and k+1.
    n may be an array (one grid length per point).
    Returns (nodes (npts, m), weights (npts, m)), m = 4 or 2.
    """
    u = np.asarray(u, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)
    if order == 1:
        k = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
        t = u - k
        return np.stack([k, k + 1], axis=1), np.stack([1.0 - t, t], axis=1)
    k0 = np.clip(np.floor(u).astype(np.int64) - 1, 0, n - 4)     # first of the 4 nodes
    t = u - k0                                                     # position among nodes 0..3
    w = np.stack([-(t - 1) * (t - 2) * (t - 3) / 6, t * (t - 2) * (t - 3) / 2,
                  -t * (t - 1) * (t - 3) / 2, t * (t - 1) * (t - 2) / 6], axis=1)
    return np.stack([k0, k0 + 1, k0 + 2, k0 + 3], axis=1), w


def stage0_inter(model, h5_database_name, delta_v_rec=1e-5, delta_v_src=0.2, dh_grid=0.04,
                 order=3, showProgress=True):
    """Distance grid of every pair of depths, written as virtual pairs for compute_gf."""
    from shakermaker import shakermaker as smod
    comm, rank, use_mpi, nprocs = getattr(smod, "comm", None), smod.rank, smod.use_mpi, smod.nprocs
    map_file = h5_database_name.replace('.h5', '') + '_map.h5'
    if rank == 0:
        t_s = perf_counter()
        nsrc, nsta = model._source.nsources, model._receivers.nstations
        sx = np.array([model._source.get_source_by_id(i).x for i in range(nsrc)], dtype=np.float64)
        rx = np.array([model._receivers.get_station_by_id(i).x for i in range(nsta)], dtype=np.float64)
        zs_lev, s_lev = _levels(sx[:, 2], delta_v_src)
        zr_lev, r_lev = _levels(rx[:, 2], delta_v_rec)
        margin_lo, margin_hi = (1, 2) if order == 3 else (0, 1)
        dp = []                                     # (L, j, first node index, number of nodes)
        for L in range(len(zs_lev)):
            s = np.flatnonzero(s_lev == L)
            for j in range(len(zr_lev)):
                r = np.flatnonzero(r_lev == j)
                if len(s) == 0 or len(r) == 0:
                    continue
                dmin, dmax = np.inf, 0.0
                for c0 in range(0, len(s), 256):
                    ss = s[c0:c0 + 256]
                    d = np.hypot(sx[ss, 0][:, None] - rx[r, 0][None, :], sx[ss, 1][:, None] - rx[r, 1][None, :])
                    dmin, dmax = min(dmin, d.min()), max(dmax, d.max())
                ib = max(0, int(np.floor(dmin / dh_grid)) - margin_lo)
                ie = int(np.ceil(dmax / dh_grid)) + margin_hi
                ie = max(ie, ib + (3 if order == 3 else 1))
                dp.append((L, j, ib, ie - ib + 1))
        dp = np.array(dp, dtype=np.int64)
        off = np.r_[0, np.cumsum(dp[:, 3])[:-1]].astype(np.int64)
        nslot = int(dp[:, 3].sum())
        pairs = np.empty((nslot, 2), dtype=np.int64)        # (virtual station, virtual source)
        dh = np.empty(nslot); zs = np.empty(nslot); zr = np.empty(nslot)
        for p, (L, j, ib, n) in enumerate(dp):
            sl = slice(off[p], off[p] + n)
            pairs[sl, 0] = np.arange(off[p], off[p] + n)     # one virtual station per node
            pairs[sl, 1] = L                                  # one virtual source per level
            dh[sl] = (ib + np.arange(n)) * dh_grid
            zs[sl] = zs_lev[L]; zr[sl] = zr_lev[j]
        with h5py.File(map_file, 'w', locking=False) as hf:
            hf.create_dataset("pairs_to_compute", data=pairs)
            hf.create_dataset("dh_of_pairs", data=dh)
            hf.create_dataset("dv_of_pairs", data=np.abs(zs - zr))
            hf.create_dataset("zsrc_of_pairs", data=zs)
            hf.create_dataset("zrec_of_pairs", data=zr)
            g = hf.create_group("inter")
            for k_, v_ in dict(dp=dp, dp_off=off, zs_lev=zs_lev, zr_lev=zr_lev,
                               s_lev=s_lev, r_lev=r_lev).items():
                g.create_dataset(k_, data=v_)
            for k_, v_ in dict(gf_mode="inter", dh_grid=dh_grid, order=order, delta_v_src=delta_v_src,
                               delta_v_rec=delta_v_rec, nsrc=nsrc, nsta=nsta).items():
                g.attrs[k_] = v_
            g.attrs["depth_tol"] = model._crust.depth_tol
        if showProgress:
            print(f"[Stage 0 inter] {nsrc} sources in {len(zs_lev)} depth levels (delta_v_src {delta_v_src * 1e3:g} m), "
                  f"{nsta} receivers at {len(zr_lev)} depths; {len(dp)} pairs of depths, {nslot} grid nodes "
                  f"(dh_grid {dh_grid * 1e3:g} m, order {order}); {perf_counter() - t_s:.1f} s -> {map_file}",
                  flush=True)
    if use_mpi and nprocs > 1:
        comm.Barrier()


def _virtual_model(model, map_file):
    """ShakerMaker whose pairs are the grid nodes of the map (for compute_gf)."""
    from shakermaker.shakermaker import ShakerMaker
    with h5py.File(map_file, 'r', locking=False) as hf:
        dh = hf["dh_of_pairs"][:]
        zr = hf["zrec_of_pairs"][:]
        zs_lev = hf["inter/zs_lev"][:]
    srcs = [PointSource([0.0, 0.0, float(z)], [0.0, 90.0, 0.0]) for z in zs_lev]
    stas = [Station([float(d), 0.0, float(z)], metadata={"save_gf": False}) for d, z in zip(dh, zr)]
    return ShakerMaker(model._crust, FaultSource(srcs, {"name": "inter_grid"}), StationList(stas, {}))


def stage1_inter(model, h5_database_name, **core_kw):
    """compute_gf on the grid nodes, with one k grid per pair of depths."""
    map_file = h5_database_name.replace('.h5', '') + '_map.h5'
    vm = _virtual_model(model, map_file)
    old = os.environ.get("SM_GF_XMAX")
    os.environ["SM_GF_XMAX"] = "common"
    try:
        vm.compute_gf(h5_database_name, **core_kw)
    finally:
        if old is None:
            os.environ.pop("SM_GF_XMAX", None)
        else:
            os.environ["SM_GF_XMAX"] = old


_FUSED = [None]


def _fused():
    """Numba kernel: like _stage2_grouped's, with the weight folded into K."""
    if _FUSED[0] is None:
        try:
            from numba import njit

            @njit(nogil=True, cache=False)
            def fused(buf, K, cg, mem, H):
                keep = cg.shape[1]
                for q in range(mem.shape[0]):
                    m, m0, k0, nk = mem[q, 0], mem[q, 1], mem[q, 2], mem[q, 3]
                    Z2, Z1, A0, T2, T1, cc, ss = K[q, 0], K[q, 1], K[q, 2], K[q, 3], K[q, 4], K[q, 5], K[q, 6]
                    for t in range(nk):
                        k = k0 + t
                        zv = 0.0; rv = 0.0; tv = 0.0
                        for j in range(4):
                            s = k - j
                            if s < 0 or s >= keep:
                                continue
                            w = H[q, j]
                            zv += w * (Z2 * cg[5, s] + Z1 * cg[2, s] - A0 * cg[0, s])
                            rv += w * (Z2 * cg[6, s] + Z1 * cg[3, s] - A0 * cg[1, s])
                            tv += w * (T2 * cg[7, s] + T1 * cg[4, s])
                        buf[m, 0, m0 + k] += zv
                        buf[m, 1, m0 + k] += -ss * rv - cc * tv
                        buf[m, 2, m0 + k] += -cc * rv + ss * tv
            _FUSED[0] = fused
        except ImportError:
            _FUSED[0] = False
    return _FUSED[0]


def stage2_inter(model, h5_database_name, dt=0.05, nfft=4096, tb=1000, tmin=0., tmax=100.,
                 writer=None, writer_mode='progressive', showProgress=True):
    """Stage 2 of run_inter: every pair is the weighted sum of its grid nodes."""
    from scipy.signal import fftconvolve
    from shakermaker import shakermaker as smod
    from shakermaker.stationlistwriter import StationListWriter
    comm, rank, nprocs, use_mpi = getattr(smod, "comm", None), smod.rank, smod.nprocs, smod.use_mpi
    t_begin = perf_counter()
    c = {k: 0.0 for k in ("read", "core", "conv", "add", "send", "recv", "wstation")}
    map_file = h5_database_name.replace('.h5', '') + '_map.h5'
    gf_file = h5_database_name.replace('.h5', '') + '_gf.h5'
    hm = h5py.File(map_file, 'r', locking=False)
    hg = h5py.File(gf_file, 'r', locking=False)
    gi = hm["inter"]
    dp, off = gi["dp"][:], gi["dp_off"][:]
    zs_lev, zr_lev, s_lev, r_lev = gi["zs_lev"][:], gi["zr_lev"][:], gi["s_lev"][:], gi["r_lev"][:]
    D, order = float(gi.attrs["dh_grid"]), int(gi.attrs["order"])
    ds = hg["/tdata"]
    nt_full = int(ds.attrs["nt_full"]) if "nt_full" in ds.attrs else None
    pid = -np.ones((len(zs_lev), len(zr_lev)), dtype=np.int64)
    pid[dp[:, 0], dp[:, 1]] = np.arange(len(dp))
    nsrc, nsta = model._source.nsources, model._receivers.nstations
    srcs = [model._source.get_source_by_id(i) for i in range(nsrc)]
    stations = [model._receivers.get_station_by_id(i) for i in range(nsta)]
    rx = np.array([s.x for s in stations], dtype=np.float64)
    # as run_fast: stations start from zero, source time functions on the dt grid
    for sta in stations:
        sta.clear_response()
    for ps in srcs:
        ps.stf.dt = dt
    if rank > 0:
        writer = None
    if writer and rank == 0:
        assert isinstance(writer, StationListWriter)
        writer.initialize(model._receivers, 2 * nfft, tmin=tmin, tmax=tmax, dt=dt, writer_mode=writer_mode)
        writer.write_metadata(model._receivers.metadata)

    crust_cache, trav = {}, {}

    def crust_for(z_src, z_rec):
        key = (float(z_src), float(z_rec))
        aux = crust_cache.get(key)
        if aux is None:
            aux = copy.deepcopy(model._crust)
            aux.split_at_depth(z_src)
            aux.split_at_depth(z_rec)
            crust_cache[key] = aux
        return aux

    tbdt = np.float32(np.float32(tb) * np.float32(dt))
    L2 = 2 * nfft
    comps = np.array([0, 1, 3, 4, 5, 6, 7, 8])
    cache, order_c, cache_bytes = {}, [], [0]
    cap = float(os.environ.get("SM_S2_CACHE_MB", "1024")) * 2**20

    def slot(k):
        g = cache.get(k)
        if g is None:
            t1 = perf_counter()
            g = np.ascontiguousarray(smod._read_tdata(ds, k, nt_full)[:, comps].T, dtype=np.float64)
            c['read'] += perf_counter() - t1
            cache[k] = g; order_c.append(k); cache_bytes[0] += g.nbytes
            while cache_bytes[0] > cap and len(order_c) > 1:
                old = order_c.pop(0); cache_bytes[0] -= cache.pop(old).nbytes
        return g

    def place(ps, t0, nout):
        a0 = (0.0 + ps.tt) + np.float64(t0)
        a1 = (dt + ps.tt) + np.float64(t0)
        dti = a1 - a0
        mu = (a0 - tmin) / dti
        m0, h = lagrange4_taps(mu)
        if h is None:
            h = np.array([1.0, 0.0, 0.0, 0.0])
        k0 = max(0, -m0)
        kend = min(L2 + 3, nout - m0)
        return dti, m0, k0, kend - k0, min(L2, kend), h

    def t0_pair(L, j, x):
        return smod._subtrav_t0(crust_for, zs_lev[L], zr_lev[j], np.float32(x), tbdt, trav)

    stf_cache = smod._STFCache()
    fused = _fused() if os.environ.get("SM_S2_FUSED", "1") == "1" else False
    nout0 = len(np.arange(tmin, tmax, dt))
    chunk = int(os.environ.get("SM_S2_CHUNK", "0") or 0)
    if chunk <= 0:
        chunk = max(1, int(float(os.environ.get("SM_S2_CHUNK_MB", "1024")) * 2**20 // (3 * (nout0 + 2) * 8)))
    chunk = min(chunk, nsta)
    lo, hi = (rank * nsrc) // nprocs, ((rank + 1) * nsrc) // nprocs     # this rank's sources
    for c0 in range(0, nsta, chunk):
        ids = np.arange(c0, min(c0 + chunk, nsta))
        t1 = perf_counter()
        dt_sta = np.empty(len(ids)); nout = np.empty(len(ids), dtype=np.int64)
        ps0 = srcs[0]
        for m, i in enumerate(ids):
            x0 = np.hypot(ps0.x[0] - rx[i, 0], ps0.x[1] - rx[i, 1])
            dt_sta[m] = place(ps0, t0_pair(s_lev[0], r_lev[i], x0), 0)[0]
            nout[m] = len(np.arange(tmin, tmax, dt_sta[m]))
        c['core'] += perf_counter() - t1
        buf = np.zeros((len(ids), 3, int(nout.max())))
        jr = r_lev[ids]
        for j in range(lo, hi):
            ps = srcs[j]
            t1 = perf_counter()
            L = s_lev[j]
            x = np.hypot(ps.x[0] - rx[ids, 0], ps.x[1] - rx[ids, 1])
            p = pid[L, jr]
            u = x / D - dp[p, 2]
            nodes, wts = interp_weights(u, dp[p, 3], order)
            kslot = off[p][:, None] + nodes
            pf, df, lf = ps.angles
            f1 = np.cos(lf)*np.cos(pf) + np.sin(lf)*np.cos(df)*np.sin(pf)
            f2 = np.cos(lf)*np.sin(pf) - np.sin(lf)*np.cos(df)*np.cos(pf)
            f3 = -np.sin(lf)*np.sin(df)
            n1 = -np.sin(pf)*np.sin(df); n2 = np.cos(pf)*np.sin(df); n3 = -np.cos(df)
            a2c, a2s = f1*n1 - f2*n2, f1*n2 + f2*n1
            a1c, a1s = f1*n3 + f3*n1, f2*n3 + f3*n2
            a0 = f3*n3
            az = np.arctan2(rx[ids, 1] - ps.x[1], rx[ids, 0] - ps.x[0])
            c1, s1, c2, s2 = np.cos(az), np.sin(az), np.cos(2*az), np.sin(2*az)
            Z2 = a2c*c2 + a2s*s2; Z1 = a1c*c1 + a1s*s1
            T2 = a2c*s2 - a2s*c2; T1 = a1c*s1 - a1s*c1
            groups = {}
            for m in range(len(ids)):
                dti, m0, k0, nk, kp, h = place(ps, t0_pair(L, jr[m], x[m]), int(nout[m]))
                if nk <= 0 or kp <= 0:
                    continue
                for col in range(kslot.shape[1]):
                    w = wts[m, col]
                    if w == 0.0:
                        continue
                    groups.setdefault((int(kslot[m, col]), dti), []).append(
                        (m, m0, k0, nk, kp, h, w * Z2[m], w * Z1[m], w * a0, w * T2[m], w * T1[m], c1[m], s1[m]))
            c['core'] += perf_counter() - t1
            for (k, dti), mem_ in groups.items():
                keep = max(mb_[4] for mb_ in mem_)
                g = slot(k)
                stf_r, scale = stf_cache.get(ps.stf, dti)
                t1 = perf_counter()
                if stf_r is None:
                    cg = g[:, :keep] * scale
                else:
                    cg = fftconvolve(g[:, :keep], stf_r[None, :], mode="full", axes=-1)[:, :keep] * dti
                c['conv'] += perf_counter() - t1
                t1 = perf_counter()
                mem = np.array([mb_[:4] for mb_ in mem_], dtype=np.int64)
                H = np.array([mb_[5] for mb_ in mem_], dtype=np.float64)
                K = np.array([mb_[6:13] for mb_ in mem_], dtype=np.float64)
                if fused:
                    fused(buf, K, np.ascontiguousarray(cg), mem, H)
                else:
                    from shakermaker.station import add_shifted
                    for q, mb_ in enumerate(mem_):
                        Z2q, Z1q, A0q, T2q, T1q, cq, sq = mb_[6:13]
                        zv = Z2q * cg[5] + Z1q * cg[2] - A0q * cg[0]
                        rv = Z2q * cg[6] + Z1q * cg[3] - A0q * cg[1]
                        tv = T2q * cg[7] + T1q * cg[4]
                        for cc_, comp in enumerate((zv, -sq * rv - cq * tv, -cq * rv + sq * tv)):
                            z = np.convolve(comp[:keep], mb_[5])
                            k0_, nk_ = mb_[2], mb_[3]
                            buf[mb_[0], cc_, mb_[1] + k0_:mb_[1] + k0_ + nk_] += z[k0_:k0_ + nk_]
                c['add'] += perf_counter() - t1
        tot = np.zeros_like(buf) if rank == 0 else None
        t1 = perf_counter()
        if use_mpi and nprocs > 1:
            comm.Reduce(buf, tot, op=smod.MPI.SUM, root=0)
        else:
            tot = buf
        c['send'] += perf_counter() - t1
        if rank == 0:
            t1 = perf_counter()
            for m, i in enumerate(ids):
                st = stations[i]
                n_ = int(nout[m])
                st.add_to_response(tot[m, 0, :n_], tot[m, 1, :n_], tot[m, 2, :n_],
                                   tmin + np.arange(n_) * dt_sta[m], tmin, tmax)
                if writer:
                    smod._timed_write(writer, st, int(i), c, t_begin)
                if writer_mode == 'progressive' and writer:
                    st.clear_response()
            c['recv'] += perf_counter() - t1
            if showProgress:
                print(f"  [rank=0] inter chunk {c0 // chunk + 1}: stations {ids[0]}..{ids[-1]} written", flush=True)
    hm.close(); hg.close()
    if rank == 0 and writer:
        writer.close()
    total = perf_counter() - t_begin
    if rank == 0 and showProgress:
        print(f"[Stage 2 inter] {nsta} stations x {nsrc} sources, order {order}: {total:.1f} s "
              f"(read {c['read']:.1f}, conv {c['conv']:.1f}, add {c['add']:.1f}, send {c['send']:.1f}, "
              f"write {c['recv']:.1f})", flush=True)
    smod._perf_dump("stage2_inter", {k: [v] for k, v in c.items()}, total)   # counters as [seconds]
    return c
