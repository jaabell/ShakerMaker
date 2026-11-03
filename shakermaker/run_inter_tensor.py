"""run_inter, scheme='tensor': interpolation in distance and source depth.

For every receiver depth z_r (exact: the rows of a DRM box are few) the Green's
functions live on a tensor grid of source-depth levels x horizontal distances, with
one wavenumber step for the whole grid (the largest distance), so any subset of nodes
is consistent and a core call with many distances returns exactly what separate calls
return. The grid is refined iteratively:

  1. coarse uniform grid (dz_src0, dh0) covering the depths and distances the pairs use,
     plus the nodes the stencil needs at the ends;
  2. every untested distance cell gets its midpoint column computed, and the column is
     compared with what the neighbouring columns interpolate; then every untested
     source-depth cell gets its midpoint row computed and compared likewise;
  3. a cell whose midpoint is predicted within eps_v (value) and, in distance, eps_s
     (error relative to the change across the cell, what a DRM layer turns into
     forces) is accepted and its midpoint is NOT stored; otherwise the midpoint
     becomes nodes and both halves are tested in the next round;
  4. until every cell passes or reaches dh_min / dz_src_min.

Errors are measured in reduced time (every node starts at its own first arrival),
on the 8 non-zero components, in the band [0, f_err]. Source-depth cells only need
the value: the two layers of a DRM box see the same source.

Storage: (slot, component, sample) float32, only the samples Stage 2 can use
(tmax - t0 + 1 s), with shuffle + zstd (store='zstd') or ZFP per component
(store='zfp<bits>') through hdf5plugin; store='f32' is plain gzip-free float32.
With freq_cut the core leaves nothing above 1.4 freq_cut, so the database keeps one
sample every q dt (decimate=True) and Stage 2 rebuilds the dt samples
with a Kaiser low-pass before using them; the output dt does not change.

Stage 2: the source depth level weights are the same for all receivers of a source,
so for every source and every distance node its receivers need, the levels are
combined first (a virtual node at the source's exact depth), convolved once with the
source time function, and every receiver interpolates in distance over those virtual
nodes with the compiled kernel of run_inter, placed at its own first arrival.
"""

import copy
import os
from time import perf_counter

import h5py
import numpy as np

COMPS = np.array([0, 1, 3, 4, 5, 6, 7, 8])
TAG_TASK, TAG_RES = 71, 72


def _require_hdf5plugin(store):
    """store='zstd'/'zfp<bits>' need the optional hdf5plugin package.

    Checked eagerly, identically on every rank, before any MPI dispatch: Stage 1's
    worker ranks block in comm.recv() as soon as they're spawned (see stage01_tensor),
    so an ImportError raised later only on rank 0 (inside _stage01_exact_streamed,
    once it starts writing the database) would leave every worker waiting forever
    instead of the job aborting.
    """
    if store == "zstd" or store.startswith("zfp"):
        try:
            import hdf5plugin  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"run_inter(..., store={store!r}) needs the optional 'hdf5plugin' "
                "package for its compressed Green's-function database "
                "(pip install hdf5plugin). Use store='f32' to avoid this dependency."
            ) from exc


def decimation(dt, freq_cut, guard=1.1, atten=120.0):
    """Stored-sample step q and reconstruction filter for a core run cut at freq_cut.

    The core leaves nothing above f_top = 1.4 freq_cut, so samples every q dt with a
    Nyquist 1/(2 q dt) >= guard * f_top keep the trace. The dt samples are rebuilt by
    zero insertion and a Kaiser low-pass at that Nyquist: pass band up to f_top, stop
    band (``atten`` dB) from the first image, 2/(2 q dt) - f_top. Returns (1, None)
    when there is no cut or nothing to gain.
    """
    if not freq_cut:
        return 1, None
    from scipy.signal import firwin, kaiserord
    fs, ftop = 1.0 / dt, 1.4 * float(freq_cut)
    q = int(np.floor(fs / (2.0 * ftop * guard)))
    if q < 2:
        return 1, None
    fn = fs / (2.0 * q)
    ntap, beta = kaiserord(atten, 2.0 * (fn - ftop) / (fs / 2.0))
    return q, firwin(ntap | 1, fn, window=("kaiser", beta), fs=fs)


def upsample(g, q, h, n):
    """dt samples (first n) of traces stored every q dt (last axis).

    The trace need not start at zero (it starts tb samples before the first arrival),
    so it is extended before its first sample by odd reflection (value and slope
    continuous) instead of the zeros resample_poly assumes; after the last stored
    sample there is the filter half-length of real samples (stage01_tensor).
    """
    from scipy.signal import resample_poly
    p = min(g.shape[-1] - 1, len(h) // (2 * q) + 2)
    ext = np.concatenate([2.0 * g[..., :1] - g[..., p:0:-1], g], axis=-1)
    return resample_poly(ext, q, 1, axis=-1, window=h)[..., p * q:p * q + n]


def lagrange_weights(x, nodes):
    """Lagrange weights of x on the given nodes (any spacing)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    w = np.ones(len(nodes))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                w[i] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return w


def stencil(x, grid, order):
    """Indices and weights of the order-3 (4 nodes) or order-1 (2 nodes) stencil of x."""
    n = len(grid)
    if n == 1:                                   # a single level (e.g. one source depth)
        return np.array([0]), np.array([1.0])
    k = int(np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n - 2))
    if order == 1 or n < 4:
        idx = np.array([k, k + 1])
    else:
        k0 = int(np.clip(k - 1, 0, n - 4))
        idx = np.arange(k0, k0 + 4)
    return idx, lagrange_weights(x, grid[idx])


def _segments(crust, zmin, zmax):
    """Source-depth segments between crust interfaces (a stencil never crosses one)."""
    tops = np.cumsum(np.r_[0.0, np.asarray(crust.d[:-1], dtype=np.float64)])
    cuts = [z for z in tops[1:] if zmin < z < zmax]
    edges = [zmin] + cuts + [zmax]
    return list(zip(edges[:-1], edges[1:]))


class _Core:
    """One core call: (z_src, z_rec, distances) -> stored float32 tdata, t0, band spectra.

    The stored traces are the first nstore samples, one every q (q = 1: no decimation),
    plus the first nhead samples at dt (the head the low-pass cannot rebuild); the
    spectra are those of the first nkeep samples at dt.
    """

    def __init__(self, model, fk, xmax_of, nkeep, f_err, q=1, nstore=None, nhead=0):
        self.m, self.fk, self.xmax_of, self.nkeep, self.f_err = model, fk, xmax_of, nkeep, f_err
        self.q, self.nstore, self.nhead = q, nstore or nkeep, nhead

    def __call__(self, zs, zr, xs, xmax=None):
        from shakermaker import core
        c = copy.deepcopy(self.m._crust)
        c.split_at_depth(zs); c.split_at_depth(zr)
        fk = self.fk
        tdata, z, e, n, t0 = core.subgreen(
            c.nlayers, c.get_layer(zs) + 1, c.get_layer(zr) + 1, 2, 0, c.d, c.a, c.b, c.rho, c.qa, c.qb,
            fk["dt"], fk["nfft"], fk["tb"], len(xs), fk["sigma"], fk["smth"], fk["wc1"], fk["wc2"],
            fk["pmin"], fk["pmax"], fk["dk"], fk["kc"], fk["taper"], np.asarray(xs, dtype=np.float64),
            0.0, np.pi / 2, 0.0, 0.0, 0.0, float(xs[0]), 0.0,
            float(self.xmax_of(zr) if xmax is None else xmax),
            float(fk.get("freq_cut", 0.0) or 0.0))
        td = np.asarray(tdata[:, :, :self.nkeep], dtype=np.float32)          # (nx, 9, nkeep)
        sp = np.fft.rfft(td[:, COMPS, :].astype(np.float64), n=2 * self.nkeep, axis=-1)
        f = np.fft.rfftfreq(2 * self.nkeep, fk["dt"])
        head = td[:, :, :self.nhead].copy()
        if self.q > 1 or self.nstore != self.nkeep:
            td = np.ascontiguousarray(np.asarray(tdata[:, :, :self.nstore:self.q], dtype=np.float32))
        return td, np.asarray(t0, dtype=np.float64), sp[..., f <= self.f_err].astype(np.complex64), head


_KERNEL2 = [None]


def _kernel2():
    """Numba kernel of Stage 2: per receiver, its 4 distance nodes are combined and
    projected on z, r, t first, then placed with its 4 time taps (64 operations per output
    sample instead of 208 when every node is added on its own with the taps inside).
    CGt: (node, sample, component) convolved node traces; kf: first node of each receiver;
    W: distance weights; R: Z2 Z1 A0 T2 T1 cos sin; mem: receiver, m0, k0, nk; H: time taps."""
    if _KERNEL2[0] is None:
        try:
            from numba import njit

            @njit(nogil=True, cache=False)
            def k2(buf, CGt, kf, W, R, mem, H, scr):
                keep = CGt.shape[1]
                for m in range(mem.shape[0]):
                    mm, m0_, k0_, nk_ = mem[m, 0], mem[m, 1], mem[m, 2], mem[m, 3]
                    Z2, Z1, A0, T2, T1, cc, ss = R[m, 0], R[m, 1], R[m, 2], R[m, 3], R[m, 4], R[m, 5], R[m, 6]
                    n0 = kf[m]
                    w0, w1, w2, w3 = W[m, 0], W[m, 1], W[m, 2], W[m, 3]
                    for s in range(keep):
                        a = CGt[n0, s]; b = CGt[n0 + 1, s]; c = CGt[n0 + 2, s]; d = CGt[n0 + 3, s]
                        g0 = w0 * a[0] + w1 * b[0] + w2 * c[0] + w3 * d[0]
                        g1 = w0 * a[1] + w1 * b[1] + w2 * c[1] + w3 * d[1]
                        g2 = w0 * a[2] + w1 * b[2] + w2 * c[2] + w3 * d[2]
                        g3 = w0 * a[3] + w1 * b[3] + w2 * c[3] + w3 * d[3]
                        g4 = w0 * a[4] + w1 * b[4] + w2 * c[4] + w3 * d[4]
                        g5 = w0 * a[5] + w1 * b[5] + w2 * c[5] + w3 * d[5]
                        g6 = w0 * a[6] + w1 * b[6] + w2 * c[6] + w3 * d[6]
                        g7 = w0 * a[7] + w1 * b[7] + w2 * c[7] + w3 * d[7]
                        scr[s, 0] = Z2 * g5 + Z1 * g2 - A0 * g0
                        scr[s, 1] = Z2 * g6 + Z1 * g3 - A0 * g1
                        scr[s, 2] = T2 * g7 + T1 * g4
                    h0, h1, h2, h3 = H[m, 0], H[m, 1], H[m, 2], H[m, 3]
                    for t in range(nk_):
                        k = k0_ + t
                        zv = 0.0; rv = 0.0; tv = 0.0
                        if k < keep:
                            zv += h0 * scr[k, 0]; rv += h0 * scr[k, 1]; tv += h0 * scr[k, 2]
                        if 1 <= k and k - 1 < keep:
                            zv += h1 * scr[k - 1, 0]; rv += h1 * scr[k - 1, 1]; tv += h1 * scr[k - 1, 2]
                        if 2 <= k and k - 2 < keep:
                            zv += h2 * scr[k - 2, 0]; rv += h2 * scr[k - 2, 1]; tv += h2 * scr[k - 2, 2]
                        if 3 <= k and k - 3 < keep:
                            zv += h3 * scr[k - 3, 0]; rv += h3 * scr[k - 3, 1]; tv += h3 * scr[k - 3, 2]
                        buf[mm, 0, m0_ + k] += zv
                        buf[mm, 1, m0_ + k] += -ss * rv - cc * tv
                        buf[mm, 2, m0_ + k] += -cc * rv + ss * tv
            _KERNEL2[0] = k2
        except ImportError:
            _KERNEL2[0] = False
    return _KERNEL2[0]


def stencil4(x, grid):
    """Vectorised order-3 stencil: first node index (4 contiguous nodes, padded with weight 0 when
    the grid has fewer than 4) and the 4 Lagrange weights, for an array of distances x."""
    x = np.asarray(x, dtype=np.float64); grid = np.asarray(grid, dtype=np.float64); n = len(grid)
    W = np.zeros((len(x), 4))
    if n == 1:
        W[:, 0] = 1.0
        return np.zeros(len(x), dtype=np.int64), W
    if n < 4:                                    # linear between the two nodes around x
        k = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n - 2)
        u = (x - grid[k]) / (grid[k + 1] - grid[k])
        W[:, 0], W[:, 1] = 1.0 - u, u
        return k.astype(np.int64), W
    k = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n - 2)
    k0 = np.clip(k - 1, 0, n - 4)
    nodes = grid[k0[:, None] + np.arange(4)[None, :]]
    for i in range(4):
        w = np.ones(len(x))
        for j in range(4):
            if i != j:
                w *= (x - nodes[:, j]) / (nodes[:, i] - nodes[:, j])
        W[:, i] = w
    return k0.astype(np.int64), W


def _bisect(levels, sel, order, eps_v, eps_s, eps_bands):
    """Distance nodes kept by bisection (indices of candidates): a cell is accepted when the
    candidate at its midpoint is predicted from the nodes within eps_v (value) and eps_s
    (relative to the change across the cell) at every level of `levels` (band spectra,
    (ncand, 8, nf)), in every frequency range of eps_bands; otherwise its midpoint becomes a node.
    Returns the selected indices, the number of cells with one candidate inside whose value
    misses eps_v (they cannot be split further: dh_min is too coarse there, e.g. in the near
    field) and the largest value error among them."""
    nf = levels[0].shape[-1]
    fsl = [slice(int(lo * (nf - 1)), int(hi * (nf - 1)) + 1) for lo, hi in eps_bands]
    sel = sorted(set(sel))
    ok = set()
    unresolved, worst_unresolved = 0, 0.0
    while True:
        new = []
        for a, b in zip(sel[:-1], sel[1:]):
            if b - a < 2 or (a, b) in ok:
                continue
            m = (a + b) // 2
            idx, w = stencil(float(m), np.array(sel, dtype=np.float64), order)
            worst_v = worst_s = 0.0
            for spl in levels:
                interp = sum(w[q] * spl[sel[idx[q]]] for q in range(len(idx)))
                span = spl[b] - spl[a]
                for sl in fsl:
                    worst_v = max(worst_v, _rel(interp[..., sl], spl[m][..., sl]))
                    worst_s = max(worst_s, float(np.sqrt(np.sum(np.abs(interp[..., sl] - spl[m][..., sl]) ** 2))
                                                 / max(np.sqrt(np.sum(np.abs(span[..., sl]) ** 2)), 1e-300)))
            if worst_v <= eps_v and worst_s <= eps_s:
                ok.add((a, b))
            else:
                new.append(m)
                if b - a == 2 and worst_v > eps_v:
                    unresolved += 1
                    worst_unresolved = max(worst_unresolved, worst_v)          # the value itself is off; eps_s alone is ill-conditioned
                                             # on cells where the trace barely changes
        if not new:
            return sel, unresolved, worst_unresolved
        sel = sorted(set(sel) | set(new))


def _rel(a, b):
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2)) / max(np.sqrt(np.sum(np.abs(b) ** 2)), 1e-300))


def stage01_tensor(model, h5_database_name, fk, tmax, delta_v_rec=1e-5, dz_src0=0.32, dh0=0.04,
                   dz_src_min=0.04, dh_min=0.02, eps_v=0.01, eps_s=0.05, f_err=20.0, order=3,
                   store="zstd", calls_max=64, src_levels="exact", decimate=True,
                   eps_bands=((0.0, 1.0), (0.5, 1.0)), showProgress=True):
    """Stages 0 and 1 of the tensor scheme (rank 0 coordinates and writes; the others compute).

    src_levels='exact' (default): one source-depth level per distinct source depth,
    not refined (above ~5 Hz the source depth cannot be interpolated: surface-reflected
    phases such as pP move the other way);
    'interp': levels every dz_src0, refined iteratively like below.

    Distance: every source-depth level is computed once at ALL candidate distances
    (every dh_min) in calls that share the wavenumber kernel (each extra distance costs a
    few percent of a call); which distances become nodes is decided afterwards, without
    new calls, by bisection on the stored band-limited spectra: starting from every
    dh0, a cell is accepted when the candidate at its midpoint is predicted within eps_v
    (value) and eps_s (relative to the change across the cell) at every level, in
    every frequency range of eps_bands (fractions of f_err): the whole band and its
    upper half by default, because the error over the whole band is ruled by the
    energetic low frequencies and left the top of the band loose (256 point sources, 10 Hz band: 5 %
    error at 5-10 Hz).
    Source depth: iterative rounds; the midpoint level of every untested cell is
    computed (one call per chunk of candidates) and kept only if the levels around it
    do not predict it within eps_v at every candidate distance.
    The traces of computed levels go to a temporary file; only the selected
    levels x distances are copied to the database (only the samples Stage 2 can use;
    store='zstd' / 'zfp<bits>'). decimate=True with freq_cut: one sample every q dt,
    see decimation(); the filter half-length is added to the stored samples so the
    rebuilt nkeep samples are exact up to the end.
    """
    _require_hdf5plugin(store)
    from shakermaker import shakermaker as smod
    from shakermaker.run_inter import _levels
    comm, rank, nprocs, use_mpi = getattr(smod, "comm", None), smod.rank, smod.nprocs, smod.use_mpi
    t_begin = perf_counter()
    nsrc, nsta = model._source.nsources, model._receivers.nstations
    sx = np.array([model._source.get_source_by_id(i).x for i in range(nsrc)], dtype=np.float64)
    rx = np.array([model._receivers.get_station_by_id(i).x for i in range(nsta)], dtype=np.float64)
    zr_lev, r_lev = _levels(rx[:, 2], delta_v_rec)
    dmin = np.full(len(zr_lev), np.inf); dmax = np.zeros(len(zr_lev))
    for j in range(len(zr_lev)):
        r = np.flatnonzero(r_lev == j)
        for c0 in range(0, nsrc, 512):
            d = np.hypot(sx[c0:c0 + 512, 0][:, None] - rx[r, 0][None, :], sx[c0:c0 + 512, 1][:, None] - rx[r, 1][None, :])
            dmin[j] = min(dmin[j], d.min()); dmax[j] = max(dmax[j], d.max())
    step = max(1, int(round(dh0 / dh_min)))                 # coarse nodes every `step` candidates
    cand = {}
    for j in range(len(zr_lev)):
        ib = max(0, int(np.floor(dmin[j] / dh0)) - (1 if order == 3 else 0)) * step
        ie = (int(np.ceil(dmax[j] / dh0)) + (2 if order == 3 else 1)) * step
        cand[j] = np.arange(ib, ie + 1) * dh_min
    xmax = {float(z): float(cand[j][-1]) for j, z in enumerate(zr_lev)}
    dt = fk["dt"]
    nkeep = int(min(2 * fk["nfft"], np.ceil((tmax + fk["tb"] * dt + 1.0) / dt)))       # samples Stage 2 uses
    q, fir = decimation(dt, fk.get("freq_cut", 0.0)) if decimate else (1, None)
    if q > 1 and nkeep + len(fir) // 2 + q > fk["nfft"]:
        # the core trace has nfft samples (zeros after): the step there would ring
        # back over the rebuilt samples, so keep the full rate
        q, fir = 1, None
    nstore = nkeep if q == 1 else nkeep + len(fir) // 2 + q
    nsamp = -(-nstore // q)                                   # samples stored per trace
    # head kept at dt: the core trace is multiplied by exp(sigma (t0 + t)), which is not
    # periodic, so near its first sample the trace is not band-limited and the low-pass
    # cannot rebuild the first half-length of samples from the coarse ones
    nhead = 0 if q == 1 else min(nkeep, len(fir) // 2 + 2 * q)
    core_call = _Core(model, fk, lambda z: xmax[float(z)], nkeep, f_err, q=q, nstore=nstore, nhead=nhead)

    # ---------------- worker side -----------------------------------------
    if use_mpi and nprocs > 1 and rank > 0:
        while True:
            task = comm.recv(source=0, tag=TAG_TASK)
            if task is None:
                break
            tid, zs, zr, xs, *xm = task
            comm.send((tid, core_call(zs, zr, xs, *xm)), dest=0, tag=TAG_RES)
        comm.Barrier()
        comm.Barrier()                       # rank 0 has written the database
        return

    # ---------------- coordinator (rank 0) --------------------------------
    if src_levels == "exact":
        return _stage01_exact_streamed(model, h5_database_name, sx, rx, zr_lev, r_lev, core_call, dict(
            fk=fk, tmax=tmax, delta_v_rec=delta_v_rec, dh0=dh0, dh_min=dh_min, eps_v=eps_v, eps_s=eps_s,
            f_err=f_err, order=order, store=store, calls_max=calls_max, eps_bands=eps_bands, nkeep=nkeep,
            q=q, nstore=nstore, nhead=nhead, nsamp=nsamp, dz_src0=dz_src0, dz_src_min=dz_src_min,
            showProgress=showProgress, t_begin=t_begin))
    stats = dict(calls=0, computed=0, levels_computed=0, levels_stored=0, candidates=0, dh_stored=0,
                 rounds=0, seconds_core=0.0)
    gf_file = h5_database_name.replace('.h5', '') + '_gf.h5'
    map_file = h5_database_name.replace('.h5', '') + '_map.h5'
    tmp_file = gf_file + ".levels.tmp"
    if os.path.dirname(gf_file):
        os.makedirs(os.path.dirname(gf_file), exist_ok=True)
    htmp = h5py.File(tmp_file, "w", locking=False)
    grids = []
    zs_all = sx[:, 2]
    for j, zr in enumerate(zr_lev):
        for (za, zb) in _segments(model._crust, zs_all.min(), zs_all.max() + 1e-9):
            if not np.any((zs_all >= za - 1e-9) & (zs_all <= zb + 1e-9)):
                continue
            if zb - za < 1e-6:
                zs = [za]
            elif src_levels == "exact":
                sel_ = (zs_all >= za - 1e-9) & (zs_all <= zb + 1e-9)
                zs = list(np.unique(np.round(zs_all[sel_], 9)))
            else:
                nz = max(1, int(np.ceil((zb - za) / dz_src0)))
                zs = list(np.linspace(za, zb, nz + 1))
                if order == 3 and len(zs) < 4:
                    zs = list(np.linspace(za, zb, 4))
            grids.append(dict(j=j, zr=float(zr), zs=zs, sp={}, ok_zs=set(),
                              sel=list(range(0, len(cand[j]), step)) + ([len(cand[j]) - 1]
                                   if (len(cand[j]) - 1) % step else [])))

    def run_levels(items):
        """items: (gi, z). Computes every candidate distance of each level; spectra kept, traces to tmp."""
        tasks = []
        for gi, z in items:
            xs = cand[grids[gi]["j"]]
            for c0 in range(0, len(xs), calls_max):
                tasks.append(((gi, round(z, 9), c0), z, grids[gi]["zr"], xs[c0:c0 + calls_max]))
        t1 = perf_counter()
        parts = {}
        if use_mpi and nprocs > 1:
            pend, it = 0, iter(tasks)
            for w in range(1, nprocs):
                t = next(it, None)
                if t is None:
                    break
                comm.send(t, dest=w, tag=TAG_TASK); pend += 1
            while pend:
                st = smod.MPI.Status()
                tid, res = comm.recv(source=smod.MPI.ANY_SOURCE, tag=TAG_RES, status=st)
                parts[tid] = res; pend -= 1
                t = next(it, None)
                if t is not None:
                    comm.send(t, dest=st.Get_source(), tag=TAG_TASK); pend += 1
        else:
            for t in tasks:
                parts[t[0]] = core_call(t[1], t[2], t[3])
        stats["seconds_core"] += perf_counter() - t1
        stats["calls"] += len(tasks)
        out = {}
        for gi, z in items:
            key = (gi, round(z, 9))
            cs = sorted(c for (g_, z_, c) in parts if (g_, z_) == key)
            td = np.concatenate([parts[key + (c,)][0] for c in cs])
            t0 = np.concatenate([parts[key + (c,)][1] for c in cs])
            sp = np.concatenate([parts[key + (c,)][2] for c in cs])
            hd = np.concatenate([parts[key + (c,)][3] for c in cs])
            name = f"g{gi}_z{round(z, 9):.9f}"
            htmp.create_dataset(name, data=td); htmp.create_dataset(name + "_t0", data=t0)
            if nhead:
                htmp.create_dataset(name + "_head", data=hd)
            out[key] = sp
            stats["computed"] += len(td); stats["levels_computed"] += 1
        return out

    def select_dh(g):
        """Bisection over the candidate distances of every stored level (no new calls)."""
        nC = len(cand[g["j"]])
        sel = sorted(set(g["sel"]))
        levels = [g["sp"][round(z, 9)] for z in g["zs"]]
        nf = levels[0].shape[-1]
        fsl = [slice(int(lo * (nf - 1)), int(hi * (nf - 1)) + 1) for lo, hi in eps_bands]
        ok = set()
        while True:
            new = []
            for a, b in zip(sel[:-1], sel[1:]):
                if b - a < 2 or (a, b) in ok:
                    continue
                m = (a + b) // 2
                grid = np.array(sel, dtype=np.float64)
                idx, w = stencil(float(m), grid, order)
                worst_v = worst_s = 0.0
                for spl in levels:
                    interp = sum(w[q] * spl[sel[idx[q]]] for q in range(len(idx)))
                    span = spl[b] - spl[a]
                    for sl in fsl:                  # every frequency range of eps_bands
                        worst_v = max(worst_v, _rel(interp[..., sl], spl[m][..., sl]))
                        worst_s = max(worst_s, float(np.sqrt(np.sum(np.abs(interp[..., sl] - spl[m][..., sl]) ** 2))
                                                     / max(np.sqrt(np.sum(np.abs(span[..., sl]) ** 2)), 1e-300)))
                if worst_v <= eps_v and worst_s <= eps_s:
                    ok.add((a, b))
                else:
                    new.append(m)
            if not new:
                break
            sel = sorted(set(sel) | set(new))
        g["sel"] = sel

    # first round: every coarse level of every grid
    res = run_levels([(gi, z) for gi, g in enumerate(grids) for z in g["zs"]])
    for (gi, zk), sp in res.items():
        grids[gi]["sp"][zk] = sp
    while True:
        stats["rounds"] += 1
        for g in grids:
            select_dh(g)
        items = []
        for gi, g in enumerate(grids):
            zs = g["zs"]
            if src_levels == "exact":
                continue
            for i in range(len(zs) - 1):
                if (round(zs[i], 9), round(zs[i + 1], 9)) in g["ok_zs"] or zs[i + 1] - zs[i] <= dz_src_min * 1.5:
                    continue
                items.append((gi, 0.5 * (zs[i] + zs[i + 1]), i))
        if not items:
            break
        res = run_levels([(gi, zm) for gi, zm, _ in items])
        changed = False
        for gi, zm, i in items:
            g = grids[gi]; zs = np.array(g["zs"]); spm = res[(gi, round(zm, 9))]
            idx, w = stencil(zm, zs, order)
            worst = max(_rel(sum(w[q] * g["sp"][round(zs[idx[q]], 9)][k] for q in range(len(idx))), spm[k])
                        for k in range(len(spm)))
            if worst <= eps_v:
                g["ok_zs"].add((round(zs[i], 9), round(zs[i + 1], 9)))
            else:
                g["sp"][round(zm, 9)] = spm
                g["zs"] = sorted(set(g["zs"]) | {zm})
                changed = True
        if showProgress:
            nn = sum(len(g["zs"]) * len(g["sel"]) for g in grids)
            print(f"[Stage 1 tensor] round {stats['rounds']}: {nn} nodes selected, {stats['levels_computed']} levels "
                  f"computed, {stats['calls']} core calls, {perf_counter() - t_begin:.0f} s", flush=True)
        if not changed:
            for g in grids:
                select_dh(g)
            break

    if use_mpi and nprocs > 1:
        for w in range(1, nprocs):
            comm.send(None, dest=w, tag=TAG_TASK)
        comm.Barrier()

    # ---------------- write the database: selected levels x distances --------
    nslot = sum(len(g["zs"]) * len(g["sel"]) for g in grids)
    kw = dict(chunks=(1, 1, nsamp), dtype=np.float32)
    if store == "zstd":
        import hdf5plugin
        kw.update(shuffle=True, **hdf5plugin.Zstd(clevel=3))
    elif store.startswith("zfp"):
        import hdf5plugin
        kw.update(**hdf5plugin.Zfp(precision=int(store[3:])))
    t1 = perf_counter()
    with h5py.File(gf_file, "w", locking=False) as hg, h5py.File(map_file, "w", locking=False) as hm:
        ds = hg.create_dataset("tdata", shape=(nslot, 9, nsamp), **kw)
        dsh = hg.create_dataset("thead", shape=(nslot, 9, nhead), **dict(kw, chunks=(1, 1, nhead))) if nhead else None
        t0s = np.empty(nslot)
        s = 0
        g_m = hm.create_group("tensor")
        for gi, g in enumerate(grids):
            zs = np.array(g["zs"]); sel = np.array(g["sel"]); dh = cand[g["j"]][sel]
            for i, z in enumerate(zs):
                name = f"g{gi}_z{round(z, 9):.9f}"
                ds[s + i * len(sel): s + (i + 1) * len(sel)] = htmp[name][sel]
                if nhead:
                    dsh[s + i * len(sel): s + (i + 1) * len(sel)] = htmp[name + "_head"][sel]
                t0s[s + i * len(sel): s + (i + 1) * len(sel)] = htmp[name + "_t0"][sel]
            gg = g_m.create_group(f"g{gi}")
            gg.create_dataset("zs", data=zs); gg.create_dataset("dh", data=dh)
            gg.attrs.update(zr=g["zr"], offset=s, zmin=zs.min(), zmax=zs.max())
            s += len(zs) * len(sel)
            stats["levels_stored"] += len(zs); stats["dh_stored"] += len(sel); stats["candidates"] += len(cand[g["j"]])
        hg.create_dataset("t0", data=t0s)
        g_m.attrs.update(gf_mode="tensor", order=order, ngrids=len(grids), nkeep=nkeep, store=store,
                         decim=q, nstore=nstore, nhead=nhead, dt=dt,
                         eps_v=eps_v, eps_s=eps_s, f_err=f_err, dz_src0=dz_src0, dh0=dh0,
                         src_levels=src_levels, freq_cut=float(fk.get("freq_cut", 0.0) or 0.0),
                         dz_src_min=dz_src_min, dh_min=dh_min, delta_v_rec=delta_v_rec, tmax=tmax,
                         eps_bands=np.asarray(eps_bands, dtype=np.float64))
        g_m.create_dataset("zr_lev", data=zr_lev); g_m.create_dataset("r_lev", data=r_lev)
        stats["stored"] = nslot
        for k_, v_ in stats.items():
            g_m.attrs[f"stat_{k_}"] = v_
    htmp.close(); os.remove(tmp_file)
    if use_mpi and nprocs > 1:
        comm.Barrier()                       # database written: Stage 2 may open it
    stats["seconds_write"] = perf_counter() - t1
    stats["seconds_total"] = perf_counter() - t_begin
    stats["gf_bytes"] = os.path.getsize(gf_file)
    if showProgress:
        print(f"[Stage 1 tensor] {nslot} nodes stored ({stats['computed']} computed, {stats['calls']} calls, "
              f"{stats['levels_computed']} levels computed, {stats['levels_stored']} stored), "
              f"{stats['gf_bytes'] / 1e6:.1f} MB ({store}), {stats['seconds_total']:.0f} s -> {gf_file}", flush=True)
    return stats


def _stage01_exact_streamed(model, h5_database_name, sx, rx, zr_lev, r_lev, core_call, o):
    """Rank-0 side of Stages 0 and 1 when the source depth is exact (the production case).

    No source-depth interpolation, so every (receiver depth, source depth) level is its own
    grid with its own distance nodes: its candidates cover only the distances its sources and
    receivers need (one wavenumber step, xmax = its last candidate), the bisection runs as
    soon as its core calls are back, its selected nodes are appended to the database and its
    traces and spectra are freed. Memory: the levels in flight (about one per worker), not
    the whole model (the first version kept every level until the end: ~650 GB for a
    DRM box of 8 179 nodes and 32 768 point sources at 2.9-13 km). Same database layout as the tensor scheme (one grid per level).
    """
    from shakermaker import shakermaker as smod
    comm, nprocs, use_mpi = getattr(smod, "comm", None), smod.nprocs, smod.use_mpi
    dh0, dh_min, order, calls_max = o["dh0"], o["dh_min"], o["order"], o["calls_max"]
    step = max(1, int(round(dh0 / dh_min)))
    zs_src = np.round(sx[:, 2], 9)
    zs_lev = np.unique(zs_src)

    # levels: (j, zs) with the distance range its sources and receivers need
    levels = []
    for j in range(len(zr_lev)):
        rr = rx[r_lev == j]
        for z in zs_lev:
            ss = sx[zs_src == z]
            dmn, dmx = np.inf, 0.0
            for c0 in range(0, len(ss), 256):
                d = np.hypot(ss[c0:c0 + 256, 0][:, None] - rr[:, 0][None, :],
                             ss[c0:c0 + 256, 1][:, None] - rr[:, 1][None, :])
                dmn, dmx = min(dmn, d.min()), max(dmx, d.max())
            ib = max(0, int(np.floor(dmn / dh0)) - (1 if order == 3 else 0)) * step
            ie = (int(np.ceil(dmx / dh0)) + (2 if order == 3 else 1)) * step
            cand = np.arange(ib, ie + 1) * dh_min
            sel0 = list(range(0, len(cand), step)) + ([len(cand) - 1] if (len(cand) - 1) % step else [])
            levels.append(dict(j=j, zr=float(zr_lev[j]), zs=float(z), cand=cand, sel0=sel0))

    nsamp, nhead = o["nsamp"], o["nhead"]
    kw = dict(chunks=(1, 1, nsamp), dtype=np.float32, maxshape=(None, 9, nsamp))
    if o["store"] == "zstd":
        import hdf5plugin
        kw.update(shuffle=True, **hdf5plugin.Zstd(clevel=3))
    elif o["store"].startswith("zfp"):
        import hdf5plugin
        kw.update(**hdf5plugin.Zfp(precision=int(o["store"][3:])))
    gf_file = h5_database_name.replace('.h5', '') + '_gf.h5'
    map_file = h5_database_name.replace('.h5', '') + '_map.h5'
    if os.path.dirname(gf_file):
        os.makedirs(os.path.dirname(gf_file), exist_ok=True)
    hg = h5py.File(gf_file, "w", locking=False)
    hm = h5py.File(map_file, "w", locking=False)
    ds = hg.create_dataset("tdata", shape=(0, 9, nsamp), **kw)
    dsh = (hg.create_dataset("thead", shape=(0, 9, nhead), **dict(kw, chunks=(1, 1, nhead), maxshape=(None, 9, nhead)))
           if nhead else None)
    g_m = hm.create_group("tensor")
    t0s, nslot = [], 0
    stats = dict(calls=0, computed=0, levels_computed=0, levels_stored=0, candidates=0, dh_stored=0,
                 rounds=1, seconds_core=0.0, unresolved_cells=0, unresolved_max_error=0.0)

    def tasks_of(li):
        lv = levels[li]
        return [((li, c0), lv["zs"], lv["zr"], lv["cand"][c0:c0 + calls_max], float(lv["cand"][-1]))
                for c0 in range(0, len(lv["cand"]), calls_max)]

    def finish(li, parts):
        nonlocal nslot
        lv = levels[li]
        cs = sorted(parts)
        td = np.concatenate([parts[c][0] for c in cs]); t0 = np.concatenate([parts[c][1] for c in cs])
        sp = np.concatenate([parts[c][2] for c in cs]); hd = np.concatenate([parts[c][3] for c in cs])
        sel, nun, wun = _bisect([sp], lv["sel0"], order, o["eps_v"], o["eps_s"], o["eps_bands"])
        sel = np.array(sel)
        stats["unresolved_cells"] += nun
        stats["unresolved_max_error"] = max(stats["unresolved_max_error"], wun)
        n = len(sel)
        ds.resize(nslot + n, axis=0); ds[nslot:nslot + n] = td[sel]
        if nhead:
            dsh.resize(nslot + n, axis=0); dsh[nslot:nslot + n] = hd[sel]
        t0s.extend(t0[sel].tolist())
        gg = g_m.create_group(f"g{li}")
        gg.create_dataset("zs", data=np.array([lv["zs"]])); gg.create_dataset("dh", data=lv["cand"][sel])
        gg.attrs.update(zr=lv["zr"], offset=nslot, zmin=lv["zs"], zmax=lv["zs"])
        nslot += n
        stats["computed"] += len(td); stats["levels_computed"] += 1; stats["levels_stored"] += 1
        stats["dh_stored"] += n; stats["candidates"] += len(lv["cand"])

    t1 = perf_counter()
    pending = {}                                   # level -> {c0: result}, levels in flight only
    left = {}                                      # level -> calls still out
    if use_mpi and nprocs > 1:
        it = (t for li in range(len(levels)) for t in tasks_of(li))
        nout = 0

        def send(w):
            nonlocal nout
            t = next(it, None)
            if t is None:
                return
            li = t[0][0]
            if li not in left:
                left[li] = len(tasks_of(li)); pending[li] = {}
            comm.send(t, dest=w, tag=TAG_TASK); nout += 1

        for w in range(1, nprocs):
            send(w)
        while nout:
            st = smod.MPI.Status()
            (li, c0), res = comm.recv(source=smod.MPI.ANY_SOURCE, tag=TAG_RES, status=st)
            nout -= 1; stats["calls"] += 1
            pending[li][c0] = res; left[li] -= 1
            send(st.Get_source())
            if left[li] == 0:
                finish(li, pending.pop(li)); del left[li]
                if o["showProgress"] and stats["levels_computed"] % 50 == 0:
                    print(f"[Stage 1 tensor] {stats['levels_computed']}/{len(levels)} levels, {nslot} nodes, "
                          f"{perf_counter() - o['t_begin']:.0f} s", flush=True)
        for w in range(1, nprocs):
            comm.send(None, dest=w, tag=TAG_TASK)
        comm.Barrier()
    else:
        for li in range(len(levels)):
            parts = {}
            for (key, zs, zr, xs, xm) in tasks_of(li):
                parts[key[1]] = core_call(zs, zr, xs, xm); stats["calls"] += 1
            finish(li, parts)
    stats["seconds_core"] = perf_counter() - t1

    hg.create_dataset("t0", data=np.array(t0s))
    g_m.attrs.update(gf_mode="tensor", order=order, ngrids=len(levels), nkeep=o["nkeep"], store=o["store"],
                     decim=o["q"], nstore=o["nstore"], nhead=nhead, dt=o["fk"]["dt"],
                     eps_v=o["eps_v"], eps_s=o["eps_s"], f_err=o["f_err"], dz_src0=o["dz_src0"], dh0=dh0,
                     src_levels="exact", freq_cut=float(o["fk"].get("freq_cut", 0.0) or 0.0),
                     dz_src_min=o["dz_src_min"], dh_min=dh_min, delta_v_rec=o["delta_v_rec"], tmax=o["tmax"],
                     eps_bands=np.asarray(o["eps_bands"], dtype=np.float64), per_level=1)
    g_m.create_dataset("zr_lev", data=zr_lev); g_m.create_dataset("r_lev", data=r_lev)
    stats["stored"] = nslot
    stats["seconds_total"] = perf_counter() - o["t_begin"]
    for k_, v_ in stats.items():
        g_m.attrs[f"stat_{k_}"] = v_
    hg.close(); hm.close()
    stats["gf_bytes"] = os.path.getsize(gf_file)
    if use_mpi and nprocs > 1:
        comm.Barrier()                             # the database is written
    if o["showProgress"]:
        print(f"[Stage 1 tensor] {nslot} nodes stored ({stats['computed']} computed, {stats['calls']} calls, "
              f"{len(levels)} levels), {stats['gf_bytes'] / 1e6:.1f} MB ({o['store']}), "
              f"{stats['seconds_total']:.0f} s -> {gf_file}", flush=True)
    if stats["unresolved_cells"]:
        print(f"[Stage 1 tensor] WARNING: {stats['unresolved_cells']} distance cells miss eps_v at their midpoint "
              f"and cannot be split below dh_min = {1000 * dh_min:.1f} m (largest value error "
              f"{100 * stats['unresolved_max_error']:.1f} %, tolerance {100 * o['eps_v']:.1f} %). A large error is "
              f"typical of the near field: use a smaller dh_min.", flush=True)
    return stats


def stage2_tensor(model, h5_database_name, dt=0.05, nfft=4096, tb=1000, tmin=0., tmax=100.,
                  writer=None, writer_mode='progressive', showProgress=True):
    """Stage 2 of the tensor scheme (virtual source-depth nodes, then distance per receiver)."""
    from scipy.signal import fftconvolve
    from shakermaker import shakermaker as smod
    from shakermaker.run_inter import _fused
    from shakermaker.station import lagrange4_taps
    from shakermaker.stationlistwriter import StationListWriter
    comm, rank, nprocs, use_mpi = getattr(smod, "comm", None), smod.rank, smod.nprocs, smod.use_mpi
    t_begin = perf_counter()
    c = {k: 0.0 for k in ("read", "combine", "conv", "add", "send", "recv", "wstation", "core")}
    hm = h5py.File(h5_database_name.replace('.h5', '') + '_map.h5', 'r', locking=False)
    hg = h5py.File(h5_database_name.replace('.h5', '') + '_gf.h5', 'r', locking=False)
    gt = hm["tensor"]; order = int(gt.attrs["order"]); nkeep = int(gt.attrs["nkeep"])
    _require_hdf5plugin(gt.attrs.get("store", "zstd"))
    q_dec = int(gt.attrs.get("decim", 1))
    fir = decimation(float(gt.attrs["dt"]), float(gt.attrs["freq_cut"]))[1] if q_dec > 1 else None
    grids = []
    for gi in range(int(gt.attrs["ngrids"])):
        gg = gt[f"g{gi}"]
        grids.append(dict(zs=gg["zs"][:], dh=gg["dh"][:], zr=float(gg.attrs["zr"]), off=int(gg.attrs["offset"]),
                          zmin=float(gg.attrs["zmin"]), zmax=float(gg.attrs["zmax"])))
    zr_lev = gt["zr_lev"][:]
    ds = hg["tdata"]
    dsh = hg["thead"] if "thead" in hg else None
    nsrc, nsta = model._source.nsources, model._receivers.nstations
    srcs = [model._source.get_source_by_id(i) for i in range(nsrc)]
    stations = [model._receivers.get_station_by_id(i) for i in range(nsta)]
    rx = np.array([s.x for s in stations], dtype=np.float64)
    r_lev = np.array([int(np.argmin(np.abs(zr_lev - z))) for z in rx[:, 2]])
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

    by_level = {(round(g["zr"], 9), round(g["zmin"], 9)): gi for gi, g in enumerate(grids)
                if abs(g["zmax"] - g["zmin"]) < 1e-12}

    def grid_of(j, zs):
        gi = by_level.get((round(float(zr_lev[j]), 9), round(float(zs), 9)))
        if gi is not None:
            return gi
        for gi, g in enumerate(grids):
            if abs(g["zr"] - zr_lev[j]) < 1e-9 and g["zmin"] - 1e-9 <= zs <= g["zmax"] + 1e-9:
                return gi
        raise ValueError(f"source depth {zs} outside the tensor grid of z_r {zr_lev[j]}")

    cache, corder, cbytes = {}, [], [0]
    cap = float(os.environ.get("SM_S2_CACHE_MB", "1024")) * 2**20

    def slot(s_):
        g = cache.get(s_)
        if g is None:
            t1 = perf_counter()
            g = np.ascontiguousarray(ds[s_][COMPS], dtype=np.float64)        # (8, nkeep) or every q
            c['read'] += perf_counter() - t1
            if q_dec > 1:
                t1 = perf_counter()
                g = np.ascontiguousarray(upsample(g, q_dec, fir, nkeep))
                nh = dsh.shape[-1]
                g[:, :nh] = dsh[s_][COMPS]
                c['upsample'] = c.get('upsample', 0.0) + perf_counter() - t1
            cache[s_] = g; corder.append(s_); cbytes[0] += g.nbytes
            while cbytes[0] > cap and len(corder) > 1:
                old = corder.pop(0); cbytes[0] -= cache.pop(old).nbytes
        return g

    crust_cache, trav = {}, {}

    def crust_for(z_src, z_rec):
        key = (float(z_src), float(z_rec))
        aux = crust_cache.get(key)
        if aux is None:
            aux = copy.deepcopy(model._crust); aux.split_at_depth(z_src); aux.split_at_depth(z_rec)
            crust_cache[key] = aux
        return aux

    tbdt = np.float32(np.float32(tb) * np.float32(dt))
    L2 = nkeep

    def place(ps, t0, nout):
        a0 = (0.0 + ps.tt) + np.float64(t0); a1 = (dt + ps.tt) + np.float64(t0)
        dti = a1 - a0
        m0, h = lagrange4_taps((a0 - tmin) / dti)
        if h is None:
            h = np.array([1.0, 0.0, 0.0, 0.0])
        k0 = max(0, -m0); kend = min(L2 + 3, nout - m0)
        return dti, m0, k0, kend - k0, min(L2, kend), h

    stf_cache = smod._STFCache()
    fused = _fused() if os.environ.get("SM_S2_FUSED", "1") == "1" else False
    kernel2 = _kernel2() if os.environ.get("SM_S2_KERNEL", "v2") == "v2" else False

    def _stage2_source_v2(ps, ids, x, Z2, Z1, a0, T2, T1, c1, s1):
        """One source, kernel v2: receivers grouped by (level, station step); per group the
        distance nodes it needs are read once, convolved once, and the kernel does the rest."""
        t1 = perf_counter()
        recs = {}
        jr = r_lev[ids]
        for j in np.unique(jr):
            sel_m = np.flatnonzero(jr == j)
            gi = grid_of(int(j), ps.x[2]); g = grids[gi]
            kf, W = stencil4(x[sel_m], g["dh"])
            for mi, m in enumerate(sel_m):
                t0 = smod._subtrav_t0(crust_for, ps.x[2], zr_lev[j], np.float32(x[m]), tbdt, trav)
                dti, m0, k0, nk, kp, h = place(ps, t0, int(nout[m]))
                if nk <= 0 or kp <= 0:
                    continue
                recs.setdefault((gi, dti), []).append(
                    (m, m0, k0, nk, kp, int(kf[mi]), W[mi], h,
                     (Z2[m], Z1[m], a0, T2[m], T1[m], c1[m], s1[m])))
        c['core'] += perf_counter() - t1
        for (gi, dti), rr in recs.items():
            g = grids[gi]
            zi, wz = stencil(ps.x[2], g["zs"], order)
            nd = len(g["dh"])
            kmin = min(r_[5] for r_ in rr); kmax = max(r_[5] for r_ in rr) + 3
            kmax = min(kmax, nd - 1)
            keep = max(r_[4] for r_ in rr)
            stf_r, scale = stf_cache.get(ps.stf, dti)
            CGt = np.zeros((kmax - kmin + 1 + 3, keep, 8))
            for k in range(kmin, kmax + 1):
                t1 = perf_counter()
                if len(zi) == 1 and wz[0] == 1.0:
                    gv = slot(g["off"] + int(zi[0]) * nd + k)                              # no copy
                else:
                    gv = sum(wz[q] * slot(g["off"] + int(zi[q]) * nd + k) for q in range(len(zi)))
                c['combine'] += perf_counter() - t1
                t1 = perf_counter()
                cg = gv[:, :keep] * scale if stf_r is None else \
                    fftconvolve(gv[:, :keep], stf_r[None, :], mode="full", axes=-1)[:, :keep] * dti
                CGt[k - kmin] = cg.T
                c['conv'] += perf_counter() - t1
            t1 = perf_counter()
            mem = np.array([r_[:4] for r_ in rr], dtype=np.int64)
            kf = np.array([r_[5] - kmin for r_ in rr], dtype=np.int64)
            W = np.array([r_[6] for r_ in rr], dtype=np.float64)
            H = np.array([r_[7] for r_ in rr], dtype=np.float64)
            R = np.array([r_[8] for r_ in rr], dtype=np.float64)
            kernel2(buf_ref[0], CGt, kf, W, R, mem, H, np.zeros((keep, 3)))
            c['add'] += perf_counter() - t1
    buf_ref = [None]
    nout0 = len(np.arange(tmin, tmax, dt))
    chunk = int(os.environ.get("SM_S2_CHUNK", "0") or 0) or max(1, int(1024 * 2**20 // (3 * (nout0 + 2) * 8)))
    chunk = min(chunk, nsta)
    lo, hi = (rank * nsrc) // nprocs, ((rank + 1) * nsrc) // nprocs
    for c0 in range(0, nsta, chunk):
        ids = np.arange(c0, min(c0 + chunk, nsta))
        # station grids: step of the pair with source 0, as the other Stage 2 paths
        dt_sta = np.empty(len(ids)); nout = np.empty(len(ids), dtype=np.int64)
        ps0 = srcs[0]
        for m, i in enumerate(ids):
            j = r_lev[i]
            x0 = np.hypot(ps0.x[0] - rx[i, 0], ps0.x[1] - rx[i, 1])
            t00 = smod._subtrav_t0(crust_for, ps0.x[2], zr_lev[j], np.float32(x0), tbdt, trav)
            dt_sta[m] = place(ps0, t00, 0)[0]
            nout[m] = len(np.arange(tmin, tmax, dt_sta[m]))
        buf = np.zeros((len(ids), 3, int(nout.max()) + 4))
        buf_ref[0] = buf
        for j_s in range(lo, hi):
            ps = srcs[j_s]
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
            Z2 = a2c*c2 + a2s*s2; Z1 = a1c*c1 + a1s*s1; T2 = a2c*s2 - a2s*c2; T1 = a1c*s1 - a1s*c1
            x = np.hypot(ps.x[0] - rx[ids, 0], ps.x[1] - rx[ids, 1])
            if kernel2 and order == 3:
                _stage2_source_v2(ps, ids, x, Z2, Z1, a0, T2, T1, c1, s1)
                continue
            groups = {}
            t1 = perf_counter()
            for m in range(len(ids)):
                j = r_lev[ids[m]]
                gi = grid_of(j, ps.x[2]); g = grids[gi]
                kidx, wd = stencil(x[m], g["dh"], order)
                t0 = smod._subtrav_t0(crust_for, ps.x[2], zr_lev[j], np.float32(x[m]), tbdt, trav)
                dti, m0, k0, nk, kp, h = place(ps, t0, int(nout[m]))
                if nk <= 0 or kp <= 0:
                    continue
                for q, k in enumerate(kidx):
                    if wd[q] == 0.0:
                        continue
                    groups.setdefault((gi, int(k), dti), []).append(
                        (m, m0, k0, nk, kp, h, wd[q] * Z2[m], wd[q] * Z1[m], wd[q] * a0,
                         wd[q] * T2[m], wd[q] * T1[m], c1[m], s1[m]))
            c['core'] += perf_counter() - t1
            zidx_cache = {}
            for (gi, k, dti), mem_ in groups.items():
                g = grids[gi]
                if gi not in zidx_cache:
                    zidx_cache[gi] = stencil(ps.x[2], g["zs"], order)
                zi, wz = zidx_cache[gi]
                t1 = perf_counter()
                nd = len(g["dh"])
                gv = sum(wz[q] * slot(g["off"] + int(zi[q]) * nd + k) for q in range(len(zi)))   # virtual node
                c['combine'] += perf_counter() - t1
                keep = max(mb_[4] for mb_ in mem_)
                stf_r, scale = stf_cache.get(ps.stf, dti)
                t1 = perf_counter()
                cg = gv[:, :keep] * scale if stf_r is None else \
                    fftconvolve(gv[:, :keep], stf_r[None, :], mode="full", axes=-1)[:, :keep] * dti
                c['conv'] += perf_counter() - t1
                t1 = perf_counter()
                mem = np.array([mb_[:4] for mb_ in mem_], dtype=np.int64)
                H = np.array([mb_[5] for mb_ in mem_], dtype=np.float64)
                K = np.array([mb_[6:13] for mb_ in mem_], dtype=np.float64)
                if fused:
                    fused(buf, K, np.ascontiguousarray(cg), mem, H)
                else:
                    for q, mb_ in enumerate(mem_):
                        Z2q, Z1q, A0q, T2q, T1q, cq, sq = mb_[6:13]
                        zv = Z2q * cg[5] + Z1q * cg[2] - A0q * cg[0]
                        rv = Z2q * cg[6] + Z1q * cg[3] - A0q * cg[1]
                        tv = T2q * cg[7] + T1q * cg[4]
                        for cc_, comp in enumerate((zv, -sq * rv - cq * tv, -cq * rv + sq * tv)):
                            zz = np.convolve(comp[:keep], mb_[5])
                            buf[mb_[0], cc_, mb_[1] + mb_[2]:mb_[1] + mb_[2] + mb_[3]] += zz[mb_[2]:mb_[2] + mb_[3]]
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
                st = stations[i]; n_ = int(nout[m])
                st.add_to_response(tot[m, 0, :n_], tot[m, 1, :n_], tot[m, 2, :n_],
                                   tmin + np.arange(n_) * dt_sta[m], tmin, tmax)
                if writer:
                    smod._timed_write(writer, st, int(i), c, t_begin)
                if writer_mode == 'progressive' and writer:
                    st.clear_response()
            c['recv'] += perf_counter() - t1
    hm.close(); hg.close()
    if rank == 0 and writer:
        writer.close()
    total = perf_counter() - t_begin
    if rank == 0 and showProgress:
        print(f"[Stage 2 tensor] {nsta} stations x {nsrc} sources: {total:.1f} s (read {c['read']:.1f}, "
              f"combine {c['combine']:.1f}, conv {c['conv']:.1f}, add {c['add']:.1f})", flush=True)
    smod._perf_dump("stage2_tensor", {k: [v] for k, v in c.items()}, total)
    return c
