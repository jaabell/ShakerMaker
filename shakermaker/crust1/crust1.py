"""CRUST 1.0 global crustal model reader (Laske et al., 2013).

Ships the 1deg x 1deg CRUST 1.0 grids (9 layers: water, ice, sediments,
crystalline crust, mantle) as package data so they travel with ``pip install``.

:class:`Crust1` loads the grids next to this file and exposes:

- Attributes: ``vp``, ``vs``, ``rho``, ``bnds`` (per layer) and ``topography_km``.
- Queries: ``profile_at`` (lat/lon -> layered column), ``type_at`` (geological
  type), ``cell_index`` / ``cell_midpoint``.
- ShakerMaker glue: ``print_shakermaker`` (ready-to-paste ``CrustModel`` snippet),
  ``print_tables``.
- Plotting (matplotlib, imported lazily): ``plot_profile``, ``plot_global_topo``,
  ``plot_regional_topo``, ``plot_regional_geological``, ``plot_global_velocity``,
  ``plot_stacked_columns``.

``__init__`` reexports :class:`Crust1` and prints ``CRUST1_CITATION`` on import;
please cite Laske et al. (2013) if you use the data.
"""

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


CRUST1_CITATION = (
    "CRUST 1.0 crustal model - Laske, G., Masters, G., Ma, Z. & Pasyanos, M. "
    "(2013), Update on CRUST1.0. Data: https://igppweb.ucsd.edu/~gabi/crust1.html "
    "(obtained there and adapted for use within ShakerMaker; please cite the source)."
)


LAYER_NAMES = ['water', 'ice', 'upper sed', 'middle sed', 'lower sed',
               'upper crust', 'middle crust', 'lower crust', 'mantle']

LAYER_COLORS = plt.cm.Pastel1(np.linspace(0, 1, 8))   # one per thickness layer
MANTLE_COLOR = (0.85, 0.85, 0.85, 1.0)               # light gray for half-space

GEOLOGICAL_GROUPS = [
    ('oceanic',              '#1849b5'),
    ('young oceanic',        '#5b8fe0'),
    ('anomalous oc.',        '#a0b8d6'),
    ('cont. shelf',          '#88c577'),
    ('cont. plateau',        '#bce17c'),
    ('thinned cont.',        '#f6c87a'),
    ('extended crust',       '#d4a45a'),
    ('inactive ridge',       '#7d99a8'),
    ('transition',           '#ddb88e'),
    ('platform',             '#e2a87f'),
    ('archean',              '#a3593f'),
    ('early/mid proteroz.',  '#cd9b6c'),
    ('late proteroz.',       '#d7a99a'),
    ('island arc',           '#9b6cd1'),
    ('forearc',              '#b794d6'),
    ('cont. arc',            '#c75a4a'),
    ('orogen',               '#e02828'),
    ('rift',                 '#222222'),
    ('phanerozoic',          '#f06d4a'),
    ('Caspian/Black sea',    '#00c8c8'),
]

CODE_TO_GROUP = {
    'A0': 'young oceanic', 'A1': 'oceanic',
    'B-': 'anomalous oc.',
    'C-': 'cont. shelf',
    'D-': 'platform', 'E-': 'platform',
    'F-': 'archean', 'G1': 'archean', 'G2': 'archean',
    'H1': 'early/mid proteroz.', 'H2': 'early/mid proteroz.',
    'I1': 'late proteroz.', 'I2': 'late proteroz.',
    'J-': 'island arc',
    'K-': 'forearc',
    'L1': 'cont. arc', 'L2': 'cont. arc',
    'M-': 'extended crust', 'N-': 'extended crust',
    'O-': 'orogen', 'P-': 'orogen', 'Q-': 'orogen',
    'R1': 'orogen', 'R2': 'orogen',
    'S-': 'transition', 'T-': 'transition', 'U-': 'transition',
    'V1': 'inactive ridge', 'V2': 'thinned cont.',
    'W-': 'cont. plateau',
    'X-': 'rift',
    'Y1': 'Caspian/Black sea', 'Y2': 'Caspian/Black sea',
    'Y3': 'Caspian/Black sea',
    'Z1': 'phanerozoic', 'Z2': 'phanerozoic',
}

SITE_COLORS = plt.cm.tab10.colors


class Crust1:
    """CRUST 1.0 global crustal model at 1deg x 1deg (Laske et al., 2013).

    Load with ``Crust1()`` (zero-config: finds the grids next to this file) and
    query by latitude/longitude, or dump a ready-to-paste ``CrustModel`` snippet.
    See the module docstring for the full attribute/method list.
    """

    BENCHMARK_SITES = [
        (-33.420003,  -70.606470, "Providencia, Santiago"),
        ( 37.771670, -122.508652, "San Francisco"),
        ( 46.231455,    6.055684, "Meyrin / CERN"),
    ]

    # ------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------
    def __init__(self, data_dir=None):
        self.data_dir = (Path(data_dir) if data_dir else
                         Path(__file__).parent / "crust1.0")
        self.nlat, self.nlon, self.nlayers = 180, 360, 9
        self.cell_size_deg = 1.0
        self.layer_names = LAYER_NAMES
        self.lats = 89.5 - np.arange(self.nlat)
        self.lons = -179.5 + np.arange(self.nlon)
        self._load_arrays()
        self._compute_derived()
        self._load_types()

    def _load_arrays(self):
        self.vp   = self._read('crust1.vp')
        self.vs   = self._read('crust1.vs')
        self.rho  = self._read('crust1.rho')
        self.bnds = self._read('crust1.bnds')

    def _read(self, name):
        arr = np.loadtxt(self.data_dir / name)
        return arr.reshape(self.nlat, self.nlon, self.nlayers)

    def _compute_derived(self):
        b = self.bnds
        self.topography_km         = b[:, :, 1]   # ocean-aware (negative=ocean)
        self.column_top_km         = b[:, :, 0]
        self.water_thickness       = b[:, :, 0] - b[:, :, 1]
        self.ice_thickness         = b[:, :, 1] - b[:, :, 2]
        self.sediment_thickness    = b[:, :, 2] - b[:, :, 5]
        self.crystalline_thickness = b[:, :, 5] - b[:, :, 8]
        self.crust_thickness       = b[:, :, 1] - b[:, :, 8]
        self.moho_elevation        = b[:, :, 8]
        self.moho_depth            = -b[:, :, 8]

    def _load_types(self):
        codes_p = self.data_dir / "CNtype1-1.txt"
        key_p   = self.data_dir / "CNtype1_key.txt"
        if not (codes_p.exists() and key_p.exists()):
            self.codes = None
            self.type_catalog = None
            self.has_types = False
            return
        self.codes = np.loadtxt(codes_p, dtype=str)
        with open(key_p) as f:
            lines = f.readlines()
        cat = {}
        for ln in lines[5:]:
            if ':' in ln and not ln.startswith((' ', '\t')):
                code = ln[:2]
                cat[code] = {'name': ln[3:].strip(),
                             'group': CODE_TO_GROUP.get(code, 'unknown')}
        self.type_catalog = cat
        self.has_types = True

    # ------------------------------------------------------------
    # Single-point lookup
    # ------------------------------------------------------------
    def cell_index(self, lat, lon):
        """(row, col) in the (180, 360) grid for any lat/lon."""
        if lon > 180: lon -= 360
        j = max(0, min(self.nlat - 1, int(90.0 - lat)))
        i = max(0, min(self.nlon - 1, int(180.0 + lon)))
        return j, i

    def cell_midpoint(self, lat, lon):
        """Midpoint (lat, lon) of the 1deg cell containing the point."""
        j, i = self.cell_index(lat, lon)
        return float(self.lats[j]), float(self.lons[i])

    def profile_at(self, lat, lon):
        """Full 9-layer profile at a point. Returns a dict."""
        j, i = self.cell_index(lat, lon)
        bnds = self.bnds[j, i, :].copy()
        thi = bnds[:-1] - bnds[1:]
        vp, vs, rho = (self.vp[j, i, :], self.vs[j, i, :], self.rho[j, i, :])

        t = thi[1:8]
        nz = t > 0
        if nz.any():
            avg_vp = float(t[nz].sum() / np.sum(t[nz] / vp[1:8][nz]))
            ok = nz & (vs[1:8] > 0)
            avg_vs = (float(t[ok].sum() / np.sum(t[ok] / vs[1:8][ok]))
                      if ok.any() else float('nan'))
            avg_rho = float((t[nz] * rho[1:8][nz]).sum() / t[nz].sum())
        else:
            avg_vp = avg_vs = avg_rho = float('nan')

        return {
            'row': j, 'col': i,
            'cell_lat': float(self.lats[j]), 'cell_lon': float(self.lons[i]),
            'bnds': bnds, 'thickness': thi,
            'vp': vp.copy(), 'vs': vs.copy(), 'rho': rho.copy(),
            'topography_km': float(bnds[0]),
            'moho_elevation_km': float(bnds[-1]),
            'moho_depth_km': float(-bnds[-1]),
            'sediment_thickness_km': float(thi[2] + thi[3] + thi[4]),
            'crystalline_thickness_km': float(thi[5] + thi[6] + thi[7]),
            'crust_thickness_km': float(thi[1:].sum()),
            'avg_vp': avg_vp, 'avg_vs': avg_vs, 'avg_rho': avg_rho,
        }

    def type_at(self, lat, lon):
        """Crustal type code + name + group, or None if addon missing."""
        if not self.has_types:
            return None
        j, i = self.cell_index(lat, lon)
        code = self.codes[j, i]
        info = self.type_catalog.get(code)
        if info is None:
            return None
        return {'code': code, **info}

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _norm(self, sites):
        if sites is None:
            return []
        if (isinstance(sites, (tuple, list)) and len(sites) in (2, 3)
                and not isinstance(sites[0], (tuple, list))):
            sites = [sites]
        out = []
        for i, s in enumerate(sites, 1):
            if len(s) == 2:
                lat, lon = s; label = f"Site {i}"
            elif len(s) == 3:
                lat, lon, label = s
            else:
                raise ValueError(f"site {i}: need 2 or 3 elements")
            out.append((float(lat), float(lon), str(label)))
        return out

    def _zoom(self, sites, zoom_deg):
        if zoom_deg != 'auto':
            return float(zoom_deg)
        if len(sites) == 1:
            return 8.0
        lats = [s[0] for s in sites]; lons = [s[1] for s in sites]
        span = max(max(lats) - min(lats), max(lons) - min(lons))
        return max(span / 2 + 2.0, 4.0)

    def _bbox(self, sites, zoom):
        lats = [s[0] for s in sites]; lons = [s[1] for s in sites]
        lat_c = (max(lats) + min(lats)) / 2
        lon_c = (max(lons) + min(lons)) / 2
        return (max(-90, lat_c - zoom), min(90, lat_c + zoom),
                max(-180, lon_c - zoom), min(180, lon_c + zoom))

    def _slice(self, grid, bbox):
        lat_lo, lat_hi, lon_lo, lon_hi = bbox
        j_lo = max(0, min(179, int(90 - lat_hi)))
        j_hi = max(0, min(179, int(90 - lat_lo)))
        i_lo = max(0, min(359, int(180 + lon_lo)))
        i_hi = max(0, min(359, int(180 + lon_hi)))
        sub = grid[j_lo:j_hi + 1, i_lo:i_hi + 1]
        extent = [-180 + i_lo, -180 + i_hi + 1,
                  90 - (j_hi + 1), 90 - j_lo]
        return sub, extent

    def _gather(self, sites):
        runs = []
        for i, (lat, lon, label) in enumerate(sites):
            runs.append({
                'lat': lat, 'lon': lon, 'label': label,
                'profile': self.profile_at(lat, lon),
                'type': self.type_at(lat, lon),
                'color': SITE_COLORS[i % len(SITE_COLORS)],
            })
        return runs

    def _mark(self, ax, runs, size=10, lw=2.0, with_label=False):
        for r in runs:
            ax.plot(r['lon'], r['lat'], 'x', color=r['color'],
                    markersize=size, markeredgewidth=lw,
                    label=r['label'] if with_label else None)

    def _finalize(self, fig, show, save_path):
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def _title(self, base, runs):
        if not runs:
            return base
        if len(runs) == 1:
            r = runs[0]
            return f"{base}  -  {r['label']}  ({r['lat']:.2f}, {r['lon']:.2f})"
        return f"{base}  -  {len(runs)} sites"

    def _step(self, p, key, mantle_extra=20.0):
        """Step-plot x/y for a profile column. Returns (xs, ys, moho)."""
        thi = p['thickness']
        b = np.concatenate([[0.0], np.cumsum(thi)])
        moho = float(b[-1])
        xs, ys = [], []
        for k in range(len(thi)):
            if thi[k] <= 0:
                continue
            xs += [p[key][k], p[key][k]]
            ys += [b[k], b[k + 1]]
        xs += [p[key][8], p[key][8]]
        ys += [moho, moho + mantle_extra]
        return xs, ys, moho

    # ------------------------------------------------------------
    # Text output
    # ------------------------------------------------------------
    def print_tables(self, sites):
        """Print one descriptive profile table per site, plus a comparison
        row when N > 1.  Text only - no ShakerMaker output here."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        for r in runs:
            self._print_one(r)
        if len(runs) > 1:
            self._print_compare(runs)

    def print_shakermaker(self, sites, *, Qa=1000.0, Qb=1000.0):
        """Print a ready-to-paste ShakerMaker `CrustModel` snippet per site.
        Water and ice are emitted commented out; zero-thickness layers are
        skipped; the mantle is the half-space (thick=0.000)."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        for r in runs:
            self._print_shakermaker(r, Qa=Qa, Qb=Qb)

    def _print_one(self, r):
        p, t = r['profile'], r['type']
        print()
        print("=" * 68)
        print(f"  {r['label']}")
        print(f"  Input:  lat = {r['lat']:+.4f}, lon = {r['lon']:+.4f}")
        print(f"  Cell:   midpoint ({p['cell_lat']:+.1f}, {p['cell_lon']:+.1f})")
        if t:
            print(f"  Type:   {t['code']}  ({t['name']})")
        print(f"  Moho:   {p['moho_depth_km']:.2f} km below sea level")
        print(f"  Crust:  {p['crust_thickness_km']:.2f} km (no water)")
        print(f"  Avg:    Vp = {p['avg_vp']:.2f}  "
              f"Vs = {p['avg_vs']:.2f}  rho = {p['avg_rho']:.2f}")
        print()
        print(f"  {'Layer':<14} {'thick(km)':>10} {'Vp(km/s)':>10} "
              f"{'Vs(km/s)':>10} {'rho(g/cc)':>10}")
        print("  " + "-" * 60)
        for k, name in enumerate(self.layer_names):
            if k < 8:
                t_str = (f"{p['thickness'][k]:>10.3f}"
                         if p['thickness'][k] > 0 else f"{'-':>10}")
            else:
                t_str = f"{'inf':>10}"
            print(f"  {name:<14} {t_str} "
                  f"{p['vp'][k]:>10.3f} {p['vs'][k]:>10.3f} "
                  f"{p['rho'][k]:>10.3f}")

    def _print_shakermaker(self, run, Qa=1000.0, Qb=1000.0):
        """Print a ready-to-paste ShakerMaker CrustModel snippet for one site.
        Water and ice are emitted commented out; zero-thickness layers are
        skipped; mantle is the half-space (thick=0.000)."""
        p = run['profile']
        layer_info = [
            ('water - skipped (ShakerMaker FK fails with Vs=0)',  0),
            ('ice - skipped',                                     1),
            ('upper sediments',                                   2),
            ('middle sediments',                                  3),
            ('lower sediments',                                   4),
            ('upper crystalline crust',                           5),
            ('middle crystalline crust',                          6),
            ('lower crystalline crust',                           7),
        ]
        n_active = 1 + sum(1 for _, k in layer_info[2:]
                           if p['thickness'][k] > 0)

        print()
        print(f"  ----- ShakerMaker CrustModel snippet for {run['label']} -----")
        print()
        print(f"crust = CrustModel({n_active})")
        for comment, k in layer_info:
            thi = p['thickness'][k]
            if k < 2:
                pre = "# "   # water and ice always commented
            else:
                if thi <= 0:
                    continue   # skip zero-thickness real layers
                pre = ""
            print(f"# {comment}")
            print(f"{pre}vp, vs, rho, thick, Qa, Qb = "
                  f"{p['vp'][k]:.2f}, {p['vs'][k]:.2f}, {p['rho'][k]:.2f}, "
                  f"{thi:.3f}, {Qa:.1f}, {Qb:.1f}")
            print(f"{pre}crust.add_layer(thick, vp, vs, rho, Qa, Qb)")
        # mantle half-space
        print(f"# mantle half-space")
        print(f"vp, vs, rho, thick, Qa, Qb = "
              f"{p['vp'][8]:.2f}, {p['vs'][8]:.2f}, {p['rho'][8]:.2f}, "
              f"0.000, {Qa:.1f}, {Qb:.1f}")
        print(f"crust.add_layer(thick, vp, vs, rho, Qa, Qb)")

    def _print_compare(self, runs):
        print()
        print("=" * 78)
        print("  Comparison")
        print(f"  {'Label':<22} {'Moho':>7} {'Crust':>7} {'Sed':>7} "
              f"{'Vp':>6} {'Vs':>6} {'Type':>6}")
        print("  " + "-" * 70)
        for r in runs:
            p, t = r['profile'], r['type']
            print(f"  {r['label'][:22]:<22} "
                  f"{p['moho_depth_km']:>7.2f} "
                  f"{p['crust_thickness_km']:>7.2f} "
                  f"{p['sediment_thickness_km']:>7.2f} "
                  f"{p['avg_vp']:>6.2f} "
                  f"{p['avg_vs']:>6.2f} "
                  f"{(t['code'] if t else '-'):>6}")

    # ============================================================
    # PLOTS  (each returns a Figure; show=True opens window;
    #         save_path=str saves PNG.)
    # ============================================================
    def plot_profile(self, sites, *, figsize=(12, 6.5),
                      halfspace_extra=20.0, show=True, save_path=None):
        """Vp, Vs, rho vs depth.

        Single site: layer-coloured backgrounds + thickness labels.
        Multi-site:  one curve per site, no backgrounds.
        """
        sites = self._norm(sites)
        runs = self._gather(sites)
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        line_colors = ['C0', 'C3', 'C2']
        keys = ['vp', 'vs', 'rho']
        labels = ['Vp (km/s)', 'Vs (km/s)', r'$\rho$ (g/cm$^3$)']

        if len(runs) == 1:
            p = runs[0]['profile']
            thi = p['thickness']
            z_top = np.concatenate([[0.0], np.cumsum(thi)])
            bottom = float(z_top[-1] + halfspace_extra)
            for ax, key, lab, lc in zip(axes, keys, labels, line_colors):
                handles = []
                for k in range(8):
                    if thi[k] <= 0:
                        continue
                    h = ax.axhspan(z_top[k], z_top[k + 1],
                                   facecolor=LAYER_COLORS[k], alpha=0.5,
                                   label=f'{self.layer_names[k]}: {thi[k]:.2f} km')
                    handles.append(h)
                handles.append(ax.axhspan(z_top[-1], bottom,
                                          facecolor=MANTLE_COLOR, alpha=0.5,
                                          label='mantle: inf'))
                xs, ys, _ = self._step(p, key, mantle_extra=halfspace_extra)
                ax.plot(xs, ys, color=lc, lw=2)
                ax.set_xlabel(lab); ax.grid(alpha=0.3)
                ax.set_ylim(bottom, 0)
            axes[2].legend(handles=handles, loc='center left',
                           bbox_to_anchor=(1.02, 0.5), fontsize=9,
                           frameon=False, title='Layer', title_fontsize=10)
        else:
            y_max = max(float(np.cumsum(r['profile']['thickness']).max())
                        for r in runs) + halfspace_extra
            for ax, key, lab in zip(axes, keys, labels):
                for r in runs:
                    xs, ys, moho = self._step(r['profile'], key,
                                              mantle_extra=halfspace_extra)
                    ax.plot(xs, ys, color=r['color'], lw=2, label=r['label'])
                    ax.axhline(moho, color=r['color'], lw=0.6,
                               ls=':', alpha=0.6)
                ax.set_xlabel(lab); ax.grid(alpha=0.3)
            axes[0].set_ylim(y_max, 0)
            axes[0].legend(fontsize=9, loc='lower right')

        axes[0].set_ylabel("Depth below surface (km)")
        fig.suptitle(self._title("Vp / Vs / rho profile", runs))
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

    def plot_global_topo(self, sites=None, *, figsize=(11, 5.5),
                          show=True, save_path=None):
        """Global topography / bathymetry, sites marked if given."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.topography_km, extent=[-180, 180, -90, 90],
                       origin='upper', cmap='terrain', aspect='auto')
        self._mark(ax, runs, size=12, lw=2.5, with_label=bool(runs))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("CRUST 1.0  -  Global topography / bathymetry")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='km')
        if runs:
            ax.legend(fontsize=8, loc='lower left')
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

    def plot_regional_topo(self, sites, *, zoom_deg='auto', figsize=(9, 7),
                           show=True, save_path=None):
        """Regional topography zoom with 1deg grid + cell outline."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        zoom = self._zoom(sites, zoom_deg)
        sub, ext = self._slice(self.topography_km, self._bbox(sites, zoom))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(sub, extent=ext, origin='upper', cmap='terrain',
                       aspect='auto', interpolation='nearest')
        for x in range(int(np.ceil(ext[0])), int(np.floor(ext[1])) + 1):
            ax.axvline(x, color='gray', lw=0.3, alpha=0.4)
        for y in range(int(np.ceil(ext[2])), int(np.floor(ext[3])) + 1):
            ax.axhline(y, color='gray', lw=0.3, alpha=0.4)
        for r in runs:
            ax.plot(r['lon'], r['lat'], 'x', color=r['color'],
                    markersize=14, markeredgewidth=2.5, label=r['label'])
            ax.add_patch(plt.Rectangle((math.floor(r['lon']),
                                        math.floor(r['lat'])), 1, 1,
                                       fill=False, edgecolor=r['color'],
                                       lw=1.8, ls='--'))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(self._title("Regional topography (1deg grid)", runs))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='km')
        if len(runs) > 1:
            ax.legend(fontsize=9, loc='lower right')
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

    def plot_regional_geological(self, sites, *, zoom_deg='auto',
                                  figsize=(11, 7),
                                  show=True, save_path=None):
        """Regional categorical map of CRUST 1.0 type groups."""
        if not self.has_types:
            raise RuntimeError("type addon missing (CNtype1-1.txt)")
        sites = self._norm(sites)
        runs = self._gather(sites)
        zoom = self._zoom(sites, zoom_deg)
        sub_codes, ext = self._slice(self.codes, self._bbox(sites, zoom))

        group_index = {g: i for i, (g, _) in enumerate(GEOLOGICAL_GROUPS)}
        group_colors = [c for _, c in GEOLOGICAL_GROUPS]
        idx = np.full(sub_codes.shape, -1, dtype=int)
        present = set()
        for j in range(sub_codes.shape[0]):
            for i in range(sub_codes.shape[1]):
                c = sub_codes[j, i]
                if c in self.type_catalog:
                    g = self.type_catalog[c]['group']
                    gi = group_index.get(g, -1)
                    if gi >= 0:
                        idx[j, i] = gi
                        present.add(g)
        masked = np.ma.masked_less(idx, 0)
        cmap = ListedColormap(group_colors); cmap.set_bad('white')

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(masked, extent=ext, origin='upper', cmap=cmap,
                  vmin=-0.5, vmax=len(GEOLOGICAL_GROUPS) - 0.5,
                  aspect='auto', interpolation='nearest')
        for r in runs:
            ax.plot(r['lon'], r['lat'], 'x', color=r['color'],
                    markersize=14, markeredgewidth=2.5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(self._title("Regional geological types", runs))
        patches = [mpatches.Patch(color=group_colors[group_index[g]], label=g)
                   for g, _ in GEOLOGICAL_GROUPS if g in present]
        ax.legend(handles=patches, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), fontsize=9,
                  frameon=False, title="Group", title_fontsize=10)
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

    def plot_global_velocity(self, sites=None, *, layer=5,
                              figsize=(18, 5),
                              show=True, save_path=None):
        """Vp, Vs, rho global maps for one layer (3 subplots)."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, key, lab, cm in zip(axes,
                                    ['vp', 'vs', 'rho'],
                                    ['Vp (km/s)', 'Vs (km/s)',
                                     r'$\rho$ (g/cm$^3$)'],
                                    ['plasma', 'viridis', 'cividis']):
            data = getattr(self, key)[:, :, layer]
            masked = np.ma.masked_equal(data, 0.0)
            im = ax.imshow(masked, extent=[-180, 180, -90, 90],
                           origin='upper', cmap=cm, aspect='auto',
                           interpolation='nearest')
            self._mark(ax, runs, size=8, lw=1.8)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.set_title(lab)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle(f"CRUST 1.0  -  global {self.layer_names[layer]} "
                     f"(layer {layer})", y=1.02)
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

    def plot_stacked_columns(self, sites, *, figsize=(9, 6),
                              show=True, save_path=None):
        """Layer-thickness stratigraphic columns, one per site."""
        sites = self._norm(sites)
        runs = self._gather(sites)
        short = ['water', 'ice', 'u sed', 'm sed', 'l sed',
                 'u crust', 'm crust', 'l crust']
        fig, ax = plt.subplots(figsize=figsize)
        for i, r in enumerate(runs):
            thi = r['profile']['thickness']
            y = 0
            for k in range(8):
                if thi[k] > 0:
                    ax.bar(i, thi[k], width=0.6, bottom=y,
                           color=LAYER_COLORS[k], edgecolor='black',
                           linewidth=0.5)
                    if thi[k] > 2.0:
                        ax.text(i, y + thi[k] / 2, short[k],
                                ha='center', va='center', fontsize=8)
                    y += thi[k]
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r['label'] for r in runs], rotation=20, ha='right')
        for tl, r in zip(ax.get_xticklabels(), runs):
            tl.set_color(r['color']); tl.set_fontweight('bold')
        ax.set_ylabel("Depth below surface (km)")
        ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.3)
        ax.set_title("CRUST 1.0  -  Layer thicknesses by site")
        patches = [mpatches.Patch(color=LAYER_COLORS[k], label=short[k])
                   for k in range(8)]
        ax.legend(handles=patches, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), fontsize=9,
                  frameon=False, title="Layer", title_fontsize=10)
        fig.tight_layout()
        return self._finalize(fig, show, save_path)

