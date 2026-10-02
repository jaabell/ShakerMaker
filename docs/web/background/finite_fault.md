# Finite faults & FFSP

From a single point to an extended rupture, and from one deterministic
rupture to a stochastic ensemble.

## Point source → finite fault

A real earthquake is not a point: it is slip over an extended fault plane.
The finite-fault response is the **linear superposition** of many point
sources (subfaults), each with its own location, mechanism, slip amount, and
**rupture time**, the instant the rupture front reaches it:

$$
u_n(\mathbf{x}_r, t) \;=\; \sum_{\alpha=1}^{N_\text{sub}}
\; G_{ni}(\mathbf{x}_r,\, t - t_{0,\alpha};\, \mathbf{\xi}_\alpha)\,
* \, \dot{M}_{ij,\alpha}(t)\,\hat n_j
$$

Each subfault is a [`PointSource`](../guides/sources.md); the collection is a
[`FaultSource`](../guides/sources.md). The rupture-time delay $t_{0,\alpha}$
is the `tt` argument on each `PointSource`, and the moment release shape is
its [source time function](../guides/source_time_functions.md). Get the
rupture timing and the slip distribution right and the superposition does the
rest.

The resolution rule: subfaults must be small versus the shortest wavelength,
$\Delta\xi,\Delta\eta \lesssim V_S^\text{min} / (N_p\,f_\text{max})$ with
$N_p \approx 5$ subfaults per wavelength.

## How big is the fault? Magnitude–area scaling

Before you can lay out subfaults you need the fault dimensions $L\times W$, and
those follow from the target magnitude through an empirical **scaling law**. The
classic regressions relate the rupture area $A = L\,W$ (km²) to the moment
magnitude $M_w$ as a log-linear fit:

$$
M_w = a + b\,\log_{10} A.
$$

| Relation | Form | Notes |
|---|---|---|
| **Wells & Coppersmith (1994)** | $M_w = 4.07 + 0.98\,\log_{10} A$ | the canonical global fit (all rupture types) |
| **Leonard (2010)** | $\log_{10} A = M_w - 4.0$ (i.e. $M_w = 4.0 + \log_{10}A$) | self-consistent moment / area / slip scaling |

These let you turn a scenario magnitude into a physically plausible
$L\times W$, the aspect ratio (and hence individual $L$, $W$) coming from the
same regressions or from a fixed $W$ once the seismogenic depth is reached. The
scalar moment closes the loop, $M_0 = 10^{1.5\,M_w + 9.1}$ N·m, and the average
slip then follows from $M_0 = \mu\,A\,\bar D$.

## Why stochastic? (FFSP)

The slip distribution of a *future* earthquake is unknowable, even the
best-recorded past events admit many slip inversions that fit the data
equally well (~30% irreducible uncertainty in local slip). The honest
forward approach is therefore an **ensemble**: generate many physically
admissible ruptures, compute each, and report a *distribution* of ground
motions rather than a single number.

The **Finite Fault Stochastic Process (FFSP)** tool does exactly this. It
splits the rupture into:

- **Constrained (deterministic):** moment / magnitude, fault dimensions
  $L\times W$, orientation (strike, dip, rake), hypocentre, target corner
  frequencies.
- **Random fields:** roughly **eight coupled stochastic fields** laid down on
  the fault plane, each with magnitude-scaled correlation lengths and
  prescribed cross-correlations to slip.

Following Liu–Archuleta–Hartzell (with Ji's refinements), the eight fields are:

| Field | What it controls |
|---|---|
| **Slip** $D(\xi,\eta)$ | where and how much the fault moves — the asperities |
| **Rupture time** $t_0$ | front arrival, from the eikonal solution + perturbations |
| **Rise time** $\tau_r$ | how long each point keeps slipping |
| **Peak time** $\tau_p$ | when the slip-rate peaks |
| **Rupture velocity** $v_r$ | how fast the front propagates (bounded, depth-dependent) |
| **Strike perturbation** | local along-strike mechanism scatter |
| **Dip perturbation** | local dip scatter (`pdip_max`) |
| **Rake perturbation** | local rake scatter (`prake_max`) |

The slip-rate *shape* (the `id_sf_type` slip-rate function) and the seeds round
out the controls: everything in the [`FFSPSource`](../guides/ffsp.md)
constructor either **fixes** one of these (the constrained inputs) or **shapes
its randomness** (the stochastic controls).

Each realisation is scored against the targets (a PDF score); you keep either
the single **best** realisation (deterministic analysis) or the **full
ensemble** (probabilistic analysis).

In ShakerMaker this is the [`FFSPSource`](../guides/ffsp.md) class, every
constrained quantity and every random-field control is a constructor
argument.

## The efficiency payoff

For an FFSP ensemble over a fixed geometry, the **FK Green's functions are
shared across all realisations** — only the slip distribution and rupture times
change between realisations, while the source–receiver geometry (and hence the
nine elementary Green's functions of every slot) stays identical. ShakerMaker
exploits this by computing the Green's functions **once** (the geometry-mapping
and kernel stages) and re-running only the cheap per-pair recombination +
convolution per realisation. The amortised cost of an extra realisation is
therefore dominated by the recombination, not the kernel, which is what makes a
50–500-member probabilistic study tractable — see the
[OP pipeline](../guides/running.md#the-op-pipeline-run_nearest).

## References

- Aki, K. & Richards, P. G. (2002). *Quantitative Seismology*, §§4.3, 10.
- Liu, P., Archuleta, R. J. & Hartzell, S. (2006). *BSSA* **96**, 2118–2130.
- Graves, R. & Pitarka, A. (2010, 2016). *BSSA*.
- Atkinson, G. (1993), double-corner-frequency source spectrum.
