# Overview

This section introduces the physics and mathematics behind ShakerMaker, just
enough to use the tool with understanding. It is a practical primer, not a
derivation; the four pages build up as:

1. **Overview** *(this page)*, the physical problem and where ShakerMaker sits.
2. [The FK method](fk_method.md), how the wave equation is actually solved.
3. [Numerical solution](numerics.md), turning the integrals into numbers, and the parameters that control it.
4. [Coordinates & conventions](conventions.md), axes, the z-up→z-down flip, units, angles.

## The physical problem

A point earthquake source buried in the Earth radiates seismic waves. We want
the **three-component ground motion** that source produces at one or many
receivers. ShakerMaker solves this for a specific, and very useful, idealised
Earth: a **horizontally layered, isotropic, anelastic half-space**.

![SCEC LOH.1 layered profile](../assets/images/crust_loh1.png){ width=420 }

Each layer has a P-wave velocity $V_P$, an S-wave velocity $V_S$, a density
$\rho$, and quality factors $Q_P, Q_S$ that set the anelastic attenuation. This
1-D layering is the assumption that makes the problem tractable in closed form
,  and it is an excellent model for sedimentary basins, regional crust, and the
near-surface where engineering ground motion matters.

## What "complete wavefield" means

ShakerMaker does **not** use the far-field ray approximation. It computes the
*complete* elastodynamic response, every term of the Green's function:

- **Body waves**, direct P, direct S, and all their reflections/refractions off
  the layer interfaces.
- **Surface waves**, Rayleigh (on the vertical/radial components) and Love (on
  the transverse), which dominate regional and basin ground motion.
- **Near-field terms**, the static and intermediate-field contributions that
  matter close to the source.

All of these emerge from a single mathematical object: the **layered-medium
Green's function**, computed by the frequency–wavenumber (FK) method.

## The four ingredients

Every simulation is built from exactly four things, the same four that
organise the [guides](../guides/first_steps.md):

| Ingredient | Physical object | ShakerMaker |
|---|---|---|
| **Medium** | 1-D layered half-space | [`CrustModel`](../guides/crust_model.md) |
| **Source** | point dislocation (moment tensor), or a sum of them | [`PointSource` / `FaultSource`](../guides/sources.md) |
| **Source time function** | how the moment is released in time | [STFs](../guides/source_time_functions.md) |
| **Receiver** | recording location(s) | [`Station`, `DRMBox`, …](../guides/receivers.md) |

…plus the **numerical parameters** of the FK integration, which are not free
knobs but discretisations of the underlying sampling theory, covered in
[Numerical solution](numerics.md).

## Where ShakerMaker sits scientifically

The FK method has a seven-decade lineage. ShakerMaker wraps **Lupei Zhu's**
`fk` Fortran code, which implements the modern, numerically stable line of that
lineage:

| Generation | Idea | Key references |
|---|---|---|
| 1950s–60s | Thomson–Haskell **propagator matrix** | Thomson (1950), Haskell (1953, 1964) |
| 1960s–80s | **compound-matrix** reformulation (numerically stable) | Knopoff (1964), Dunkin (1965), Wang & Herrmann (1980) |
| 2000s– | **unified static–dynamic** formulation | Hisada (1994/95), **Zhu & Rivera (2002)** |

The discrete-wavenumber integration follows **Bouchon (1981)**; the
decomposition into nine elementary Green's functions follows **Helmberger
(1983)**. ShakerMaker adds the modern engineering layer on top: DRM boxes, an
MPI pipeline, and H5DRM/SW4 export.

Continue to [**The FK method →**](fk_method.md).
