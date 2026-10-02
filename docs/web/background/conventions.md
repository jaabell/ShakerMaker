# Coordinates & conventions

ShakerMaker inherits its engine from a seismology code but exposes an
**engineering** convention to the user. The two differ in the sign of the
vertical axis, and getting this right is the difference between a sensible
seismogram and an upside-down one. This page is the single reference for axes,
units, and angles.

## Two coordinate frames

### The theory frame (Zhu's `fk`)

The FK derivation, and Lupei Zhu's original Fortran, uses the standard
**seismological** frame: a right-handed cylindrical system with the vertical
axis $\mathbf{e}_z$ pointing **upward**. The free surface is at $z = 0$, and a
source at depth $h$ sits at

$$
z_s = -h < 0
$$

i.e. **negative $z$ is down**. This is the convention in every equation of the
[FK method](fk_method.md) page.

### The ShakerMaker frame (what you type)

The Python API flips the vertical axis to the **engineering / structural**
convention, where **depth is positive**:

$$
z_{\text{ShakerMaker}} = +\,\text{depth (down)}
$$

So a source 4 km deep is `PointSource([0, 0, 4], ...)`, a positive `z`. The
engine performs the flip to Zhu's $z$-up frame internally; you never apply it
yourself. **Always give depths as positive numbers.**

!!! warning "The one thing to remember"
    In the **equations**, $z$ is *up* (source at $z = -h$).
    In the **code / your scripts**, $z$ is *down* (source at depth $= +h$).
    Same physics, flipped axis, handled by the engine.

### Horizontal axes

ShakerMaker is right-handed with

$$
x = \text{North}, \qquad y = \text{East}, \qquad z = \text{down}.
$$

A receiver 4 km north of the epicentre is `Station([4, 0, 0])`; 4 km east is
`Station([0, 4, 0])`. (Note SW4 uses the opposite horizontal pairing,
`x = East, y = North`, the [SW4 exporter](../guides/sw4_export.md) swaps the
columns for you.)

## Output components: `(Z, E, N)`

The kernel produces motion in the **radial–transverse** frame $(Z, R, T)$
natural to a single source–receiver pair: $R$ points from epicentre to
receiver, $T = \hat{Z}\times\hat{R}$, $Z$ is vertical. ShakerMaker rotates this
to the **geographic** frame by the source–receiver azimuth $\phi$ (clockwise
from North):

$$
u_E = u_R\sin\phi - u_T\cos\phi, \qquad u_N = u_R\cos\phi + u_T\sin\phi
$$

and stores the three components on each station as `s.z, s.e, s.n`, **vertical,
east, north**. `ZENTPlot` labels them $u_Z, u_E, u_N$ (velocity), or their
integral/derivative for displacement/acceleration.

## Units

ShakerMaker is **not** SI, it uses the units customary in regional seismology.
Be consistent with these everywhere:

| Quantity | Unit | Example |
|---|---|---|
| Length / coordinates | km | `Station([0, 4, 0])` |
| Depth `z` | km, **positive down** | source at `z = 4` |
| P/S velocity | km/s | `vp = 6.0` |
| Density | g/cm³ | `rho = 2.7` |
| Quality factor $Q$ | dimensionless | `qs = 10000.` |
| Time | s | `dt = 0.005` |
| Frequency | Hz | `f0 = 2.0` |
| Angles | degrees | `[strike, dip, rake]` |
| Output motion | velocity — units follow the STF source | `s.z, s.e, s.n` |

## Source angles: strike, dip, rake

The mechanism is the Aki–Richards triple, in **degrees**:

- **Strike** $\phi$, azimuth of the fault trace, clockwise from North (0–360°).
- **Dip** $\delta$, inclination of the fault plane from horizontal (0–90°).
- **Rake** $\lambda$, slip direction of the hanging wall in the fault plane
  (−180–180°): 0° = left-lateral strike-slip, 90° = pure reverse, −90° = pure
  normal.

`PointSource` takes them as `angles=[strike, dip, rake]` and converts to
radians internally.

| Mechanism | `[strike, dip, rake]` |
|---|---|
| Vertical strike-slip | `[90, 90, 0]` |
| Normal (dip-slip) | `[0, 45, -90]` |
| Reverse / thrust | `[0, 45, 90]` |

## Quick reference card

| | Theory (equations) | ShakerMaker (code) |
|---|---|---|
| Vertical axis | $z$ up, source at $z=-h$ | $z$ down, depth $=+h$ |
| Horizontal | radial $R$, transverse $T$ (per source–receiver pair) | $x$ = North, $y$ = East |
| Output | $(Z, R, T)$ kernels | $(Z, E, N)$ on the station |
| Lengths | km | km |
| Angles | strike $\phi$, dip $\delta$, rake $\lambda$ (Aki–Richards) | degrees (Aki–Richards) |

Back to [**Overview**](overview.md) · on to [**Finite faults & FFSP →**](finite_fault.md).
