# Numerical solution

The FK method ends in two integrals, over wavenumber $k$ and over frequency
$\omega$. Turning those into numbers introduces a handful of parameters. They
are **not** free knobs: each one is a discretisation choice with a theorem
behind it, and each maps to an argument of [`run`](../guides/running.md). This
page connects the two.

## The wavenumber integral

The receiver displacement is the inverse Hankel transform

$$
u(r, \omega) = \int_0^{\infty} U(k, \omega)\, J_m(kr)\, k\, dk
$$

evaluated, in practice, as a finite sum on a midpoint grid:

$$
u(r, \omega) \approx \sum_{l=1}^{N_k} U(k_l, \omega)\, J_m(k_l r)\, k_l\, \Delta k,
\qquad k_l = \big(l - \tfrac12\big)\,\Delta k.
$$

Two numbers define this sum: the **spacing** $\Delta k$ and the **upper cutoff**
$k_c = N_k\,\Delta k$. Everything below is about choosing them well.

## Discrete wavenumber: `dk`

Bouchon (1981) showed that sampling $k$ at a finite spacing $\Delta k$ is
*exactly* equivalent to surrounding the real source with a periodic lattice of
image sources spaced $L = 2\pi/\Delta k$ apart. The method is exact **as long as
the wavefield from those image sources has not yet arrived** within your time
window. If $\Delta k$ is too large, $L$ is too small, the images arrive early,
and a spurious oscillatory tail appears after the S arrival, **spatial
aliasing**.

| | |
|---|---|
| Parameter | `dk` (in units of $\pi/r_{\max}$) |
| Symptom of too large | long ringing tail after $t \sim r/V_S$ |
| Fix | halve `dk` until the trace stops changing |

This is [Exercise 2, Part A](../exercises/02_convergence.md#part-a-convergence-in-dk).

## Truncation: `kc`, `pmin`, `pmax`

The integrand decays like $e^{-k h_s}$ at high $k$ (with $h_s$ the
source–receiver depth separation), so the sum can stop at a finite **cutoff**
$k_c$. Too small a cutoff and high-frequency, high-wavenumber S energy is lost.

| Parameter | Meaning |
|---|---|
| `kc` | wavenumber cutoff $k_c$ (in $1/h_s$); needs $k_c \gtrsim 10$ |
| `pmin`, `pmax` | the slowness window $p = k/\omega$ to integrate (in $1/V_{s,\text{src}}$, the source-layer shear velocity) |

## Sigma damping: the complex frequency

In an elastic medium the Rayleigh/Love **poles sit on the real $k$-axis** and
the integral is singular (see [the FK kernels](fk_method.md#the-surface-kernels)).
The fix is to evaluate the inverse Fourier transform along a line *below* the
real axis, i.e. with a **complex frequency** $\omega \to \omega - i\sigma$. This
moves every pole off the real axis and makes the integrand smooth.

The price: the recovered trace is multiplied by $e^{-\sigma t}$, which also
tames the **wrap-around** from the finite FFT, but if $\sigma$ is too large it
visibly decays the late signal into noise.

| Parameter | `sigma` (in $1/T$) |
|---|---|
| Too large | noisy, decaying tail |
| Too small | wrap-around / aliasing not suppressed |

## Time discretisation: `dt`, `nfft`, `tb`, `smth`

The frequency integral is an FFT, so the time axis is discrete:

- **`dt`**, output time step. The Nyquist frequency is $f_{\text{Nyq}} = 1/(2\,dt)$;
  to resolve $f_{\max} = 10$ Hz you need $dt \le 0.05$ s.
- **`nfft`**, number of FFT samples, so the record length is
  $T = N_{\text{FFT}}\cdot dt$. Too short and the wave train **wraps around**.
- **`tb`**, pre-arrival padding samples, so the first arrival is not clipped
  and causality is preserved.
- **`smth`**, spectral densification factor that interpolates the output to a
  finer step without recomputing the kernel.

This is [Exercise 2, Part B](../exercises/02_convergence.md#part-b-convergence-in-nfft).

## Low-pass taper: `taper`

A smooth spectral taper near the high-frequency end suppresses Gibbs ringing
from the hard FFT cutoff. Set by `taper` (0–1); raise it if high-frequency
content looks lost.

## Low-frequency high-pass: `wc1`, `wc2`

At the other end of the band, a raised-cosine window over the lowest frequency
bins controls how the **DC / near-static** component is handled: it tapers the
spectrum to zero across the first few bins so that an unconstrained zero-frequency
term does not leak into the trace as a spurious baseline offset. `wc1` and `wc2`
are the start and end bins of that low-frequency taper; the defaults (1, 2) leave
the physical low-frequency content intact and only need raising if you are
deliberately **removing a static offset**.

| Parameter | `wc1`, `wc2` (low-frequency bins) |
|---|---|
| Default | 1, 2 (keep the low-frequency content) |
| Raise | to suppress a near-static baseline / DC offset |

## The map: theory → parameter

| `run` parameter | Theory quantity | Governs |
|---|---|---|
| `dt` | $\Delta t$ | output rate / Nyquist |
| `nfft` | $N_{\text{FFT}}$ | record length $T = N\,dt$ |
| `tb` | pre-arrival samples | causality padding |
| `dk` | $\Delta k$ | discrete-wavenumber spacing (aliasing) |
| `kc` | $k_c$ | wavenumber cutoff (truncation) |
| `sigma` | $\sigma$ in $\omega - i\sigma$ | pole regularisation + wrap-around |
| `pmin`, `pmax` | slowness window | which phases are integrated |
| `taper` | spectral taper | high-frequency Gibbs ringing |
| `wc1`, `wc2` | low-frequency window | high-pass / DC handling |
| `smth` | interpolation factor | output densification |

Every entry in the [diagnostic checklist](../guides/running.md#diagnostic-checklist)
is a symptom of one of these being mis-set, now you know *why*.

For the operational detail — exact defaults, the source line behind each knob,
and the `check_parameters` pre-flight that derives them for you — see the
[FK parameters reference](../guides/parameters.md).

## References

- Bouchon, M. (1981). *BSSA* **71**, 959–971, the discrete-wavenumber theorem.
- Zhu, L. (2011). *Synthetic Seismograms and Seismic Waveform Modeling.*
- Aki, K. & Richards, P. G. (2002). *Quantitative Seismology*, 2nd ed., §7.4.

Continue to [**Coordinates & conventions →**](conventions.md).
