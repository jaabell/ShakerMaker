# FK parameters

This page documents **every numerical input** of `ShakerMaker.run` (and of the
identical core arguments in `compute_gf` / `run_fast` / `run_nearest`), what
each one controls in the Fortran FK kernel, the exact formula behind it, and
the **source line** that implements it. It is the verbose companion to the
shorter [Running a simulation](running.md) page and to the theory in
[Numerical solution](../background/numerics.md).

!!! abstract "The one idea that organises everything"
    Of the fifteen FK control inputs, **you only choose two** — `dt` (the
    frequency band) and `tmax` (the output window). Three are **derived** from
    those plus the model geometry (`nfft`, `dk`, `tb`). The remaining ten are
    **numerical regularisers whose defaults are correct in essentially every
    case** — including `pmax`, which we justify with a controlled experiment
    below.

    $$
    \underbrace{\texttt{dt},\ \texttt{tmax}}_{\text{you choose}}
    \;\rightarrow\;
    \underbrace{\texttt{nfft},\ \texttt{dk},\ \texttt{tb}}_{\text{derived}}
    \;\rightarrow\;
    \underbrace{\texttt{sigma},\,\texttt{taper},\,\texttt{wc1},\,\texttt{wc2},\,\texttt{pmin},\,\texttt{pmax},\,\texttt{kc},\,\texttt{nx},\,\texttt{smth}}_{\text{defaults hold}}
    $$

## Where the code lives

`ShakerMaker.run` → `_call_core` → `subgreen` (`shakermaker/core/subgreen.f`)
→ `subfk` (`shakermaker/core/subfk.f`). `subgreen` builds the per-pair geometry
and `subfk` performs the double integral over frequency $\omega$ and wavenumber
$k$. The stand-alone driver `shakermaker/core/fk.f` carries the **identical**
logic in one readable file, so the citations below use `fk.f` line numbers as
the canonical reference and name `subfk.f` / `subgreen.f` where the operative
copy differs.

!!! info "What the core reads from the geometry (and two common mistakes)"
    - **$h_s$ is the source–receiver *vertical separation*** $|z_s - z_r|$,
      accumulated over the layers between them (`subgreen.f:32-37`) — **not** the
      total crust thickness. It sets the evanescent cutoff $k_c = \texttt{kc}/h_s$.
    - **$V_{s,\text{src}}$ is the shear velocity of the *source layer***
      (`subgreen.f:49`, `subfk.f:64`) — it normalises the slowness window
      `pmin`/`pmax`, **not** the global minimum velocity.
    - $x_\text{max} = \max(h_s, r)$ (`subgreen.f:66-70`) is the horizontal scale
      of the wavenumber step.

---

## The two you choose

### `dt` — output time step → the frequency band

`dt` is the only parameter that fixes the **highest believable frequency**. The
Nyquist frequency is

$$
f_\text{Nyq} = \frac{1}{2\,\texttt{dt}},
$$

but the *usable* band is narrower because the driver applies a raised-cosine
low-pass (the `taper`). The flat passband edge is

$$
f_\text{max} = (1 - \texttt{taper})\,f_\text{Nyq}
\qquad\text{(fk.f:74: } w_c = \tfrac{N_\text{FFT}}{2}(1-\texttt{taper})\text{)},
$$

above which the spectrum rolls smoothly to zero at $f_\text{Nyq}$ (`fk.f:169`).

| quantity | formula | example (`dt`=0.0025, `taper`=0.9) |
|---|---|---|
| Nyquist | $1/(2\,\texttt{dt})$ | 200 Hz |
| usable $f_\text{max}$ | $(1-\texttt{taper})\,f_\text{Nyq}$ | **20 Hz** |
| shortest wavelength | $\lambda_\text{min} = V_{s,\text{min}}/f_\text{max}$ | 37.5 m |

!!! tip "Halving the band runs ~2× faster"
    Since $f_\text{max}\propto 1/\texttt{dt}$, to resolve half the frequency
    double the step: $f_\text{max}=20\to10$ Hz means `dt` $=0.0025\to0.005$.

#### The mesh `dt` implies (DRM / SW4 downstream)

$\lambda_\text{min}$ is the bridge to any finite-element/-difference model the
motion will drive. Resolve it with $N$ points per wavelength:

$$
\Delta x \le \frac{\lambda_\text{min}}{N} = \frac{V_{s,\text{min}}}{f_\text{max}\,N},
\qquad
\Delta t_\text{FEM} \le C\,\frac{\Delta x}{V_{p,\text{surf}}}.
$$

!!! warning "Use $V_{p,\text{surf}}$, not $V_{p,\text{max}}$, in the CFL step"
    The small elements ($\Delta x$) live in the **slow surface layer**, so the
    stable step uses that layer's P velocity $V_{p,\text{surf}}$, *not* the deep
    basement $V_{p,\text{max}}$. Pairing the smallest element with the fastest
    deep velocity is a combination that exists nowhere in the model and yields
    an absurdly tiny $\Delta t_\text{FEM}$. Example: $V_{s,\text{min}}=750$ m/s,
    $f_\text{max}=20$ Hz, $N=10$ → $\Delta x=3.75$ m; with
    $V_{p,\text{surf}}\approx1.5$ km/s, $\Delta t_\text{FEM}\approx0.0025$ s
    (usable), versus $\approx0.0005$ s if $V_{p,\text{max}}=7$ km/s is misused.

### `tmax` (and `tmin`) — the output window

`tmax` is how much of the wave train you keep. Three times bound it:

$$
\begin{aligned}
\text{MAX (record end)} &: \quad t_\text{end} = t_\text{first} - \texttt{pad} + N_\text{FFT}\,\texttt{dt} && \text{above it = garbage},\\
\text{full signal}      &: \quad t_\text{full} = \max_\text{pairs}(t_0 + r/V_\text{Ray}) + t_\text{coda} && \text{end of surface-wave coda},\\
\text{gate}             &: \quad 0 \le \texttt{tmin} < \texttt{tmax} \le t_\text{end}. &&
\end{aligned}
$$

| verdict | condition | meaning |
|---|---|---|
| hard error | $\texttt{tmax} > t_\text{end}$ or gate violated | result is wrong |
| soft (cuts coda) | $\texttt{tmax} < t_\text{full}$ | window is correct but incomplete |

Recommended value: $\texttt{tmax} = \min(t_\text{full},\ t_\text{end})$.

---

## The three derived

### `nfft` — FFT length → record length

`nfft` is the number of FFT samples (in *both* time and frequency); the record
length is $T = N_\text{FFT}\,\texttt{dt}$ (`fk.f:66,72`). It must be a **power of
two** (radix-2 FFT) and long enough that the wave train does not wrap around.
The required length is the larger of the physics and the requested window:

$$
T_\text{need} = \max\!\big(\underbrace{(t_\text{last}-t_\text{first})+\texttt{pad}}_{\text{physics}},\ \underbrace{(\texttt{tmax}-t_\text{first})+\texttt{pad}}_{\text{window}}\big),
\qquad
N_\text{FFT} = 2^{\lceil \log_2(T_\text{need}/\texttt{dt})\rceil}.
$$

!!! info "Why a small `tmax` does not buy a short record"
    Even if you only keep 20 s, the FFT record must still contain the **full**
    signal (e.g. 60 s of surface-wave coda) or the late energy aliases into the
    retained window. `nfft` is driven by whichever is longer.

### `dk` — wavenumber spacing → resolution + the ghost source

The physical step is $\Delta k = \texttt{dk}\cdot\pi/x_\text{max}$ (`fk.f:107`).
Two constraints:

$$
N_k \approx \frac{\texttt{kc}\,x_\text{max}}{\pi\,h_s\,\texttt{dk}} \gtrsim 10
\quad\text{(quadrature resolved)},
\qquad
\texttt{dk} < 0.5 \quad\text{(≥4 samples/Bessel period, fk.f:82-84)}.
$$

By **Bouchon's theorem**, sampling $k$ at $\Delta k$ surrounds the source with
image sources spaced $L = 2\pi/\Delta k = 2\,x_\text{max}/\texttt{dk}$ apart; the
nearest spurious image arrives at

$$
t_\text{img} \approx \frac{L - r}{V_\text{Ray}},\qquad V_\text{Ray}\approx 0.92\,V_{s,\text{min}}.
$$

The method is exact only while $t_\text{img} > \texttt{tmax}$. Choose the
**coarsest** `dk` that keeps $N_k\ge10$ on every pair and the image beyond the
window — coarser is faster.

### `tb` — pre-arrival padding (samples)

`tb` zero-pads before the first arrival and shifts the reduction time
$t_0 \leftarrow t_0 - \texttt{tb}\cdot\texttt{dt}$ (`subgreen.f:73`, `fk.f:105`),
preventing the bandlimited onset and filter ringing from wrapping to the trace
start. A pad of 1–2 s ($\texttt{tb}\approx 1.5/\texttt{dt}$) is right almost
always; too small clips the pulse, too large pushes the train out of the record.

---

## The ten defaults (and why they hold)

| param | default | controls | formula / source | revisit only if… |
|---|---|---|---|---|
| `sigma` | 2 | wrap-around damping $e^{-\sigma t}$ | rescaled by $\Delta\omega$ (`fk.f:73`) | coda contaminated (↑) / late signal dies (↓) |
| `taper` | 0.9 | low-pass, sets $f_\text{max}$ | $f_\text{max}=(1-\texttt{taper})f_\text{Nyq}$ (`fk.f:74`) | want a specific $f_\text{max}$: $\texttt{taper}=1-f_\text{max}/f_\text{Nyq}$ |
| `wc1`,`wc2` | 1, 2 | low-freq high-pass (DC) | raised cosine over low bins (`fk.f:171-172`) | removing a static offset |
| `pmin` | 0 | max phase velocity captured | $c_\text{max}=V_{s,\text{src}}/\texttt{pmin}$ (`fk.f:140`) | excluding vertical body waves |
| `pmax` | 1 | min phase velocity captured | $c_\text{min}=V_{s,\text{src}}/\texttt{pmax}$ (`fk.f:141`) | see experiment below — **leave at 1** |
| `kc` | 15 | evanescent cutoff $k_c=\texttt{kc}/h_s$ | decay $e^{-\texttt{kc}}\approx3\times10^{-7}$ (`fk.f:89`) | needs $>10$; default is safe |
| `nx` | 1 | distance ranges per call | structural (`fk.f:95`) | never — always 1 per pair |
| `smth` | 1 | output densification | $\Delta t_\text{out}=\texttt{dt}/\texttt{smth}$ (`fk.f:186`) | want finer output sampling |

!!! note "`sigma` is effectively dimensionless"
    Because `sigma` is rescaled by $\Delta\omega = 2\pi/(N_\text{FFT}\texttt{dt})$
    at `fk.f:73`, the attenuation at the **end of the record** is exactly
    $e^{-\sigma}$, independent of `dt` and `nfft`: `sigma`=2 leaves
    $e^{-2}\approx13.5\%$, `sigma`=3 leaves $\approx5\%$. That is why the default
    is universal.

---

## Case study: is `pmax` = 1 safe?

A paper argument suggests it might not be. The integration upper bound is

$$
k_\text{max}(\omega) = \sqrt{\Big(\tfrac{k_c}{h_s}\Big)^2 + \Big(\tfrac{\texttt{pmax}}{V_{s,\text{src}}}\omega\Big)^2}
\qquad\text{(fk.f:141)},
$$

so a wave of phase velocity $c=\omega/k$ is captured only if $c \ge
V_{s,\text{src}}/\texttt{pmax}$. With `pmax`=1 the minimum captured phase
velocity equals $V_{s,\text{src}}$, while the Rayleigh wave travels at
$\approx0.92\,V_s$ (slowness $1.11/V_s$) — *outside* the window. The original FK
author even left a commented default `pmax = 1.11` (`subfk.f:51-52`), exactly
$1/0.9$. So with a **deep, fast** source, should `pmax` be raised?

**We tested it.** Worst case: slow surface layer ($V_s=0.75$ km/s) over fast
half-space ($V_s=3.5$ km/s), source 4 km deep in the fast rock, receiver at
10 km, band to 5–10 Hz. The paper formula predicts `pmax`$\approx5$ is needed.
Running the actual core and comparing three-component velocity traces:

| `pmax` | RMS difference vs `pmax`=8 |
|---|---|
| 1.0 | 0.1 % |
| 2.0 | 0.0 % |
| 4.0 | 0.0 % |

A harder stress test ($f_\text{max}=10$ Hz, $V_s=0.5$ km/s surface, source 8 km
deep, 15 km range) gives **0.00 %** across `pmax` ∈ [1, 8], and a `kc` sweep
(5→60) at fixed `pmax`=1 also gives 0.00 %. **The waveforms are identical.**

!!! success "Verdict: leave `pmax` = 1"
    The paper argument neglected (1) the static floor $k_c/h_s$, which already
    extends $k_\text{max}$ past where the integrand has energy, and (2) the
    evanescent decay $e^{-k h_s}$ — there is nothing left to integrate out
    there. Physically, the only regime where `pmax`=1 could truncate energy
    (deep fast source, high-frequency surface waves) is the regime where those
    surface waves are weakly excited. `pmax = 1.11` is the textbook-safe value;
    `pmax = 1` is empirically identical.

---

## The `check_parameters` pre-flight

`ShakerMaker.check_parameters` reproduces every formula above with **no FK
run** (pure arithmetic) and prints a report organised by the dependency
hierarchy. Call it right after building the model:

```python
model = ShakerMaker(crust, fault, stations)
report = model.check_parameters(dt=0.0025, nfft=32768, dk=0.2, tb=800, tmax=20)
# report == {"passed": bool,
#            "recommended": {"dk":..., "tb":..., "nfft":..., "tmax":...}}
```

It prints, top to bottom: a **header** (your `dt`/`tmax` + geometry +
physical signal window); one **block per first-class parameter** (`dt`,
`nfft`, `dk`, `tb`, `tmax`) with its equation, `fk.f` line, value, verdict and
`>> recommend`; a **RESULT** block that separates *hard errors* from
*recommended changes*; and a **READY-TO-RUN** block — a full `run()` call with
every input labelled `YOU SET` / `RECOMMEND` / `DERIVED` / `default` /
`structural`.

```text
 tmax = 20 s     output window                          [OK, cuts coda]
   MAX  = t_first - pad + nfft*dt = 4.2 - 2.0 + 81.9 = 84.1 s [fk.f:72]
   full = max(t0 + r/V_Ray) + coda = 54.7 s
   gate = tmin < tmax <= 84.1   (tmin = 0.0)                  [fk.f:105]
   >> recommend tmax = 54.7 s
 ...
 RESULT: all hard checks passed
   >> recommended change:  tmax  20 -> 54.7 s   (capture full signal)
```

The verdict logic is two-tier: `nfft`, `dk`, `tb` and the `tmax` record-gate
are **hard checks** (failure corrupts the result); clipping the coda is a
**soft recommendation** flagged `[OK, cuts coda]`.

#### What a *failing* report looks like

When a hard check fails, the block is tagged `[ERROR]` instead of `[OK]`, and
the RESULT block lists the error and refuses to bless the run. The most common
hard error is asking for a window longer than the record can hold,
$\texttt{tmax} > t_\text{end}$ — e.g. requesting `tmax`=120 s with an `nfft`
that only reaches $t_\text{end}=84.1$ s:

```text
 tmax = 120 s    output window                           [ERROR, exceeds record]
   MAX  = t_first - pad + nfft*dt = 4.2 - 2.0 + 81.9 = 84.1 s [fk.f:72]
   full = max(t0 + r/V_Ray) + coda = 54.7 s
   gate = tmin < tmax <= 84.1   (tmin = 0.0)              VIOLATED  [fk.f:105]
   >> tmax 120 > t_end 84.1  -> everything past 84.1 s is wrap-around garbage
   >> recommend tmax = 54.7 s   (or raise nfft to extend the record)
 ...
 RESULT: 1 hard error — fix before running
   [ERROR] tmax 120 s exceeds record end 84.1 s   (gate violated)
   >> recommended change:  tmax  120 -> 54.7 s    (capture full signal)
```

Here `report["passed"]` is `False`; a calling script should read
`report["recommended"]` and re-run with the corrected values rather than
proceeding. The fix is either to lower `tmax` to $\le t_\text{end}$ or to raise
`nfft` so the record is long enough to hold the requested window.

!!! tip "Extra knobs of `check_parameters`"
    `n_per_wavelength` (mesh points/wavelength, default 10), `courant`
    (CFL number $C$, default 1.0), `fem_fmax` (mesh target frequency; default
    `None` → use the FK band $f_\text{max}$ so the mesh matches the motion),
    and `coda` (fixed coda margin in seconds, internal — not an FK parameter).

---

## Full parameter map

| `run` arg | default | core symbol | line | role |
|---|---|---|---|---|
| `dt` | 0.05 | $\Delta t$ | `fk.f:72` | output step / Nyquist / band |
| `nfft` | 4096 | $N_\text{FFT}$ | `fk.f:66,72` | record length $T=N\,\texttt{dt}$ |
| `tb` | 1000 | pre-arrival samples | `fk.f:105`, `subgreen.f:73` | causal pad / reduction shift |
| `dk` | 0.3 | $\Delta k=\texttt{dk}\,\pi/x_\text{max}$ | `fk.f:107` | wavenumber spacing / aliasing |
| `kc` | 15.0 | $k_c=\texttt{kc}/h_s$ | `fk.f:89`, `subfk.f:67` | evanescent cutoff |
| `sigma` | 2 | $\sigma$ in $\omega-\mathrm{i}\sigma$ | `fk.f:73` | pole regularisation + wrap-around |
| `taper` | 0.9 | spectral taper | `fk.f:74,169` | high-frequency Gibbs control |
| `wc1`,`wc2` | 1, 2 | low-freq window | `fk.f:171-172` | high-pass / DC handling |
| `pmin`,`pmax` | 0, 1 | slowness $/V_{s,\text{src}}$ | `fk.f:90-91,140-141` | which phases are integrated |
| `nx` | 1 | distance ranges | `fk.f:95` | always 1 per pair |
| `smth` | 1 | upsampling | `fk.f:186` | output densification |
| `tmin`,`tmax` | 0, 100 | output window | `fk.f:105` | retained time window |

## Calibration recipe

1. **Band → `dt`**: from your $f_\text{max}$, $\texttt{dt}=(1-\texttt{taper})/(2 f_\text{max})$.
2. **Window → `tmax`**: keep the full train, $\texttt{tmax}\approx t_\text{full}$.
3. **Record → `nfft`**: smallest power of two with $N_\text{FFT}\texttt{dt}\ge T_\text{need}$.
4. **Spacing → `dk`**: coarsest with $N_k\ge10$ and $t_\text{img}>\texttt{tmax}$.
5. **Pad → `tb`**: $\approx 1.5/\texttt{dt}$.
6. **Mesh check**: $\lambda_\text{min}/N=\Delta x$, $\Delta t_\text{FEM}=C\,\Delta x/V_{p,\text{surf}}$.
7. **Everything else**: defaults (including `pmax`=1).

Let `check_parameters` do steps 3–6 for you and print the ready-to-run call.

## See also

- [Running a simulation](running.md) — the short version + the OP pipeline.
- [Numerical solution](../background/numerics.md) — the theory each parameter regularises.
- [The FK method](../background/fk_method.md) — the kernels $U(\omega,k)$.
- [ShakerMaker engine API](../api/shakermaker.md) — full signatures.
