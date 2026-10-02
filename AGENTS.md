# AGENTS.md

Shared guide for AI coding agents (Claude Code, Codex, Copilot, etc.) working in
ShakerMaker. Humans are welcome to read it too. Keep it current: if you change
how the project builds, runs, or is organized, update this file in the same commit.

## What this project is

ShakerMaker computes synthetic seismograms with the frequency–wavenumber (FK)
method and writes Domain Reduction Method (DRM) input motions (H5DRM format).

- **Fortran/C core** (`shakermaker/core/`): FK kernel, originally from L. Zhu's
  `fk` code, modified. Exposed to Python via f2py using the hand-maintained
  signature file `core.pyf`.
- **Python layer** (`shakermaker/`):
  - `shakermaker.py` — `ShakerMaker` driver. Direct pipeline: `run`.
    Green's-function (OP) pipeline: `gen_pairs` → `compute_gf` → `run_fast`
    (or `run_nearest`); `check_parameters` validates dt/nfft/dk/tb. Exporters:
    `export_drm_geometry`, `export_sw4`, `export_sw4_topo`. Work is distributed
    over MPI (`mpi4py`, optional); `numba` accelerates pair grouping when
    available, with a pure-Python fallback. Windows runs the core in a
    large-stack thread (`_win_run`).
  - `crustmodel.py` (+ `cm_library/` predefined models), `pointsource.py`,
    `faultsource.py`, `station.py`, `stationlist.py`, `sourcetimefunction.py`,
    `stationlistwriter.py`.
  - Plug-ins: `sl_extensions/` (receiver layouts: `DRMBox`, `SurfaceGrid`,
    `PointCloudDRMReceiver`), `slw_extensions/` (HDF5 / H5DRM writers),
    `stf_extensions/` (Brune, Dirac, Discrete, Gaussian, SRF2 source time
    functions), `tools/plotting.py`.
- **Examples** (`examples/`): the de-facto test suite. There are no unit tests.
- **Docs** (`docs/`): Sphinx, built on Read the Docs (`.readthedocs.yml`).

## Build & run

```bash
pip install numpy scipy h5py mpi4py numba matplotlib  # gfortran + MPI required
python setup.py build_ext --inplace               # compiles shakermaker.core via numpy.distutils/f2py
python examples/example1_simple.py                # smoke test
mpirun -np 4 python examples/example2_drm.py      # parallel run
```

`setup.py` depends on `numpy.distutils`, which is removed in NumPy ≥ 1.26 /
Python ≥ 3.12. If a build fails for that reason, report it rather than
silently rewriting the build system — migrating (e.g. to meson-python) is a
deliberate project decision.

## Conventions

- **Fortran:** fixed-form, lines up to 132 columns (`-ffixed-line-length-132`).
  Any change to a subroutine signature must be mirrored in `core.pyf` and in the
  `_call_core*` wrappers in `shakermaker.py`.
- **Units:** km, km/s, g/cm³, seconds; Z-E-N component ordering in outputs.
  Don't mix in SI meters without explicit conversion.
- **Optional deps:** code must still run without `mpi4py` (serial) and without
  `numba`. Only MPI rank 0 writes files / prints summaries.
- **Portability:** Linux is primary, but Windows is supported — don't break
  `_win_run` or assume POSIX-only paths. `.gitattributes` normalizes line endings.
- **Physics changes need evidence:** when touching the kernel, Green's
  functions, filtering, or DRM output, run an example before and after and
  state in the commit body what was compared (traces, peak values, plots).
- Don't commit build products (`build/`, `dist/`, `*.egg-info`, `*.so`, `*.o`)
  or large output data (`*.h5`, `*.h5drm`).
- Keep diffs focused; match the surrounding style rather than reformatting.

## Agents

Project subagents live in `.claude/agents/` and are versioned with the repo.
Add or edit them via normal commits.

| Agent | Model | Use for |
|-------|-------|---------|
| `fk-core` | opus | Fortran/C kernel, `core.pyf`, f2py build problems |
| `pr-reviewer` | opus | Reviewing PRs / PR series and proposing merge order (read-only) |
| `drm-pipeline` | sonnet | Python API, MPI driver, Green's function DB, H5DRM writers |
| `docs-writer` | sonnet | Sphinx docs, README, docstrings |
| `example-runner` | haiku | Running examples to verify a change; reporting numerical diffs (read-only) |

Model tiers: **opus** where a subtle mistake silently corrupts physics or
lets a bad merge through (kernel numerics, threading, review); **sonnet** for
routine feature and docs work; **haiku** for mechanical, well-specified jobs
(run, compare, report). Use the alias (`opus`/`sonnet`/`haiku`) in the
agent's `model:` frontmatter so it tracks the latest model of that tier. If a
cheap agent's output looks shaky, rerun the task on a stronger tier rather
than lowering the bar.

## Signing commits (required for agents)

Every commit made by an AI agent must be signed with trailers identifying the
agent and model. The `commit-msg` hook in `.githooks/` enforces this.

```
<type>(<scope>): <subject, imperative, ≤ 72 chars>

<body: what and why; for physics changes, how it was verified>

Agent: <agent-name>                 # e.g. drm-pipeline, or "main" for the top-level session
Co-Authored-By: <Model Name> <noreply@anthropic.com>
```

- Subjects follow Conventional Commits as used in the history
  (`feat`, `fix`, `perf`, `refactor`, `chore`, `docs`; scope such as
  `engine`, `core`, `stf`, `receivers`, `crust`, `station`).
- `Agent:` names which agent from the table above made the change (`main` if
  it was the primary session rather than a subagent).
- `Co-Authored-By:` names the model, so authorship stays with the human who ran
  the session and the agent is credited as co-author.
- Tool-specific extra trailers (e.g. a session link) are fine to add.
- The hook rejects a commit that has an AI `Co-Authored-By:` without `Agent:`,
  or an `Agent:` without a `Co-Authored-By:`. Human-only commits are unaffected.

Enable the hook once per clone:

```bash
git config core.hooksPath .githooks
```

Never bypass it with `--no-verify`.
