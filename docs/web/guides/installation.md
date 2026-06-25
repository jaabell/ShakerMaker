# Installation & compilation

ShakerMaker is not a pure-Python package. Its heart is a **Fortran** engine, 
Lupei Zhu's FK code (the `shakermaker.core` extension) plus the FFSP
stochastic-rupture generator (`shakermaker.ffsp.ffsp_core`). Those two must be
**compiled for your platform and your exact Python version** before the Python
API will import. This page walks through that compilation on Linux and on
Windows, end to end, explaining *why* each step is there.

!!! info "What actually gets built"
    `setup.py` drives NumPy's `f2py` to wrap the Fortran sources into two
    importable extensions:

    - `shakermaker/core.*.so` (Linux) / `core.*.pyd` (Windows), the FK Green's-function kernel.
    - `shakermaker/ffsp/ffsp_core.*`, the FFSP finite-fault generator.

    The file suffix encodes the Python version (e.g. `core.cpython-310-x86_64-linux-gnu.so`),
    so a build for Python 3.10 will **not** load under 3.11. Match your
    interpreter.

---

## Linux (native: recommended)

Linux is the smoothest path: the GNU Fortran compiler `gfortran` is all you
need, and MPI comes from OpenMPI. The example below uses Arch/OMARCHY
(`pacman`); on Debian/Ubuntu substitute `apt` (`gfortran`, `libopenmpi-dev`,
`python3-venv`).

### 1. System dependencies

```bash
sudo pacman -S gcc-fortran openmpi make python-setuptools
```

`gcc-fortran` provides `gfortran` (compiles the FK and FFSP sources);
`openmpi` provides `mpirun` and the headers `mpi4py` links against; `make` is
used by the FFSP build step.

### 2. A clean virtual environment

```bash
python -m venv ~/shakermaker_env
source ~/shakermaker_env/bin/activate
```

Isolating the install matters because of the next point, the build is
**version-sensitive**.

### 3. Python dependencies

ShakerMaker **runs on NumPy 2.x** — the engine was ported to NumPy 2 + Numba.
NumPy is only constrained when you **compile the Fortran from source**: `setup.py`
drives `f2py` through `numpy.distutils`, which NumPy removed in 2.0 and Python
removed from its standard library in 3.12. So the *build* step (and only the
build step) needs a NumPy 1.x together with `setuptools<60`, under Python
3.8–3.11:

```bash
pip install "setuptools<60.0" "numpy<2.0" wheel scipy h5py mpi4py matplotlib
```

`setuptools<60` keeps the classic `setup.py`/`numpy.distutils` path working
(later setuptools shadows the stdlib `distutils` and breaks it). Once the
extensions are compiled — or if you install the prebuilt `.so`/`.pyd` shipped in
the repository — you can run under NumPy 2.x.

### 4. Build & install

From the repository root:

```bash
cd /path/to/ShakerMaker
pip install . --no-build-isolation
```

`--no-build-isolation` tells pip to use the NumPy/setuptools you just pinned for
the compile, instead of fetching fresh (newer) build deps into a throwaway
environment, which would re-introduce the `numpy.distutils` problem at build time.

For development you usually want the compiled extensions **in the source tree**
so you can edit Python and re-import without reinstalling:

```bash
python setup.py build_ext --inplace
```

That drops `core.*.so` and `ffsp/ffsp_core.*.so` next to the sources.

### 5. Let FFSP run

The FFSP generator ships a small Fortran executable that needs the execute
bit:

```bash
chmod +x ~/shakermaker_env/lib/python3.10/site-packages/shakermaker/ffsp/ffsp_dcf_v2
```

(Adjust the path to your environment and Python version.)

### 6. Verify

```bash
python -c "import shakermaker; print('ShakerMaker OK')"
python -c "from shakermaker import core; print('FK core OK')"
```

If both print cleanly, the Fortran extensions loaded.

### 7. Run with MPI

```bash
mpirun -np 4 python examples/06_nearest_method/nearest_all.py
```

One MPI rank per CPU core (the FK kernel is single-threaded within a rank, see
[the pipeline](running.md#the-op-pipeline-run_nearest)).

---

## Windows (Intel oneAPI + MSVC)

Windows is harder because there is no free, drop-in Fortran compiler that
`f2py` recognises out of the box. The supported toolchain is **Intel oneAPI's
`ifx`** for Fortran together with **MSVC** (`cl.exe`) for the C sources. Plan
for an hour the first time. The steps below are condensed from the full
internal build log; the order matters.

!!! warning "Read these first, they cause most failures"
    - **Paths with spaces break the build.** If the repo lives under
      `C:\Dropbox\…` (a space in `01. Brain`), create a junction to a
      space-free path and build from there.
    - **Use CMD, not PowerShell, for build steps.** PowerShell quoting breaks
      the compiler invocations.
    - **To compile, use NumPy 1.x** (same `numpy.distutils` reason as Linux);
      once built, ShakerMaker runs fine under NumPy 2.x.
    - **Don't install MSYS2/gfortran** alongside Intel, it contaminates PATH.

### Required tools

| Tool | Version | Provides |
|---|---|---|
| Visual Studio 2022 Community | 17.x | MSVC `cl.exe` (C workload) |
| Intel oneAPI Base Toolkit | 2025.x | MKL |
| Intel oneAPI HPC Toolkit | 2025.x | the `ifx` Fortran compiler |
| Python | 3.10.x | matches the `.pyd` suffix; build needs 3.8–3.11 |
| NumPy | 1.26.4 (build) | 1.x only to compile; runs under 2.x |

### 1. Junction around the space-in-path problem

In an **Administrator CMD**:

```cmd
mkdir C:\shakermaker_compiler
mklink /J "C:\shakermaker_compiler\ShakerMaker" "C:\Dropbox\01. Brain\11. GitHub\ShakerMaker"
```

All build commands then run from `C:\shakermaker_compiler\ShakerMaker`.

### 2. Environment + venv (fresh CMD every time)

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
C:\Users\you\shaker_env\Scripts\activate.bat
where cl & where ifx & python --version
```

Never re-run `VsDevCmd.bat`/`setvars.bat` twice in the same window, PATH
overflows and CMD dies with *"The input line is too long"*. Open a fresh
window instead.

### 3. The `ifort.bat` shim

`f2py` searches for `ifort.exe`, but oneAPI 2025 ships only `ifx.exe`. Create a
one-line wrapper (**Administrator CMD**):

```cmd
echo @ifx %* > "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifort.bat"
```

### 4. Build

```cmd
set LIB=%LIB%;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib
cd C:\shakermaker_compiler\ShakerMaker
python setup.py install
```

The manual `LIB` line prevents `LNK1104: cannot open file 'ifconsol.lib'`,
which appears when `setvars.bat` doesn't export the Intel library path. The
build compiles `ffsp_core.*.pyd`, then `core.*.pyd`, then installs the Python
files.

### 5. Make the Intel DLLs findable at startup (`sitecustomize.py`)

The compiled `core.pyd` links Intel oneAPI DLLs (`libiomp5md.dll`, …). If
Python starts without oneAPI initialised (e.g. from Jupyter), the import fails
with *"DLL load failed while importing core"*. The permanent fix is a
`sitecustomize.py` in site-packages, Python runs it automatically on every
startup. Create it with **PowerShell** (CMD's `echo` injects characters that
break the parser):

```powershell
$content = @"
import os
os.add_dll_directory(r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin")
os.add_dll_directory(r"C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin")
"@
[System.IO.File]::WriteAllText("C:\Users\you\shaker_env\Lib\site-packages\sitecustomize.py", $content)
```

!!! note "Recreate it after every reinstall"
    `python setup.py install` / `pip uninstall` deletes `sitecustomize.py`.
    Re-run the snippet above each time you rebuild.

### 6. MPI needs `locking=False`

On Windows, several MPI ranks opening the same HDF5 database deadlock on file
locking. The build patches `shakermaker.py` so every
`h5py.File(h5_database_name, …)` call passes `locking=False`. Do **not** set
the environment variable `HDF5_USE_FILE_LOCKING=FALSE`, that one blocks Python
from starting under MPI.

### 7. Smoke test

```cmd
C:\Users\you\shaker_env\Scripts\activate.bat
python -c "import shakermaker; print('shakermaker OK')"
python -c "from shakermaker import core; print('core OK')"
```

### 8. Run with MPI

```cmd
call "...\VsDevCmd.bat" -arch=x64 -host_arch=x64
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
C:\Users\you\shaker_env\Scripts\activate.bat
chcp 65001
cd "C:\path\to\your\model"
mpiexec -n 10 python your_script.py
```

`chcp 65001` switches CMD to UTF-8 so ShakerMaker's banner renders. Always
`cd` into the model directory first, relative paths resolve from the working
directory.

---

## Troubleshooting (Windows)

| Symptom | Cause | Fix |
|---|---|---|
| `DLL load failed while importing core` | Intel runtime not on the DLL path | (re)create `sitecustomize.py` |
| `CompilerNotFound: intelvem` | `ifort.bat` shim missing | create it, check `where ifort` |
| `LNK1104: cannot open 'ifconsol.lib'` | Intel `LIB` not set | the `set LIB=…` line before building |
| `The input line is too long` | PATH overflow | open a fresh CMD window |
| `error #6404` in `haskell.f` | long Fortran lines | `/extend-source:132` (already in `setup.py`) |

---

## Which platform?

| | Linux | Windows |
|---|---|---|
| Compiler | `gfortran` (free, built-in) | Intel `ifx` + MSVC |
| Setup effort | minutes | ~1 hour, first time |
| MPI | OpenMPI, native | Intel MPI, needs `locking=False` |
| Recommended for | clusters, production runs | desktop development |

For large production runs, **build natively on Linux**, it is simpler and the
MPI scaling is cleaner. Windows is convenient for desktop exploration.

## Next

[Getting started](first_steps.md) · [Running a simulation](running.md)
