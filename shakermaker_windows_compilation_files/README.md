# ShakerMaker — Windows Compilation Files
```         
                                                           ▄█████▀ 

 ▄█          ▄████████ ████████▄     ▄████████ ███    █▄  ███▄▄▄▄    ▄██████▄  
███         ███    ███ ███   ▀███   ███    ███ ███    ███ ███▀▀▀██▄ ███    ███ 
███         ███    ███ ███    ███   ███    ███ ███    ███ ███   ███ ███    ███ 
███         ███    ███ ███    ███  ▄███▄▄▄▄██▀ ███    ███ ███   ███ ███    ███ 
███       ▀███████████ ███    ███ ▀▀███▀▀▀▀▀   ███    ███ ███   ███ ███    ███ 
███         ███    ███ ███    ███ ▀███████████ ███    ███ ███   ███ ███    ███ 
███▌    ▄   ███    ███ ███   ▄███   ███    ███ ███    ███ ███   ███ ███    ███ 
█████▄▄██   ███    █▀  ████████▀    ███    ███ ████████▀   ▀█   █▀   ▀██████▀  
▀                                   ███    ███                                 
      ▄█    █▄     ▄████████  ███     ███    ▄████████    ▄████████ 
     ███    ███   ███         ███     ███   ███          ███    ███ 
     ███    ███   ███         ███     ███   ███          ███    ███ 
    ▄███▄▄▄▄███  ▄███▄▄▄▄     ███     ███  ▄███▄▄▄▄     ▄███▄▄▄▄██▀
   ▀▀███▀▀▀▀███  ▀███▀▀▀▀      ███   ███   ▀███▀▀▀▀    ▀▀███▀▀▀▀▀ 
            ███   ███           ███ ███     ███        ▀███████████
            ███   ███            █████      ███          ███    ███
            ███   ███            █████      ███          ███    ███
            █▀     ▀████████      ███        ▀████████   ███    ███
```
This repository contains the automated scripts to compile and install **ShakerMaker 1.0** on Windows 11 using Python 3.10, Intel oneAPI compilers, and Visual Studio 2022.

---

## Requirements

Before running anything, make sure your machine has internet access and enough disk space. The full installation requires approximately 20 GB for Visual Studio 2022 and the Intel oneAPI toolkits.

| Tool | Version | Purpose |
|------|---------|---------|
| Windows 11 | Any | Operating system |
| Python | 3.10.x | Runtime and build environment |
| Git | 2.x | Repository access |
| Visual Studio 2022 Community | 17.x | C++ compiler (`cl.exe`) |
| Intel oneAPI Base Toolkit | 2025.x | Math Kernel Library (MKL) |
| Intel oneAPI HPC Toolkit | 2025.x | Fortran compiler (`ifx`) and Intel MPI |
| NumPy | 1.26.4 | Build dependency — do NOT use 2.x |

---

## Repository Structure

```
shakermaker_windows_compilation_files/
└── SCRIPTS/
    ├── shakermaker.cfg               ← Central configuration file (edit this first)
    ├── 00_shakermaker_common.ps1     ← Shared functions used by all scripts (do not edit)
    ├── 01_shakermaker_setup.ps1      ← Step 1: Install prerequisites and create virtual environment
    ├── 02_shakermaker_junction.ps1   ← Step 2: Create junction to the ShakerMaker repository
    ├── 03_shakermaker_build.ps1      ← Step 3: Compile ShakerMaker and configure DLL loading
    └── RUN_ME.bat                    ← Main launcher with interactive menu
```

---

## Quick Start

### Step 1 — Edit the configuration file

Open `SCRIPTS\shakermaker.cfg` and fill in your settings. This is the only file you need to edit.

```ini
# Python version
PYTHON_VERSION      = 3.10
PYTHON_FULL_VERSION = 3.10.9

# Name of the virtual environment to create
# Keep this SHORT to avoid Windows command-line length errors during compilation
VENV_NAME           = shakermaker_venv

# Base folder for the virtual environment
# Result will be: C:\Users\your_username\shakermaker_venv
VENV_BASE           = C:\Users\%USERNAME%

# Folder where the junction will be created (must have no spaces)
COMPILER_DIR        = C:\shakermaker_compiler

# Full path to your ShakerMaker source repository
# This path may contain spaces — the junction script handles it
SHAKERMAKER_SOURCE  = C:\path\to\your\ShakerMaker

# Python packages to install in the virtual environment
# Add any additional packages you need here
PYTHON_DEPS = numpy==1.26.4 setuptools wheel h5py mpi4py matplotlib scipy pandas
```

> **Important:** Keep `VENV_NAME` short (e.g. `sm_venv`, `shaker`). Long names cause Windows command-line length errors during compilation.

---

### Step 2 — Open PowerShell as Administrator and run the scripts

All scripts must be run from a **PowerShell window opened as Administrator**. This is required because Step 3 adds Intel MPI to the Windows system PATH permanently — without Administrator rights, `mpiexec` will not work and you will only be able to run sequential simulations.

1. Press `Win + X` and select **Windows PowerShell (Admin)** or **Terminal (Admin)**
2. Navigate to the `shakermaker_windows_compilation_files` folder:
   ```powershell
   cd "C:\path\to\shakermaker_windows_compilation_files"
   ```
3. Unblock the scripts (only needed the first time, or after downloading updates):
   ```powershell
   Get-ChildItem SCRIPTS | Unblock-File
   ```
4. Launch the menu:
   ```powershell
   .\SCRIPTS\RUN_ME.bat
   ```

You will see an interactive menu:

```
+==========================================================+
|             ShakerMaker - Windows Setup Menu             | 
+==========================================================+
|                                                          |
|   [1]  Step 1 - Install Prerequisites                   |
|         (Python, Git, VS2022, Intel oneAPI, venv, deps) |
|                                                          |
|   [2]  Step 2 - Create Junction                         |
|         (link your repo to a space-free build path)     |
|                                                          |
|   [3]  Step 3 - Build and Compile                       |
|         (compile ShakerMaker + smoke test)              |
|                                                          |
|   [4]  Run All Steps in Order (1 then 2 then 3)         |
|                                                          |
|   [Q]  Quit                                             |
|                                                          |
+==========================================================+
```

For a fresh installation on a new machine, choose **[4] Run All Steps in Order**. All confirmation prompts accept **Enter** as Yes.

---

## What Each Script Does

### `shakermaker.cfg` — Central Configuration

The single source of truth for all scripts. All paths, version numbers, virtual environment names, and Python dependencies are defined here. No other file needs to be edited.

---

### `00_shakermaker_common.ps1` — Shared Functions

Loaded automatically by the other three scripts via dot-sourcing. Contains shared helper functions for reading the config, writing logs, and printing colored output. Do not run this file directly.

---

### `01_shakermaker_setup.ps1` — Prerequisites and Virtual Environment

**What it does:**

- **STEP 0** — Cleans any stale `sitecustomize.py` files left by previous installations that could cause DLL errors at startup
- **STEP 1** — Checks whether each required tool is already installed: Git, Python 3.10, Visual Studio 2022, Intel oneAPI Base and HPC Toolkits. Shows `[OK]` for installed tools and `[--]` for missing ones
- **STEP 2** — Checks whether the virtual environment already exists. If it does, reuses it. If not, marks it for creation
- **STEP 3** — Checks which Python packages from `PYTHON_DEPS` are already installed at the correct version. Missing or wrong-version packages are marked for installation
- **STEP 4** — Shows a full summary of everything that will be installed and asks for confirmation (Enter = Yes)
- **STEP 5** — Installs only the missing system components via `winget`
- **STEP 6** — Upgrades `pip` inside the virtual environment
- **STEP 7** — Installs missing Python packages from `PYTHON_DEPS`
- **STEP 8** — Writes a clean `sitecustomize.py` to the Python base so Intel DLLs and MPI environment variables are configured at every Python startup
- **STEP 9** — Final verification of all installed packages

> **Note:** Visual Studio and Intel oneAPI are large downloads (5–10 GB each). The script waits for each installer to finish before proceeding.

---

### `02_shakermaker_junction.ps1` — Junction Setup

**Why this is needed:**

The ShakerMaker source repository may live inside a path with spaces (for example inside Dropbox or OneDrive). Windows compiler tools cannot handle spaces in paths and will fail. The solution is to create a Windows junction — a filesystem link — from a clean path with no spaces to the actual repository location.

**What it does:**

- **STEP 1** — Checks if a junction already exists at `COMPILER_DIR\ShakerMaker`. If it does, asks whether to replace it (Enter = Yes)
- **STEP 2** — If `SHAKERMAKER_SOURCE` is defined in `shakermaker.cfg`, offers to use it directly. Otherwise asks the user to enter the path interactively. Validates that the path exists and contains `setup.py`
- **STEP 3** — Creates the `COMPILER_DIR` folder if it does not exist, then creates the junction using `mklink /J`
- **STEP 4** — Saves the validated source path back to `shakermaker.cfg` so the build script can find it automatically
- **STEP 5** — Verifies the junction is a valid reparse point and that `setup.py` is reachable through it

**Result:** `C:\shakermaker_compiler\ShakerMaker\` → your actual repository

---

### `03_shakermaker_build.ps1` — Build and Compile

**What it does:**

- **STEP 1** — Pre-build checks: verifies junction, virtual environment, Visual Studio, and Intel `ifx` compiler are all present
- **STEP 2** — Creates the `ifort.bat` wrapper — Intel oneAPI 2025.x ships `ifx.exe` but NumPy's f2py looks for `ifort.exe`; this wrapper redirects `ifort` calls to `ifx`
- **STEP 3** — Launches a CMD subprocess with the full build environment initialized (VsDevCmd + setvars + venv + Intel paths) and runs `python setup.py install`. If the first attempt fails due to Windows path-length limits (`WinError 206`), automatically retries using the compilation cache
- **STEP 4** — Cleans any stale `sitecustomize.py` files and writes a fresh one to both the venv and the Python base site-packages. This file sets Intel MPI environment variables (`I_MPI_FABRICS`, `FI_PROVIDER`) and DLL paths so ShakerMaker can be imported from Jupyter, VS Code, or any launcher without manual setup
- **STEP 4b** — Adds Intel compiler and MPI paths to the Windows system PATH permanently. This allows `mpiexec` to be called from any CMD window without running `setvars.bat` first. **Requires Administrator rights**
- **STEP 5** — Smoke test: runs `from shakermaker import core` inside an initialized CMD to verify the build succeeded

**Why CMD and not PowerShell for the build:**

`VsDevCmd.bat` and `setvars.bat` modify PATH and environment variables of the calling shell. This only works correctly in CMD. The script launches a temporary `.bat` file as a subprocess to handle this.

> **Note on first-time compilation:** If the virtual environment name is long, the first build attempt may fail with `WinError 206`. The script detects this automatically and retries — the second attempt uses the compilation cache and succeeds.

---

## After Installation

Once the build completes, ShakerMaker is ready to use.

### Using in Jupyter or VS Code

Select the virtual environment as your Python kernel. The imports work directly:

```python
from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.stationlist import StationList
import numpy as np
import matplotlib.pyplot as plt
```

### Running parallel simulations with MPI

Open a **fresh CMD** window, activate your virtual environment, and run:

```cmd
C:\Users\your_username\shakermaker_venv\Scripts\activate.bat
cd "C:\path\to\your\model"
mpiexec -n 10 python your_script.py
```

After Step 3, `mpiexec` is available from any CMD window without needing `setvars.bat` or `VsDevCmd.bat`.

---

## Log File

All three scripts append to a single transcript log at:

```
SCRIPTS\shakermaker.log
```

If you encounter any issue, send this file for support.

---

## Troubleshooting

### Scripts are blocked and cannot run

Open PowerShell as Administrator, navigate to `shakermaker_windows_compilation_files` and run:

```powershell
Get-ChildItem SCRIPTS | Unblock-File
```

### `mpiexec` is not recognized

Step 3 was not run as Administrator so the Intel MPI path was not added to the system PATH. Open PowerShell as Administrator and run Step 3 again.

### Build fails with `WinError 206: The filename or extension is too long`

The script retries automatically. If it still fails, use a shorter virtual environment name in `shakermaker.cfg` (e.g. `sm_venv`).

### `DLL load failed while importing core`

Run Step 3 again — it recreates `sitecustomize.py` automatically.

### `CompilerNotFound: intelvem`

The `ifort.bat` wrapper is missing. Run Step 3 again.

### `The input line is too long` in CMD

The PATH overflowed from running `VsDevCmd.bat` or `setvars.bat` multiple times in the same window. Close it and open a fresh one.

### `FileNotFoundError: barry_allen\Scripts` or similar stale venv error

A `sitecustomize.py` from a previous installation is still active. Run Step 1 — STEP 0 cleans all stale files automatically.

---

## Reinstalling ShakerMaker

Run Step 3 again. It recompiles, reinstalls, and recreates `sitecustomize.py` automatically.

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| Windows | 11 (26200.x) |
| Python | 3.10.11 |
| Visual Studio 2022 Community | 17.14.28 |
| Intel oneAPI Base Toolkit | 2025.x |
| Intel oneAPI HPC Toolkit | 2025.x |
| NumPy | 1.26.4 |
| ShakerMaker | 1.0 (branch: optimize_process) |