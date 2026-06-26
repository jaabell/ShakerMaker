# ==============================================================================
#  ShakerMaker - Step 1: Prerequisites + Virtual Environment + Python Deps
#  Version : 2.0
# ==============================================================================
#  Run from RUN_ME.bat or manually:
#      PowerShell -ExecutionPolicy Bypass -File .\shakermaker_setup.ps1
# ==============================================================================

param([switch]$NonInteractive)

. "$PSScriptRoot\00_shakermaker_common.ps1"

# Wraps "Press Enter to exit" so it is skipped when called from Run All
function Wait-Enter { if (-not $NonInteractive) { Read-Host "  Press Enter to exit" } }

# --- Load config --------------------------------------------------------------
$cfg            = Read-ShakerConfig "$PSScriptRoot\shakermaker.cfg"
$PYTHON_VERSION      = $cfg["PYTHON_VERSION"]
$PYTHON_FULL_VERSION = $cfg["PYTHON_FULL_VERSION"]
$VENV_NAME           = $cfg["VENV_NAME"]
$VENV_BASE           = $cfg["VENV_BASE"]
$PYTHON_DEPS         = $cfg["PYTHON_DEPS"] -split "\s+"
$LOG_FILE            = "$VENV_BASE\shakermaker_setup.log"
$VENV_PATH           = "$VENV_BASE\$VENV_NAME"

# Derived
$PYTHON_MAJOR = ($PYTHON_VERSION -split "\.")[0]
$PYTHON_MINOR = ($PYTHON_VERSION -split "\.")[1]
$PY_CMD       = "-$PYTHON_VERSION"
$WINGET_PYID  = "Python.Python.$PYTHON_MAJOR.$PYTHON_MINOR"
$PY_REGEX     = "Python $PYTHON_MAJOR\.$PYTHON_MINOR"

function Log { param([string]$m) Write-ShakerLog $LOG_FILE $m }

# --- Banner -------------------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |      ShakerMaker - Step 1: Prerequisites Setup          |" -ForegroundColor Cyan
Write-Host "  |      Python $PYTHON_FULL_VERSION  |  Venv: $VENV_NAME  |  User: $env:USERNAME" -ForegroundColor Cyan
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Log "Setup started - user: $env:USERNAME - Python: $PYTHON_FULL_VERSION - Venv: $VENV_NAME"
Start-Transcript -Path "$PSScriptRoot\shakermaker.log" -Append -Force | Out-Null

# ==============================================================================
Print-Header "STEP 0 - Clean stale sitecustomize.py"
# ==============================================================================
# Remove any stale sitecustomize.py files before doing anything else.
# Stale files from previous installations cause errors when creating or using any venv.

$INTEL_BIN_PATH = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
$INTEL_MPI_PATH = "C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin"

$cleanSiteContent = @"
import os
os.environ["I_MPI_FABRICS"] = "shm"
os.environ["FI_PROVIDER"] = "sockets"
os.add_dll_directory(r"$INTEL_BIN_PATH")
os.add_dll_directory(r"$INTEL_MPI_PATH")
"@

try {
    $pyBaseExe = (py -3.10 -c "import sys; print(sys.executable)" 2>&1)
    $pyBaseDir  = Split-Path $pyBaseExe

    # Check root of Python base (e.g. C:\Users\...\Python310\sitecustomize.py)
    $siteBaseRoot = "$pyBaseDir\sitecustomize.py"
    if (Test-Path $siteBaseRoot) {
        Remove-Item $siteBaseRoot -Force
        Print-INFO "Removed stale sitecustomize.py from Python root: $siteBaseRoot"
        Log "[--] Removed stale $siteBaseRoot"
    }

    # Check Lib\site-packages
    $siteBase = "$pyBaseDir\Lib\site-packages\sitecustomize.py"
    if (Test-Path $siteBase) {
        Remove-Item $siteBase -Force
        Print-INFO "Removed stale sitecustomize.py from Python base: $siteBase"
        Log "[--] Removed stale $siteBase"
    }

    [System.IO.File]::WriteAllText($siteBase, $cleanSiteContent)
    Print-OK "Clean sitecustomize.py written to Python base"
    Log "[OK] Clean sitecustomize.py written to $siteBase"
} catch {
    Print-FAIL "Could not clean Python base sitecustomize.py: $_"
    Log "[!!] sitecustomize.py clean failed: $_"
}

$siteVenv = "$VENV_PATH\Lib\site-packages\sitecustomize.py"
if (Test-Path $siteVenv) {
    Remove-Item $siteVenv -Force
    Print-INFO "Removed stale sitecustomize.py from venv"
    Log "[--] Removed stale $siteVenv"
}

# ==============================================================================
Print-Header "STEP 1 - System Checklist"
# ==============================================================================

$missing = [System.Collections.ArrayList]@()

# Git
$gitExe = Get-Command git -ErrorAction SilentlyContinue
if ($gitExe) {
    $gitVer = (git --version 2>&1)
    Print-OK "Git found: $gitVer"
    Log "[OK] Git: $gitVer"
} else {
    Print-SKIP "Git not found - will install"
    Log "[--] Git not found"
    $missing.Add("Git") | Out-Null
}

# Python
$pyExe = Get-Command py -ErrorAction SilentlyContinue
$pythonFound = $false
if ($pyExe) {
    try {
        $pyVer = (py $PY_CMD --version 2>&1)
        if ($pyVer -match $PY_REGEX) {
            Print-OK "Python $PYTHON_VERSION found: $pyVer"
            Log "[OK] Python $PYTHON_VERSION"
            $pythonFound = $true
        }
    } catch {}
}
if (-not $pythonFound) {
    Print-SKIP "Python $PYTHON_VERSION not found - will install"
    Log "[--] Python $PYTHON_VERSION not found"
    $missing.Add("Python") | Out-Null
}

# Visual Studio 2022
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if (Test-Path $vsPath) {
    Print-OK "Visual Studio 2022 Community found"
    Log "[OK] Visual Studio 2022"
} else {
    Print-SKIP "Visual Studio 2022 not found - will install"
    Log "[--] Visual Studio 2022 not found"
    $missing.Add("VisualStudio") | Out-Null
}

# Intel oneAPI Base
$oneapiBase = "C:\Program Files (x86)\Intel\oneAPI\mkl"
if (Test-Path $oneapiBase) {
    Print-OK "Intel oneAPI Base Toolkit found (MKL present)"
    Log "[OK] Intel oneAPI Base"
} else {
    Print-SKIP "Intel oneAPI Base Toolkit not found - will install"
    Log "[--] Intel oneAPI Base not found"
    $missing.Add("IntelBase") | Out-Null
}

# Intel oneAPI HPC
$ifxExe = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe"
if (Test-Path $ifxExe) {
    Print-OK "Intel oneAPI HPC Toolkit found (ifx present)"
    Log "[OK] Intel oneAPI HPC"
} else {
    Print-SKIP "Intel oneAPI HPC Toolkit not found - will install"
    Log "[--] Intel oneAPI HPC not found"
    $missing.Add("IntelHPC") | Out-Null
}

# ==============================================================================
Print-Header "STEP 2 - Virtual Environment"
# ==============================================================================

if (Test-Path "$VENV_PATH\Scripts\activate.bat") {
    Print-OK "Virtual environment $VENV_NAME already exists - will use it"
    Print-INFO "Path: $VENV_PATH"
    Log "[OK] Venv exists at $VENV_PATH"
} else {
    Print-SKIP "Virtual environment $VENV_NAME not found - will create it"
    Print-INFO "Path: $VENV_PATH"
    Log "[--] Venv not found - will create"
    $missing.Add("VirtualEnv") | Out-Null
}

# ==============================================================================
Print-Header "STEP 3 - Python Dependencies Checklist"
# ==============================================================================

$missingDeps = [System.Collections.ArrayList]@()

if (Test-Path "$VENV_PATH\Scripts\python.exe") {
    $pipList = & "$VENV_PATH\Scripts\python.exe" -m pip list --format=freeze 2>&1
    foreach ($dep in $PYTHON_DEPS) {
        $pkgName    = ($dep -split "==")[0]
        $pkgVersion = if ($dep -match "==") { ($dep -split "==")[1] } else { $null }
        $found      = $pipList | Where-Object { $_ -match "(?i)^$pkgName==" }
        if ($found) {
            $installedVer = ($found -split "==")[1]
            if ($pkgVersion -and $installedVer -ne $pkgVersion) {
                Print-SKIP "$pkgName installed ($installedVer) but need $pkgVersion - will reinstall"
                Log "[--] $pkgName wrong version: $installedVer vs $pkgVersion"
                $missingDeps.Add($dep) | Out-Null
            } else {
                Print-OK "$pkgName $installedVer already installed"
                Log "[OK] $pkgName $installedVer"
            }
        } else {
            Print-SKIP "$pkgName not found - will install"
            Log "[--] $pkgName not found"
            $missingDeps.Add($dep) | Out-Null
        }
    }
} else {
    Print-INFO "Virtual environment not yet created - all deps will be installed after"
    foreach ($dep in $PYTHON_DEPS) { $missingDeps.Add($dep) | Out-Null }
}

# ==============================================================================
Print-Header "STEP 4 - Installation Summary"
# ==============================================================================

if ($missing.Count -eq 0 -and $missingDeps.Count -eq 0) {
    Write-Host ""
    Write-Host "  Everything is already installed. Nothing to do!" -ForegroundColor Green
    Write-Host ""
    Log "Nothing to install - all checks passed"
    Wait-Enter
    exit 0
}

Write-Host ""
Write-Host "  System components to install:" -ForegroundColor White
if ($missing.Count -eq 0) {
    Write-Host "    (none)" -ForegroundColor Gray
} else {
    foreach ($item in $missing) { Write-Host "    - $item" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "  Python packages to install/update:" -ForegroundColor White
if ($missingDeps.Count -eq 0) {
    Write-Host "    (none)" -ForegroundColor Gray
} else {
    foreach ($dep in $missingDeps) { Write-Host "    - $dep" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "  Virtual environment : $VENV_NAME" -ForegroundColor White
Write-Host "  Location            : $VENV_PATH" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "  Do you want to proceed with the installation? (Y/N - Enter = Yes)"
if ($confirm -ne "" -and $confirm -notmatch "^[Yy]$") {
    Write-Host "  Aborted by user." -ForegroundColor Red
    Log "Aborted by user"
    exit 0
}

# ==============================================================================
Print-Header "STEP 5 - Installing System Components"
# ==============================================================================

foreach ($item in $missing) {
    switch ($item) {

        "Git" {
            Write-Host "  Installing Git..." -ForegroundColor White
            Log "Installing Git"
            winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -eq 0) { Print-OK "Git installed"; Log "[OK] Git installed" }
            else { Print-FAIL "Git failed (exit $LASTEXITCODE)"; Log "[!!] Git failed" }
        }

        "Python" {
            Write-Host "  Installing Python $PYTHON_FULL_VERSION..." -ForegroundColor White
            Log "Installing Python $PYTHON_FULL_VERSION (winget id: $WINGET_PYID)"
            winget install --id $WINGET_PYID -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -eq 0) { Print-OK "Python $PYTHON_FULL_VERSION installed"; Log "[OK] Python installed" }
            else { Print-FAIL "Python failed (exit $LASTEXITCODE)"; Log "[!!] Python failed" }
        }

        "VisualStudio" {
            Write-Host "  Installing Visual Studio 2022 (10-20 min)..." -ForegroundColor White
            Log "Installing Visual Studio 2022"
            winget install --id Microsoft.VisualStudio.2022.Community -e `
                --accept-source-agreements --accept-package-agreements `
                --override "--wait --quiet --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
            if ($LASTEXITCODE -eq 0) { Print-OK "Visual Studio 2022 installed"; Log "[OK] VS2022 installed" }
            else { Print-FAIL "Visual Studio failed (exit $LASTEXITCODE)"; Log "[!!] VS2022 failed" }
        }

        "IntelBase" {
            Write-Host "  Installing Intel oneAPI Base Toolkit (~5 GB)..." -ForegroundColor White
            Log "Installing Intel oneAPI Base"
            winget install --id Intel.OneAPI.BaseToolkit -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -eq 0) { Print-OK "Intel oneAPI Base installed"; Log "[OK] Intel Base installed" }
            else { Print-FAIL "Intel Base failed (exit $LASTEXITCODE)"; Log "[!!] Intel Base failed" }
        }

        "IntelHPC" {
            Write-Host "  Installing Intel oneAPI HPC Toolkit (~5 GB)..." -ForegroundColor White
            Log "Installing Intel oneAPI HPC"
            winget install --id Intel.OneAPI.HPCToolkit -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -eq 0) { Print-OK "Intel oneAPI HPC installed"; Log "[OK] Intel HPC installed" }
            else { Print-FAIL "Intel HPC failed (exit $LASTEXITCODE)"; Log "[!!] Intel HPC failed" }
        }

        "VirtualEnv" {
            Write-Host "  Creating virtual environment $VENV_NAME ..." -ForegroundColor White
            Log "Creating venv at $VENV_PATH with Python $PYTHON_VERSION"
            try {
                py $PY_CMD -m venv $VENV_PATH
                Print-OK "Virtual environment created at $VENV_PATH"
                Log "[OK] Venv created at $VENV_PATH"
            } catch {
                Print-FAIL "Failed to create venv: $_"
                Log "[!!] Venv creation failed: $_"
            }
        }
    }
}

# ==============================================================================
Print-Header "STEP 6 - Upgrading pip"
# ==============================================================================

$pythonExe = "$VENV_PATH\Scripts\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe -m pip install --upgrade pip --quiet
    $pipVer = (& $pythonExe -m pip --version 2>&1)
    Print-OK "pip upgraded: $pipVer"
    Log "[OK] pip: $pipVer"
} else {
    Print-FAIL "Python not found at $pythonExe"
    Log "[!!] pip upgrade failed - python not found"
}

# ==============================================================================
Print-Header "STEP 7 - Installing Python Dependencies"
# ==============================================================================

if ($missingDeps.Count -gt 0) {
    foreach ($dep in $missingDeps) {
        Write-Host "  Installing $dep ..." -ForegroundColor White
        Log "pip install $dep"
        & $pythonExe -m pip install $dep --quiet
        if ($LASTEXITCODE -eq 0) { Print-OK "$dep installed"; Log "[OK] $dep" }
        else { Print-FAIL "$dep failed"; Log "[!!] $dep failed" }
    }
} else {
    Print-OK "All Python dependencies already satisfied"
}

# ==============================================================================
Print-Header "STEP 8 - Fix Python Base sitecustomize.py"
# ==============================================================================

# Write sitecustomize.py to the Python base so that any Python process
# (Jupyter, VS Code, CMD) loads Intel DLL paths automatically at startup.
# This also prevents errors from stale sitecustomize.py files left by old venvs.

$INTEL_BIN_PATH = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
$INTEL_MPI_PATH = "C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin"

$siteContent = @"
import os
os.environ["I_MPI_FABRICS"] = "shm"
os.environ["FI_PROVIDER"] = "sockets"
os.add_dll_directory(r"$INTEL_BIN_PATH")
os.add_dll_directory(r"$INTEL_MPI_PATH")
"@

$pythonBaseExe = & $pythonExe -c "import sys; print(sys._base_executable)" 2>&1
$pythonBaseDir = Split-Path $pythonBaseExe
$siteCustomizeBase = "$pythonBaseDir\Lib\site-packages\sitecustomize.py"

# Remove stale sitecustomize.py from Python base if it exists
if (Test-Path $siteCustomizeBase) {
    Remove-Item $siteCustomizeBase -Force
    Print-INFO "Removed stale sitecustomize.py from Python base"
    Log "[--] Removed stale sitecustomize.py from $siteCustomizeBase"
}

# Remove stale sitecustomize.py from venv if it exists
$siteCustomizeVenv = "$VENV_PATH\Lib\site-packages\sitecustomize.py"
if (Test-Path $siteCustomizeVenv) {
    Remove-Item $siteCustomizeVenv -Force
    Print-INFO "Removed stale sitecustomize.py from venv"
    Log "[--] Removed stale sitecustomize.py from $siteCustomizeVenv"
}

try {
    [System.IO.File]::WriteAllText($siteCustomizeBase, $siteContent)
    Print-OK "sitecustomize.py written to Python base: $siteCustomizeBase"
    Log "[OK] sitecustomize.py base written"
} catch {
    Print-FAIL "Failed to write sitecustomize.py to Python base: $_"
    Log "[!!] sitecustomize.py base failed: $_"
}

# ==============================================================================
Print-Header "STEP 9 - Final Verification"
# ==============================================================================

$venvPyVer = (& $pythonExe --version 2>&1)
Print-OK "Virtual env Python: $venvPyVer"
Log "Verification - Python: $venvPyVer"

foreach ($dep in $PYTHON_DEPS) {
    $pkgName = ($dep -split "==")[0]
    $result  = & $pythonExe -m pip show $pkgName 2>&1 | Select-String "^Version:"
    if ($result) {
        $ver = ($result -split ":")[1].Trim()
        Print-OK "$pkgName $ver"
        Log "[OK] $pkgName $ver"
    } else {
        Print-FAIL "$pkgName could not be verified"
        Log "[!!] $pkgName verify failed"
    }
}

# ==============================================================================
$line = "=" * 70
Write-Host ""
Write-Host $line -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Venv : $VENV_NAME  at  $VENV_PATH" -ForegroundColor White
Write-Host "  Log  : $LOG_FILE" -ForegroundColor White
Write-Host "  Next : Run Step 2 - Junction Setup" -ForegroundColor Yellow

Write-Host $line -ForegroundColor Cyan
Write-Host ""
Log "Setup complete"
Stop-Transcript | Out-Null
Wait-Enter