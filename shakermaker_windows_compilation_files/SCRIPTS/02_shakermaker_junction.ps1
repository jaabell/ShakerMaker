# ==============================================================================
#  ShakerMaker - Step 2: Junction Setup
#  Version : 2.0
# ==============================================================================
#  Creates a space-free junction at C:\shakermaker_compiler\ShakerMaker
#  pointing to your actual repository (which may have spaces in its path).
#
#  Run from RUN_ME.bat or manually:
#      PowerShell -ExecutionPolicy Bypass -File .\shakermaker_junction.ps1
# ==============================================================================

param([switch]$NonInteractive)

. "$PSScriptRoot\00_shakermaker_common.ps1"

# Wraps "Press Enter to exit" so it is skipped when called from Run All
function Wait-Enter { if (-not $NonInteractive) { Read-Host "  Press Enter to exit" } }

# --- Load config --------------------------------------------------------------
$cfg             = Read-ShakerConfig "$PSScriptRoot\shakermaker.cfg"
$COMPILER_DIR    = $cfg["COMPILER_DIR"]
$JUNCTION_PATH   = "$COMPILER_DIR\ShakerMaker"
$LOG_FILE        = "C:\Users\$env:USERNAME\shakermaker_junction.log"

# SHAKERMAKER_SOURCE is optional in cfg (may be commented out)
$SOURCE_FROM_CFG = if ($cfg.ContainsKey("SHAKERMAKER_SOURCE")) { $cfg["SHAKERMAKER_SOURCE"] } else { "" }

function Log { param([string]$m) Write-ShakerLog $LOG_FILE $m }

# --- Banner -------------------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |      ShakerMaker - Step 2: Junction Setup               |" -ForegroundColor Cyan
Write-Host "  |      User: $env:USERNAME" -ForegroundColor Cyan
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Log "Junction setup started - user: $env:USERNAME"
Start-Transcript -Path "$PSScriptRoot\shakermaker.log" -Append -Force | Out-Null

# ==============================================================================
Print-Header "STEP 1 - Check Existing Junction"
# ==============================================================================

$existingTarget = ""

if (Test-Path $JUNCTION_PATH) {
    $item = Get-Item $JUNCTION_PATH -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        $existingTarget = $item.Target
        Print-OK "Junction already exists at $JUNCTION_PATH"
        Print-INFO "Currently pointing to: $existingTarget"
        Log "[OK] Junction exists - target: $existingTarget"

        Write-Host ""
        $overwrite = Read-Host "  Do you want to replace the existing junction? (Y/N - Enter = Yes)"
        if ($overwrite -ne "" -and $overwrite -notmatch "^[Yy]$") {
            Write-Host ""
            Print-OK "Keeping existing junction. Nothing changed."
            Log "User kept existing junction"
            Wait-Enter
            exit 0
        }
        Remove-Item $JUNCTION_PATH -Force
        Print-INFO "Existing junction removed"
        Log "Existing junction removed"
    }
} else {
    Print-SKIP "No junction found at $JUNCTION_PATH - will create"
    Log "[--] No junction found"
}

# ==============================================================================
Print-Header "STEP 2 - ShakerMaker Source Location"
# ==============================================================================

Write-Host ""

$sourcePath = ""

# Pre-fill from config if available
if ($SOURCE_FROM_CFG -ne "") {
    Write-Host "  Source path found in shakermaker.cfg:" -ForegroundColor White
    Write-Host "  $SOURCE_FROM_CFG" -ForegroundColor Yellow
    Write-Host ""
    $useCfg = Read-Host "  Use this path? (Y/N - Enter = Yes)"
    if ($useCfg -eq "" -or $useCfg -match "^[Yy]$") {
        $sourcePath = $SOURCE_FROM_CFG
    }
}

# Ask user if not set
if ($sourcePath -eq "") {
    Write-Host "  Enter the full path to your ShakerMaker repository." -ForegroundColor White
    Write-Host "  This folder must contain setup.py and the shakermaker/ subfolder." -ForegroundColor Gray
    Write-Host "  Example: C:\Dropbox\01. Brain\11. GitHub\ShakerMaker_OP" -ForegroundColor Gray
    Write-Host ""

    $attempts = 0
    while ($true) {
        $attempts++
        if ($attempts -gt 5) {
            Print-FAIL "Too many invalid attempts. Exiting."
            Log "Aborted - too many invalid path attempts"
            Wait-Enter
            exit 1
        }

        $input = Read-Host "  ShakerMaker source path"
        $input = $input.Trim().Trim('"')

        if ([string]::IsNullOrWhiteSpace($input)) {
            Print-FAIL "Path cannot be empty. Try again."
            continue
        }

        if (-not (Test-Path $input)) {
            Print-FAIL "Path does not exist: $input"
            Print-INFO "Check the path and try again."
            continue
        }

        if (-not (Test-Path "$input\setup.py")) {
            Print-FAIL "setup.py not found in: $input"
            Print-INFO "Make sure this is the root of the ShakerMaker repository."
            continue
        }

        $sourcePath = $input
        break
    }
}

# Validate the path from cfg if it was used
if ($SOURCE_FROM_CFG -ne "" -and $sourcePath -eq $SOURCE_FROM_CFG) {
    if (-not (Test-Path $sourcePath)) {
        Print-FAIL "Path from config does not exist: $sourcePath"
        Print-INFO "Update SHAKERMAKER_SOURCE in shakermaker.cfg and try again."
        Log "[!!] Source path from cfg not found: $sourcePath"
        Wait-Enter
        exit 1
    }
    if (-not (Test-Path "$sourcePath\setup.py")) {
        Print-FAIL "setup.py not found in: $sourcePath"
        Print-INFO "Update SHAKERMAKER_SOURCE in shakermaker.cfg and try again."
        Log "[!!] setup.py not found in cfg path: $sourcePath"
        Wait-Enter
        exit 1
    }
    Print-OK "Source path validated: $sourcePath"
    Log "[OK] Source path validated: $sourcePath"
}

# ==============================================================================
Print-Header "STEP 3 - Create Junction"
# ==============================================================================

# Create compiler dir if needed
if (-not (Test-Path $COMPILER_DIR)) {
    try {
        New-Item -ItemType Directory -Path $COMPILER_DIR -Force | Out-Null
        Print-OK "Created directory: $COMPILER_DIR"
        Log "[OK] Created $COMPILER_DIR"
    } catch {
        Print-FAIL "Failed to create $COMPILER_DIR : $_"
        Log "[!!] Failed to create $COMPILER_DIR"
        Wait-Enter
        exit 1
    }
} else {
    Print-OK "Directory already exists: $COMPILER_DIR"
    Log "[OK] $COMPILER_DIR already exists"
}

# Create junction
try {
    $result = cmd /c "mklink /J `"$JUNCTION_PATH`" `"$sourcePath`"" 2>&1
    if (Test-Path $JUNCTION_PATH) {
        Print-OK "Junction created successfully"
        Print-INFO "  $JUNCTION_PATH"
        Print-INFO "  => $sourcePath"
        Log "[OK] Junction created: $JUNCTION_PATH => $sourcePath"
    } else {
        Print-FAIL "Junction creation failed: $result"
        Log "[!!] Junction creation failed: $result"
        Wait-Enter
        exit 1
    }
} catch {
    Print-FAIL "Error creating junction: $_"
    Log "[!!] Exception creating junction: $_"
    Wait-Enter
    exit 1
}

# ==============================================================================
Print-Header "STEP 4 - Update Config File"
# ==============================================================================

# Write SHAKERMAKER_SOURCE back to cfg so build script picks it up
$cfgPath    = "$PSScriptRoot\shakermaker.cfg"
$cfgContent = Get-Content $cfgPath

# Remove existing SHAKERMAKER_SOURCE lines (commented or not)
$cfgContent = $cfgContent | Where-Object { $_ -notmatch "^\s*#?\s*SHAKERMAKER_SOURCE\s*=" }

# Append the validated value
$cfgContent += "SHAKERMAKER_SOURCE = $sourcePath"
$cfgContent | Set-Content $cfgPath

Print-OK "SHAKERMAKER_SOURCE saved to shakermaker.cfg"
Log "[OK] Config updated with SHAKERMAKER_SOURCE = $sourcePath"

# ==============================================================================
Print-Header "STEP 5 - Verify Junction"
# ==============================================================================

$verify = Get-Item $JUNCTION_PATH -Force -ErrorAction SilentlyContinue
if ($verify -and ($verify.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    Print-OK "Junction verified as reparse point"
    Print-OK "setup.py reachable at $JUNCTION_PATH\setup.py : $(Test-Path "$JUNCTION_PATH\setup.py")"
    Log "[OK] Junction verified"
} else {
    Print-FAIL "Junction verification failed"
    Log "[!!] Junction verification failed"
}

# ==============================================================================
$line = "=" * 70
Write-Host ""
Write-Host $line -ForegroundColor Cyan
Write-Host "  Junction setup complete!" -ForegroundColor Green
Write-Host "  Junction : $JUNCTION_PATH" -ForegroundColor White
Write-Host "  Source   : $sourcePath" -ForegroundColor White
Write-Host "  Log      : $LOG_FILE" -ForegroundColor White
Write-Host "  Next     : Run Step 3 - Build and Compile" -ForegroundColor Yellow
Write-Host $line -ForegroundColor Cyan
Write-Host ""
Log "Junction setup complete"
Stop-Transcript | Out-Null
Wait-Enter