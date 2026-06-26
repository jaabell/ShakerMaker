# ==============================================================================
#  ShakerMaker - Shared Config Reader
#  This file is dot-sourced by all scripts:
#      . "$PSScriptRoot\shakermaker_common.ps1"
# ==============================================================================

function Read-ShakerConfig {
    param([string]$ConfigPath)

    if (-not (Test-Path $ConfigPath)) {
        Write-Host "  [!!]  Config file not found: $ConfigPath" -ForegroundColor Red
        Write-Host "        Please make sure shakermaker.cfg is in the same folder as the scripts." -ForegroundColor Gray
        Read-Host "  Press Enter to exit"
        exit 1
    }

    $cfg = @{}

    Get-Content $ConfigPath | ForEach-Object {
        $line = $_.Trim()
        # Skip empty lines and comments
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -match "^([^=]+)=(.*)$") {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Skip lines where value is empty (e.g. commented-out SHAKERMAKER_SOURCE)
            if ($value -ne "") {
                # Expand %USERNAME%
                $value = $value -replace "%USERNAME%", $env:USERNAME
                $cfg[$key] = $value
            }
        }
    }

    # --- Validate required keys -----------------------------------------------
    $required = @("PYTHON_VERSION", "PYTHON_FULL_VERSION", "VENV_NAME", "VENV_BASE", "COMPILER_DIR", "PYTHON_DEPS")
    foreach ($key in $required) {
        if (-not $cfg.ContainsKey($key)) {
            Write-Host "  [!!]  Missing required config key: $key" -ForegroundColor Red
            Write-Host "        Please check shakermaker.cfg" -ForegroundColor Gray
            Read-Host "  Press Enter to exit"
            exit 1
        }
    }

    return $cfg
}

function Write-ShakerLog {
    param([string]$LogFile, [string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LogFile -Append | Out-Null
}

function Print-Header {
    param([string]$title)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Print-OK   { param([string]$msg) Write-Host "  [OK]  $msg" -ForegroundColor Green  }
function Print-SKIP { param([string]$msg) Write-Host "  [--]  $msg" -ForegroundColor Yellow }
function Print-FAIL { param([string]$msg) Write-Host "  [!!]  $msg" -ForegroundColor Red    }
function Print-INFO { param([string]$msg) Write-Host "        $msg" -ForegroundColor Gray   }
