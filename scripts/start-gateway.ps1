# Starts the GPT-6 Sol MCP gateway as a hidden background process.
#
# Used by the Startup-folder shortcut so the gateway survives logoff/reboot.
# Safe to run repeatedly: an existing listener on the port is left alone.

$ErrorActionPreference = "Stop"

$gwDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $gwDir "venv\Scripts\pythonw.exe"
$logDir = Join-Path $gwDir "logs"
$logFile = Join-Path $logDir "gateway.log"

if (-not (Test-Path $python)) {
    Write-Error "Interpreter not found: $python"
    exit 1
}

# Read the port from .env so this script never hardcodes it.
$port = 7420
$envFile = Join-Path $gwDir ".env"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern '^\s*GATEWAY_PORT\s*=\s*(\d+)' |
        Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

# Already listening? Nothing to do.
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Gateway already listening on port $port (PID $($existing[0].OwningProcess))."
    exit 0
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Rotate the log so it cannot grow without bound.
if (Test-Path $logFile) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Move-Item $logFile (Join-Path $logDir "gateway-$stamp.log") -Force
    Get-ChildItem $logDir -Filter "gateway-*.log" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip 5 |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

$env:PYTHONPATH = Join-Path $gwDir "src"

$proc = Start-Process -FilePath $python `
    -ArgumentList @(
        "-m", "uvicorn", "gpt6_sol_mcp.gateway.app:app",
        "--host", "127.0.0.1", "--port", "$port"
    ) `
    -WorkingDirectory $gwDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError (Join-Path $logDir "gateway.err.log") `
    -PassThru

# Wait briefly so a crash-on-startup is visible in the exit code.
Start-Sleep -Seconds 4
if ($proc.HasExited) {
    Write-Error "Gateway exited immediately (code $($proc.ExitCode)). See $logDir"
    exit 1
}

Write-Output "Gateway started on port $port (PID $($proc.Id)). Log: $logFile"
