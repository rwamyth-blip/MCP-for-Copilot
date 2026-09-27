# Stops the GPT-6 Sol MCP gateway listening on the configured port.
#
# Only terminates the process that owns the listening socket, so an unrelated
# python process is never killed by name.

$ErrorActionPreference = "Stop"

$gwDir = Split-Path -Parent $PSScriptRoot

$port = 7420
$envFile = Join-Path $gwDir ".env"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern '^\s*GATEWAY_PORT\s*=\s*(\d+)' |
        Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Output "Nothing listening on port $port."
    exit 0
}

foreach ($ownerPid in ($listeners.OwningProcess | Select-Object -Unique)) {
    Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
    Write-Output "Stopped PID $ownerPid."
}

Start-Sleep -Seconds 2
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "Port $port is still in use."
    exit 1
}
Write-Output "Port $port is free."
