param([string]$PythonLauncher = "py")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
    & $PythonLauncher -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python 3.11 venv" }
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip "setuptools>=68"
if ($LASTEXITCODE -ne 0) { throw "pip/setuptools bootstrap failed" }
& $python -m pip install -e "$root[dev]"
if ($LASTEXITCODE -ne 0) { throw "local package installation failed" }
Write-Host "Local environment ready: $python"
Write-Host "Copy external\local.example.ps1 to external\local.ps1 if you need custom paths."
