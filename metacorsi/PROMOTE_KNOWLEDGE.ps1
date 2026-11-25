param([Parameter(Mandatory=$true)][string]$Knowledge)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$source = (Resolve-Path -LiteralPath $Knowledge).Path
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $root "runs")).Path
if (-not $source.StartsWith($runsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Promotion source must be a run-local brain below $runsRoot"
}
foreach ($name in @('current.json','candidates.json','grounding_graph.json','ltm')) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $name))) { throw "Missing promotion artifact: $name" }
}
$active = Join-Path $root "knowledge\active"
$backup = Join-Path $root ("backups\active_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
[System.IO.Directory]::CreateDirectory($backup) | Out-Null
Copy-Item -Path (Join-Path $active "*") -Destination $backup -Recurse -Force
Copy-Item -Path (Join-Path $source "*") -Destination $active -Recurse -Force
Write-Host "Promoted: $source"
Write-Host "Recoverable backup: $backup"
