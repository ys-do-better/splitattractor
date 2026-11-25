param([switch]$Force)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$output = Join-Path $root "knowledge\releases\k29_cls_nearmiss3"

if ((Test-Path -LiteralPath $output) -and -not $Force) {
    Write-Host "K29 already exists: $output"
    Write-Host "Use -Force to rebuild it from K28 + K29 near-miss episodes."
    exit 0
}

if ($Force) {
    & $runtime.Python (Join-Path $root "scripts\build_cls_k28_k29.py") --replace
} else {
    & $runtime.Python (Join-Path $root "scripts\build_cls_k28_k29.py")
}
if ($LASTEXITCODE -ne 0) { throw "K29 build failed: $LASTEXITCODE" }
Write-Host "K29 build complete: $output"
