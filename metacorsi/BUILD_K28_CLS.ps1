param([switch]$SkipAudit)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"

& $runtime.Python (Join-Path $root "scripts\build_cls_k20_k28.py")
if ($LASTEXITCODE -ne 0) { throw "CLS build failed: $LASTEXITCODE" }
if (-not $SkipAudit) {
    & $runtime.Python (Join-Path $root "scripts\audit_cls_k20_k28.py") `
        --knowledge (Join-Path $root "knowledge\active") `
        --output (Join-Path $root "evidence\RETRIEVAL_AUDIT.json")
    if ($LASTEXITCODE -ne 0) { throw "CLS retrieval audit failed: $LASTEXITCODE" }
}
Write-Host "K28-CLS active brain built and audited under $root\knowledge\active"
