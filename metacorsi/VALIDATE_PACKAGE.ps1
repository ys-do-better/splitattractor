$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$bad = $false
Get-ChildItem $root -Filter *.ps1 -Recurse | ForEach-Object {
    try { $null = [scriptblock]::Create((Get-Content $_.FullName -Raw)) }
    catch { Write-Host "PowerShell parse failure: $($_.FullName)"; $bad = $true }
}
if ($bad) { throw "PowerShell validation failed" }
& $runtime.Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
$smoke = Join-Path $root "runs\package_validation\knowledge"
& $runtime.Python (Join-Path $root "scripts\build_cls_k20_k28.py") `
    --k20 (Join-Path $root "knowledge_sources\k20_215") `
    --k28 (Join-Path $root "knowledge_sources\k28_251") --output $smoke
if ($LASTEXITCODE -ne 0) { throw "local CLS rebuild failed" }
& $runtime.Python (Join-Path $root "scripts\consolidate_active_cls.py") --knowledge $smoke
if ($LASTEXITCODE -ne 0) { throw "local CLS consolidation failed" }
& $runtime.Python (Join-Path $root "scripts\audit_cls_k20_k28.py") `
    --knowledge $smoke --output (Join-Path $root "runs\package_validation\RETRIEVAL_AUDIT.json")
if ($LASTEXITCODE -ne 0) { throw "local retrieval audit failed" }
Write-Host "Package validation: PASS"
