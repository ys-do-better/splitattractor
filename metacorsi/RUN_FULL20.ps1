param(
    [string]$Knowledge = "",
    [string]$RunRoot = "",
    [int]$Concurrency = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
if (-not $Knowledge) { $Knowledge = Join-Path $root "knowledge\active" }
$Knowledge = (Resolve-Path -LiteralPath $Knowledge).Path
if (-not $RunRoot) { $RunRoot = Join-Path $root ("runs\full20_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
if ($Concurrency -le 0) { $Concurrency = $runtime.Concurrency }
[System.IO.Directory]::CreateDirectory($RunRoot) | Out-Null
$current = Join-Path $Knowledge "current.json"
$manifestInfo = Get-Content (Join-Path $Knowledge "ltm\CURRENT.json") -Raw | ConvertFrom-Json
$manifest = Join-Path (Join-Path $Knowledge "ltm") (Join-Path $manifestInfo.path "manifest.json")
$beforeCurrent = (Get-FileHash -Algorithm SHA256 -LiteralPath $current).Hash
$beforeManifest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash
$job = "cls_full20_" + (Split-Path -Leaf $RunRoot)

& $runtime.Python -m metacorsi.harbor_agents.dsh_runner `
    --tasks-file (Join-Path $root "data\deepswe20_ids.txt") `
    --model $runtime.Model --dotenv $runtime.Dotenv `
    --ltm-policy modular_reverse_v23 --memory-strategy cls_dual_store `
    --control-mode rca_only --evolution-policy grounding_feedback `
    --dsh-patch (Join-Path $root "configs\dsh.materialize.patch.yml") `
    --dsh-materialize-patch (Join-Path $root "configs\dsh.materialize-8192.patch.yml") `
    --dsh-command-max-chars 1600 --dsh-max-steps 24 --dsh-max-extension 6 `
    --concurrency $Concurrency --read-only-knowledge --disable-candidate-replay `
    --result-dir $RunRoot --knowledge-store $current `
    --candidate-registry (Join-Path $Knowledge "candidates.json") --job-name $job
if ($LASTEXITCODE -ne 0) { throw "Full20 runner failed: $LASTEXITCODE" }

$afterCurrent = (Get-FileHash -Algorithm SHA256 -LiteralPath $current).Hash
$afterManifest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash
if ($beforeCurrent -ne $afterCurrent -or $beforeManifest -ne $afterManifest) {
    throw "Read-only full20 mutated knowledge"
}
& $runtime.Python (Join-Path $root "scripts\finalize_k28_cls_full20.py") $RunRoot
if ($LASTEXITCODE -ne 0) { throw "Full20 finalizer failed: $LASTEXITCODE" }
Write-Host "Full20 complete: $RunRoot"
