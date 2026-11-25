param(
    [Parameter(Mandatory=$true)][string]$TasksFile,
    [string]$PanelName = "retention",
    [int]$Concurrency = 5,
    [int]$MaxSteps = 36,
    [int]$MaxExtension = 12,
    [string]$RunRoot = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$env:METACORSI_DOTENV = "D:\.ds"
$env:METACORSI_MODEL = "deepseek-flash"
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$knowledge = Join-Path $root "knowledge\releases\k29_cls_nearmiss3"
$current = Join-Path $knowledge "current.json"
$tasks = (Resolve-Path -LiteralPath $TasksFile).Path
if (-not (Test-Path -LiteralPath $current)) { throw "Build K29 first: $current" }
if (-not $RunRoot) {
    $RunRoot = Join-Path $root ("runs\k29_" + $PanelName + "_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
[System.IO.Directory]::CreateDirectory($RunRoot) | Out-Null
$manifestInfo = Get-Content (Join-Path $knowledge "ltm\CURRENT.json") -Raw | ConvertFrom-Json
$manifest = Join-Path (Join-Path $knowledge "ltm") (Join-Path $manifestInfo.path "manifest.json")
$beforeCurrent = (Get-FileHash -Algorithm SHA256 -LiteralPath $current).Hash
$beforeManifest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash
$job = "k29_" + $PanelName

& $runtime.Python -m metacorsi.harbor_agents.dsh_runner `
    --tasks-file $tasks --model $runtime.Model --dotenv $runtime.Dotenv `
    --ltm-policy modular_reverse_v23 --memory-strategy cls_dual_store `
    --control-mode rca_only --evolution-policy grounding_feedback `
    --dsh-patch (Join-Path $root "configs\dsh.deepseek-official.patch.yml") `
    --dsh-materialize-patch (Join-Path $root "configs\dsh.deepseek-official-materialize.patch.yml") `
    --dsh-command-max-chars 2400 --dsh-call-timeout 600 `
    --dsh-max-steps $MaxSteps --dsh-max-extension $MaxExtension `
    --concurrency $Concurrency --read-only-knowledge --disable-candidate-replay `
    --result-dir $RunRoot --knowledge-store $current `
    --candidate-registry (Join-Path $knowledge "candidates.json") --job-name $job
if ($LASTEXITCODE -ne 0) { throw "K29 panel failed: $LASTEXITCODE" }

$afterCurrent = (Get-FileHash -Algorithm SHA256 -LiteralPath $current).Hash
$afterManifest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash
if ($beforeCurrent -ne $afterCurrent -or $beforeManifest -ne $afterManifest) {
    throw "Read-only K29 panel mutated knowledge"
}
Write-Host "K29 $PanelName panel complete: $RunRoot"
