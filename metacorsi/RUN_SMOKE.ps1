param([string]$Task = "abs-stepped-slices")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$run = Join-Path $root ("runs\smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
[System.IO.Directory]::CreateDirectory($run) | Out-Null
$taskFile = Join-Path $run "task.txt"
[System.IO.File]::WriteAllText($taskFile, $Task + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
$job = "smoke_" + $Task
& $runtime.Python -m metacorsi.harbor_agents.dsh_runner `
    --tasks-file $taskFile --model $runtime.Model --dotenv $runtime.Dotenv `
    --ltm-policy modular_reverse_v23 --memory-strategy cls_dual_store `
    --control-mode rca_only --evolution-policy grounding_feedback `
    --dsh-patch (Join-Path $root "configs\dsh.materialize.patch.yml") `
    --dsh-materialize-patch (Join-Path $root "configs\dsh.materialize-8192.patch.yml") `
    --dsh-command-max-chars 1600 --dsh-max-steps 2 --dsh-max-extension 0 `
    --concurrency 1 --read-only-knowledge --disable-candidate-replay `
    --result-dir $run --knowledge-store (Join-Path $root "knowledge\active\current.json") `
    --candidate-registry (Join-Path $root "knowledge\active\candidates.json") --job-name $job
if ($LASTEXITCODE -ne 0) { throw "Smoke runner failed: $LASTEXITCODE" }
$aggregate = Get-Content (Join-Path $run ("aggregate." + $job + ".json")) -Raw | ConvertFrom-Json
if ([int]$aggregate.tasks_found -ne 1) { throw "Smoke did not produce one verifier result" }
Write-Host "End-to-end smoke complete: $run"
Write-Host "Reward is not a package-validity criterion because smoke uses only 2 agent steps."
