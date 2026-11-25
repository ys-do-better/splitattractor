param(
    [string]$TasksFile = "",
    [string]$RunRoot = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
if (-not $TasksFile) { $TasksFile = Join-Path $root "data\evolution_queue.txt" }
$TasksFile = (Resolve-Path -LiteralPath $TasksFile).Path
$tasks = @(Get-Content -LiteralPath $TasksFile | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') })
if (-not $tasks.Count) { throw "Evolution queue is empty: $TasksFile" }
if (-not $RunRoot) { $RunRoot = Join-Path $root ("runs\evolution_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
[System.IO.Directory]::CreateDirectory($RunRoot) | Out-Null
$working = Join-Path $RunRoot "knowledge"
[System.IO.Directory]::CreateDirectory($working) | Out-Null
Copy-Item -Path (Join-Path $root "knowledge\active\*") -Destination $working -Recurse -Force

$index = 0
foreach ($task in $tasks) {
    $index += 1
    $slug = ($task -replace '[^A-Za-z0-9_.-]', '_')
    $episode = Join-Path $RunRoot (("episode_{0:D2}_" -f $index) + $slug)
    [System.IO.Directory]::CreateDirectory($episode) | Out-Null
    $single = Join-Path $episode "task.txt"
    [System.IO.File]::WriteAllText($single, $task + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    $job = ("evolve_{0:D2}_" -f $index) + $slug
    & $runtime.Python -m metacorsi.harbor_agents.dsh_runner `
        --tasks-file $single --model $runtime.Model --dotenv $runtime.Dotenv `
        --ltm-policy modular_reverse_v23 --memory-strategy cls_dual_store `
        --control-mode rca_only --evolution-policy grounding_feedback `
        --dsh-patch (Join-Path $root "configs\dsh.materialize.patch.yml") `
        --dsh-materialize-patch (Join-Path $root "configs\dsh.materialize-8192.patch.yml") `
        --dsh-command-max-chars 1600 --dsh-max-steps 24 --dsh-max-extension 6 `
        --concurrency 1 --result-dir $episode `
        --knowledge-store (Join-Path $working "current.json") `
        --candidate-registry (Join-Path $working "candidates.json") --job-name $job
    if ($LASTEXITCODE -ne 0) { throw "Evolution episode failed: $task (exit $LASTEXITCODE)" }
    & $runtime.Python (Join-Path $root "scripts\consolidate_active_cls.py") --knowledge $working
    if ($LASTEXITCODE -ne 0) { throw "CLS consolidation failed after: $task" }
}
& $runtime.Python (Join-Path $root "scripts\audit_cls_k20_k28.py") `
    --knowledge $working --output (Join-Path $RunRoot "RETRIEVAL_AUDIT.json")
if ($LASTEXITCODE -ne 0) { throw "Evolved brain retrieval audit failed" }
[System.IO.File]::WriteAllText(
    (Join-Path $RunRoot "EVOLUTION_FINISHED.json"),
    (@{status='complete'; episodes=$tasks.Count; knowledge=$working; finished_at=(Get-Date).ToString('o')} | ConvertTo-Json),
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "Evolution complete. Working brain: $working"
Write-Host "Run RUN_FULL20.ps1 -Knowledge `"$working`" before promotion."
