param(
    [string]$TasksFile = ".\data\deepswe20_ids.txt",
    [int]$MaxAttempts = 5,
    [int]$Concurrency = 5,
    [int]$MaxSteps = 42,
    [int]$MaxExtension = 12,
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$panelRunner = Join-Path $root "RUN_K29_PANEL.ps1"
$sourceTasks = (Resolve-Path -LiteralPath $TasksFile).Path

if (-not $RunRoot) {
    $RunRoot = Join-Path $root ("runs\k29_passat5_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
[System.IO.Directory]::CreateDirectory($RunRoot) | Out-Null

$remaining = @(Get-Content -LiteralPath $sourceTasks | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$allTasks = @($remaining)
$firstPass = @{}
$attempts = @()

for ($attempt = 1; $attempt -le $MaxAttempts -and $remaining.Count -gt 0; $attempt++) {
    $attemptTasks = Join-Path $RunRoot ("attempt_{0}_tasks.txt" -f $attempt)
    [System.IO.File]::WriteAllLines($attemptTasks, [string[]]$remaining)
    $attemptRoot = Join-Path $RunRoot ("attempt_{0}" -f $attempt)

    & $panelRunner `
        -TasksFile $attemptTasks `
        -PanelName ("passat5_a{0}" -f $attempt) `
        -Concurrency ([Math]::Min($Concurrency, $remaining.Count)) `
        -MaxSteps $MaxSteps `
        -MaxExtension $MaxExtension `
        -RunRoot $attemptRoot
    if ($LASTEXITCODE -ne 0) { throw "Pass@5 attempt $attempt failed: $LASTEXITCODE" }

    $aggregatePath = Get-ChildItem -LiteralPath $attemptRoot -Filter "aggregate*.json" | Select-Object -First 1 -ExpandProperty FullName
    if (-not $aggregatePath) { throw "Missing aggregate for attempt $attempt" }
    $aggregate = Get-Content -LiteralPath $aggregatePath -Raw | ConvertFrom-Json
    $attemptResult = [ordered]@{}

    foreach ($property in $aggregate.per_task.PSObject.Properties) {
        $task = $property.Name.Split('/')[-1]
        $reward = [double]$property.Value.reward
        $attemptResult[$task] = $reward
        if ($reward -eq 1.0 -and -not $firstPass.ContainsKey($task)) {
            $firstPass[$task] = $attempt
        }
    }

    $attempts += [ordered]@{
        attempt = $attempt
        tasks = @($remaining)
        rewards = $attemptResult
        aggregate = $aggregatePath
    }
    $remaining = @($remaining | Where-Object { -not $firstPass.ContainsKey($_) })
}

$perTask = [ordered]@{}
foreach ($task in $allTasks) {
    $passed = $firstPass.ContainsKey($task)
    $perTask[$task] = [ordered]@{
        pass_at_5 = $passed
        first_pass_attempt = if ($passed) { [int]$firstPass[$task] } else { $null }
    }
}

$summary = [ordered]@{
    schema_version = 1
    finished_at = [DateTimeOffset]::UtcNow.ToString("o")
    metric = "Pass@5"
    knowledge = "K29-CLS-NearMiss3 (read-only)"
    max_attempts = $MaxAttempts
    tasks = $allTasks.Count
    passed = $firstPass.Count
    pass_at_5 = if ($allTasks.Count) { $firstPass.Count / $allTasks.Count } else { 0 }
    per_task = $perTask
    attempts = $attempts
}
$summaryPath = Join-Path $RunRoot "PASS_AT_5_SUMMARY.json"
[System.IO.File]::WriteAllText($summaryPath, ($summary | ConvertTo-Json -Depth 12))
Write-Host "K29 Pass@5 complete: $summaryPath"
