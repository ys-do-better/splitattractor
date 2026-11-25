function Get-MetaCoRSIRuntime {
    param([Parameter(Mandatory=$true)][string]$Root)

    $vendorRca = Join-Path $Root "vendor\rca-repair"
    $pathParts = @($Root, $vendorRca)
    if ($env:PYTHONPATH) { $pathParts += $env:PYTHONPATH }
    $env:PYTHONPATH = $pathParts -join [System.IO.Path]::PathSeparator

    $localConfig = Join-Path $Root "external\local.ps1"
    if (Test-Path -LiteralPath $localConfig) { . $localConfig }

    $python = $env:METACORSI_PYTHON
    if (-not $python -and $MetaCoRSIPython) { $python = $MetaCoRSIPython }
    if (-not $python) {
        $localPython = Join-Path $Root ".venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $localPython) { $python = $localPython }
    }
    if (-not $python -and (Test-Path -LiteralPath "D:\YYJ_MetaCoRSI\.venv-win\Scripts\python.exe")) {
        $python = "D:\YYJ_MetaCoRSI\.venv-win\Scripts\python.exe"
    }

    $dotenv = $env:METACORSI_DOTENV
    if (-not $dotenv -and $MetaCoRSIDotenv) { $dotenv = $MetaCoRSIDotenv }
    if (-not $dotenv) { $dotenv = "D:\.env" }
    $model = $env:METACORSI_MODEL
    if (-not $model -and $MetaCoRSIModel) { $model = $MetaCoRSIModel }
    if (-not $model) { $model = "DeepSeek-V4.1-Flash-1017" }
    $concurrency = if ($MetaCoRSIConcurrency) { [int]$MetaCoRSIConcurrency } else { 5 }

    if (-not $python -or -not (Test-Path -LiteralPath $python)) {
        throw "Python not found. Run SETUP_LOCAL.ps1 or set METACORSI_PYTHON."
    }
    if (-not (Test-Path -LiteralPath $dotenv)) {
        throw "Provider dotenv not found: $dotenv"
    }
    [pscustomobject]@{
        Python = (Resolve-Path -LiteralPath $python).Path
        Dotenv = (Resolve-Path -LiteralPath $dotenv).Path
        Model = $model
        Concurrency = $concurrency
    }
}
