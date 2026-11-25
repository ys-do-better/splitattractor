$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "external\Resolve-Runtime.ps1")
$runtime = Get-MetaCoRSIRuntime -Root $root
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "Root       : $root"
Write-Host "Python     : $($runtime.Python)"
Write-Host "Dotenv     : $($runtime.Dotenv) (secret values are not printed)"
Write-Host "Model      : $($runtime.Model)"
Write-Host "Concurrency: $($runtime.Concurrency)"
& $runtime.Python -c "import metacorsi, metacorsi_dsh_agent; print('Python imports: OK')"
if ($LASTEXITCODE -ne 0) { throw "MetaCoRSI import failed" }
docker version --format "Docker server: {{.Server.Version}}"
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop is unavailable" }
& $runtime.Python (Join-Path $root "scripts\audit_cls_k20_k28.py") `
    --knowledge (Join-Path $root "knowledge\active") `
    --output (Join-Path $root "evidence\ENV_RETRIEVAL_AUDIT.json")
if ($LASTEXITCODE -ne 0) { throw "Active brain audit failed" }
Write-Host "Environment check: PASS"
