# Copy this file to local.ps1 to override machine-specific paths.
$MetaCoRSIPython = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"
$MetaCoRSIDotenv = "D:\.env"
$MetaCoRSIModel = "DeepSeek-V4.1-Flash-1017"
$MetaCoRSIConcurrency = 5
