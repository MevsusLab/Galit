$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "GALIT preflight: $Python"
& $Python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python release_manifest.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Preflight complete. Evidence: reports/release-manifest.json and reports/release-manifest.md'
