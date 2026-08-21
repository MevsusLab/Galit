$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }
& $Python -c "import sys, fastapi, uvicorn; assert sys.version_info >= (3, 10); print('GALIT API preflight OK')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m uvicorn api:app --host 127.0.0.1 --port 8000
