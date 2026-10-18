# Full pipeline: simulate -> load PostgreSQL -> experiment -> media response -> forecast -> recovery -> tests.
# Needs the database: docker compose up -d --wait
# The Monte Carlo study is separate (about 15 min): .\.venv\Scripts\python.exe scripts\06_recovery.py --reps 100
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'
foreach ($s in '01_simulate', '02_load_db', '03_experiment', '04_response', '05_forecast', '06_recovery') {
    Write-Host "== $s" -ForegroundColor Cyan
    & $py "scripts\$s.py"
    if ($LASTEXITCODE) { throw "$s failed" }
}
& $py -m pytest -q tests
