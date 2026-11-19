# One-command verification for Re:Play.
# Run from the repository root after activating the Python environment:
#   powershell -ExecutionPolicy Bypass -File scripts/verify.ps1

$ErrorActionPreference = "Stop"

Write-Host "== Backend: ruff =="
Push-Location backend
try {
    python -m ruff check .
    Write-Host "== Backend: pytest =="
    python -m pytest -q
    Write-Host "== 60 deterministic scenarios =="
    python -m noname.evaluation
    Write-Host "== Transparent baseline =="
    python -m noname.benchmark
}
finally {
    Pop-Location
}

Write-Host "== Frontend: production build =="
Push-Location frontend
try {
    npm run build
}
finally {
    Pop-Location
}

Write-Host "All checks passed."
