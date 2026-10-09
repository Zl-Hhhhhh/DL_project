param([switch]$DownloadModels)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { throw 'Run setup.ps1 first.' }
    $referenceRequirements = if (Test-Path -LiteralPath 'requirements-references-lock.txt') { 'requirements-references-lock.txt' } else { 'requirements-references.txt' }
    & uv pip install --python .venv/Scripts/python.exe --no-cache --link-mode copy --build-constraint configs/legacy-build-constraints.txt --constraint requirements-lock.txt -r $referenceRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Reference dependency installation failed.' }
    & ./.venv/Scripts/python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    if ($DownloadModels) {
        & ./.venv/Scripts/python.exe -X utf8 scripts/prepare_instantstyle.py --ranged
        if ($LASTEXITCODE -ne 0) { throw 'Model download failed; rerun to resume verified chunks.' }
    }
} finally { Pop-Location }
