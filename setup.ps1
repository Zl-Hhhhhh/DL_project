param([string]$Python = 'D:/anaconda/envs/espnet_env/python.exe')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & uv venv .venv --python $Python --no-cache
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
    }
    $requirementsFile = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
    $wheelArgs = @()
    $torchWheel = '.cache/wheels/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
    if (Test-Path -LiteralPath $torchWheel) {
        $wheelHash = (Get-FileHash -LiteralPath $torchWheel -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($wheelHash -ne '9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f') {
            throw 'Cached PyTorch wheel does not match the official SHA256.'
        }
        $wheelArgs = @($torchWheel)
    }
    & uv pip install --python .venv/Scripts/python.exe --no-cache --link-mode copy --index-strategy unsafe-best-match -r $requirementsFile @wheelArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & uv pip install --python .venv/Scripts/python.exe --no-cache --link-mode copy --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
    & ./.venv/Scripts/python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    & ./.venv/Scripts/object-collage.exe doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
} finally { Pop-Location }
